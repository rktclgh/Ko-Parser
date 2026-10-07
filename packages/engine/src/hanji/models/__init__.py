"""모델 파일 목록과 찾기(스펙 §5). 가중치는 git에 없고 우리가 따로 올리지도 않는다: models.toml에 적은 업스트림 고정
주소에서 `hanji models fetch`로 받는다(파일마다 크기·SHA-256 고정).
찾는 순서: HANJI_MODEL_DIR(직접 둔 파일) → `models fetch`가 받아 둔 사용자 캐시(HANJI_CACHE_DIR 또는
platformdirs 사용자 캐시/models). 두 곳 모두 그 아래 상대 경로(ocr/det.onnx …)가 같다. 실행 중에는 아무것도 내려받지
않는다. 찾은 파일은 resolve()가 크기와 SHA-256을 확인한다(같은 파일은 프로세스 안에서 한 번만 읽는다)."""

import hashlib
import os
import shutil
import threading
import time
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

from ..errors import ModelError

__all__ = ["CACHE_DIR_ENV", "MANIFEST", "MODEL_DIR_ENV", "ModelError", "ModelFile", "cache_dir", "candidates",
           "fetch", "fetch_hint", "find", "names", "resolve", "variants"]

MODEL_DIR_ENV = "HANJI_MODEL_DIR"
CACHE_DIR_ENV = "HANJI_CACHE_DIR"
_CHUNK = 1 << 20
TIMEOUT = 60  # 받기: 소켓 한 번을 기다리는 초
BACKOFF = (2, 4, 8)  # 받기: 잠깐의 오류(HTTP 429·5xx, 연결·시간 초과·끊김)면 이만큼(초) 쉬고 같은 주소를 다시(3번)
STALE_PART = 3600  # 받기: 이보다 오래(초) 된 같은 파일의 조각(죽은 프로세스가 남긴 것)은 지운다


@dataclass(frozen=True, slots=True)
class ModelFile:
    """모델 파일 하나. path는 HANJI_MODEL_DIR·캐시·`--to` 폴더 안의 상대 경로(/로 나눈다), sources는 받을 주소
    (앞에서부터 시도한다)."""

    name: str
    variant: str
    path: str
    size: int
    sha256: str
    sources: tuple[str, ...]


def _load(path: Path) -> dict[str, ModelFile]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {name: ModelFile(name=name, variant=e["variant"], path=e["path"], size=e["size"], sha256=e["sha256"],
                            sources=tuple(e["sources"]))
            for name, e in data["files"].items()}


MANIFEST: dict[str, ModelFile] = _load(Path(__file__).with_name("models.toml"))
_lock = threading.Lock()
_hashes: dict[tuple[str, int, int], str] = {}


def variants() -> list[str]:
    """목록에 있는 변형(ocr·layout …)."""
    return sorted({e.variant for e in MANIFEST.values()})


def _entry(name: str) -> ModelFile:
    """목록의 모델 파일. 목록에 없는 이름이면 ModelError(KeyError가 아니다)."""
    entry = MANIFEST.get(name)
    if entry is None:
        raise ModelError(f"unknown model file {name!r}; known: {', '.join(MANIFEST)}")
    return entry


def names(selected: Iterable[str] | None = None) -> list[str]:
    """고른 변형(None이면 모두)의 모델 이름, 목록 순서. 목록에 없는 변형, 변형 하나를 그냥 문자열로 준 것("ocr"은
    o·c·r로 풀린다)은 ModelError(조용히 아무것도 고르지 않는 대신)."""
    if isinstance(selected, str):
        raise ModelError(f"model variants must be a list such as [{selected!r}], not a string")
    wanted = None if selected is None else set(selected)
    unknown = sorted(wanted - set(variants())) if wanted else []
    if unknown:
        raise ModelError(f"unknown model variant {', '.join(map(repr, unknown))}; known: {', '.join(variants())}")
    return [name for name, e in MANIFEST.items() if wanted is None or e.variant in wanted]


def fetch_hint(name: str) -> str:
    """그 모델 파일을 갖추는 방법(오류 문구에 넣는다)."""
    return f"run `hanji models fetch {_entry(name).variant}` or set {MODEL_DIR_ENV}"


def cache_dir() -> Path:
    """`models fetch`가 받는 사용자 캐시 폴더(HANJI_CACHE_DIR로 바꿀 수 있다, `~`는 홈 폴더로 푼다)."""
    env = os.environ.get(CACHE_DIR_ENV)
    return Path(env).expanduser() if env else Path(user_cache_dir("hanji", appauthor=False)) / "models"


def candidates(name: str) -> list[Path]:
    """찾는 순서대로의 후보 경로: HANJI_MODEL_DIR(있으면) → 사용자 캐시. 파일마다 따로 찾는다: 일부만 둔
    HANJI_MODEL_DIR은 없는 파일을 캐시에서 채운다(스펙 §5 순서). `~`는 홈 폴더로 푼다(.env·Docker ENV는 셸이
    풀지 않는다)."""
    entry = _entry(name)
    env = os.environ.get(MODEL_DIR_ENV)
    roots = [Path(env).expanduser()] if env else []
    return [root / entry.path for root in (*roots, cache_dir())]


def find(name: str) -> Path | None:
    """처음 있는 후보 파일. 크기·해시는 보지 않는다(available()이 쓴다: 빠르게). 없으면 None. 열 수 없는 폴더(권한)는
    없는 것으로 보고 다음 후보로 넘어간다(Python 3.12의 is_file()은 PermissionError를 그대로 낸다)."""
    for path in candidates(name):
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    """파일 해시. (실제 경로, 크기, mtime)이 같으면 프로세스 안에서 기억한 값을 쓴다(130MB 레이아웃 모델도 한 번에
    약 0.05초라 디스크에는 기억하지 않는다)."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _lock:
        known = _hashes.get(key)
    if known is None:
        known = _digest(path)
        with _lock:
            _hashes[key] = known
    return known


def _matches(entry: ModelFile, path: Path) -> bool:
    """크기와 SHA-256이 목록과 같은가(크기가 다르면 읽지 않는다)."""
    return path.stat().st_size == entry.size and _sha256(path) == entry.sha256


def resolve(name: str) -> Path:
    """찾은 파일의 크기·SHA-256을 확인해 돌려준다. 못 찾으면, 읽을 수 없으면(권한, 찾은 뒤 사라짐), 또는 고정한
    파일과 다르면 ModelError(설정 오류)."""
    entry = _entry(name)
    path = find(name)
    if path is None:
        raise ModelError(f"model file {entry.path} not found; {fetch_hint(name)}")
    try:
        matches = _matches(entry, path)
    except OSError as exc:
        raise ModelError(f"model file {path} could not be read ({exc.strerror or exc}); check its permissions, or "
                         f"{fetch_hint(name)}") from exc
    if not matches:
        raise ModelError(f"model file {path} does not match the pinned size and SHA-256 of {entry.path} (a newer "
                         f"hanji may pin different model files); replace it with `hanji models fetch "
                         f"{entry.variant}` (add --to <folder> for {MODEL_DIR_ENV})")
    return path


def _urlopen(url: str):
    """받기 연결(표준 라이브러리 urllib, 리디렉션을 따라간다). 받을 때만 urllib.request를 가져온다(찾기·파싱은 쓰지 않는다).
    테스트가 바꿔 끼운다."""
    import urllib.request

    return urllib.request.urlopen(url, timeout=TIMEOUT)


def _sleep(seconds: float) -> None:
    """다시 받기 전 쉬기(테스트가 바꿔 끼운다)."""
    time.sleep(seconds)


def _transient(exc: BaseException) -> bool:
    """같은 주소를 다시 받을 만한 오류인가: HTTP 429·5xx, 연결 오류, 시간 초과, 받다가 끊김(TLS EOF 포함). 그 밖의 HTTP
    오류(404 등)·인증서 오류 등 TLS 오류·디스크 오류는 다시 받아도 같다. urlopen은 연결 중 TLS 오류를
    URLError(reason=SSLError)로 감싸므로 감싼 오류로 판단한다."""
    import http.client
    import urllib.error

    try:
        import ssl
    except ImportError:  # ssl 없이 빌드한 Python
        ssl = None
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    if isinstance(exc, urllib.error.URLError) and ssl is not None and isinstance(exc.reason, ssl.SSLError):
        exc = exc.reason
    elif isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead)):
        return True
    if ssl is None or not isinstance(exc, ssl.SSLError):
        return False
    if isinstance(exc, (ssl.SSLEOFError, ssl.SSLZeroReturnError)):
        return True
    return "EOF" in str(getattr(exc, "reason", None) or "")


def _expected_length(response, pinned: int) -> int:
    """받을 바이트 수: 응답의 Content-Length(있고 읽히면), 없으면 고정한 크기."""
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    try:
        return int(value) if value is not None else pinned
    except ValueError:
        return pinned


def _download(entry: ModelFile, url: str, part: Path) -> None:
    """url을 part에 받는다. 고정한 크기를 넘으면 그 자리에서 멈추고, 다 받으면 크기·SHA-256을 본다. 다르면 ModelError.
    기대한 길이(Content-Length, 없으면 고정 크기)보다 적게 받고 끝나면 연결이 끊긴 것이라 IncompleteRead(다시 받는다):
    Content-Length가 있으면 http.client는 끊겨도 예외 없이 b""를 준다."""
    import http.client

    digest, size = hashlib.sha256(), 0
    with _urlopen(url) as response, part.open("wb") as f:
        expected = _expected_length(response, entry.size)
        while chunk := response.read(_CHUNK):
            size += len(chunk)
            if size > entry.size:
                raise ModelError(f"more than the pinned {entry.size} bytes")
            digest.update(chunk)
            f.write(chunk)
    if size < expected:
        raise http.client.IncompleteRead(b"", expected - size)
    if size != entry.size or digest.hexdigest() != entry.sha256:
        raise ModelError(f"{size} bytes with SHA-256 {digest.hexdigest()} that do not match the pinned {entry.size} "
                         f"bytes and SHA-256 {entry.sha256}")


def _sweep_parts(target: Path) -> None:
    """같은 파일의 조각(`<이름>.<pid>.<tid>.part`) 중 STALE_PART초보다 오래된 것을 지운다(끊긴 프로세스가 남긴 것).
    지우지 못하면 넘어간다(Windows에서 다른 프로세스가 연 조각). 아직 받는 중인 아주 느린 조각을 지우면 그 받기는
    바꾸기에서 조각이 없음을 보고 다시 받는다(_fetch_one)."""
    now = time.time()
    for old in target.parent.glob(f"{target.name}.*.part"):
        try:
            if now - old.stat().st_mtime > STALE_PART:
                old.unlink()
        except OSError:
            pass


def _fetch_one(entry: ModelFile, target: Path, report: Callable[[ModelFile, str], None] | None) -> None:
    """주소를 목록 순서대로 시도한다. 잠깐의 오류는 같은 주소를 BACKOFF만큼 쉬며 3번 더 시도하고, 그래도 안 되거나
    다시 받아도 같은 오류(404·다른 바이트 등)면 다음 주소. `<이름>.<pid>.<tid>.part`에 받아 맞을 때만 target으로
    이름을 바꾼다(끊기거나 바이트가 달라도 깨진 파일이 남지 않는다). 모두 실패하면 주소마다 이유를 담은 ModelError.
    target을 바꾸지 못하면(Windows에서 다른 프로세스가 연 파일·쓰기 금지) 받기 실패가 아니라 쓰는 중·쓰기 금지라고 알린다."""
    import http.client

    target.parent.mkdir(parents=True, exist_ok=True)
    _sweep_parts(target)
    part = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.part")
    reasons = []
    for url in entry.sources:
        for attempt in range(len(BACKOFF) + 1):
            if report is not None:
                report(entry, url)
            try:
                _download(entry, url, part)
            except ModelError as exc:  # 다른 바이트·너무 큼: 다시 받아도 같다
                reasons.append(f"{url}: {exc}")
                break
            except (OSError, http.client.HTTPException) as exc:  # URLError·HTTPError·끊김·디스크
                if attempt < len(BACKOFF) and _transient(exc):
                    _sleep(BACKOFF[attempt])
                    continue
                reasons.append(f"{url}: {exc}")
                break
            else:
                try:
                    os.replace(part, target)
                except FileNotFoundError as exc:
                    if part.exists():
                        raise
                    # 조각이 사라졌다: 한 시간 넘게 받는 사이 다른 프로세스의 _sweep_parts가 지웠다. 잠깐의 오류처럼 다시
                    if attempt < len(BACKOFF):
                        _sleep(BACKOFF[attempt])
                        continue
                    reasons.append(f"{url}: {exc}")
                    break
                except PermissionError as exc:
                    raise ModelError(f"{target} is in use or not writable and cannot be replaced; close other "
                                     f"hanji processes (or check its permissions) and run `hanji models "
                                     f"fetch` again") from exc
                return
            finally:
                try:  # 지우지 못해도(Windows 백신 잠금) 원래 오류를 가리지 않는다. 남은 조각은 _sweep_parts가 치운다
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
    raise ModelError(f"could not download {entry.path} ({'; '.join(reasons)})")


def _free_space(root: Path) -> int:
    """root(아직 없으면 가장 가까운 있는 위 폴더)가 있는 디스크의 빈 바이트."""
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _fetched(entry: ModelFile, target: Path) -> bool:
    """target이 이미 받은 그 파일인가(기억한 해시 없이 다시 읽는다). 읽지 못하면(권한 등) 아니라고 보고 새로 받는다."""
    try:
        return target.is_file() and target.stat().st_size == entry.size and _digest(target) == entry.sha256
    except OSError:
        return False


def fetch(selected: Iterable[str] | None = None, to: Path | str | None = None,
          report: Callable[[ModelFile, str], None] | None = None) -> list[Path]:
    """고른 변형(None이면 모두)의 파일을 to(None이면 사용자 캐시) 아래 상대 경로로 받는다. 이미 있고 크기·SHA-256이
    맞으면 건너뛰고, 다르면 새로 받아 바꾼다(있는 파일은 기억한 해시를 쓰지 않고 다시 읽는다: 같은 크기·mtime으로 바뀐
    파일도 잡는다). 받을 파일 크기의 합보다 디스크 빈 곳이 적으면 받기 전에 ModelError. report(파일, 주소)는 받기를
    시작할 때마다 부른다(CLI 진행 표시). 반환: 목록 순서의 파일 경로. 실행 중 자동으로 부르지 않는다(`hanji models
    fetch`·CI만)."""
    root = cache_dir() if to is None else Path(to)
    plan = [(MANIFEST[name], root / MANIFEST[name].path) for name in names(selected)]
    todo = [(entry, target) for entry, target in plan if not _fetched(entry, target)]
    need = sum(entry.size for entry, _ in todo)
    if todo and _free_space(root) < need:
        raise ModelError(f"not enough disk space for the model files under {root}: {need:,} bytes needed, "
                         f"{_free_space(root):,} free")
    for entry, target in todo:
        _fetch_one(entry, target, report)
    return [target for _, target in plan]
