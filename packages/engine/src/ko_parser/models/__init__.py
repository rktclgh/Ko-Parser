"""모델 파일 목록과 찾기(스펙 §5). 가중치는 git에 없고 우리가 따로 올리지도 않는다: models.toml에 적은 업스트림 고정
주소에서 `ko-parser models fetch`로 받는다(파일마다 크기·SHA-256 고정).
찾는 순서: KO_PARSER_MODEL_DIR(직접 둔 파일) → `models fetch`가 받아 둔 사용자 캐시(KO_PARSER_CACHE_DIR 또는
platformdirs 사용자 캐시/models). 두 곳 모두 그 아래 상대 경로(ocr/det.onnx …)가 같다. 실행 중에는 아무것도 내려받지
않는다. 찾은 파일은 resolve()가 크기와 SHA-256을 확인한다(같은 파일은 프로세스 안에서 한 번만 읽는다)."""

import hashlib
import os
import threading
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

from ..errors import ModelError

__all__ = ["CACHE_DIR_ENV", "MANIFEST", "MODEL_DIR_ENV", "ModelError", "ModelFile", "cache_dir", "candidates",
           "fetch_hint", "find", "names", "resolve", "variants"]

MODEL_DIR_ENV = "KO_PARSER_MODEL_DIR"
CACHE_DIR_ENV = "KO_PARSER_CACHE_DIR"
_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class ModelFile:
    """모델 파일 하나. path는 KO_PARSER_MODEL_DIR·캐시·`--to` 폴더 안의 상대 경로(/로 나눈다), sources는 받을 주소
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


def names(selected: Iterable[str] | None = None) -> list[str]:
    """고른 변형(None이면 모두)의 모델 이름, 목록 순서."""
    wanted = None if selected is None else set(selected)
    return [name for name, e in MANIFEST.items() if wanted is None or e.variant in wanted]


def fetch_hint(name: str) -> str:
    """그 모델 파일을 갖추는 방법(오류 문구에 넣는다)."""
    return f"run `ko-parser models fetch {MANIFEST[name].variant}` or set {MODEL_DIR_ENV}"


def cache_dir() -> Path:
    """`models fetch`가 받는 사용자 캐시 폴더(KO_PARSER_CACHE_DIR로 바꿀 수 있다)."""
    env = os.environ.get(CACHE_DIR_ENV)
    return Path(env) if env else Path(user_cache_dir("ko-parser", appauthor=False)) / "models"


def candidates(name: str) -> list[Path]:
    """찾는 순서대로의 후보 경로: KO_PARSER_MODEL_DIR(있으면) → 사용자 캐시. 파일마다 따로 찾는다: 일부만 둔
    KO_PARSER_MODEL_DIR은 없는 파일을 캐시에서 채운다(스펙 §5 순서)."""
    entry = MANIFEST[name]
    env = os.environ.get(MODEL_DIR_ENV)
    roots = [Path(env)] if env else []
    return [root / entry.path for root in (*roots, cache_dir())]


def find(name: str) -> Path | None:
    """처음 있는 후보 파일. 크기·해시는 보지 않는다(available()이 쓴다: 빠르게). 없으면 None."""
    for path in candidates(name):
        if path.is_file():
            return path
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
    """찾은 파일의 크기·SHA-256을 확인해 돌려준다. 못 찾으면, 또는 고정한 파일과 다르면 ModelError(설정 오류)."""
    entry = MANIFEST[name]
    path = find(name)
    if path is None:
        raise ModelError(f"model file {entry.path} not found; {fetch_hint(name)}")
    if not _matches(entry, path):
        raise ModelError(f"model file {path} does not match the pinned size and SHA-256 of {entry.path} (a newer "
                         f"ko-parser may pin different model files); replace it with `ko-parser models fetch "
                         f"{entry.variant}` (add --to <folder> for {MODEL_DIR_ENV})")
    return path
