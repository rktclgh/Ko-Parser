"""`ko-parser models fetch`: 목록의 주소를 차례로 시도해 받고(잠깐의 오류는 쉬었다 다시), 크기·SHA-256이 맞을 때만
캐시(또는 --to 폴더)에 둔다. 네트워크 없이 본다: 받기 연결(models._urlopen)과 쉬기(models._sleep)를 가짜로 바꿔 끼운다."""

import hashlib
import http.client
import io
import os
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error

import pytest

from ko_parser import models
from ko_parser.cli import main
from ko_parser.errors import ModelError

A, B = b"model a bytes", b"model b"
GOOD_A, GOOD_B, MIRROR_A = "https://up.invalid/a.onnx", "https://up.invalid/b.onnx", "https://mirror.invalid/a.onnx"


def entry(name: str, path: str, variant: str, data: bytes, *sources: str) -> models.ModelFile:
    return models.ModelFile(name=name, variant=variant, path=path, size=len(data),
                            sha256=hashlib.sha256(data).hexdigest(), sources=sources)


class Response(io.BytesIO):
    """urlopen 응답 흉내(with 문과 read(n)). cut이면 그만큼 준 뒤 연결이 끊긴다."""

    def __init__(self, data: bytes, cut: int | None = None) -> None:
        super().__init__(data)
        self.cut = cut

    def read(self, n: int = -1) -> bytes:
        if self.cut is not None and self.tell() >= self.cut:
            raise http.client.IncompleteRead(b"")
        return super().read(n if self.cut is None else min(n, self.cut - self.tell()))


@pytest.fixture
def web(monkeypatch, tmp_path):
    """가짜 목록(ocr 하나·layout 하나)·빈 캐시와 가짜 서버. 반환: (주소 → 응답 바이트·예외 또는 그 목록(요청마다 앞에서
    하나씩), 받은 주소 목록, 쉰 초 목록)."""
    served = {GOOD_A: A, GOOD_B: B}
    requests, sleeps = [], []

    def urlopen(url):
        requests.append(url)
        found = served.get(url, urllib.error.URLError("connection refused"))
        if isinstance(found, list):
            found = found.pop(0)
        if isinstance(found, Exception):
            raise found
        return Response(found) if isinstance(found, bytes) else found

    monkeypatch.setattr(models, "_urlopen", urlopen)
    monkeypatch.setattr(models, "_sleep", sleeps.append)
    monkeypatch.setattr(models, "MANIFEST", {"a": entry("a", "ocr/a.onnx", "ocr", A, GOOD_A),
                                             "b": entry("b", "layout/b.onnx", "layout", B, GOOD_B)})
    monkeypatch.setattr(models, "_hashes", {})
    monkeypatch.setenv("KO_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KO_PARSER_MODEL_DIR", raising=False)
    return served, requests, sleeps


def leftovers(root) -> list:
    return sorted(p.name for p in root.rglob("*.part"))


def test_fetch_downloads_into_the_cache_and_resolve_finds_it(web, tmp_path):
    _, requests, _ = web
    paths = models.fetch()
    cache = tmp_path / "cache"
    assert paths == [cache / "ocr" / "a.onnx", cache / "layout" / "b.onnx"] and requests == [GOOD_A, GOOD_B]
    assert paths[0].read_bytes() == A and models.resolve("a") == paths[0] and leftovers(cache) == []


def test_fetch_to_a_folder_makes_a_model_dir(web, tmp_path, monkeypatch):
    """폐쇄망: 연결된 기계에서 --to로 받아 옮긴 폴더를 KO_PARSER_MODEL_DIR로 가리킨다."""
    assert models.fetch(["ocr"], tmp_path / "carry") == [tmp_path / "carry" / "ocr" / "a.onnx"]
    assert not (tmp_path / "cache").exists()
    monkeypatch.setenv("KO_PARSER_MODEL_DIR", str(tmp_path / "carry"))
    assert models.resolve("a") == tmp_path / "carry" / "ocr" / "a.onnx"


def test_fetch_skips_verified_files_and_replaces_a_wrong_one(web, tmp_path):
    _, requests, _ = web
    models.fetch(["ocr"])
    requests.clear()
    assert [p.name for p in models.fetch(["ocr"])] == ["a.onnx"] and requests == []
    (tmp_path / "cache" / "ocr" / "a.onnx").write_bytes(b"model A BYTES")  # 같은 크기, 다른 바이트
    assert models.fetch(["ocr"])[0].read_bytes() == A and requests == [GOOD_A]


def test_fetch_tries_the_next_source(web, monkeypatch, tmp_path):
    """앞 주소가 실패하거나(연결·HTTP 오류) 다른 바이트를 주면 다음 주소. 실패한 주소의 조각은 남지 않는다."""
    served, requests, _ = web
    served[MIRROR_A] = A
    served[GOOD_A] = b"model A BYTES"  # 같은 크기, 다른 해시
    monkeypatch.setitem(models.MANIFEST, "a", entry("a", "ocr/a.onnx", "ocr", A, GOOD_A, MIRROR_A))
    assert models.fetch(["ocr"])[0].read_bytes() == A and requests == [GOOD_A, MIRROR_A]
    served[GOOD_A] = urllib.error.HTTPError(GOOD_A, 404, "Not Found", {}, None)
    (tmp_path / "cache" / "ocr" / "a.onnx").unlink()
    requests.clear()
    assert models.fetch(["ocr"])[0].read_bytes() == A and requests == [GOOD_A, MIRROR_A]
    assert leftovers(tmp_path / "cache") == []


@pytest.mark.parametrize("data, cut", [(b"model A BYTES", None), (A * 1000, None), (A, 5)],
                         ids=["other-bytes", "too-long", "cut-off"])
def test_fetch_keeps_nothing_when_the_bytes_are_wrong(web, tmp_path, data, cut):
    """다른 바이트·너무 긴 응답은 다시 받지 않고, 끊긴 응답은 세 번 더 받아도 끊기면 실패. 어느 쪽도 파일·조각을 남기지
    않는다."""
    served, requests, _ = web
    served[GOOD_A] = [Response(data, cut) for _ in range(4)]
    with pytest.raises(ModelError, match=r"could not download ocr/a\.onnx \(https://up\.invalid/a\.onnx: "):
        models.fetch(["ocr"])
    assert not (tmp_path / "cache" / "ocr" / "a.onnx").exists() and leftovers(tmp_path / "cache") == []
    assert len(requests) == (4 if cut else 1)


def test_fetch_error_names_every_source(web, monkeypatch):
    two = entry("a", "ocr/a.onnx", "ocr", A, "https://x.invalid/1", "https://x.invalid/2")
    monkeypatch.setitem(models.MANIFEST, "a", two)
    with pytest.raises(ModelError, match=r"x\.invalid/1: .*connection refused.*; https://x\.invalid/2: "):
        models.fetch(["ocr"])


def test_cli_models_fetch_prints_paths_without_opening_a_state_file(web, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("KO_PARSER_DB", str(tmp_path / "db" / "state.db"))
    code = main(["models", "fetch", "layout"])
    out, err = capsys.readouterr()
    assert (code, out) == (0, f"{tmp_path / 'cache' / 'layout' / 'b.onnx'}\n")
    assert err == f"ko-parser: downloading layout/b.onnx (7 bytes) from {GOOD_B}\n"
    assert not (tmp_path / "db").exists()
    assert main(["models", "fetch", "--to", str(tmp_path / "carry")]) == 0
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / "carry" / "ocr" / "a.onnx"),
                                                    str(tmp_path / "carry" / "layout" / "b.onnx")]


def test_cli_models_fetch_failure_exit_1_and_unknown_model_exit_2(web, capsys):
    web[0][GOOD_A] = urllib.error.URLError("connection refused")
    code = main(["models", "fetch"])
    out, err = capsys.readouterr()
    assert (code, out) == (1, "") and "ko-parser: could not download ocr/a.onnx" in err and "Traceback" not in err
    assert main(["models", "fetch", "nope"]) == 2


def http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "error", {}, None)


def test_fetch_retries_a_temporary_error_after_a_pause(web):
    """HTTP 429·5xx, 연결 오류, 시간 초과, 끊김은 같은 주소를 2·4·8초 쉬며 다시 받는다."""
    served, requests, sleeps = web
    served[GOOD_A] = [http_error(GOOD_A, 429), TimeoutError("timed out"), Response(A, cut=5), A]
    assert models.fetch(["ocr"])[0].read_bytes() == A
    assert requests == [GOOD_A] * 4 and sleeps == [2, 4, 8]


def test_fetch_moves_to_the_next_source_after_three_retries(web, monkeypatch):
    served, requests, sleeps = web
    served[GOOD_A] = [http_error(GOOD_A, 503)] * 4
    served[MIRROR_A] = A
    monkeypatch.setitem(models.MANIFEST, "a", entry("a", "ocr/a.onnx", "ocr", A, GOOD_A, MIRROR_A))
    assert models.fetch(["ocr"])[0].read_bytes() == A
    assert requests == [GOOD_A] * 4 + [MIRROR_A] and sleeps == [2, 4, 8]


def test_fetch_sweeps_stale_parts_and_names_its_part_by_process_and_thread(web, tmp_path, monkeypatch):
    """끊긴 프로세스가 남긴 오래된 조각은 지우고 새 조각은 둔다(다른 프로세스가 받는 중일 수 있다). 조각 이름은
    `<이름>.<pid>.<tid>.part`."""
    folder = tmp_path / "cache" / "ocr"
    folder.mkdir(parents=True)
    old, fresh = folder / "a.onnx.11.22.part", folder / "a.onnx.33.44.part"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    names = []
    real = models._download
    monkeypatch.setattr(models, "_download", lambda e, url, part: names.append(part.name) or real(e, url, part))
    models.fetch(["ocr"])
    assert not old.exists() and fresh.exists() and names == [f"a.onnx.{os.getpid()}.{threading.get_ident()}.part"]


def test_fetch_checks_free_disk_space_before_downloading(web, monkeypatch):
    _, requests, _ = web
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(100, 95, 5))
    with pytest.raises(ModelError, match=r"not enough disk space for the model files under .*: 20 bytes needed, 5 free"):
        models.fetch()
    assert requests == []


def test_fetch_says_a_model_file_in_use_cannot_be_replaced(web, monkeypatch, tmp_path):
    """Windows: 다른 ko-parser 프로세스가 연 모델 파일은 바꿀 수 없다(PermissionError): 받기 실패가 아니라 쓰는 중."""
    real = os.replace

    def locked(src, dst):
        if str(dst).endswith("a.onnx"):
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(ModelError, match=r"a\.onnx is in use or not writable and cannot be replaced; close other ko-parser processes"):
        models.fetch(["ocr"])
    assert list((tmp_path / "cache").rglob("*.part")) == []


class FakeSocket:
    """http.client.HTTPResponse가 읽는 소켓 흉내(makefile만)."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def makefile(self, mode: str) -> io.BytesIO:
        return io.BytesIO(self.raw)


def real_response(body: bytes, length: int) -> http.client.HTTPResponse:
    """진짜 HTTPResponse. Content-Length가 body보다 길면 연결이 끊긴 것: read는 예외 없이 b""를 준다."""
    response = http.client.HTTPResponse(FakeSocket(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % length + body))
    response.begin()
    return response


def test_a_real_cut_with_content_length_is_retried_then_fails(web, tmp_path):
    """받다가 끊기면(Content-Length보다 적게 받고 b"") 잠깐의 오류로 보고 2·4·8초 쉬며 다시 받는다. 끝내 끊기면 실패,
    파일·조각은 남지 않는다."""
    served, requests, sleeps = web
    served[GOOD_A] = [real_response(A[:5], len(A)) for _ in range(4)]
    with pytest.raises(ModelError, match=r"could not download ocr/a\.onnx \(https://up\.invalid/a\.onnx: IncompleteRead"):
        models.fetch(["ocr"])
    assert requests == [GOOD_A] * 4 and sleeps == [2, 4, 8]
    assert not (tmp_path / "cache" / "ocr" / "a.onnx").exists() and leftovers(tmp_path / "cache") == []


def test_a_real_cut_succeeds_on_the_second_attempt(web):
    served, requests, sleeps = web
    served[GOOD_A] = [real_response(A[:5], len(A)), real_response(A, len(A))]
    assert models.fetch(["ocr"])[0].read_bytes() == A
    assert requests == [GOOD_A] * 2 and sleeps == [2]


def test_a_complete_shorter_file_is_a_mismatch_not_a_cut(web):
    """Content-Length만큼 다 받았는데 고정 크기보다 짧으면(다른 파일) 다시 받지 않는다. 오류에 받은 SHA-256이 있다."""
    served, requests, sleeps = web
    served[GOOD_A] = real_response(A[:5], 5)
    with pytest.raises(ModelError, match=rf"5 bytes with SHA-256 {hashlib.sha256(A[:5]).hexdigest()}"):
        models.fetch(["ocr"])
    assert requests == [GOOD_A] and sleeps == []


def test_mismatch_error_names_the_received_sha256(web):
    served, _, _ = web
    served[GOOD_A] = b"model A BYTES"
    with pytest.raises(ModelError, match=rf"13 bytes with SHA-256 {hashlib.sha256(b'model A BYTES').hexdigest()} "):
        models.fetch(["ocr"])


def test_tls_eof_is_temporary_but_a_certificate_error_is_not():
    assert models._transient(ssl.SSLEOFError(8, "EOF occurred in violation of protocol"))
    assert not models._transient(ssl.SSLCertVerificationError(1, "certificate verify failed"))


def test_a_failing_part_cleanup_does_not_hide_the_download_error(web, monkeypatch):
    """Windows: 백신이 잡은 조각을 지우지 못해도(PermissionError) 원래 오류(받기 실패)가 그대로 나온다."""
    served, _, _ = web
    served[GOOD_A] = b"model A BYTES"
    real = models.Path.unlink

    def locked(self, missing_ok=False):
        if self.name.endswith(".part"):
            raise PermissionError(13, "locked")
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(models.Path, "unlink", locked)
    with pytest.raises(ModelError, match=r"could not download ocr/a\.onnx"):
        models.fetch(["ocr"])


def test_cli_import_loads_no_network_modules():
    """CLI를 가져와도 urllib.request·http.client는 가져오지 않는다(받을 때만, 하위 프로세스에서 본다)."""
    code = ("import sys\n"
            "import ko_parser.cli\n"
            "print(sorted(m for m in ('http.client', 'urllib.request') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
