"""모델 파일 목록과 찾기(ko_parser.models): 목록이 업스트림 고정 주소·크기·SHA-256을 갖는지, 찾는 순서(환경 변수 →
캐시), 크기·SHA-256 확인과 해시 기억. 가짜 목록·가짜 파일로 돈다(네트워크·실제 모델 없음)."""

import hashlib
import os
import re
from pathlib import Path

import pytest

from ko_parser import models
from ko_parser.errors import KoParserError, ModelError

DATA = b"fake model bytes"
ENTRY = models.ModelFile(name="m", variant="ocr", path="ocr/m.onnx", size=len(DATA),
                         sha256=hashlib.sha256(DATA).hexdigest(), sources=("https://example.invalid/m.onnx",))


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """가짜 목록 하나, 빈 KO_PARSER_MODEL_DIR·캐시 폴더. 반환: (환경 변수 폴더, 캐시 폴더)."""
    monkeypatch.setattr(models, "MANIFEST", {"m": ENTRY})
    monkeypatch.setattr(models, "_hashes", {})
    env, cache = tmp_path / "env", tmp_path / "cache"
    monkeypatch.setenv("KO_PARSER_MODEL_DIR", str(env))
    monkeypatch.setenv("KO_PARSER_CACHE_DIR", str(cache))
    return env, cache


def put(root: Path, rel: str, data: bytes = DATA) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_manifest_pins_every_file_to_an_upstream_address():
    """모든 모델 파일은 Hugging Face 고정 커밋 주소(40자)에서 받는다. OCR은 PaddlePaddle 공식 ONNX."""
    assert "ocr" in models.variants() and models.names(["ocr"]) == ["ocr-det", "ocr-rec", "ocr-rec-config"]
    for entry in models.MANIFEST.values():
        assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256) and entry.size > 0
        assert entry.path.startswith(entry.variant + "/") and entry.sources
        assert all(re.fullmatch(r"https://huggingface\.co/PaddlePaddle/[\w.-]+/resolve/[0-9a-f]{40}/inference\.(onnx|yml)",
                                url) for url in entry.sources), entry.sources
    rec = models.MANIFEST["ocr-rec"]
    assert (rec.size, rec.sha256) == (13418787, "92f0b7785e64fc9090106a241cf4c1eb97472824558272751b88a2a4476d3a08")
    assert rec.sources == ("https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx/resolve/"
                           "5c6f574b8e2230adf4287b33e736d71b9fabd28e/inference.onnx",)
    assert issubclass(ModelError, KoParserError)


def test_model_dir_comes_before_the_cache(fake, monkeypatch):
    env, cache = fake
    assert models.find("m") is None
    put(cache, "ocr/m.onnx")
    assert models.find("m") == cache / "ocr" / "m.onnx"
    put(env, "ocr/m.onnx")
    assert models.find("m") == env / "ocr" / "m.onnx"
    monkeypatch.delenv("KO_PARSER_MODEL_DIR")
    assert models.candidates("m") == [cache / "ocr" / "m.onnx"] and models.find("m") == cache / "ocr" / "m.onnx"


def test_each_file_falls_back_to_the_cache_on_its_own(fake, monkeypatch):
    """일부 파일만 둔 KO_PARSER_MODEL_DIR: 있는 파일은 거기서, 없는 파일은 캐시에서(파일마다 스펙 §5 순서)."""
    env, cache = fake
    other = models.ModelFile(name="n", variant="ocr", path="ocr/n.onnx", size=len(DATA), sha256=ENTRY.sha256,
                             sources=("https://example.invalid/n.onnx",))
    monkeypatch.setattr(models, "MANIFEST", {"m": ENTRY, "n": other})
    put(env, "ocr/m.onnx")
    put(cache, "ocr/n.onnx")
    assert (models.resolve("m"), models.resolve("n")) == (env / "ocr" / "m.onnx", cache / "ocr" / "n.onnx")


def test_cache_dir_defaults_to_the_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("KO_PARSER_CACHE_DIR", raising=False)
    default = models.cache_dir()
    assert default.name == "models" and "ko-parser" in default.parts
    monkeypatch.setenv("KO_PARSER_CACHE_DIR", str(tmp_path))
    assert models.cache_dir() == tmp_path


def test_resolve_checks_size_and_sha256_and_remembers_the_hash(fake, monkeypatch):
    env, _ = fake
    path = put(env, "ocr/m.onnx")
    assert models.resolve("m") == path
    reads = []
    real = models._digest
    monkeypatch.setattr(models, "_digest", lambda p: reads.append(p) or real(p))
    assert models.resolve("m") == path and reads == []  # 같은 파일(경로·크기·mtime)은 다시 읽지 않는다
    put(env, "ocr/m.onnx", b"fake model BYTES")  # 같은 크기, 다른 바이트
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))  # mtime이 바뀌면 다시 읽는다
    with pytest.raises(ModelError):
        models.resolve("m")
    assert reads == [path]


def test_resolve_rejects_a_found_file_with_other_bytes(fake):
    """찾기(available이 쓴다)는 되지만 resolve가 크게 알린다: 깨진 파일은 조용히 물러나지 않는다."""
    env, _ = fake
    put(env, "ocr/m.onnx", b"x" * len(DATA))  # 크기는 같고 바이트가 다르다
    assert models.find("m") is not None
    with pytest.raises(ModelError, match=r"does not match the pinned size and SHA-256 of ocr/m\.onnx \(a newer "
                                         r"ko-parser may pin different model files\); .*ko-parser models fetch ocr"):
        models.resolve("m")
    put(env, "ocr/m.onnx", b"short")
    with pytest.raises(ModelError, match="does not match"):
        models.resolve("m")


def test_resolve_without_any_file_names_fetch_and_the_env_var(fake):
    with pytest.raises(ModelError, match=r"ocr/m\.onnx not found; run `ko-parser models fetch ocr` or set "
                                         r"KO_PARSER_MODEL_DIR"):
        models.resolve("m")


def test_names_follow_the_list_order_and_the_chosen_variants(monkeypatch):
    other = models.ModelFile(name="l", variant="layout", path="layout/l.onnx", size=1, sha256="0" * 64,
                             sources=("https://example.invalid/l.onnx",))
    monkeypatch.setattr(models, "MANIFEST", {"l": other, "m": ENTRY})
    assert models.variants() == ["layout", "ocr"] and models.names() == ["l", "m"]
    assert models.names(["ocr"]) == ["m"] and models.names([]) == []
    assert models.fetch_hint("l") == "run `ko-parser models fetch layout` or set KO_PARSER_MODEL_DIR"
