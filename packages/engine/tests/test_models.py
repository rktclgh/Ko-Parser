"""모델 파일 목록과 찾기(hanji.models): 목록이 업스트림 고정 주소·크기·SHA-256을 갖는지, 찾는 순서(환경 변수 →
캐시), 크기·SHA-256 확인과 해시 기억. 가짜 목록·가짜 파일로 돈다(네트워크·실제 모델 없음)."""

import hashlib
import importlib.metadata
import os
import re
import sys
from pathlib import Path

import pytest

from hanji import models
from hanji.errors import HanjiError, ModelError

DATA = b"fake model bytes"
ENTRY = models.ModelFile(name="m", variant="ocr", path="ocr/m.onnx", size=len(DATA),
                         sha256=hashlib.sha256(DATA).hexdigest(), sources=("https://example.invalid/m.onnx",))


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """가짜 목록 하나, 빈 HANJI_MODEL_DIR·캐시 폴더. 반환: (환경 변수 폴더, 캐시 폴더)."""
    monkeypatch.setattr(models, "MANIFEST", {"m": ENTRY})
    monkeypatch.setattr(models, "_hashes", {})
    env, cache = tmp_path / "env", tmp_path / "cache"
    monkeypatch.setenv("HANJI_MODEL_DIR", str(env))
    monkeypatch.setenv("HANJI_CACHE_DIR", str(cache))
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
    assert issubclass(ModelError, HanjiError)


def test_model_dir_comes_before_the_cache(fake, monkeypatch):
    env, cache = fake
    assert models.find("m") is None
    put(cache, "ocr/m.onnx")
    assert models.find("m") == cache / "ocr" / "m.onnx"
    put(env, "ocr/m.onnx")
    assert models.find("m") == env / "ocr" / "m.onnx"
    monkeypatch.delenv("HANJI_MODEL_DIR")
    assert models.candidates("m") == [cache / "ocr" / "m.onnx"] and models.find("m") == cache / "ocr" / "m.onnx"


def test_each_file_falls_back_to_the_cache_on_its_own(fake, monkeypatch):
    """일부 파일만 둔 HANJI_MODEL_DIR: 있는 파일은 거기서, 없는 파일은 캐시에서(파일마다 스펙 §5 순서)."""
    env, cache = fake
    other = models.ModelFile(name="n", variant="ocr", path="ocr/n.onnx", size=len(DATA), sha256=ENTRY.sha256,
                             sources=("https://example.invalid/n.onnx",))
    monkeypatch.setattr(models, "MANIFEST", {"m": ENTRY, "n": other})
    put(env, "ocr/m.onnx")
    put(cache, "ocr/n.onnx")
    assert (models.resolve("m"), models.resolve("n")) == (env / "ocr" / "m.onnx", cache / "ocr" / "n.onnx")


def test_cache_dir_defaults_to_the_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("HANJI_CACHE_DIR", raising=False)
    default = models.cache_dir()
    assert default.name == "models" and "hanji" in default.parts
    monkeypatch.setenv("HANJI_CACHE_DIR", str(tmp_path))
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
                                         r"hanji may pin different model files\); .*hanji models fetch ocr"):
        models.resolve("m")
    put(env, "ocr/m.onnx", b"short")
    with pytest.raises(ModelError, match="does not match"):
        models.resolve("m")


def test_resolve_without_any_file_names_fetch_and_the_env_var(fake):
    with pytest.raises(ModelError, match=r"ocr/m\.onnx not found; run `hanji models fetch ocr` or set "
                                         r"HANJI_MODEL_DIR"):
        models.resolve("m")


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: -1)() == 0,
                    reason="chmod 000 does not stop reading on Windows or as root")
def test_resolve_reports_an_unreadable_file_as_a_model_error(fake):
    """찾았지만 읽을 수 없는 파일(권한): 날 PermissionError가 아니라 경로와 받기 안내를 담은 ModelError."""
    env, _ = fake
    path = put(env, "ocr/m.onnx")
    path.chmod(0)
    try:
        with pytest.raises(ModelError, match=r"model file .*m\.onnx could not be read \(Permission denied\); check its "
                                             r"permissions, or run `hanji models fetch ocr` or set "
                                             r"HANJI_MODEL_DIR$") as info:
            models.resolve("m")
    finally:
        path.chmod(0o644)
    assert isinstance(info.value.__cause__, PermissionError)


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: -1)() == 0,
                    reason="chmod 000 does not stop reading on Windows or as root")
def test_find_skips_a_folder_it_cannot_open(fake):
    """열 수 없는 모델 폴더(권한)는 날 PermissionError가 아니라 '없음': 다음 후보(캐시)로 넘어가고, 어디에도 없으면
    None(find는 '보이는가'만 본다)."""
    env, cache = fake
    put(env, "ocr/m.onnx")
    (env / "ocr").chmod(0)
    try:
        assert models.find("m") is None
        with pytest.raises(ModelError, match=r"ocr/m\.onnx not found; run `hanji models fetch ocr`"):
            models.resolve("m")
        put(cache, "ocr/m.onnx")
        assert models.find("m") == cache / "ocr" / "m.onnx" and models.resolve("m") == cache / "ocr" / "m.onnx"
    finally:
        (env / "ocr").chmod(0o755)


def test_resolve_reports_a_file_that_vanished_after_find_as_a_model_error(fake, monkeypatch):
    """찾은 뒤 사라진 파일(다른 프로세스가 지움)도 ModelError. 어느 OS에서나 돈다."""
    env, _ = fake
    monkeypatch.setattr(models, "find", lambda name: env / "ocr" / "gone.onnx")
    with pytest.raises(ModelError, match=r"gone\.onnx could not be read \(.+\); .*models fetch ocr`") as info:
        models.resolve("m")
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_unknown_names_and_variants_are_model_errors(fake):
    """목록에 없는 이름·변형은 KeyError나 빈 결과가 아니라 ModelError. 변형 하나를 문자열로 주면("ocr" → o·c·r)
    조용히 아무것도 고르지 않는 대신 알린다."""
    for call in (lambda: models.resolve("nope"), lambda: models.find("nope"), lambda: models.candidates("nope"),
                 lambda: models.fetch_hint("nope")):
        with pytest.raises(ModelError, match=r"unknown model file 'nope'; known: m"):
            call()
    for call in (lambda: models.names(["layout"]), lambda: models.fetch(["layout"]),
                 lambda: models.names(["ocr", "nope"])):
        with pytest.raises(ModelError, match=r"unknown model variant"):
            call()
    with pytest.raises(ModelError, match=r"unknown model variant 'layout', 'nope'; known: ocr"):
        models.names(["nope", "layout"])
    for call in (lambda: models.names("ocr"), lambda: models.fetch("ocr")):
        with pytest.raises(ModelError, match=r"model variants must be a list such as \['ocr'\], not a string"):
            call()


def test_model_and_cache_dirs_expand_the_home_folder(fake, monkeypatch, tmp_path):
    """.env·Docker ENV·systemd처럼 셸을 거치지 않은 `~`도 홈 폴더로 푼다."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows의 expanduser
    monkeypatch.setenv("HANJI_MODEL_DIR", "~/env")
    monkeypatch.setenv("HANJI_CACHE_DIR", "~/cache")
    assert models.cache_dir() == tmp_path / "cache"
    assert models.candidates("m") == [tmp_path / "env" / "ocr" / "m.onnx", tmp_path / "cache" / "ocr" / "m.onnx"]
    put(tmp_path / "env", "ocr/m.onnx")
    assert models.find("m") == tmp_path / "env" / "ocr" / "m.onnx" and models.resolve("m") == models.find("m")


def test_names_follow_the_list_order_and_the_chosen_variants(monkeypatch):
    other = models.ModelFile(name="l", variant="layout", path="layout/l.onnx", size=1, sha256="0" * 64,
                             sources=("https://example.invalid/l.onnx",))
    monkeypatch.setattr(models, "MANIFEST", {"l": other, "m": ENTRY})
    assert models.variants() == ["layout", "ocr"] and models.names() == ["l", "m"]
    assert models.names(["ocr"]) == ["m"] and models.names([]) == []
    assert models.fetch_hint("l") == "run `hanji models fetch layout` or set HANJI_MODEL_DIR"


def require_models(*names: str) -> None:
    """실제 모델 파일이 필요한 테스트: 찾을 수 없으면 건너뛰고, HANJI_CI_REQUIRE_MODELS=1이면 실패한다."""
    missing = [name for name in names if models.find(name) is None]
    if missing:
        if os.environ.get("HANJI_CI_REQUIRE_MODELS") == "1":
            pytest.fail(f"model files not found: {missing}; run `hanji models fetch`")
        pytest.skip(f"model files not found: {missing} (hanji models fetch)")


@pytest.mark.parametrize("name", list(models.MANIFEST))
def test_found_model_file_is_the_pinned_one(name):
    require_models(name)
    path = models.resolve(name)  # 크기·SHA-256 확인
    assert path.stat().st_size == models.MANIFEST[name].size


def test_model_license_and_notice_ship_with_the_engine():
    """모델 파일은 업스트림에서 받지만 출처·라이선스 고지는 엔진 패키지에 둔다(목록의 파일마다 고지가 있다)."""
    root = Path(models.__file__).parent
    text = (root / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    # 고지 바이트 그대로(.gitattributes -text: Windows autocrlf 체크아웃도 바꾸지 않는다)
    assert hashlib.sha256((root / "LICENSE").read_bytes()).hexdigest() == (
        "3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e")
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    assert all(entry.path in notice for entry in models.MANIFEST.values())
    assert "PaddlePaddle/PP-OCRv5_mobile_det_onnx (commit e6f4fa85f00e168c862bc462aebca69eef9b3d3d)" in notice
    assert "PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx (commit 5c6f574b8e2230adf4287b33e736d71b9fabd28e)" in notice
    assert "PaddlePaddle/PP-DocLayout_plus-L_onnx (commit feb74619326f634e0e883218598096a3733ad9f7)" in notice


def test_engine_extras_are_runtime_dependencies_only():
    """모델 패키지는 없다: extra는 런타임 의존성만 깔고 모델 파일은 `hanji models fetch`가 받는다."""
    requires = importlib.metadata.requires("hanji")
    ocr = sorted(r.split(";")[0] for r in requires if "extra == 'ocr'" in r)
    assert ocr == ["numpy<3,>=1.26", "onnxruntime<2,>=1.20", "pyclipper<2,>=1.3"]
    layout = sorted(r.split(";")[0] for r in requires if "extra == 'layout'" in r)
    assert layout == ["numpy<3,>=1.26", "onnxruntime<2,>=1.20"]
    assert sorted(r.split(";")[0] for r in requires if "extra == 'all'" in r) == [  # 빌드가 fonts·ocr·layout을 펼친다
        "hanji-fonts<0.2,>=0.1", "numpy<3,>=1.26", "onnxruntime<2,>=1.20", "pyclipper<2,>=1.3"]
    assert not [r for r in requires if "models" in r]


def test_layout_files_are_the_official_paddle_onnx():
    """레이아웃 모델은 PaddlePaddle 공식 ONNX(fp32)와 설정을 Hugging Face 커밋으로 고정해 받는다(변환·fp16 없음)."""
    base = ("https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L_onnx/resolve/"
            "feb74619326f634e0e883218598096a3733ad9f7/")
    assert models.names(["layout"]) == ["layout", "layout-config"]
    model, config = models.MANIFEST["layout"], models.MANIFEST["layout-config"]
    assert (model.path, model.size, model.sources) == ("layout/inference.onnx", 129736329, (base + "inference.onnx",))
    assert model.sha256 == "77afb2caa74dd13240d087d2eced91d7fcd2caebd16006a0a66162fc8707ff0e"
    assert (config.path, config.size, config.sources) == ("layout/inference.yml", 1838, (base + "inference.yml",))
    assert config.sha256 == "d60f782a16f96afb27e8280399899a94c3e9ffc694ffb2f913ea00af1c522f1e"
