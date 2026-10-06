"""인식 글자 목록(ocr/charset.py): 설정 파일 inference.yml의 character_dict 블록을 PyYAML 없이 읽고, 이은 바이트를
SHA-256으로 확인한다. 합성 설정으로 보고, 실제 설정 파일은 있을 때만."""

import hashlib
import os

import pytest

from ko_parser import models
from ko_parser.errors import ModelError, OcrUnavailable
from ko_parser.formats.pdf import ocr
from ko_parser.formats.pdf.ocr import charset

# 실제 설정 파일에 나오는 모양: 그대로 쓴 글자(백슬래시 포함), 작은따옴표로 감싼 글자(작은따옴표 자신은 '''',
# 큰따옴표는 '"'). 블록 앞뒤의 다른 키는 읽지 않는다
CONFIG = """Global:
  model_name: korean_PP-OCRv5_mobile_rec
PostProcess:
  name: CTCLabelDecode
  character_dict:
  - ᄀ
  - 가
  - '!'
  - '"'
  - ''''
  - '#'
  - \\
  - '0'
  - 힣
PreProcess:
  transform_ops:
  - DecodeImage:
"""
SYMBOLS = ["ᄀ", "가", "!", '"', "'", "#", "\\", "0", "힣"]
# 설정 파일은 resolve()가 크기·SHA-256을 확인한 뒤에 읽는다: 글자 목록이 다르면 다시 받을 일이 아니라 이 빌드의
# models.toml과 DICT_SHA256이 서로 맞지 않는 것이다
MISMATCH = (r"character list in .*inference\.yml \(9 symbols\) does not match the pinned SHA-256: this ko-parser "
            r"build pins a recognition config \(models\.toml\) and a character list \(charset\.DICT_SHA256\) that "
            r"disagree; reinstall ko-parser or report it")


def require_models(*names: str) -> None:
    """실제 모델 파일이 필요한 테스트: 찾을 수 없으면 건너뛰고, KO_PARSER_CI_REQUIRE_MODELS=1이면 실패한다."""
    missing = [name for name in names if models.find(name) is None]
    if missing:
        if os.environ.get("KO_PARSER_CI_REQUIRE_MODELS") == "1":
            pytest.fail(f"model files not found: {missing}; run `ko-parser models fetch`")
        pytest.skip(f"model files not found: {missing} (ko-parser models fetch)")


def test_reads_the_character_dict_block_with_its_quoting_forms():
    assert charset.read_character_dict(CONFIG) == SYMBOLS
    assert charset.read_character_dict(CONFIG.rstrip("\n").split("PreProcess:")[0].rstrip("\n")) == SYMBOLS  # 파일 끝
    assert charset.read_character_dict("PostProcess:\n  name: CTCLabelDecode\n") == []
    assert charset.read_character_dict(CONFIG.replace("  - 가\n", "  - 가\u2028\n"))[1] == "가\u2028"  # 줄은 LF로만


def test_load_symbols_checks_the_pinned_sha256(tmp_path, monkeypatch):
    config = tmp_path / "inference.yml"
    config.write_text(CONFIG, encoding="utf-8")
    with pytest.raises(ModelError, match=MISMATCH):
        charset.load_symbols(config)
    pinned = hashlib.sha256(("\n".join(SYMBOLS) + "\n").encode("utf-8")).hexdigest()
    monkeypatch.setattr(charset, "DICT_SHA256", pinned)
    assert charset.load_symbols(config) == SYMBOLS


def test_charset_mismatch_while_building_the_reader_is_ocr_unavailable(tmp_path, monkeypatch):
    """읽개를 만들다 글자 목록이 다르면(모델 파일 확인은 지났다) 날 ModelError가 아니라 --no-ocr 안내를 담은
    OcrUnavailable이고 읽개 모듈(onnxruntime)까지 가지 않는다. 실제 모델·onnxruntime 없이 돈다."""
    config = tmp_path / "ocr" / "inference.yml"
    config.parent.mkdir()
    config.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(ocr, "MODULES", ())
    monkeypatch.setattr(ocr, "_reader", None)
    monkeypatch.setattr(models, "find", lambda name: config)
    monkeypatch.setattr(models, "resolve", lambda name: config)
    monkeypatch.setattr(charset, "DICT_SHA256", "0" * 64)
    with pytest.raises(OcrUnavailable, match=MISMATCH + r", or run with --no-ocr") as info:
        ocr.get_reader()
    assert isinstance(info.value.__cause__, ModelError) and ocr._reader is None


def test_real_config_gives_the_pinned_dictionary():
    """공식 인식 설정의 글자 목록 11,945자를 줄마다 이은 바이트가 예전 dict.txt와 같다(SHA-256 a88071c6…)."""
    require_models("ocr-rec-config")
    symbols = charset.load_symbols(models.resolve("ocr-rec-config"))
    assert len(symbols) == 11945 and all(len(s) == 1 for s in symbols) and "가" in symbols and "힣" in symbols
