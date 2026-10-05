"""번들 OCR 모델·사전·라이선스가 고정한 파일 그대로인지."""

import hashlib

import pytest

import ko_parser_ocr_models as m

FILES = [m.DET_FILE, m.REC_FILE, m.DICT_FILE, m.LICENSE_FILE]


@pytest.mark.parametrize("name", FILES)
def test_file_matches_its_pinned_sha256(name):
    data = (m.model_dir() / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m.SHA256[name]


def test_model_dir_holds_only_the_pinned_files():
    """방향 판정 모델처럼 쓰지 않는 파일이 섞이지 않는다."""
    names = sorted(p.name for p in m.model_dir().iterdir() if not p.name.startswith("."))  # .DS_Store 등
    assert names == sorted(FILES)
    assert sorted(m.SHA256) == sorted(m.SOURCES) == sorted(FILES)
    assert all(url.startswith("https://") for url in m.SOURCES.values())


def test_license_is_apache_2():
    text = (m.model_dir() / m.LICENSE_FILE).read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text


def test_dict_has_one_symbol_per_line():
    """인식 사전: 줄마다 한 글자, 줄바꿈은 LF(CRLF로 바뀌면 SHA-256도 깨진다)."""
    raw = (m.model_dir() / m.DICT_FILE).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    symbols = raw.decode("utf-8").removesuffix("\n").split("\n")
    assert len(symbols) == 11945 and all(len(s) == 1 for s in symbols)
    assert "가" in symbols and "힣" in symbols
