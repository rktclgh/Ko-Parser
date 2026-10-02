import codecs

import pytest

from ko_parser.errors import ParseError
from ko_parser.formats.text import decode_text


def test_utf8_plain_and_bom():
    assert decode_text("가나\n".encode("utf-8"), "a.md") == "가나\n"
    assert decode_text(codecs.BOM_UTF8 + "가나".encode("utf-8"), "a.md") == "가나"


def test_cp949_fallback():
    assert decode_text("한글 문서\n둘째 줄".encode("cp949"), "a.md") == "한글 문서\n둘째 줄"


def test_newlines_normalized():
    assert decode_text(b"a\r\nb\rc\n", "a.md") == "a\nb\nc\n"
    assert decode_text("가\r\n나".encode("cp949"), "a.md") == "가\n나"


@pytest.mark.parametrize("data", [b"\xff\xfe", b"\x80", "가".encode("utf-16")])
def test_undecodable_raises_parse_error(data):
    with pytest.raises(ParseError) as info:
        decode_text(data, "깨진.md")
    assert info.value.location == "깨진.md"
