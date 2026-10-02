import pytest
from pydantic import ValidationError

from ko_parser.formats.base import ParsedSource, Parser


def test_parsed_source_defaults_and_freeze():
    parsed = ParsedSource(mime="text/markdown", blocks=[{"kind": "paragraph", "text": "가"}])
    assert parsed.pages == () and parsed.blocks[0]["text"] == "가"
    with pytest.raises(ValidationError):
        parsed.mime = "x"
    with pytest.raises(ValidationError):
        ParsedSource(mime="")
    with pytest.raises(ValidationError):
        ParsedSource(mime="text/plain", extra=1)


def test_parser_protocol_is_structural():
    class Dummy:
        mimes = ("text/x-dummy",)
        extensions = (".dummy",)

        def parse(self, data: bytes, name: str) -> ParsedSource:
            return ParsedSource(mime="text/x-dummy")

    parser: Parser = Dummy()
    assert parser.parse(b"", "a.dummy").mime == "text/x-dummy"
