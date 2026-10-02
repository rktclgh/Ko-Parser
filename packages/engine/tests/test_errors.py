import pytest

from ko_parser.errors import (
    DocumentNotFound, KoParserError, ParseError, StoreConflict, UnsupportedFormat, VersionNotFound, VlmUnavailable,
)


@pytest.mark.parametrize("kind", [UnsupportedFormat, ParseError, DocumentNotFound, VersionNotFound, StoreConflict,
                                  VlmUnavailable])
def test_all_errors_share_base(kind):
    assert issubclass(kind, KoParserError)


def test_parse_error_keeps_reason_and_location():
    err = ParseError("invalid table", "보고서.md:3-5")
    assert (err.reason, err.location) == ("invalid table", "보고서.md:3-5")
    assert str(err) == "invalid table (보고서.md:3-5)"
    assert str(ParseError("empty")) == "empty" and ParseError("empty").location is None
