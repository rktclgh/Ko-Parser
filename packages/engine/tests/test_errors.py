import pytest

import ko_parser
from ko_parser.errors import (
    DocumentNotFound, KoParserError, ModelError, ParseError, StoreConflict, UnsupportedFormat, VersionNotFound,
    VlmUnavailable,
)


@pytest.mark.parametrize("kind", [UnsupportedFormat, ParseError, DocumentNotFound, VersionNotFound, StoreConflict,
                                  VlmUnavailable, ModelError])
def test_all_errors_share_base(kind):
    assert issubclass(kind, KoParserError)


def test_package_exports_the_model_error():
    """models fetch·resolve의 오류도 다른 오류처럼 패키지에서 바로 가져온다."""
    assert ko_parser.ModelError is ModelError and "ModelError" in ko_parser.__all__


def test_parse_error_keeps_reason_and_location():
    err = ParseError("invalid table", "보고서.md:3-5")
    assert (err.reason, err.location) == ("invalid table", "보고서.md:3-5")
    assert str(err) == "invalid table (보고서.md:3-5)"
    assert str(ParseError("empty")) == "empty" and ParseError("empty").location is None
