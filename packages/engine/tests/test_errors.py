import pytest

import hanji
from hanji.errors import (
    DocumentNotFound, HanjiError, ModelError, ParseError, StoreConflict, UnsupportedFormat, VersionNotFound,
    VlmUnavailable,
)


@pytest.mark.parametrize("kind", [UnsupportedFormat, ParseError, DocumentNotFound, VersionNotFound, StoreConflict,
                                  VlmUnavailable, ModelError])
def test_all_errors_share_base(kind):
    assert issubclass(kind, HanjiError)


def test_package_exports_the_model_error():
    """models fetch·resolve의 오류도 다른 오류처럼 패키지에서 바로 가져온다."""
    assert hanji.ModelError is ModelError and "ModelError" in hanji.__all__


def test_parse_error_keeps_reason_and_location():
    err = ParseError("invalid table", "보고서.md:3-5")
    assert (err.reason, err.location) == ("invalid table", "보고서.md:3-5")
    assert str(err) == "invalid table (보고서.md:3-5)"
    assert str(ParseError("empty")) == "empty" and ParseError("empty").location is None
