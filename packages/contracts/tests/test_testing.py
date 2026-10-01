import asyncio

import pytest

from ko_parser_contracts.geometry import BBox
from ko_parser_contracts.provenance import ErrorInfo
from ko_parser_contracts.testing import (
    Recording, RecordingMeta, ReplayDriver, ReplayMiss, ScriptedDriver, request_fingerprint,
)
from ko_parser_contracts.vlm import ImagePayload, PageRef, VlmDriver, VlmError, VlmRequest, VlmResult

BOX = BBox(x0=0, y0=0, x1=1, y1=1)


def req(request_id="req-1", png=b"img-a", anchor=None, hints=("ko",)) -> VlmRequest:
    return VlmRequest(request_id=request_id, task="REGION_TABLE", image=ImagePayload.from_png(png, BOX, 144),
                      region_id="r1", page_ref=PageRef(document_id="d", page=1), anchor_text=anchor,
                      language_hints=hints)


def result(request_id="recorded") -> VlmResult:
    return VlmResult(request_id=request_id, blocks=[], raw_text="<table></table>", output_format="html",
                     driver_id="omlx", model_id="dots.mocr-8bit")


META = RecordingMeta(model_id="dots.mocr-8bit", profile_id="dots_ocr", prompt="Extract the text content from this image.",
                     runtime="oMLX", recorded_at="2026-10-02T00:00:00+09:00")


def test_fingerprint_ignores_request_id_but_not_content():
    base = request_fingerprint(req())
    assert base == request_fingerprint(req(request_id="other"))
    assert base == request_fingerprint(req(hints=("ko",)))
    assert base != request_fingerprint(req(png=b"img-b"))
    assert base != request_fingerprint(req(anchor="텍스트 레이어"))
    assert request_fingerprint(req(hints=("en", "ko"))) == request_fingerprint(req(hints=("ko", "en")))


def test_replay_hit_rewrites_request_id():
    rec = Recording(fingerprint=request_fingerprint(req()), result=result(), meta=META)
    driver = ReplayDriver([rec])
    assert isinstance(driver, VlmDriver)
    out = asyncio.run(driver.run(req(request_id="req-now")))
    assert out.request_id == "req-now" and out.raw_text == "<table></table>"


def test_replay_miss_is_a_test_error_not_vlm_error():
    driver = ReplayDriver([])
    with pytest.raises(ReplayMiss):
        asyncio.run(driver.run(req()))
    assert not issubclass(ReplayMiss, VlmError)


def test_replay_rejects_duplicate_fingerprints():
    rec = Recording(fingerprint=request_fingerprint(req()), result=result(), meta=META)
    with pytest.raises(ValueError, match="duplicate"):
        ReplayDriver([rec, rec])


def test_replay_from_dir(tmp_path):
    rec = Recording(fingerprint=request_fingerprint(req()), result=result(), meta=META)
    (tmp_path / "a.json").write_text(rec.model_dump_json(), encoding="utf-8")
    driver = ReplayDriver.from_dir(tmp_path)
    assert asyncio.run(driver.run(req())).model_id == "dots.mocr-8bit"


def test_scripted_driver_sequence():
    timeout = ErrorInfo(code="TIMEOUT", retryable=True, message="slow")
    driver = ScriptedDriver([timeout, result()])
    with pytest.raises(VlmError) as exc:
        asyncio.run(driver.run(req(request_id="a")))
    assert exc.value.info.code == "TIMEOUT"
    assert asyncio.run(driver.run(req(request_id="b"))).request_id == "b"
    assert [r.request_id for r in driver.calls] == ["a", "b"]
    with pytest.raises(AssertionError, match="exhausted"):
        asyncio.run(driver.run(req(request_id="c")))


def test_drivers_report_capabilities_and_health():
    for driver in (ReplayDriver([]), ScriptedDriver([])):
        assert "REGION_TABLE" in driver.capabilities().tasks
        assert asyncio.run(driver.health()) is True
