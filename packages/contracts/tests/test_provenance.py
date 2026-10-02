import pytest
from pydantic import ValidationError

from ko_parser_contracts.provenance import (
    Attempt, CorrectionSummary, ErrorInfo, GateCheck, GateResult, ProcessingHistory, RegionRecord, Usage,
)

LOC = {"kind": "page", "page": 1, "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1}}


def summary(**kw):
    base = dict(segments_total=4, fixed=1, unchanged=1, unmatched=1, skipped=1,
                chars_replaced=2, chars_dropped=0, unmatched_with_digits=1)
    base.update(kw)
    return CorrectionSummary(**base)


def test_correction_summary_totals():
    assert summary().segments_total == 4
    with pytest.raises(ValidationError, match="segments_total"):
        summary(segments_total=5)
    with pytest.raises(ValidationError, match="unmatched_with_digits"):
        summary(unmatched_with_digits=2)


def test_gate_result_consistency():
    ok = GateCheck(name="char_f1", passed=True, value=0.97, threshold=">=0.90")
    bad = GateCheck(name="numbers_preserved", passed=False)
    assert GateResult(passed=False, checks=[ok, bad]).passed is False
    with pytest.raises(ValidationError, match="passed"):
        GateResult(passed=True, checks=[ok, bad])
    assert GateResult(passed=True, checks=[]).passed is True


def test_usage_allows_unknown_values():
    assert Usage().input_tokens is None
    with pytest.raises(ValidationError):
        Usage(latency_ms=-1)


def test_error_info_excerpt_limit():
    ErrorInfo(code="BAD_OUTPUT", retryable=False, message="x", raw_excerpt="a" * 2000)
    with pytest.raises(ValidationError):
        ErrorInfo(code="BAD_OUTPUT", retryable=False, message="x", raw_excerpt="a" * 2001)
    with pytest.raises(ValidationError):
        ErrorInfo(code="EXPLODED", retryable=False, message="x")


def test_region_record_and_history():
    det = Attempt(layer="det")
    vlm = Attempt(layer="vlm_small", driver_id="omlx", model_id="dots.mocr-8bit", usage=Usage(latency_ms=2300),
                  correction=summary())
    region = RegionRecord(region_id="r1", locator=LOC, kind="table", attempts=[det, vlm], chosen="det",
                          gate=GateResult(passed=False, checks=[GateCheck(name="numbers_preserved", passed=False)]),
                          fallback_reason="unmatched segment with digits")
    h = ProcessingHistory(document_id="doc-1", version=3, regions=[region])
    assert h.schema_version == "0.2"
    with pytest.raises(ValidationError, match="region_id"):
        ProcessingHistory(document_id="doc-1", version=3, regions=[region, region])
    with pytest.raises(ValidationError):
        RegionRecord(region_id="r2", locator=LOC, kind="table", attempts=[], chosen="det")


FAILED_GATE = GateResult(passed=False, checks=[GateCheck(name="numbers_preserved", passed=False)])
PASSED_GATE = GateResult(passed=True, checks=[GateCheck(name="char_f1", passed=True)])


def region(**kw):
    base = dict(region_id="r1", locator=LOC, kind="table", attempts=[Attempt(layer="det")], chosen="det")
    base.update(kw)
    return RegionRecord(**base)


def test_attempt_error_excludes_correction():
    err = ErrorInfo(code="TIMEOUT", retryable=True, message="x")
    Attempt(layer="vlm_small", error=err)
    with pytest.raises(ValidationError, match="must not carry a correction"):
        Attempt(layer="vlm_small", error=err, correction=summary())


def test_region_id_not_empty():
    with pytest.raises(ValidationError):
        region(region_id="")


def test_chosen_layer_needs_successful_attempt():
    with pytest.raises(ValidationError, match="no successful attempt"):
        region(chosen="large", gate=PASSED_GATE)
    timeout = Attempt(layer="vlm_small", error=ErrorInfo(code="TIMEOUT", retryable=True, message="x"))
    with pytest.raises(ValidationError, match="no successful attempt"):
        region(attempts=[Attempt(layer="det"), timeout], chosen="vlm", gate=PASSED_GATE)


def test_non_det_choice_requires_passed_gate():
    attempts = [Attempt(layer="det"), Attempt(layer="vlm_small")]
    with pytest.raises(ValidationError, match="passed gate"):
        region(attempts=attempts, chosen="vlm", gate=FAILED_GATE)
    with pytest.raises(ValidationError, match="passed gate"):
        region(attempts=attempts, chosen="vlm")


def test_fallback_reason_only_with_det():
    attempts = [Attempt(layer="det"), Attempt(layer="vlm_small")]
    with pytest.raises(ValidationError, match="fallback_reason"):
        region(attempts=attempts, chosen="vlm", gate=PASSED_GATE, fallback_reason="x")


def test_valid_choices_accepted():
    attempts = [Attempt(layer="det"), Attempt(layer="vlm_small", correction=summary())]
    assert region(attempts=attempts, chosen="vlm", gate=PASSED_GATE).chosen == "vlm"
    fallback = region(attempts=attempts, chosen="det", gate=FAILED_GATE, fallback_reason="unmatched digits")
    assert fallback.fallback_reason == "unmatched digits"
