"""처리 이력: 영역마다 어떤 레이어·모델로 처리했고 무엇이 채택됐는지."""

from typing import Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel, VersionedModel
from .document import BlockKind
from .locator import Locator

ErrorCode = Literal["UNAVAILABLE", "TIMEOUT", "RATE_LIMITED", "INPUT_TOO_LARGE", "BAD_OUTPUT", "AUTH"]


class Usage(ContractModel):
    """API가 돌려준 사용량. 모르면 None. 비용은 계산하지 않는다."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)


class ErrorInfo(ContractModel):
    code: ErrorCode
    retryable: bool
    message: str
    raw_excerpt: str | None = Field(default=None, max_length=2000)


class CorrectionSummary(ContractModel):
    """글자 보정 1회의 요약. 단위는 조각(segments)과 글자(chars)."""

    segments_total: int = Field(ge=0)
    fixed: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    unmatched: int = Field(ge=0)
    skipped: int = Field(ge=0)
    chars_replaced: int = Field(ge=0)
    chars_dropped: int = Field(ge=0)
    unmatched_with_digits: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_totals(self) -> Self:
        if self.segments_total != self.fixed + self.unchanged + self.unmatched + self.skipped:
            raise ValueError("segments_total must equal fixed + unchanged + unmatched + skipped")
        if self.unmatched_with_digits > self.unmatched:
            raise ValueError("unmatched_with_digits must be <= unmatched")
        return self


class GateCheck(ContractModel):
    name: str
    passed: bool
    value: float | None = None
    threshold: str | None = None


class GateResult(ContractModel):
    passed: bool
    checks: tuple[GateCheck, ...]

    @model_validator(mode="after")
    def _check_passed(self) -> Self:
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("passed must equal all(check.passed)")
        return self


class Attempt(ContractModel):
    layer: Literal["det", "vlm_small", "vlm_large"]
    driver_id: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    usage: Usage | None = None
    correction: CorrectionSummary | None = None
    error: ErrorInfo | None = None


class RegionRecord(ContractModel):
    region_id: str
    locator: Locator
    kind: BlockKind
    attempts: tuple[Attempt, ...] = Field(min_length=1)
    chosen: Literal["det", "vlm", "large"]
    gate: GateResult | None = None
    fallback_reason: str | None = None


class ProcessingHistory(VersionedModel):
    document_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    regions: tuple[RegionRecord, ...] = ()

    @model_validator(mode="after")
    def _check_regions(self) -> Self:
        ids = [r.region_id for r in self.regions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate region_id")
        return self
