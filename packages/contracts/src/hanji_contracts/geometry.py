"""좌표: 페이지 기준 0~1 정규화, 원점 왼쪽 위, 회전 보정 후."""

from typing import Annotated, Literal, Self, get_args

from pydantic import ConfigDict, Field, model_validator

from .base import ContractModel

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
TextLayerState = Literal["digital", "scanned", "unreliable"]
_NEEDS_STATS = [state for state in get_args(TextLayerState) if state != "digital"]


class BBox(ContractModel):
    x0: Unit
    y0: Unit
    x1: Unit
    y1: Unit

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("bbox requires x0 < x1 and y0 < y1")
        return self


class TextLayerStats(ContractModel):
    """쪽 판정 근거. 비율의 분모는 공백이 아닌 글자 전체(숨은 글자 포함), 글자가 없으면 0."""

    chars: int = Field(ge=0)  # 보이는 글자 수(공백 제외)
    invisible_ratio: Unit  # 렌더 모드 3(숨은) 글자 비율
    unmapped_ratio: Unit  # 유니코드 0·U+FFFD·매핑 오류 비율
    pua_ratio: Unit  # 사용자 정의 영역(PUA) 글자 비율
    max_image_coverage: Unit  # 가장 큰 그림이 덮는 쪽 면적 비율


class TextCoverage(ContractModel):
    """쪽 하나의 글자 장부. 글자는 PDFium 텍스트 쪽이 돌려준 글자 객체 가운데 공백이 아닌 것이다(너비 0 글리프·같은
    자리 중복 객체처럼 PDFium이 텍스트 쪽에 넣지 않은 것과 쪽 밖 글자는 세지 않는다).
    layer_chars = in_blocks + hidden + replaced, rescued <= in_blocks."""

    layer_chars: int = Field(ge=0)  # 텍스트 레이어 글자 수(숨은 글자 포함. TextLayerStats.chars는 보이는 글자만)
    in_blocks: int = Field(ge=0)  # 블록에 배정된 글자 id 수(블록 글자 길이가 아니다: NFC·끼운 공백·U+FFFD)
    hidden: int = Field(ge=0)  # 렌더 모드 3(숨은) 글자: 쪽 상태와 상관없이 블록에 넣지 않는다
    replaced: int = Field(default=0, ge=0)  # unreliable 쪽을 OCR로 다시 읽어 블록에 넣지 않은 글자
    rescued: int = Field(default=0, ge=0)  # in_blocks 가운데 구조 문단으로 살린 글자

    @model_validator(mode="after")
    def _check_ledger(self) -> Self:
        if self.layer_chars != self.in_blocks + self.hidden + self.replaced:
            raise ValueError("layer_chars must equal in_blocks + hidden + replaced")
        if self.rescued > self.in_blocks:
            raise ValueError("rescued must be <= in_blocks")
        return self


class PageInfo(ContractModel):
    """width_pt·height_pt는 회전 보정 후 크기, rotation은 원본 PDF의 /Rotate 값.

    text_layer는 글자층 판정(scanned 쪽은 보이는 글자만 블록이 되고, unreliable 쪽 블록은 깨진 글자층에서 나와 믿기
    어렵다), text_stats는 그 근거. coverage는 글자 장부(텍스트 레이어가 있는 형식, 장부가 맞지 않으면 None).
    """

    # 아래 검증기 규칙을 JSON Schema에도 싣는다(digital이 아니면 text_stats가 null이 아닌 값으로 있어야 한다)
    model_config = ConfigDict(json_schema_extra={
        "if": {"properties": {"text_layer": {"enum": _NEEDS_STATS}}, "required": ["text_layer"]},
        "then": {"properties": {"text_stats": {"not": {"type": "null"}}}, "required": ["text_stats"]},
    })

    page: int = Field(ge=1)
    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)
    rotation: Literal[0, 90, 180, 270] = 0
    render_dpi: int = Field(gt=0)
    text_layer: TextLayerState = "digital"
    text_stats: TextLayerStats | None = None
    coverage: TextCoverage | None = None

    @model_validator(mode="after")
    def _check_text_layer(self) -> Self:
        if self.text_layer != "digital" and self.text_stats is None:
            raise ValueError("scanned and unreliable pages require text_stats")
        if self.coverage is not None:
            if self.coverage.replaced and self.text_layer != "unreliable":
                raise ValueError("coverage.replaced requires an unreliable page")
            stats = self.text_stats
            if stats is not None and self.coverage.layer_chars != stats.chars + self.coverage.hidden:
                raise ValueError("coverage.layer_chars must equal text_stats.chars + coverage.hidden")
        return self
