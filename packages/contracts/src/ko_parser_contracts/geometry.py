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


class PageInfo(ContractModel):
    """width_pt·height_pt는 회전 보정 후 크기, rotation은 원본 PDF의 /Rotate 값.

    text_layer는 글자층 판정(scanned·unreliable 쪽에는 블록이 없다), text_stats는 그 근거.
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

    @model_validator(mode="after")
    def _check_text_layer(self) -> Self:
        if self.text_layer != "digital" and self.text_stats is None:
            raise ValueError("scanned and unreliable pages require text_stats")
        return self
