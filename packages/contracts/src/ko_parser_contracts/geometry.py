"""좌표: 페이지 기준 0~1 정규화, 원점 왼쪽 위, 회전 보정 후."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel

Unit = Annotated[float, Field(ge=0.0, le=1.0)]


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


class PageInfo(ContractModel):
    """width_pt·height_pt는 회전 보정 후 크기, rotation은 원본 PDF의 /Rotate 값."""

    page: int = Field(ge=1)
    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)
    rotation: Literal[0, 90, 180, 270] = 0
    render_dpi: int = Field(gt=0)
