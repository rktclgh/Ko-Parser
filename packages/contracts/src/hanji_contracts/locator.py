"""형식별 출처 표기. kind로 구분하는 판별 유니온."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel
from .geometry import BBox


class PageLocator(ContractModel):
    """PDF·이미지: 쪽 번호(1부터) + bbox."""

    kind: Literal["page"] = "page"
    page: int = Field(ge=1)
    bbox: BBox


class FlowLocator(ContractModel):
    """DOCX·HWPX: 섹션 경로 + 문단 순번(0부터)."""

    kind: Literal["flow"] = "flow"
    section_path: tuple[str, ...] = ()
    paragraph_index: int = Field(ge=0)


class SlideLocator(ContractModel):
    """PPTX: 슬라이드 번호(1부터) + 도형 순번(0부터)."""

    kind: Literal["slide"] = "slide"
    slide: int = Field(ge=1)
    shape_index: int = Field(ge=0)


class LinesLocator(ContractModel):
    """MD·TXT: 섹션 경로 + 줄 범위(1부터, 양 끝 포함)."""

    kind: Literal["lines"] = "lines"
    section_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


Locator = Annotated[PageLocator | FlowLocator | SlideLocator | LinesLocator, Field(discriminator="kind")]
