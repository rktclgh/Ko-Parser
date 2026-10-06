"""그림 블록의 이미지 참조(계약 0.3). 이미지 바이트는 트리에 넣지 않고 저장소 자산으로 두며 내용 해시로 가리킨다."""

from typing import Literal

from pydantic import Field

from .base import ContractModel

MAX_FIGURE_SIDE = 4000  # 그림 PNG의 긴 변 상한(화소)
MAX_DOCUMENT_ASSET_BYTES = 512 * 1024 * 1024  # 문서 하나의 그림 바이트 총합 상한(서로 다른 자산마다 한 번 센다)
FigureCategory = Literal["image", "chart"]


class FigureImage(ContractModel):
    """잘라 낸 그림 PNG의 참조. asset은 PNG 바이트의 sha256(저장소 get_asset의 키).
    caption_block_id는 같은 트리의 caption 블록(한 캡션은 그림 하나에만). 내용 해시에는 asset만 들어간다."""

    asset: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mime: Literal["image/png"]
    width_px: int = Field(gt=0, le=MAX_FIGURE_SIDE)
    height_px: int = Field(gt=0, le=MAX_FIGURE_SIDE)
    dpi: int = Field(gt=0)
    category: FigureCategory
    caption_block_id: str | None = Field(default=None, min_length=1)
