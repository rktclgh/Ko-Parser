"""VLM 계약. VLM이 주는 종류·좌표·구조는 제안이며 최종 확정은 커널이 한다."""

import base64
import binascii
import hashlib
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, PlainSerializer, PlainValidator, WithJsonSchema, model_validator

from .base import ContractModel, VersionedModel
from .document import BlockKind, check_kind_fields
from .geometry import BBox
from .provenance import ErrorInfo, Usage
from .table import Table

Task = Literal["REGION_TABLE", "REGION_FIGURE", "REGION_TEXT", "PAGE_FULL"]


def _decode_png(value: object) -> bytes:
    """파이썬에서는 원시 bytes, JSON에서는 표준 base64 문자열을 받는다."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError("png must be standard base64") from exc
    raise ValueError("png must be bytes or a base64 string")


PngBytes = Annotated[
    bytes,
    PlainValidator(_decode_png),
    PlainSerializer(lambda b: base64.b64encode(b).decode("ascii"), return_type=str, when_used="json"),
    WithJsonSchema({"type": "string", "contentEncoding": "base64"}),
]


class ImagePayload(ContractModel):
    """커널 렌더러가 만든 PNG. 같은 이미지는 같은 바이트·같은 해시."""

    png: PngBytes
    page_bbox: BBox
    dpi: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check_hash(self) -> Self:
        if hashlib.sha256(self.png).hexdigest() != self.sha256:
            raise ValueError("sha256 does not match png bytes")
        return self

    @classmethod
    def from_png(cls, png: bytes, page_bbox: BBox, dpi: int) -> "ImagePayload":
        return cls(png=png, page_bbox=page_bbox, dpi=dpi, sha256=hashlib.sha256(png).hexdigest())


class PageRef(ContractModel):
    document_id: str = Field(min_length=1)
    page: int = Field(ge=1)


class VlmRequest(VersionedModel):
    request_id: str = Field(min_length=1)
    task: Task
    image: ImagePayload
    region_id: str | None = None
    page_ref: PageRef
    anchor_text: str | None = None
    language_hints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_region(self) -> Self:
        if (self.task == "PAGE_FULL") != (self.region_id is None):
            raise ValueError("region_id must be None if and only if task == 'PAGE_FULL'")
        return self


class VlmBlock(ContractModel):
    """제안 블록. bbox는 요청 이미지 기준 0~1이고 페이지 좌표 변환은 커널이 한다."""

    kind: BlockKind
    text: str = ""
    table: Table | None = None
    level: int | None = Field(default=None, ge=1, le=6)
    bbox: BBox | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        check_kind_fields(self.kind, self.table, self.level)
        return self


class VlmResult(VersionedModel):
    """표는 엄격한 Table만. 정규화하지 못하면 결과 대신 VlmError(BAD_OUTPUT)."""

    request_id: str = Field(min_length=1)
    blocks: tuple[VlmBlock, ...]
    raw_text: str
    output_format: Literal["html", "otsl", "markdown", "json", "text"]
    usage: Usage = Usage()
    driver_id: str
    model_id: str
    model_revision: str | None = None
    warnings: tuple[str, ...] = ()


class Capabilities(ContractModel):
    tasks: frozenset[Task]
    languages: tuple[str, ...]
    max_image_pixels: int = Field(gt=0)
    max_concurrency: int = Field(ge=1)
    structured_output: bool
    location: Literal["local", "lan", "cloud"]
    tier: Literal["small", "large"]


class VlmError(Exception):
    """드라이버의 모든 실패는 이 예외 하나로 전달한다."""

    def __init__(self, info: ErrorInfo) -> None:
        super().__init__(f"{info.code}: {info.message}")
        self.info = info


@runtime_checkable
class VlmDriver(Protocol):
    id: str

    def capabilities(self) -> Capabilities: ...

    async def health(self) -> bool: ...

    async def run(self, req: VlmRequest) -> VlmResult: ...
