"""문서: 블록의 평평한 목록. 버전마다 불변 스냅숏이다."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel, VersionedModel
from .geometry import PageInfo
from .ids import compute_block_id, compute_content_hash
from .locator import Locator, PageLocator
from .table import Table

BlockKind = Literal["heading", "paragraph", "list_item", "table", "figure", "caption", "page_header", "page_footer"]
BlockState = Literal["det", "unverified", "vlm"]
TextSource = Literal["native", "text_layer", "ocr", "vlm", "mixed"]
LayerState = Literal["det", "vlm_running", "vlm_done", "vlm_failed"]
# 레이어 상태별로 허용되는 블록 상태. VLM 실패 시 문서 전체가 det로 복귀한다.
ALLOWED_BLOCK_STATES: dict[str, frozenset[str]] = {
    "det": frozenset({"det"}),
    "vlm_running": frozenset({"det", "unverified"}),
    "vlm_done": frozenset({"det", "vlm"}),
    "vlm_failed": frozenset({"det"}),
}


def check_kind_fields(kind: str, table: Table | None, level: int | None) -> None:
    """kind와 table·level의 조합을 검사한다(Block과 VlmBlock이 함께 쓴다)."""
    if (kind == "table") != (table is not None):
        raise ValueError("table must be set if and only if kind == 'table'")
    if (kind == "heading") != (level is not None):
        raise ValueError("level must be set if and only if kind == 'heading'")


class Block(ContractModel):
    block_id: str
    content_hash: str
    order: int = Field(ge=0)
    kind: BlockKind
    text: str
    table: Table | None = None
    level: int | None = Field(default=None, ge=1, le=6)
    section_path: tuple[str, ...] = ()
    locator: Locator
    confidence: float = Field(ge=0.0, le=1.0)
    state: BlockState
    text_source: TextSource
    region_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        check_kind_fields(self.kind, self.table, self.level)
        if self.table is not None:
            if self.text != self.table.plain_text():
                raise ValueError("table block text must equal table.plain_text()")
            sources = {cell.text_source for cell in self.table.cells}
            expected = next(iter(sources)) if len(sources) == 1 else "mixed"
            if self.text_source != expected:
                raise ValueError(f"table block text_source must be {expected!r}")
        elif self.text_source == "mixed":
            raise ValueError("only table blocks may use text_source 'mixed'")
        has_vlm_text = self.text_source == "vlm" or (
            self.table is not None and any(c.text_source == "vlm" for c in self.table.cells))
        if has_vlm_text and self.state != "vlm":
            raise ValueError("vlm text requires state 'vlm'")
        if self.content_hash != compute_content_hash(self.kind, self.text, self.level, self.table):
            raise ValueError("content_hash does not match block content")
        return self


class SourceInfo(ContractModel):
    name: str
    mime: str
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    page_count: int | None = Field(default=None, ge=0)


class DocumentTree(VersionedModel):
    document_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    layer_state: LayerState
    source: SourceInfo
    pages: tuple[PageInfo, ...] = ()
    blocks: tuple[Block, ...] = ()

    @model_validator(mode="after")
    def _check_document(self) -> Self:
        page_numbers = [p.page for p in self.pages]
        if len(set(page_numbers)) != len(page_numbers):
            raise ValueError("duplicate page numbers")
        orders = [b.order for b in self.blocks]
        if any(a >= b for a, b in zip(orders, orders[1:])):
            raise ValueError("blocks must be sorted by strictly increasing order")
        seen: Counter[str] = Counter()
        for block in self.blocks:
            expected = compute_block_id(self.document_id, block.content_hash, seen[block.content_hash])
            seen[block.content_hash] += 1
            if block.state not in ALLOWED_BLOCK_STATES[self.layer_state]:
                raise ValueError(f"block state {block.state!r} not allowed when layer_state is {self.layer_state!r}")
            if block.block_id != expected:
                raise ValueError(f"block_id mismatch at order {block.order}")
            if isinstance(block.locator, PageLocator) and block.locator.page not in page_numbers:
                raise ValueError(f"block at order {block.order} refers to unknown page {block.locator.page}")
        return self


def build_blocks(document_id: str, specs: Iterable[Mapping[str, Any]]) -> tuple[Block, ...]:
    """order(나열 순서), content_hash, block_id를 계산해 블록을 만든다.

    spec에는 block_id·content_hash·order를 넣지 않는다. 표 블록은 text를 생략하면 table.plain_text()로 채운다.
    """
    seen: Counter[str] = Counter()
    blocks: list[Block] = []
    for order, spec in enumerate(specs):
        fields = dict(spec)
        for key in ("block_id", "content_hash", "order"):
            if key in fields:
                raise ValueError(f"spec must not contain {key!r}")
        table = fields.get("table")
        if isinstance(table, Mapping):
            table = Table.model_validate(table)
            fields["table"] = table
        if table is not None and "text" not in fields:
            fields["text"] = table.plain_text()
        content_hash = compute_content_hash(fields["kind"], fields.get("text", ""), fields.get("level"), table)
        block_id = compute_block_id(document_id, content_hash, seen[content_hash])
        seen[content_hash] += 1
        blocks.append(Block(block_id=block_id, content_hash=content_hash, order=order, **fields))
    return tuple(blocks)
