"""파서 출력 → 결정론 레이어 문서 트리."""

from collections.abc import Mapping
from typing import Any

from hanji_contracts import Block, DocumentTree, SourceInfo, build_blocks

from ..formats.base import ParsedSource

CAPTION_REF = "caption_ref"  # 파서 명세의 figure 안 키: 짝 캡션 명세의 순번(트리를 만들 때 block_id로 바꾼다)


def build_tree(parsed: ParsedSource, document_id: str, version: int, source: SourceInfo) -> DocumentTree:
    """계약 build_blocks로 order·content_hash·block_id를 계산하고 layer_state='det' 트리를 만든다.
    그림 명세의 caption_ref(캡션 명세 순번)는 그 캡션 블록의 block_id(figure.caption_block_id)로 바꾼다."""
    specs: list[Mapping[str, Any]] = []
    refs: dict[int, object] = {}
    for index, spec in enumerate(parsed.blocks):
        figure = spec.get("figure")
        if isinstance(figure, Mapping) and CAPTION_REF in figure:
            figure = dict(figure)
            ref = figure.pop(CAPTION_REF)
            if ref is not None:
                refs[index] = ref
            spec = {**spec, "figure": figure}
        specs.append(spec)
    blocks = build_blocks(document_id, specs)
    if refs:
        blocks = tuple(_with_caption(b, blocks, refs[b.order]) if b.order in refs else b for b in blocks)
    return DocumentTree(document_id=document_id, version=version, layer_state="det", source=source,
                        pages=parsed.pages, blocks=blocks)


def _with_caption(block: Block, blocks: tuple[Block, ...], ref: object) -> Block:
    """그림 블록의 figure.caption_block_id를 짝 캡션 블록의 id로(해시에 들어가지 않아 id는 그대로)."""
    if (not isinstance(ref, int) or isinstance(ref, bool) or not 0 <= ref < len(blocks)
            or block.figure is None):
        raise ValueError(f"invalid caption_ref {ref!r} at block {block.order}")
    figure = {**block.figure.model_dump(), "caption_block_id": blocks[ref].block_id}
    return Block.model_validate({**block.model_dump(), "figure": figure})
