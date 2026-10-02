"""파서 출력 → 결정론 레이어 문서 트리."""

from ko_parser_contracts import DocumentTree, SourceInfo, build_blocks

from ..formats.base import ParsedSource


def build_tree(parsed: ParsedSource, document_id: str, version: int, source: SourceInfo) -> DocumentTree:
    """계약 build_blocks로 order·content_hash·block_id를 계산하고 layer_state='det' 트리를 만든다."""
    return DocumentTree(document_id=document_id, version=version, layer_state="det", source=source,
                        pages=parsed.pages, blocks=build_blocks(document_id, parsed.blocks))
