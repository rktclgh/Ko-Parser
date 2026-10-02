"""DocumentTree → 마크다운. 원본 복원이 아니라 RAG용 텍스트 뷰다."""

import re

from ko_parser_contracts import DocumentTree

_NUMBERED = re.compile(r"\d+[.)] ")


def to_markdown(tree: DocumentTree) -> str:
    """블록 사이 빈 줄 하나. page_header·page_footer는 생략. 블록이 없으면 빈 문자열."""
    parts: list[str] = []
    for block in tree.blocks:
        match block.kind:
            case "page_header" | "page_footer":
                continue
            case "heading":
                parts.append("#" * (block.level or 1) + " " + block.text)
            case "list_item":
                parts.append(block.text if _NUMBERED.match(block.text) else "- " + block.text)
            case "table" if block.table is not None:
                parts.append(block.table.to_markdown())
            case _:
                parts.append(block.text)
    return "\n\n".join(parts) + "\n" if parts else ""
