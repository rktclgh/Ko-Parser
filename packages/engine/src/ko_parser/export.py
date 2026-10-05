"""DocumentTree → 마크다운. 원본 복원이 아니라 RAG용 텍스트 뷰다."""

import re

from ko_parser_contracts import DocumentTree

_NUMBERED = re.compile(r"\d+[.)] ")
_ALT_SPECIAL = re.compile(r"([\\\[\]])")
_NEEDS_ANGLE = re.compile(r"[\s()<>]")


def asset_name(asset: str) -> str:
    """그림 이미지 파일 이름: sha256 앞 16자 + .png."""
    return asset.removeprefix("sha256:")[:16] + ".png"


def _alt(text: str) -> str:
    """이미지 대체 글자(캡션): 한 줄로, 역슬래시·대괄호는 이스케이프."""
    return _ALT_SPECIAL.sub(r"\\\1", " ".join(text.split()))


def _target(folder: str, asset: str) -> str:
    """링크 주소. 공백·괄호가 있으면 <…>로 감싼다."""
    path = f"{folder.rstrip('/')}/{asset_name(asset)}" if folder.rstrip("/") else asset_name(asset)
    return f"<{path.replace('<', '%3C').replace('>', '%3E')}>" if _NEEDS_ANGLE.search(path) else path


def to_markdown(tree: DocumentTree, assets_dir: str | None = None) -> str:
    """블록 사이 빈 줄 하나. page_header·page_footer는 생략. 블록이 없으면 빈 문자열.
    assets_dir를 주면 이미지가 있는 그림 블록은 ![캡션](assets_dir/<sha256 앞 16자>.png)과 그 아래 그림 글자(대체
    글자)로 쓴다(이미지 파일은 부르는 쪽이 쓴다). 없으면 그림은 글자만, 글자 없는 그림은 건너뛴다."""
    captions = {b.block_id: b.text for b in tree.blocks if b.kind == "caption"}
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
            case "figure" if assets_dir is not None and block.figure is not None:
                alt = _alt(captions.get(block.figure.caption_block_id or "", ""))
                image = f"![{alt}]({_target(assets_dir, block.figure.asset)})"
                parts.append(f"{image}\n\n{block.text}" if block.text.strip() else image)
            case "figure" if not block.text.strip():
                continue
            case _:
                parts.append(block.text)
    return "\n\n".join(parts) + "\n" if parts else ""
