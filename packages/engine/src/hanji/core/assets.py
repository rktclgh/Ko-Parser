"""파서가 낸 그림 이미지(ParsedSource.assets)와 트리의 그림 참조가 맞는지 본다."""

import hashlib
from collections.abc import Mapping

from hanji_contracts import MAX_DOCUMENT_ASSET_BYTES, DocumentTree


def asset_key(data: bytes) -> str:
    """이미지 바이트의 내용 해시(FigureImage.asset 모양)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def check_assets(tree: DocumentTree, assets: Mapping[str, bytes]) -> None:
    """키는 바이트의 sha256, 그림 블록이 가리키는 이미지는 모두 있고 남는 것이 없으며, 총합은 MAX_DOCUMENT_ASSET_BYTES
    이하. 어긋나면 ValueError(파서 버그: 저장하지 않는다)."""
    for key, data in assets.items():
        if key != asset_key(data):
            raise ValueError(f"asset {key} does not match its bytes")
    referenced = {b.figure.asset for b in tree.blocks if b.figure is not None}
    if missing := sorted(referenced - assets.keys()):
        raise ValueError(f"figure asset {missing[0]} is missing from the parsed assets")
    if extra := sorted(assets.keys() - referenced):
        raise ValueError(f"asset {extra[0]} is not referenced by any figure block")
    if sum(len(data) for data in assets.values()) > MAX_DOCUMENT_ASSET_BYTES:
        raise ValueError(f"document assets exceed MAX_DOCUMENT_ASSET_BYTES ({MAX_DOCUMENT_ASSET_BYTES})")
