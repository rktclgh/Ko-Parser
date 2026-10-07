"""저장 포트. 엔진은 무엇을 저장할지만 알고, 어디에 어떻게는 구현이 정한다(D3)."""

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Protocol

from hanji_contracts import (
    MAX_DOCUMENT_ASSET_BYTES, ChangeBatch, DocRef, DocumentChange, DocumentTree, ProcessingHistory,
)

from ..core.assets import asset_key
from ..errors import StoreConflict

__all__ = ["NO_ASSETS", "Store", "StoreConflict", "check_commit", "check_query", "make_batch"]

NO_ASSETS: Mapping[str, bytes] = MappingProxyType({})


class Store(Protocol):
    def latest(self, document_id: str) -> DocumentTree | None: ...

    def get_tree(self, document_id: str, version: int | None = None) -> DocumentTree:
        """없는 문서는 DocumentNotFound, 없는 버전은 VersionNotFound."""
        ...

    def documents(self) -> tuple[DocRef, ...]:
        """문서마다 최신 버전. document_id 순."""
        ...

    def commit(self, tree: DocumentTree, change: DocumentChange, history: ProcessingHistory,
               assets: Mapping[str, bytes] = NO_ASSETS) -> int:
        """새 버전·변경 로그 한 건·처리 이력·그림 이미지를 원자적으로 넣고 커서 번호(1부터 전역 단조 증가)를 돌려준다.

        tree.version이 저장된 최신 버전 + 1(첫 버전은 1)이 아니면 StoreConflict. assets(sha256 → PNG)는 트리의 그림이
        가리키는 것만 받고, 가리키는데 주지 않은 이미지는 이미 저장돼 있어야 한다(아니면 ValueError). 해시가 같은
        이미지는 한 번만 저장한다(문서끼리도). 트리가 가리키는 이미지(주는 것 + 이미 저장된 것) 총합이
        MAX_DOCUMENT_ASSET_BYTES를 넘으면 ValueError. 이미지는 지우지 않는다(참조 수로 정리하는 것은 저장 레이어 몫).
        """
        ...

    def get_asset(self, asset: str) -> bytes:
        """저장된 그림 이미지 바이트. 없으면 AssetNotFound."""
        ...

    def changes_after(self, cursor: int | None, limit: int) -> ChangeBatch: ...

    def history(self, document_id: str, version: int | None = None) -> ProcessingHistory: ...


def check_commit(tree: DocumentTree, change: DocumentChange, history: ProcessingHistory, current: int | None,
                 assets: Mapping[str, bytes] = NO_ASSETS,
                 stored: Callable[[str], int | None] = lambda asset: None) -> dict[str, str]:
    """두 구현이 같이 쓰는 커밋 전 검사. current는 저장된 최신 버전(없으면 None), stored는 이미 저장된 이미지의 바이트
    수(없으면 None). 트리가 가리키는 이미지 총합(서로 다른 이미지마다 한 번)은 MAX_DOCUMENT_ASSET_BYTES 이하: 엔진을
    거치지 않고 commit을 바로 불러도 상한을 지킨다. 반환: 이 트리의 그림이 가리키는 이미지 → mime."""
    expected = 1 if current is None else current + 1
    if tree.version != expected:
        raise StoreConflict(f"{tree.document_id}: expected version {expected}, got {tree.version}")
    if (change.document_id, change.version) != (tree.document_id, tree.version):
        raise ValueError("change does not match tree")
    if (history.document_id, history.version) != (tree.document_id, tree.version):
        raise ValueError("history does not match tree")
    if change.previous_version != current:
        raise ValueError("change.previous_version must equal the stored latest version")
    referenced = {b.figure.asset: b.figure.mime for b in tree.blocks if b.figure is not None}
    for key, data in assets.items():
        if key not in referenced:
            raise ValueError(f"asset {key} is not referenced by the tree")
        if key != asset_key(data):
            raise ValueError(f"asset {key} does not match its bytes")
    sizes = {a: len(assets[a]) if a in assets else stored(a) for a in referenced}
    missing = sorted(a for a, size in sizes.items() if size is None)
    if missing:
        raise ValueError(f"figure asset {missing[0]} is neither given nor stored")
    if sum(size for size in sizes.values() if size is not None) > MAX_DOCUMENT_ASSET_BYTES:
        raise ValueError(f"document assets exceed MAX_DOCUMENT_ASSET_BYTES ({MAX_DOCUMENT_ASSET_BYTES})")
    return referenced


def check_query(cursor: int | None, limit: int) -> None:
    if cursor is not None and cursor < 0:
        raise ValueError("cursor must be >= 0")
    if limit < 1:
        raise ValueError("limit must be >= 1")


def make_batch(cursor: int | None, last_seq: int, rows: Sequence[tuple[int, DocumentChange]]) -> ChangeBatch:
    """rows는 cursor 뒤의 (seq, 변경)을 seq 순으로. cursor가 last_seq보다 크면 다른 저장소의 커서다(D7)."""
    start = cursor or 0
    if start > last_seq:
        return ChangeBatch(cursor_from=cursor, next_cursor=start, resync_required=True)
    next_cursor = rows[-1][0] if rows else start
    return ChangeBatch(cursor_from=cursor, next_cursor=next_cursor, changes=tuple(change for _, change in rows))
