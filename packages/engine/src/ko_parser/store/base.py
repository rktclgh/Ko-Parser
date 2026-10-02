"""저장 포트. 엔진은 무엇을 저장할지만 알고, 어디에 어떻게는 구현이 정한다(D3)."""

from collections.abc import Sequence
from typing import Protocol

from ko_parser_contracts import ChangeBatch, DocRef, DocumentChange, DocumentTree, ProcessingHistory

from ..errors import StoreConflict

__all__ = ["Store", "StoreConflict", "check_commit", "check_query", "make_batch"]


class Store(Protocol):
    def latest(self, document_id: str) -> DocumentTree | None: ...

    def get_tree(self, document_id: str, version: int | None = None) -> DocumentTree:
        """없는 문서는 DocumentNotFound, 없는 버전은 VersionNotFound."""
        ...

    def documents(self) -> tuple[DocRef, ...]:
        """문서마다 최신 버전. document_id 순."""
        ...

    def commit(self, tree: DocumentTree, change: DocumentChange, history: ProcessingHistory) -> int:
        """새 버전·변경 로그 한 건·처리 이력을 원자적으로 넣고 커서 번호(1부터 전역 단조 증가)를 돌려준다.

        tree.version이 저장된 최신 버전 + 1(첫 버전은 1)이 아니면 StoreConflict.
        """
        ...

    def changes_after(self, cursor: int | None, limit: int) -> ChangeBatch: ...

    def history(self, document_id: str, version: int | None = None) -> ProcessingHistory: ...


def check_commit(tree: DocumentTree, change: DocumentChange, history: ProcessingHistory,
                 current: int | None) -> None:
    """두 구현이 같이 쓰는 커밋 전 검사. current는 저장된 최신 버전(없으면 None)."""
    expected = 1 if current is None else current + 1
    if tree.version != expected:
        raise StoreConflict(f"{tree.document_id}: expected version {expected}, got {tree.version}")
    if (change.document_id, change.version) != (tree.document_id, tree.version):
        raise ValueError("change does not match tree")
    if (history.document_id, history.version) != (tree.document_id, tree.version):
        raise ValueError("history does not match tree")
    if change.previous_version != current:
        raise ValueError("change.previous_version must equal the stored latest version")


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
