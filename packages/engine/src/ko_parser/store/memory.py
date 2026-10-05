"""프로세스 메모리 저장소. 테스트·일회성 실행용이며 스레드 안전하지 않다."""

from collections.abc import Mapping

from ko_parser_contracts import ChangeBatch, DocRef, DocumentChange, DocumentTree, ProcessingHistory

from ..errors import AssetNotFound, DocumentNotFound, VersionNotFound
from .base import NO_ASSETS, check_commit, check_query, make_batch


class MemoryStore:
    def __init__(self) -> None:
        self._trees: dict[str, list[DocumentTree]] = {}
        self._histories: dict[str, list[ProcessingHistory]] = {}
        self._log: list[DocumentChange] = []
        self._assets: dict[str, bytes] = {}

    def latest(self, document_id: str) -> DocumentTree | None:
        trees = self._trees.get(document_id)
        return trees[-1] if trees else None

    def get_tree(self, document_id: str, version: int | None = None) -> DocumentTree:
        return self._pick(self._trees, document_id, version)

    def documents(self) -> tuple[DocRef, ...]:
        return tuple(DocRef(document_id=doc_id, version=trees[-1].version, layer_state=trees[-1].layer_state)
                     for doc_id, trees in sorted(self._trees.items()))

    def commit(self, tree: DocumentTree, change: DocumentChange, history: ProcessingHistory,
               assets: Mapping[str, bytes] = NO_ASSETS) -> int:
        current = self.latest(tree.document_id)
        check_commit(tree, change, history, None if current is None else current.version, assets,
                     self._assets.__contains__)
        # 검사를 모두 마친 뒤에만 바꾼다(실패하면 아무것도 남지 않는다)
        self._trees.setdefault(tree.document_id, []).append(tree)
        self._histories.setdefault(tree.document_id, []).append(history)
        self._log.append(change)
        for key, data in assets.items():
            self._assets.setdefault(key, bytes(data))
        return len(self._log)

    def get_asset(self, asset: str) -> bytes:
        try:
            return self._assets[asset]
        except KeyError:
            raise AssetNotFound(asset) from None

    def changes_after(self, cursor: int | None, limit: int) -> ChangeBatch:
        check_query(cursor, limit)
        start = cursor or 0
        rows = [(seq, change) for seq, change in enumerate(self._log[start:start + limit], start + 1)]
        return make_batch(cursor, len(self._log), rows)

    def history(self, document_id: str, version: int | None = None) -> ProcessingHistory:
        return self._pick(self._histories, document_id, version)

    @staticmethod
    def _pick[T](items: dict[str, list[T]], document_id: str, version: int | None) -> T:
        found = items.get(document_id)
        if not found:
            raise DocumentNotFound(document_id)
        if version is None:
            return found[-1]
        if not 1 <= version <= len(found):
            raise VersionNotFound(f"{document_id} v{version}")
        return found[version - 1]
