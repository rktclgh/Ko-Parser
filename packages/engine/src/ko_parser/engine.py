"""계약 Engine의 로컬 구현. ingest는 동기(D6), VLM 경로는 없다."""

import hashlib
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from ko_parser_contracts import (
    ChangeBatch, DocFilter, DocRef, DocumentTree, JobRef, JobStatus, ProcessingHistory, SourceInfo,
)

from .core import build_tree, diff_trees
from .errors import StoreConflict, VlmUnavailable
from .formats.base import Parser
from .formats.detect import default_parsers, detect_parser
from .store.base import Store


def _ref(tree: DocumentTree) -> DocRef:
    return DocRef(document_id=tree.document_id, version=tree.version, layer_state=tree.layer_state)


class LocalEngine:
    def __init__(self, store: Store, parsers: Sequence[Parser] | None = None) -> None:
        self._store = store
        self._parsers = default_parsers() if parsers is None else tuple(parsers)

    def ingest(self, path: str, document_id: str | None = None, force: bool = False) -> DocRef:
        """document_id가 없으면 doc_ + 원본 sha256 앞 24자리(D1). 원본이 같으면 force가 아닌 한 그대로(D2)."""
        file = Path(path)
        name = unicodedata.normalize("NFC", file.name)  # 전체 경로는 저장하지 않는다
        data = file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        doc_id = "doc_" + digest[:24] if document_id is None else document_id
        if not doc_id:
            raise ValueError("document_id must not be empty")
        try:
            return self._ingest_once(data, name, doc_id, "sha256:" + digest, force)
        except StoreConflict:  # 다른 쓰기와 겹쳤다: 최신 버전을 다시 읽고 한 번만 재시도
            return self._ingest_once(data, name, doc_id, "sha256:" + digest, force)

    def _ingest_once(self, data: bytes, name: str, doc_id: str, content_hash: str, force: bool) -> DocRef:
        latest = self._store.latest(doc_id)
        if latest is not None and latest.source.content_hash == content_hash and not force:
            return _ref(latest)
        parsed = detect_parser(name, self._parsers).parse(data, name)  # 형식 판별은 원본이 달라진 뒤에만
        source = SourceInfo(name=name, mime=parsed.mime, content_hash=content_hash,
                            page_count=len(parsed.pages) or None)
        tree = build_tree(parsed, doc_id, 1 if latest is None else latest.version + 1, source)
        change = diff_trees(latest, tree)
        if change is None:  # 원본은 달라도 블록과 쪽 정보가 같다
            assert latest is not None
            return _ref(latest)
        self._store.commit(tree, change, ProcessingHistory(document_id=doc_id, version=tree.version))
        return _ref(tree)

    def documents(self) -> tuple[DocRef, ...]:
        return self._store.documents()

    def get_tree(self, document_id: str, version: int | None = None) -> DocumentTree:
        return self._store.get_tree(document_id, version)

    def run_vlm(self, target: DocFilter) -> JobRef:
        raise VlmUnavailable("VLM layer is not available in this engine")

    def job(self, job_id: str) -> JobStatus:
        raise VlmUnavailable("VLM layer is not available in this engine")

    def changes(self, cursor: int | None, limit: int = 100) -> ChangeBatch:
        return self._store.changes_after(cursor, limit)

    def history(self, document_id: str, version: int | None = None) -> ProcessingHistory:
        return self._store.history(document_id, version)
