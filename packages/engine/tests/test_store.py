"""저장소 공통 테스트 묶음: 모든 Store 구현이 같은 결과를 내야 한다."""

import pytest

from ko_parser.core import build_tree, diff_trees
from ko_parser.errors import DocumentNotFound, StoreConflict, VersionNotFound
from ko_parser.formats.base import ParsedSource
from ko_parser.store import MemoryStore, SqliteStore
from ko_parser_contracts import DocRef, DocumentTree, ProcessingHistory, SourceInfo

SOURCE = SourceInfo(name="메모.md", mime="text/markdown", content_hash="sha256:" + "0" * 64)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryStore()
        return
    s = SqliteStore(tmp_path / "한글 폴더" / "state.db")
    yield s
    s.close()


def make_tree(doc: str, version: int, *texts: str) -> DocumentTree:
    blocks = [{"kind": "paragraph", "text": t, "confidence": 1.0, "state": "det", "text_source": "native",
               "locator": {"kind": "lines", "line_start": i + 1, "line_end": i + 1}} for i, t in enumerate(texts)]
    return build_tree(ParsedSource(mime="text/markdown", blocks=blocks), doc, version, SOURCE)


def commit(store, doc: str, *texts: str) -> int:
    """최신 버전 다음 버전으로 커밋하고 커서를 돌려준다."""
    prev = store.latest(doc)
    tree = make_tree(doc, 1 if prev is None else prev.version + 1, *texts)
    return store.commit(tree, diff_trees(prev, tree), ProcessingHistory(document_id=doc, version=tree.version))


def test_empty_store(store):
    assert store.latest("없음") is None
    assert store.documents() == ()
    batch = store.changes_after(None, 10)
    assert (batch.cursor_from, batch.next_cursor, batch.changes, batch.resync_required) == (None, 0, (), False)


def test_commit_and_read_back(store):
    assert commit(store, "d1", "가", "나") == 1
    tree = store.latest("d1")
    assert tree == make_tree("d1", 1, "가", "나")
    assert store.get_tree("d1") == tree and store.get_tree("d1", 1) == tree
    assert store.history("d1") == ProcessingHistory(document_id="d1", version=1)


def test_versions_and_cursor_are_monotonic(store):
    cursors = [commit(store, "d1", "가"), commit(store, "d2", "나"), commit(store, "d1", "가", "다")]
    assert cursors == [1, 2, 3]
    assert store.get_tree("d1", 1).version == 1 and store.get_tree("d1").version == 2
    assert store.history("d1", 2).version == 2


def test_version_conflict_leaves_store_unchanged(store):
    commit(store, "d1", "가")
    stale = make_tree("d1", 1, "나")  # 최신이 1인데 다시 1
    with pytest.raises(StoreConflict):
        store.commit(stale, diff_trees(None, stale), ProcessingHistory(document_id="d1", version=1))
    skip = make_tree("d1", 3, "나")  # 2를 건너뜀
    with pytest.raises(StoreConflict):
        store.commit(skip, diff_trees(store.latest("d1"), skip), ProcessingHistory(document_id="d1", version=3))
    first = make_tree("new", 2, "가")  # 첫 버전은 1이어야 한다
    with pytest.raises(StoreConflict):
        store.commit(first, diff_trees(make_tree("new", 1), first), ProcessingHistory(document_id="new", version=2))
    assert store.latest("d1").version == 1 and store.latest("new") is None
    assert store.changes_after(None, 10).next_cursor == 1


def test_mismatched_change_or_history_rejected_atomically(store):
    tree = make_tree("d1", 1, "가")
    with pytest.raises(ValueError):
        store.commit(tree, diff_trees(None, tree), ProcessingHistory(document_id="d1", version=2))
    other = make_tree("d2", 1, "가")
    with pytest.raises(ValueError):
        store.commit(tree, diff_trees(None, other), ProcessingHistory(document_id="d1", version=1))
    assert store.latest("d1") is None and store.documents() == ()
    with pytest.raises(DocumentNotFound):
        store.history("d1")
    assert store.changes_after(None, 10).changes == ()


def test_changes_paging_with_limit(store):
    for i, doc in enumerate(("a", "b", "a", "c", "a")):
        commit(store, doc, f"{doc}-{i}")
    first = store.changes_after(None, 2)
    assert (first.cursor_from, first.next_cursor) == (None, 2)
    assert [(c.document_id, c.version) for c in first.changes] == [("a", 1), ("b", 1)]
    second = store.changes_after(first.next_cursor, 2)
    assert (second.cursor_from, second.next_cursor) == (2, 4)
    assert [(c.document_id, c.version, c.previous_version) for c in second.changes] == [("a", 2, 1), ("c", 1, None)]
    last = store.changes_after(4, 2)
    assert last.next_cursor == 5 and len(last.changes) == 1
    done = store.changes_after(5, 2)
    assert (done.cursor_from, done.next_cursor, done.changes, done.resync_required) == (5, 5, (), False)


def test_same_document_chain_in_one_batch(store):
    commit(store, "d1", "가")
    commit(store, "d1", "가", "나")
    commit(store, "d1", "나")
    batch = store.changes_after(0, 10)
    assert [(c.version, c.previous_version) for c in batch.changes] == [(1, None), (2, 1), (3, 2)]
    assert batch.cursor_from == 0 and batch.next_cursor == 3


def test_cursor_beyond_latest_requires_resync(store):
    commit(store, "d1", "가")
    batch = store.changes_after(7, 10)
    assert batch.resync_required and batch.changes == ()
    assert (batch.cursor_from, batch.next_cursor) == (7, 7)


def test_bad_query_rejected(store):
    with pytest.raises(ValueError):
        store.changes_after(None, 0)
    with pytest.raises(ValueError):
        store.changes_after(-1, 10)


def test_documents_sorted_by_id_with_latest_version(store):
    commit(store, "문서-나", "가")
    commit(store, "doc-b", "가")
    commit(store, "문서-가", "가")
    commit(store, "doc-b", "나")
    assert store.documents() == (
        DocRef(document_id="doc-b", version=2, layer_state="det"),
        DocRef(document_id="문서-가", version=1, layer_state="det"),
        DocRef(document_id="문서-나", version=1, layer_state="det"),
    )


def test_missing_document_and_version(store):
    commit(store, "d1", "가")
    with pytest.raises(DocumentNotFound):
        store.get_tree("없음")
    with pytest.raises(DocumentNotFound):
        store.history("없음", 1)
    for version in (0, 2):
        with pytest.raises(VersionNotFound):
            store.get_tree("d1", version)
        with pytest.raises(VersionNotFound):
            store.history("d1", version)


def test_commit_rejects_mismatched_previous_version(store):
    commit(store, "d1", "가")
    v2 = make_tree("d1", 2, "나")
    with pytest.raises(ValueError):  # 변경의 previous_version이 None인데 최신은 1
        store.commit(v2, diff_trees(None, v2), ProcessingHistory(document_id="d1", version=2))
    assert store.latest("d1").version == 1
    assert len(store.changes_after(None, 100).changes) == 1
    with pytest.raises(VersionNotFound):
        store.history("d1", 2)


def test_commit_returns_the_cursor_of_its_change(store):
    first = commit(store, "d1", "가")
    batch = store.changes_after(None, 1)
    assert [(c.document_id, c.version) for c in batch.changes] == [("d1", 1)] and first == batch.next_cursor
    second = commit(store, "d1", "가", "나")
    batch = store.changes_after(first, 1)
    assert [(c.document_id, c.version) for c in batch.changes] == [("d1", 2)] and second == batch.next_cursor
