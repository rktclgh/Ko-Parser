import sqlite3

import pytest

from ko_parser.core import build_tree, diff_trees
from ko_parser.errors import KoParserError, StoreConflict
from ko_parser.formats.base import ParsedSource
from ko_parser.store import SqliteStore
from ko_parser_contracts import DocumentTree, ProcessingHistory, SourceInfo

SOURCE = SourceInfo(name="메모.md", mime="text/markdown", content_hash="sha256:" + "0" * 64)


def make_tree(doc: str, version: int, *texts: str) -> DocumentTree:
    blocks = [{"kind": "paragraph", "text": t, "confidence": 1.0, "state": "det", "text_source": "native",
               "locator": {"kind": "lines", "line_start": i + 1, "line_end": i + 1}} for i, t in enumerate(texts)]
    return build_tree(ParsedSource(mime="text/markdown", blocks=blocks), doc, version, SOURCE)


def commit(store: SqliteStore, doc: str, *texts: str) -> int:
    prev = store.latest(doc)
    tree = make_tree(doc, 1 if prev is None else prev.version + 1, *texts)
    return store.commit(tree, diff_trees(prev, tree), ProcessingHistory(document_id=doc, version=tree.version))


@pytest.fixture
def db(tmp_path):
    return tmp_path / "상태 폴더" / "state.db"


def test_reopen_gives_same_results(db):
    with SqliteStore(db) as store:
        commit(store, "d1", "가")
        commit(store, "d2", "나")
        commit(store, "d1", "가", "다")
        before = (store.documents(), store.changes_after(None, 10), store.get_tree("d1", 1), store.history("d1"))
    with SqliteStore(db) as store:
        after = (store.documents(), store.changes_after(None, 10), store.get_tree("d1", 1), store.history("d1"))
        assert after == before
        assert commit(store, "d2", "라") == 4  # 커서는 재시작 뒤에도 이어진다


def test_wal_mode_and_format_meta(db):
    with SqliteStore(db):
        pass
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("SELECT value FROM meta WHERE key = 'format'").fetchone()[0] == "1"
    finally:
        conn.close()


def test_unknown_format_rejected(db):
    with SqliteStore(db):
        pass
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE meta SET value = '999' WHERE key = 'format'")
    conn.close()
    with pytest.raises(KoParserError, match="unsupported store format"):
        SqliteStore(db)


def test_failure_inside_commit_rolls_back_everything(db):
    with SqliteStore(db) as store:
        commit(store, "d1", "가")
        raw = sqlite3.connect(db)
        with raw:  # 버전 2의 이력 자리를 미리 차지해 커밋 중간(histories 삽입)에서 실패시킨다
            raw.execute("INSERT INTO histories VALUES ('d1', 2, '{}')")
        raw.close()
        with pytest.raises(sqlite3.IntegrityError):
            commit(store, "d1", "나")
        assert store.latest("d1").version == 1
        assert store.changes_after(None, 10).next_cursor == 1
        assert [r.version for r in store.documents()] == [1]
        assert commit(store, "d2", "다") == 2  # 실패한 커밋은 커서 번호를 쓰지 않는다


def test_two_instances_conflict_on_same_version(db):
    with SqliteStore(db) as a, SqliteStore(db) as b:
        commit(a, "d1", "가")
        prev = b.latest("d1")
        mine = make_tree("d1", 2, "나")
        theirs = make_tree("d1", 2, "다")
        a.commit(theirs, diff_trees(prev, theirs), ProcessingHistory(document_id="d1", version=2))
        with pytest.raises(StoreConflict):
            b.commit(mine, diff_trees(prev, mine), ProcessingHistory(document_id="d1", version=2))
        assert b.latest("d1") == theirs
