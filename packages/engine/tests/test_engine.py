import hashlib
import inspect
import unicodedata

import pytest

from ko_parser import LocalEngine, MemoryStore, SqliteStore
from ko_parser.core import build_tree, diff_trees
from ko_parser.errors import ParseError, StoreConflict, UnsupportedFormat, VlmUnavailable
from ko_parser.formats.base import ParsedSource
from ko_parser_contracts import DocFilter, Engine, ProcessingHistory, SourceInfo


class FakeParser:
    """테스트용: 비어 있지 않은 줄마다 문단 하나. '!fail' 줄이 있으면 ParseError."""

    mimes = ("text/x-fake",)
    extensions = (".fake",)

    def __init__(self, suffix: str = "") -> None:
        self.calls = 0
        self.suffix = suffix  # 파서 버전이 바뀐 것을 흉내 낸다

    def parse(self, data: bytes, name: str) -> ParsedSource:
        self.calls += 1
        blocks = []
        for i, line in enumerate(data.decode("utf-8").split("\n"), 1):
            if line == "!fail":
                raise ParseError("fake failure", f"{name}:{i}")
            if line.strip():
                blocks.append({"kind": "paragraph", "text": line + self.suffix, "confidence": 1.0, "state": "det",
                               "text_source": "native",
                               "locator": {"kind": "lines", "line_start": i, "line_end": i}})
        return ParsedSource(mime="text/x-fake", blocks=blocks)


@pytest.fixture
def parser():
    return FakeParser()


@pytest.fixture
def engine(parser):
    return LocalEngine(MemoryStore(), parsers=[parser])


def write(path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def test_signature_matches_contract():
    ours = inspect.signature(LocalEngine.ingest).parameters
    theirs = inspect.signature(Engine.ingest).parameters
    assert [(p.name, p.default) for p in ours.values()] == [(p.name, p.default) for p in theirs.values()]
    assert {"ingest", "documents", "get_tree", "run_vlm", "job", "changes", "history"} <= set(dir(LocalEngine))


def test_first_ingest_creates_version_one(engine, tmp_path):
    path = write(tmp_path / "보고서.fake", "가\n나\n")
    ref = engine.ingest(str(path))
    data = path.read_bytes()
    assert ref.document_id == "doc_" + hashlib.sha256(data).hexdigest()[:24]
    assert (ref.version, ref.layer_state) == (1, "det")
    tree = engine.get_tree(ref.document_id)
    assert [b.text for b in tree.blocks] == ["가", "나"]
    assert tree.source.name == "보고서.fake" and tree.source.mime == "text/x-fake"
    assert tree.source.content_hash == "sha256:" + hashlib.sha256(data).hexdigest()
    assert tree.source.page_count is None
    assert engine.history(ref.document_id).regions == ()


def test_default_id_is_deterministic_across_paths(engine, tmp_path):
    a = engine.ingest(str(write(tmp_path / "a" / "x.fake", "같은 내용\n")))
    b = engine.ingest(str(write(tmp_path / "b" / "다른 이름.fake", "같은 내용\n")))
    assert a == b and len(engine.documents()) == 1


def test_same_file_again_makes_no_version(engine, parser, tmp_path):
    path = write(tmp_path / "x.fake", "가\n")
    first = engine.ingest(str(path), document_id="문서1")
    again = engine.ingest(str(path), document_id="문서1")
    assert again == first and parser.calls == 1  # 원본 해시가 같으면 파싱하지 않는다
    assert len(engine.changes(None).changes) == 1


def test_modified_file_gives_exact_change_and_keeps_ids(engine, tmp_path):
    path = write(tmp_path / "x.fake", "가\n나\n다\n")
    engine.ingest(str(path), document_id="d")
    v1 = engine.get_tree("d")
    write(path, "가\n\n다\n라\n")  # 나 삭제(빈 줄로 줄 번호 유지), 라 추가
    ref = engine.ingest(str(path), document_id="d")
    v2 = engine.get_tree("d")
    assert ref.version == 2
    change = engine.changes(1).changes[0]
    by_text = {b.text: b.block_id for b in v1.blocks}
    assert change.removed == (by_text["나"],)
    assert change.added == (v2.blocks[2].block_id,)
    assert change.updated == (by_text["다"],)  # order가 2 → 1
    assert v2.blocks[0].block_id == by_text["가"] and v2.blocks[1].block_id == by_text["다"]


def test_changed_source_with_same_blocks_makes_no_version(engine, parser, tmp_path):
    path = write(tmp_path / "x.fake", "가\n")
    engine.ingest(str(path), document_id="d")
    write(path, "가\n\n\n")  # 바이트는 다르지만 블록은 같다
    assert engine.ingest(str(path), document_id="d").version == 1 and parser.calls == 2


def test_force_reparses_and_versions_only_on_difference(tmp_path):
    store = MemoryStore()
    path = write(tmp_path / "x.fake", "가\n")
    old = FakeParser()
    LocalEngine(store, parsers=[old]).ingest(str(path), document_id="d")
    assert LocalEngine(store, parsers=[old]).ingest(str(path), document_id="d", force=True).version == 1
    assert old.calls == 2
    upgraded = LocalEngine(store, parsers=[FakeParser(suffix=".")])
    assert upgraded.ingest(str(path), document_id="d").version == 1  # force 없이는 재파싱하지 않는다
    assert upgraded.ingest(str(path), document_id="d", force=True).version == 2


def test_change_feed_mixes_documents_in_order(engine, tmp_path):
    a = write(tmp_path / "a.fake", "가\n")
    b = write(tmp_path / "b.fake", "나\n")
    engine.ingest(str(a), document_id="A")
    engine.ingest(str(b), document_id="B")
    write(a, "가\n다\n")
    engine.ingest(str(a), document_id="A")
    first = engine.changes(None, limit=2)
    assert [(c.document_id, c.version) for c in first.changes] == [("A", 1), ("B", 1)]
    rest = engine.changes(first.next_cursor)
    assert [(c.document_id, c.version, c.previous_version) for c in rest.changes] == [("A", 2, 1)]
    assert engine.changes(rest.next_cursor).changes == ()


def test_parse_failure_leaves_store_untouched(engine, tmp_path):
    path = write(tmp_path / "x.fake", "가\n")
    engine.ingest(str(path), document_id="d")
    write(path, "가\n!fail\n")
    with pytest.raises(ParseError) as info:
        engine.ingest(str(path), document_id="d")
    assert info.value.location == "x.fake:2"
    assert engine.get_tree("d").version == 1 and engine.changes(None).next_cursor == 1


def test_unsupported_format(engine, tmp_path):
    with pytest.raises(UnsupportedFormat):
        engine.ingest(str(write(tmp_path / "x.txt", "가\n")))
    with pytest.raises(UnsupportedFormat):
        engine.ingest(str(write(tmp_path / "noext", "가\n")))
    assert engine.documents() == ()


def test_extension_match_is_case_insensitive(engine, tmp_path):
    assert engine.ingest(str(write(tmp_path / "X.FAKE", "가\n"))).version == 1


def test_empty_document_id_rejected(engine, tmp_path):
    with pytest.raises(ValueError):
        engine.ingest(str(write(tmp_path / "x.fake", "가\n")), document_id="")


def test_source_name_is_nfc_file_name_only(engine, tmp_path):
    nfd = unicodedata.normalize("NFD", "보고서.fake")
    path = write(tmp_path / "한글 폴더" / nfd, "가\n")
    tree = engine.get_tree(engine.ingest(str(path)).document_id)
    assert tree.source.name == "보고서.fake"
    assert tmp_path.name not in tree.model_dump_json()  # 전체 경로는 저장하지 않는다


def test_vlm_paths_unavailable(engine):
    with pytest.raises(VlmUnavailable):
        engine.run_vlm(DocFilter())
    with pytest.raises(VlmUnavailable):
        engine.job("j1")


class ConflictOnce(MemoryStore):
    """첫 커밋 직전에 다른 쓰기가 같은 문서의 버전 1을 먼저 넣은 상황을 흉내 낸다."""

    def __init__(self) -> None:
        super().__init__()
        self.conflicts = 0
        self.raced = False

    def commit(self, tree, change, history):
        if not self.raced:
            self.raced = True
            rival = build_tree(FakeParser().parse("경쟁\n".encode("utf-8"), "r.fake"), tree.document_id, 1,
                               SourceInfo(name="r.fake", mime="text/x-fake", content_hash="sha256:" + "1" * 64))
            super().commit(rival, diff_trees(None, rival),
                           ProcessingHistory(document_id=tree.document_id, version=1))
        try:
            return super().commit(tree, change, history)
        except StoreConflict:
            self.conflicts += 1
            raise


def test_store_conflict_is_retried_once(tmp_path):
    store = ConflictOnce()
    engine = LocalEngine(store, parsers=[FakeParser()])
    ref = engine.ingest(str(write(tmp_path / "x.fake", "가\n")), document_id="d")
    assert store.conflicts == 1 and ref.version == 2  # 최신(경쟁 버전)을 다시 읽고 그다음 버전으로
    assert [(c.version, c.previous_version) for c in engine.changes(None).changes] == [(1, None), (2, 1)]
    assert [b.text for b in engine.get_tree("d").blocks] == ["가"]


def test_store_conflict_twice_propagates(tmp_path):
    class AlwaysConflict(MemoryStore):
        def commit(self, tree, change, history):
            raise StoreConflict("busy")

    engine = LocalEngine(AlwaysConflict(), parsers=[FakeParser()])
    with pytest.raises(StoreConflict):
        engine.ingest(str(write(tmp_path / "x.fake", "가\n")))


def test_reingest_against_sqlite_keeps_unchanged_blocks_out_of_updated(tmp_path):
    path = write(tmp_path / "x.fake", "가\n나\n")
    with SqliteStore(tmp_path / "state.db") as store:
        engine = LocalEngine(store, parsers=[FakeParser()])
        engine.ingest(str(path), document_id="d")
        write(path, "가\n나\n다\n")
        engine.ingest(str(path), document_id="d")
        change = engine.changes(1).changes[0]
        assert change.updated == () and len(change.added) == 1  # JSON 왕복한 블록도 같다고 본다
