import importlib.util
import shutil
from pathlib import Path

import pytest

from ko_parser import LocalEngine, MemoryStore
from ko_parser.export import to_markdown
from ko_parser_contracts import DocumentTree

HERE = Path(__file__).resolve().parent
ROOT = HERE / "fixtures"


def _load_builder():
    # 계약 패키지의 build_fixtures와 모듈 이름이 겹치지 않게 따로 이름 붙인다
    spec = importlib.util.spec_from_file_location("engine_build_fixtures", HERE / "build_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()
NAMES = sorted(BUILDER.SAMPLES)


def _expected(name: str) -> DocumentTree:
    return DocumentTree.model_validate_json((ROOT / "expected" / f"{Path(name).stem}.json").read_bytes())


def _content(tree: DocumentTree) -> list[tuple]:
    return [(b.kind, b.text, b.level, b.section_path, b.locator) for b in tree.blocks]


def test_expected_fixture_files_exist():
    assert {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()} >= {
        "inputs/report.md", "inputs/memo_bom_crlf.md", "inputs/memo_cp949.md", "inputs/empty.md",
        "expected/report.json", "expected/report.md", "expected/empty.json", "expected/empty.md",
    }


def test_fixtures_are_up_to_date():
    for rel, data in BUILDER.build_all().items():
        assert (ROOT / rel).read_bytes() == data, rel


@pytest.mark.parametrize("name", NAMES)
def test_expected_tree_roundtrips_and_exports(name):
    tree = _expected(name)
    assert DocumentTree.model_validate_json(tree.model_dump_json()) == tree
    assert (ROOT / "expected" / f"{Path(name).stem}.md").read_bytes().decode("utf-8") == to_markdown(tree)


@pytest.mark.parametrize("name", NAMES)
def test_engine_ingest_matches_golden(name):
    engine = LocalEngine(MemoryStore())
    ref = engine.ingest(str(ROOT / "inputs" / name), document_id=BUILDER.document_id(name))
    assert engine.get_tree(ref.document_id) == _expected(name)


def test_bom_crlf_and_cp949_match():
    assert _content(_expected("memo_bom_crlf.md")) == _content(_expected("memo_cp949.md"))
    assert (ROOT / "inputs" / "memo_bom_crlf.md").read_bytes().startswith(b"\xef\xbb\xbf# ")
    assert b"\r\n" in (ROOT / "inputs" / "memo_bom_crlf.md").read_bytes()


def test_report_covers_every_rule():
    tree = _expected("report.md")
    assert {b.kind for b in tree.blocks} == {"heading", "paragraph", "list_item", "table", "figure"}
    texts = [b.text for b in tree.blocks]
    assert "3. 셋째 단계" in texts and "공고문 게시" in texts
    assert texts.count("자세한 내용은 별첨을 참고한다.") == 2
    assert len({b.block_id for b in tree.blocks}) == len(tree.blocks)


def test_korean_file_name_in_korean_directory(tmp_path):
    target = tmp_path / "한글 경로" / "하위 폴더" / "회의록.md"
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "inputs" / "memo_cp949.md", target)
    engine = LocalEngine(MemoryStore())
    tree = engine.get_tree(engine.ingest(str(target)).document_id)
    assert tree.source.name == "회의록.md"
    assert _content(tree) == _content(_expected("memo_cp949.md"))
