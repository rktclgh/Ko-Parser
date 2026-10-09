import importlib.util
import sys
from pathlib import Path

import pytest

from hanji import LocalEngine, MemoryStore
from hanji.formats.detect import default_parsers
from hanji_contracts import DocumentTree

HERE = Path(__file__).resolve().parent
ROOT = HERE / "fixtures" / "pdf"


def _load_builder():
    # 다른 생성기와 모듈 이름이 겹치지 않게 따로 이름 붙인다
    spec = importlib.util.spec_from_file_location("engine_build_pdf_fixtures", HERE / "build_pdf_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()
NAMES = sorted(BUILDER.SAMPLES)


def _expected(name: str) -> DocumentTree:
    return DocumentTree.model_validate_json((ROOT / "expected" / f"{Path(name).stem}.json").read_bytes())


def test_expected_fixture_files_exist():
    expected = {f"inputs/{name}" for name in NAMES} | {f"expected/{Path(name).stem}.json" for name in NAMES}
    assert {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()} >= expected


def test_fixtures_are_up_to_date():
    """reportlab invariant 바이트와 추출 결과(반올림한 bbox)가 OS와 관계없이 같아야 한다."""
    for rel, data in BUILDER.build_all().items():
        assert (ROOT / rel).read_bytes() == data, rel


@pytest.mark.parametrize("name", NAMES)
def test_engine_ingest_matches_golden(name):
    engine = LocalEngine(MemoryStore(), default_parsers(ocr=False, layout=False))
    ref = engine.ingest(str(ROOT / "inputs" / name), document_id=BUILDER.document_id(name))
    assert engine.get_tree(ref.document_id) == _expected(name)


def test_report_headings_lists_and_paragraphs():
    tree = _expected("report.pdf")
    headings = [(b.level, b.text) for b in tree.blocks if b.kind == "heading"]
    assert headings == [(1, "2026년 사업 계획 (초안)"), (2, "1. 추진 배경"), (3, "가. 세부 목표"), (2, "2. 예산"),
                        (4, "핵심 지표")]
    lists = [b.text for b in tree.blocks if b.kind == "list_item"]
    assert lists == ["□ 첫째, 참여 기관을 모집한다.\n모집 공고는 1분기에 낸다.", "○ 둘째, 선정 결과를 통보한다.",
                     "- 셋째, 협약을 체결한다."]
    texts = [b.text for b in tree.blocks]
    assert "큰 글자로 쓴 강조 문단은\n세 줄 이상 이어지면\n제목이 아니라 문단이다." in texts
    *_, last = tree.blocks  # 두 줄 두 열 정렬 글자는 2×2 선 없는 표다(스펙 결정 D2)
    assert (last.kind, last.confidence, [c.text for c in last.table.cells]) == (
        "table", 0.4, ["구분", "금액(원)", "인건비", "4,250,000"])
    assert tree.blocks[4].section_path == ("2026년 사업 계획 (초안)", "1. 추진 배경", "가. 세부 목표")
    assert {p.text_layer for p in tree.pages} == {"digital"} and tree.source.page_count == 2


def test_header_footer_repeat_but_one_off_margin_text_is_paragraph():
    tree = _expected("header_footer.pdf")
    assert [(b.kind, b.text) for b in tree.blocks if b.kind.startswith("page_")] == [
        ("page_header", "2026년 사업 계획 보고"), ("page_footer", "- 1 -"),
        ("page_header", "2026년 사업 계획 보고"), ("page_footer", "- 2 -"),
        ("page_header", "2026년 사업 계획 보고"), ("page_footer", "- 3 -")]
    assert ("paragraph", "대외비") in [(b.kind, b.text) for b in tree.blocks]


@pytest.mark.parametrize("name,invisible,coverage", [("scanned_invisible.pdf", 0.9268, 0.0),
                                                     ("image_page.pdf", 0.0, 0.2954)])
def test_scanned_pages_keep_reasons_and_only_visible_text(name, invisible, coverage):
    """숨은 글자층은 블록이 되지 않고 보이는 쪽 번호만 블록으로 남는다."""
    tree = _expected(name)
    page_number = {"scanned_invisible.pdf": "- 1 -", "image_page.pdf": "- 3 -"}[name]
    assert [(b.text, b.locator.page) for b in tree.blocks] == [(page_number, 1)]
    (page,) = tree.pages
    assert page.text_layer == "scanned"
    assert (page.text_stats.chars, page.text_stats.invisible_ratio, page.text_stats.max_image_coverage) == (
        3, invisible, coverage)


def test_empty_page_is_digital_without_blocks():
    tree = _expected("empty.pdf")
    assert tree.blocks == () and tree.pages[0].text_layer == "digital" and tree.pages[0].text_stats.chars == 0


def test_table_fixture_has_one_table_block_between_paragraphs():
    tree = _expected("table.pdf")
    assert [b.kind for b in tree.blocks] == ["heading", "paragraph", "table", "paragraph"]
    block = tree.blocks[2]
    assert [(c.row, c.col, c.rowspan, c.colspan, c.text, c.header) for c in block.table.cells] == [
        (0, 0, 1, 1, "구분", "column"), (0, 1, 1, 2, "상반기", "column"), (0, 3, 1, 1, "비고", "column"),
        (1, 0, 2, 1, "사업", "none"), (1, 1, 1, 1, "1분기", "none"), (1, 2, 1, 1, "2분기", "none"),
        (1, 3, 1, 1, "", "none"), (2, 1, 1, 1, "3건", "none"), (2, 2, 1, 1, "5건", "none"), (2, 3, 1, 1, "", "none"),
        (3, 0, 1, 1, "합계", "none"), (3, 1, 1, 1, "3건", "none"), (3, 2, 1, 1, "5건", "none"),
        (3, 3, 1, 1, "누적\n8건", "none")]
    assert block.section_path == ("1. 추진 실적",) and block.text == block.table.plain_text()
    assert (block.locator.page, block.locator.bbox.x0, block.locator.bbox.y1) == (1, 0.121, 0.259)
    assert "상반기" not in "".join(b.text for b in tree.blocks if b.kind != "table")


def test_golden_pages_carry_a_ledger_that_adds_up():
    """골든의 모든 쪽에 글자 장부가 있고 보이는 글자는 모두 블록에 들었다. 숨은 글자는 scanned_invisible만 38."""
    hidden = {}
    for name in NAMES:
        for p in _expected(name).pages:
            assert p.coverage is not None and p.coverage.in_blocks == p.text_stats.chars, (name, p.page)
            assert (p.coverage.replaced, p.coverage.rescued) == (0, 0)
            hidden[(name, p.page)] = p.coverage.hidden
    assert {k: v for k, v in hidden.items() if v} == {("scanned_invisible.pdf", 1): 38}


def test_borderless_fixture_has_two_borderless_tables_and_a_list():
    tree = _expected("borderless.pdf")
    assert [(b.kind, b.confidence) for b in tree.blocks] == [
        ("heading", 0.6), ("paragraph", 0.7), ("table", 0.4), ("heading", 0.6), ("table", 0.4), ("list_item", 0.7),
        ("list_item", 0.7), ("list_item", 0.7)]
    first, second = (b.table for b in tree.blocks if b.kind == "table")
    assert [[c.text for c in first.cells if c.row == r] for r in range(first.n_rows)] == [
        ["구분", "2025년", "2026년"], ["인건비", "1,200", "1,350"], ["운영비", "850", "900"], ["합계", "2,050", "2,250"]]
    assert [[c.text for c in second.cells if c.row == r] for r in range(second.n_rows)] == [
        ["분기", "건수", "금액"], ["1분기", "3건", "120"], ["2분기", "5건", "210"], ["3분기", "4건", "180"]]
    assert tree.blocks[4].section_path == ("2. 분기 실적",)


def test_core_install_without_numpy_or_onnxruntime_gives_the_borderless_golden(monkeypatch):
    """hanji-core(레이아웃·OCR 추가 설치 없음): 두 모듈을 import할 수 없게 막아도 검출기 경로로 같은 골든이 나온다."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    from hanji.formats.pdf import layout

    assert not layout.available()  # 막기가 듣지 않으면 이 테스트가 조용히 통과하지 못하게 한다
    engine = LocalEngine(MemoryStore(), default_parsers())  # OCR·레이아웃 자동: 설치가 없는 것으로 본다
    ref = engine.ingest(str(ROOT / "inputs" / "borderless.pdf"), document_id=BUILDER.document_id("borderless.pdf"))
    assert engine.get_tree(ref.document_id) == _expected("borderless.pdf")
