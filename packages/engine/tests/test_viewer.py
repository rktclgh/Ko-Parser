import json
import re

import pytest

from ko_parser.core import build_tree
from ko_parser.formats.base import ParsedSource
from ko_parser.viewer import render_html
from ko_parser_contracts import DocumentTree, PageInfo, SourceInfo, TextLayerStats

DATA = re.compile(r'<script type="application/json" id="ko-data">(.*?)</script>', re.DOTALL)
A4 = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144)
SCANNED = PageInfo(page=2, width_pt=595.0, height_pt=842.0, render_dpi=144, text_layer="scanned",
                   text_stats=TextLayerStats(chars=3, invisible_ratio=0.0, unmapped_ratio=0.0, pua_ratio=0.0,
                                             max_image_coverage=0.6))


def spec(text: str, y: float, kind: str = "paragraph", page: int = 1, **kw) -> dict:
    return {"kind": kind, "text": text, "confidence": 0.7, "state": "det", "text_source": "text_layer",
            "locator": {"kind": "page", "page": page, "bbox": {"x0": 0.1, "y0": y, "x1": 0.9, "y1": round(y + 0.02, 3)}},
            **kw}


def pdf_tree(version: int, *specs: dict, name: str = "보고서.pdf") -> DocumentTree:
    source = SourceInfo(name=name, mime="application/pdf", content_hash="sha256:" + "0" * 64, page_count=2)
    return build_tree(ParsedSource(mime="application/pdf", pages=(A4, SCANNED), blocks=specs), "d1", version, source)


def native(locator: dict, text: str = "가") -> dict:
    return {"kind": "paragraph", "text": text, "confidence": 1.0, "state": "det", "text_source": "native",
            "locator": locator}


def data_of(page: str) -> dict:
    (raw,) = DATA.findall(page)
    return json.loads(raw)


def test_structure_pages_blocks_and_page_states():
    # scanned 쪽(2쪽)도 보이는 글자(쪽 번호)는 블록이다(스펙 2026-10-03 보정)
    tree = pdf_tree(1, spec("제목", 0.1, "heading", level=1), spec("본문", 0.2), spec("- 2 -", 0.9, page=2))
    page = render_html(tree, {1: b"\xff\xd8jpeg-1", 2: b"\xff\xd8jpeg-2"})
    data = data_of(page)
    assert page.startswith("<!doctype html>") and "<title>보고서.pdf — ko-parser 뷰어</title>" in page
    assert [p["page"] for p in data["pages"]] == [1, 2] and len(data["blocks"]) == 3
    assert data["pages"][0]["image"] == "data:image/jpeg;base64,/9hqcGVnLTE="
    assert data["pages"][1]["state"] == "scanned" and data["pages"][1]["stats"]["max_image_coverage"] == 0.6
    assert data["document"]["page_states"] == {"digital": 1, "scanned": 1}
    first = data["blocks"][0]
    assert (first["kind"], first["level"], first["page"], first["bbox"], first["where"]) == (
        "heading", 1, 1, [0.1, 0.1, 0.9, 0.12], "1쪽")
    assert (first["text_source"], first["state"], first["confidence"], first["change"]) == ("text_layer", "det", 0.7,
                                                                                          None)
    assert [(b["text"], b["page"], b["where"]) for b in data["blocks"] if b["page"] == 2] == [("- 2 -", 2, "2쪽")]
    assert data["pages"][0]["notice"] is None
    notice = data["pages"][1]["notice"]
    assert notice == "그림 속 글자는 OCR 필요(보이는 글자만 블록)" and "만들지 않았다" not in notice


def test_unreliable_page_has_no_blocks_and_says_so():
    broken = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144, text_layer="unreliable",
                      text_stats=TextLayerStats(chars=120, invisible_ratio=0.0, unmapped_ratio=0.4, pua_ratio=0.0,
                                                max_image_coverage=0.0))
    source = SourceInfo(name="깨짐.pdf", mime="application/pdf", content_hash="sha256:" + "2" * 64, page_count=1)
    tree = build_tree(ParsedSource(mime="application/pdf", pages=(broken,)), "u1", 1, source)
    data = data_of(render_html(tree))
    assert data["blocks"] == [] and data["document"]["page_states"] == {"unreliable": 1}
    assert data["pages"][0]["state"] == "unreliable" and data["pages"][0]["stats"]["unmapped_ratio"] == 0.4
    assert data["pages"][0]["notice"] == "글자가 깨져 블록을 만들지 않았다"


def test_where_formats_slide_and_flow_locators():
    source = SourceInfo(name="발표.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        content_hash="sha256:" + "3" * 64)
    tree = build_tree(ParsedSource(mime=source.mime, blocks=[
        native({"kind": "slide", "slide": 3, "shape_index": 1}),
        native({"kind": "flow", "paragraph_index": 4}, "나")]), "s1", 1, source)
    data = data_of(render_html(tree))
    assert data["pages"] == []
    assert [(b["where"], b["page"], b["bbox"]) for b in data["blocks"]] == [("슬라이드 3", None, None),
                                                                          ("문단 4", None, None)]


def test_without_images_pages_have_no_image():
    data = data_of(render_html(pdf_tree(1, spec("본문", 0.2))))
    assert [p["image"] for p in data["pages"]] == [None, None]


def test_document_without_pages_lists_blocks_only():
    source = SourceInfo(name="메모.md", mime="text/markdown", content_hash="sha256:" + "1" * 64)
    tree = build_tree(ParsedSource(mime="text/markdown", blocks=[
        {"kind": "paragraph", "text": "가", "confidence": 1.0, "state": "det", "text_source": "native",
         "locator": {"kind": "lines", "line_start": 3, "line_end": 4}}]), "md", 1, source)
    data = data_of(render_html(tree))
    assert data["pages"] == [] and data["blocks"][0]["where"] == "3~4줄" and data["blocks"][0]["page"] is None


def test_previous_version_marks_added_updated_and_lists_removed():
    v1 = pdf_tree(1, spec("유지", 0.1), spec("위치만 바뀜", 0.2), spec("사라짐", 0.3))
    v2 = pdf_tree(2, spec("유지", 0.1), spec("위치만 바뀜", 0.25), spec("새 문단", 0.4))
    data = data_of(render_html(v2, previous=v1))
    assert [(b["text"], b["change"]) for b in data["blocks"]] == [
        ("유지", None), ("위치만 바뀜", "updated"), ("새 문단", "added")]
    assert [(r["text"], r["where"]) for r in data["removed"]] == [("사라짐", "1쪽")]
    assert data["document"]["previous_version"] == 1


@pytest.mark.parametrize("evil", ["</script><script>alert(1)</script>", "<!--<script>", "<img src=x onerror=alert(1)>",
                                  "__DATA__ __TITLE__ & \"따옴표\""])
def test_document_text_is_escaped(evil):
    tree = pdf_tree(1, spec(evil, 0.1), name=f"{evil}.pdf")
    page = render_html(tree)
    data = data_of(page)
    assert data["blocks"][0]["text"] == evil and data["document"]["name"] == f"{evil}.pdf"
    assert page.count("<script") == 2 and page.count("</script>") == 2  # 데이터 블록과 코드 블록뿐
    raw = DATA.findall(page)[0]
    assert "<" not in raw  # JSON 안의 '<'는 모두 \\u003c
    title = re.search(r"<title>(.*?)</title>", page, re.DOTALL).group(1)
    assert "<" not in title and ">" not in title


def test_no_external_resources_and_deterministic():
    tree = pdf_tree(1, spec("본문", 0.2))
    page = render_html(tree, {1: b"\xff\xd8x"})
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//|url\(\s*["']?(https?:)?//|@import|<link""", page)
    assert "default-src 'none'" in page
    assert render_html(tree, {1: b"\xff\xd8x"}) == page


def test_table_block_shows_markdown_table():
    from ko_parser_contracts import Cell, Table

    table = Table(n_rows=2, n_cols=2, cells=[
        Cell(row=0, col=0, text="구분", header="column", text_source="text_layer"),
        Cell(row=0, col=1, text="금액", header="column", text_source="text_layer"),
        Cell(row=1, col=0, text="인건비", text_source="text_layer"),
        Cell(row=1, col=1, text="4,250,000", text_source="text_layer")])
    tree = pdf_tree(1, dict(spec(table.plain_text(), 0.1, "table"), table=table))
    assert data_of(render_html(tree))["blocks"][0]["text"] == table.to_markdown()
    # 사라진 표 블록도 살아 있는 블록과 같은 글자 규칙(마크다운 표)
    removed = data_of(render_html(pdf_tree(2, spec("본문", 0.2)), previous=tree))["removed"]
    assert [(r["kind"], r["text"]) for r in removed] == [("table", table.to_markdown())]


def test_page_notice_is_not_overlaid_on_page_image():
    # 안내가 그림 위에 겹치면 scanned 쪽 위쪽의 보이는 글자 상자를 가린다
    page = render_html(pdf_tree(1, spec("- 2 -", 0.05, page=2)))
    (rule,) = re.findall(r"\.notice \{([^}]*)\}", page)
    assert "absolute" not in rule and "inset" not in rule


def merged_table(text: str = "실적"):
    from ko_parser_contracts import Cell, Table

    return Table(n_rows=2, n_cols=3, cells=[  # 칸 순서를 일부러 섞는다
        Cell(row=1, col=2, text="나", text_source="text_layer"),
        Cell(row=0, col=0, rowspan=2, text="구분", header="column", text_source="text_layer"),
        Cell(row=0, col=1, colspan=2, text=text, header="column", text_source="text_layer"),
        Cell(row=1, col=1, text="첫 줄\n둘째 줄", text_source="text_layer")])


def table_tree(version: int, text: str = "실적") -> DocumentTree:
    table = merged_table(text)
    return pdf_tree(version, dict(spec(table.plain_text(), 0.1, "table"), table=table))


def test_table_block_has_grid_cells_in_row_order():
    data = data_of(render_html(table_tree(1)))
    assert data["blocks"][0]["table"] == {"n_rows": 2, "n_cols": 3, "cells": [
        {"row": 0, "col": 0, "rowspan": 2, "colspan": 1, "text": "구분", "header": "column"},
        {"row": 0, "col": 1, "rowspan": 1, "colspan": 2, "text": "실적", "header": "column"},
        {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "첫 줄\n둘째 줄", "header": "none"},
        {"row": 1, "col": 2, "rowspan": 1, "colspan": 1, "text": "나", "header": "none"}]}
    # 사라진 표 블록도 같은 격자
    removed = data_of(render_html(pdf_tree(2, spec("본문", 0.2)), previous=table_tree(1)))["removed"]
    assert removed[0]["table"] == data["blocks"][0]["table"]
    assert data_of(render_html(pdf_tree(1, spec("본문", 0.2))))["blocks"][0]["table"] is None


@pytest.mark.parametrize("evil", ["</script><script>alert(1)</script>", "<img src=x onerror=alert(1)>", "<!--<script>"])
def test_table_cell_text_is_escaped(evil):
    page = render_html(table_tree(1, evil))
    assert data_of(page)["blocks"][0]["table"]["cells"][1]["text"] == evil
    assert page.count("<script") == 2 and page.count("</script>") == 2
    assert "<" not in DATA.findall(page)[0]


def test_grid_is_built_with_dom_and_text_content_only():
    """문서 글자를 HTML로 해석하는 API를 쓰지 않고, 칸 글자의 줄바꿈은 CSS pre-wrap으로 보인다(병합·머리 칸 모양은
    PR 마무리의 브라우저 확인)."""
    page = render_html(table_tree(1))
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page and "document.write" not in page
    (rule,) = re.findall(r"\.grid td, \.grid th \{([^}]*)\}", page)
    assert "white-space: pre-wrap" in rule
