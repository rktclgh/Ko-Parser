import pytest

from hanji import LocalEngine, MemoryStore
from hanji.core import build_tree
from hanji.errors import ParseError
from hanji.export import to_markdown
from hanji.formats.detect import default_parsers, detect_parser
from hanji.formats.markdown import MarkdownParser
from hanji_contracts import SourceInfo

SOURCE = SourceInfo(name="t.md", mime="text/markdown", content_hash="sha256:" + "0" * 64)


def parse(src: str) -> list[tuple]:
    """(kind, text, section_path, line_start, line_end). 계약 검증(build_tree)까지 통과해야 한다."""
    tree = build_tree(MarkdownParser().parse(src.encode("utf-8"), "t.md"), "d", 1, SOURCE)
    return [(b.kind, b.text, b.section_path, b.locator.line_start, b.locator.line_end) for b in tree.blocks]


def test_headings_atx_setext_and_section_stack():
    src = "# 가\n\n## 나\n\n### 다\n\n본문\n\n## 라\n\n마\n===\n\n끝\n"
    assert parse(src) == [
        ("heading", "가", (), 1, 1),
        ("heading", "나", ("가",), 3, 3),
        ("heading", "다", ("가", "나"), 5, 5),
        ("paragraph", "본문", ("가", "나", "다"), 7, 7),
        ("heading", "라", ("가",), 9, 9),
        ("heading", "마", (), 11, 12),
        ("paragraph", "끝", ("마",), 14, 14),
    ]


def test_heading_levels_and_empty_heading_skipped():
    tree = build_tree(MarkdownParser().parse("###### 여섯\n\n#\n\n본문\n".encode("utf-8"), "t.md"), "d", 1, SOURCE)
    assert [(b.kind, b.level) for b in tree.blocks] == [("heading", 6), ("paragraph", None)]
    assert tree.blocks[1].section_path == ("여섯",)


def test_paragraph_inline_marks_removed_and_breaks_kept():
    src = "본 문서는 **2026년** [추진 현황](https://example.com)을 *정리*한다.\n둘째 줄 `코드`.  \n강제 줄바꿈 &amp; \\* 끝\n"
    assert parse(src) == [
        ("paragraph", "본 문서는 2026년 추진 현황을 정리한다.\n둘째 줄 코드.\n강제 줄바꿈 & * 끝", (), 1, 3)]


def test_list_items_nested_numbered_and_empty():
    src = "- 첫 항목\n  - 중첩 항목\n- \n\n3. 셋째\n4. 넷째\n\n7) 괄호\n"
    assert parse(src) == [
        ("list_item", "첫 항목", (), 1, 1),
        ("list_item", "중첩 항목", (), 2, 2),
        ("list_item", "3. 셋째", (), 5, 5),
        ("list_item", "4. 넷째", (), 6, 6),
        ("list_item", "7) 괄호", (), 8, 8),
    ]


def test_loose_list_item_keeps_only_leading_paragraph():
    src = "- 첫 문단\n\n  둘째 문단\n\n  ```\n  코드\n  ```\n- # 제목 항목\n"
    assert parse(src) == [
        ("list_item", "첫 문단", (), 1, 1),
        ("paragraph", "둘째 문단", (), 3, 3),
        ("paragraph", "코드", (), 5, 7),
        ("heading", "제목 항목", (), 8, 8),
    ]


def test_table_cells_header_and_escapes():
    src = "| 분기 | 매출 |\n|---|---|\n| *1분기* | 1\\|250 |\n| 2분기 |\n"
    tree = build_tree(MarkdownParser().parse(src.encode("utf-8"), "t.md"), "d", 1, SOURCE)
    (block,) = tree.blocks
    assert (block.kind, block.text_source, block.locator.line_start, block.locator.line_end) == ("table", "native", 1, 4)
    assert block.table.to_grid() == (("분기", "매출"), ("1분기", "1|250"), ("2분기", ""))
    assert {c.header for c in block.table.cells if c.row == 0} == {"column"}
    assert {c.header for c in block.table.cells if c.row > 0} == {"none"}
    assert {c.text_source for c in block.table.cells} == {"native"}
    assert block.text == "분기\t매출\n1분기\t1|250\n2분기\t"


def test_header_only_table():
    tree = build_tree(MarkdownParser().parse("| 가 | 나 |\n|---|---|\n".encode("utf-8"), "t.md"), "d", 1, SOURCE)
    assert tree.blocks[0].table.n_rows == 1 and tree.blocks[0].text == "가\t나"


def test_code_blocks_html_and_rule():
    src = "```python\nprint(\"안녕\")\n\n```\n\n    들여쓰기 코드\n\n<div>\nHTML 블록\n</div>\n\n---\n"
    assert parse(src) == [
        ("paragraph", 'print("안녕")', (), 1, 4),
        ("paragraph", "들여쓰기 코드", (), 6, 6),
        ("paragraph", "<div>\nHTML 블록\n</div>", (), 8, 10),
    ]


def test_blockquote_unwrapped():
    src = "> # 인용 제목\n> 인용 *문단*\n> - 인용 항목\n"
    assert parse(src) == [
        ("heading", "인용 제목", (), 1, 1),
        ("paragraph", "인용 문단", ("인용 제목",), 2, 2),
        ("list_item", "인용 항목", ("인용 제목",), 3, 3),
    ]


def test_figures():
    src = "![조직도 그림](org.png)\n\n![](빈.png)\n\n![가](a.png) ![나](b.png)\n\n글 ![사진](c.png) 뒤\n"
    assert parse(src) == [
        ("figure", "조직도 그림", (), 1, 1),
        ("figure", "가\n나", (), 5, 5),
        ("paragraph", "글 사진 뒤", (), 7, 7),
    ]


def test_linked_image_paragraph_is_figure():
    src = "[![배지](b.png)](https://x)\n\n[![가](a.png)](u) [![나](c.png)](v)\n\n[글 ![사진](c.png)](u)\n"
    assert parse(src) == [
        ("figure", "배지", (), 1, 1),
        ("figure", "가\n나", (), 3, 3),
        ("paragraph", "글 사진", (), 5, 5),
    ]


def test_multiline_setext_heading_folded_to_one_line():
    tree = build_tree(MarkdownParser().parse("마\n바\n===\n\n본문\n".encode("utf-8"), "t.md"), "d", 1, SOURCE)
    assert [(b.kind, b.text, b.section_path, b.locator.line_start, b.locator.line_end) for b in tree.blocks] == [
        ("heading", "마 바", (), 1, 3),
        ("paragraph", "본문", ("마 바",), 5, 5),
    ]
    assert to_markdown(tree) == "# 마 바\n\n본문\n"


def test_block_fields_fixed():
    tree = build_tree(MarkdownParser().parse("# 가\n\n나\n".encode("utf-8"), "t.md"), "d", 1, SOURCE)
    for block in tree.blocks:
        assert (block.state, block.text_source, block.confidence, block.locator.kind) == ("det", "native", 1.0, "lines")
        assert block.locator.section_path == block.section_path
    assert tree.pages == ()


@pytest.mark.parametrize("src", ["", "\n\n   \n", "---\n", "- \n"])
def test_empty_documents_have_no_blocks(src):
    assert parse(src) == []


def test_crlf_and_cp949_give_same_lines():
    lf = "# 가\n\n나\n다\n"
    parser = MarkdownParser()
    expected = parser.parse(lf.encode("utf-8"), "t.md")
    assert parser.parse(lf.replace("\n", "\r\n").encode("cp949"), "t.md") == expected


def test_deep_inline_nesting_does_not_recurse():
    src = "*" * 3000 + "가" + "*" * 3000 + "\n"
    assert parse(src) == [("paragraph", "가", (), 1, 1)]


def test_oversized_table_is_parse_error(monkeypatch):
    monkeypatch.setattr("hanji_contracts.table.MAX_TABLE_CELLS", 4)
    with pytest.raises(ParseError) as info:
        MarkdownParser().parse("앞\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n".encode("utf-8"), "큰표.md")
    assert info.value.location == "큰표.md:3-6" and "invalid table" in info.value.reason


def test_undecodable_is_parse_error():
    with pytest.raises(ParseError):
        MarkdownParser().parse(b"\xff\xfe\x00", "t.md")


def test_detect_by_extension():
    parsers = default_parsers()
    for name in ("a.md", "B.MD", "c.markdown", "보고서.Markdown"):
        assert isinstance(detect_parser(name, parsers), MarkdownParser)


def test_default_engine_parses_markdown(tmp_path):
    path = tmp_path / "메모.md"
    path.write_bytes("# 제목\n\n본문\n".encode("utf-8"))
    engine = LocalEngine(MemoryStore())
    tree = engine.get_tree(engine.ingest(str(path)).document_id)
    assert tree.source.mime == "text/markdown" and [b.kind for b in tree.blocks] == ["heading", "paragraph"]


def _nested_list(depth: int) -> str:
    return "".join("  " * i + f"- a{i}\n" for i in range(depth)) + "\n뒤 문단\n"


@pytest.mark.parametrize("src", [_nested_list(12), "> " * 20 + "깊은\n", "> " * 5000 + "가\n"])
def test_block_nesting_limit_is_parse_error(src):
    with pytest.raises(ParseError) as info:
        MarkdownParser().parse(src.encode("utf-8"), "깊은.md")
    assert info.value.reason == "block nesting too deep" and info.value.location == "깊은.md"


def test_deep_but_within_limit_parses_fully():
    assert [b[1] for b in parse(_nested_list(5))] == ["a0", "a1", "a2", "a3", "a4", "뒤 문단"]
    assert parse("> " * 8 + "가\n") == [("paragraph", "가", (), 1, 1)]
    assert [b[1] for b in parse(_nested_list(8))][-2:] == ["a7", "뒤 문단"]


def test_many_tables_scan_is_linear():
    import time

    src = "| a | b |\n|---|---|\n| 1 | 2 |\n\n문단\n\n" * 3000
    start = time.perf_counter()
    blocks = parse(src)
    assert len(blocks) == 6000 and time.perf_counter() - start < 5


def test_removed_image_leaves_no_edge_whitespace():
    assert parse("![](x.png) 뒤\n\n앞 ![](x.png)\n\n가 ![](x.png) 나\n") == [
        ("paragraph", "뒤", (), 1, 1),
        ("paragraph", "앞", (), 3, 3),
        ("paragraph", "가  나", (), 5, 5),
    ]
