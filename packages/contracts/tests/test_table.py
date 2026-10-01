import re

import pytest
from pydantic import ValidationError

from ko_parser_contracts.table import Cell, Table


def c(row, col, text="", rs=1, cs=1, header="none", src="text_layer"):
    return Cell(row=row, col=col, rowspan=rs, colspan=cs, text=text, header=header, text_source=src)


def merged_table() -> Table:
    # 분류(2행 병합) | 2025년(2열 병합)
    #                | 상반기 | 하반기
    # 인건비(행 머리) | 4,250,000 | 4,310,000
    return Table(
        n_rows=3,
        n_cols=3,
        cells=[
            c(0, 0, "분류", rs=2, header="column"),
            c(0, 1, "2025년", cs=2, header="column"),
            c(1, 1, "상반기", header="column"),
            c(1, 2, "하반기", header="column"),
            c(2, 0, "인건비", header="row"),
            c(2, 1, "4,250,000"),
            c(2, 2, "4,310,000"),
        ],
    )


def test_valid_merged_table():
    t = merged_table()
    assert isinstance(t.cells, tuple)


def test_overlap_rejected():
    with pytest.raises(ValidationError, match="overlap"):
        Table(n_rows=1, n_cols=2, cells=[c(0, 0, cs=2), c(0, 1)])


def test_uncovered_rejected():
    with pytest.raises(ValidationError, match="uncovered"):
        Table(n_rows=1, n_cols=2, cells=[c(0, 0)])


def test_out_of_bounds_rejected():
    with pytest.raises(ValidationError, match="bounds"):
        Table(n_rows=1, n_cols=1, cells=[c(0, 0, cs=2)])


def test_duplicate_anchor_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        Table(n_rows=1, n_cols=1, cells=[c(0, 0), c(0, 0)])


def test_to_grid_repeats_merged_text():
    assert merged_table().to_grid() == (
        ("분류", "2025년", "2025년"),
        ("분류", "상반기", "하반기"),
        ("인건비", "4,250,000", "4,310,000"),
    )


def test_plain_text_lists_anchor_cells_once():
    assert merged_table().plain_text() == "분류\t2025년\n상반기\t하반기\n인건비\t4,250,000\t4,310,000"


def test_to_html_golden():
    assert merged_table().to_html() == (
        "<table><thead>"
        '<tr><th rowspan="2">분류</th><th colspan="2">2025년</th></tr>'
        "<tr><th>상반기</th><th>하반기</th></tr>"
        "</thead><tbody>"
        "<tr><th>인건비</th><td>4,250,000</td><td>4,310,000</td></tr>"
        "</tbody></table>"
    )


def test_html_escapes_and_breaks():
    t = Table(n_rows=1, n_cols=1, cells=[c(0, 0, 'a<b & "c"\nd')])
    assert t.to_html() == "<table><tbody><tr><td>a&lt;b &amp; &quot;c&quot;<br>d</td></tr></tbody></table>"


def test_header_only_table():
    t = Table(n_rows=1, n_cols=2, cells=[c(0, 0, "가", header="column"), c(0, 1, "", header="column")])
    assert t.to_html() == "<table><thead><tr><th>가</th><th></th></tr></thead><tbody></tbody></table>"


def test_to_markdown_is_lossy_grid():
    assert merged_table().to_markdown() == (
        "| 분류 | 2025년 | 2025년 |\n"
        "| --- | --- | --- |\n"
        "| 분류 | 상반기 | 하반기 |\n"
        "| 인건비 | 4,250,000 | 4,310,000 |"
    )


def test_markdown_escapes_pipe_and_newline():
    t = Table(n_rows=1, n_cols=1, cells=[c(0, 0, "a|b\nc")])
    assert t.to_markdown() == "| a\\|b<br>c |\n| --- |"


def test_cell_requires_text_source():
    with pytest.raises(ValidationError):
        Cell(row=0, col=0)


def test_grid_over_cell_cap_rejected():
    with pytest.raises(ValidationError, match="table grid"):
        Table(n_rows=1000, n_cols=101, cells=[c(0, 0, rs=1000, cs=101)])


def test_uncovered_reports_first_positions():
    with pytest.raises(ValidationError, match=re.escape("uncovered grid positions: [(0, 1), (1, 0), (1, 1)]")):
        Table(n_rows=2, n_cols=2, cells=[c(0, 0, "A")])


def test_empty_cells_at_cap_rejected_fast():
    with pytest.raises(ValidationError, match="uncovered"):
        Table(n_rows=1, n_cols=100_000, cells=[])


def test_to_html_keeps_rowspan_in_one_row_group():
    t = Table(
        n_rows=2,
        n_cols=2,
        cells=[c(0, 0, "A", rs=2, header="column"), c(0, 1, "B", header="column"), c(1, 1, "C")],
    )
    assert t.to_html() == '<table><tbody><tr><th rowspan="2">A</th><th>B</th></tr><tr><td>C</td></tr></tbody></table>'


def test_to_html_header_shrinks_to_uncrossed_boundary():
    t = Table(
        n_rows=3,
        n_cols=2,
        cells=[
            c(0, 0, "H1", header="column"),
            c(0, 1, "H2", header="column"),
            c(1, 0, "X", rs=2, header="column"),
            c(1, 1, "Y", header="column"),
            c(2, 1, "Z"),
        ],
    )
    assert t.to_html() == (
        "<table><thead><tr><th>H1</th><th>H2</th></tr></thead>"
        '<tbody><tr><th rowspan="2">X</th><th>Y</th></tr><tr><td>Z</td></tr></tbody></table>'
    )


def test_markdown_escapes_backslash_before_pipe():
    t = Table(n_rows=1, n_cols=1, cells=[c(0, 0, "a\\|b")])
    assert t.to_markdown() == "| a\\\\\\|b |\n| --- |"
    t = Table(n_rows=1, n_cols=1, cells=[c(0, 0, "x\r\ny")])
    assert t.to_markdown() == "| x<br>y |\n| --- |"


def test_plain_text_many_rows():
    t = Table(n_rows=20_000, n_cols=1, cells=[c(r, 0, "x") for r in range(20_000)])
    assert t.plain_text() == "\n".join(["x"] * 20_000)


def test_html_normalizes_cr_line_breaks():
    t = Table(n_rows=1, n_cols=1, cells=[c(0, 0, "a\r\nb\rc")])
    assert "<td>a<br>b<br>c</td>" in t.to_html()
