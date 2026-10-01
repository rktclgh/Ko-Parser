import re

import pytest

from ko_parser_contracts.ids import compute_block_id, compute_content_hash, normalize_text
from ko_parser_contracts.table import Cell, Table


def table(text: str) -> Table:
    return Table(n_rows=1, n_cols=2, cells=[
        Cell(row=0, col=1, text="b", text_source="vlm"),
        Cell(row=0, col=0, text=text, text_source="vlm"),
    ])


def test_normalize_text():
    assert normalize_text("  가  나\n다\t") == "가 나 다"
    assert normalize_text("１２３") == "123"


def test_hash_format_and_determinism():
    h = compute_content_hash("paragraph", "본문", None, None)
    assert re.fullmatch(r"c_[0-9a-f]{32}", h)
    assert h == compute_content_hash("paragraph", "본문", None, None)


def test_hash_normalizes_fullwidth_and_whitespace():
    assert compute_content_hash("paragraph", "금액 １２ 원", None, None) == compute_content_hash(
        "paragraph", "금액  12\n원", None, None
    )


def test_hash_depends_on_kind_level_and_table():
    base = compute_content_hash("paragraph", "가", None, None)
    assert base != compute_content_hash("heading", "가", 1, None)
    assert compute_content_hash("heading", "가", 1, None) != compute_content_hash("heading", "가", 2, None)
    assert compute_content_hash("table", "x", None, table("a")) != compute_content_hash("table", "x", None, table("z"))


def test_hash_ignores_cell_order():
    t1 = table("a")
    t2 = Table(n_rows=1, n_cols=2, cells=list(reversed(t1.cells)))
    assert compute_content_hash("table", "x", None, t1) == compute_content_hash("table", "x", None, t2)


def test_block_id_uses_occurrence():
    h = compute_content_hash("paragraph", "같은 문단", None, None)
    a, b = compute_block_id("doc-1", h, 0), compute_block_id("doc-1", h, 1)
    assert a != b and re.fullmatch(r"b_[0-9a-f]{24}", a)
    assert compute_block_id("doc-2", h, 0) != a


def test_block_id_rejects_negative_occurrence():
    with pytest.raises(ValueError):
        compute_block_id("doc-1", "c_" + "0" * 32, -1)
