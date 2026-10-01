import hashlib
import math

import pytest
from pydantic import ValidationError

from ko_parser_contracts.document import Block, DocumentTree, SourceInfo, build_blocks
from ko_parser_contracts.ids import compute_block_id, compute_content_hash
from ko_parser_contracts.table import Cell, Table

FLOW = {"kind": "flow", "section_path": [], "paragraph_index": 0}
PAGE1 = {"kind": "page", "page": 1, "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.2}}
SRC = SourceInfo(name="a.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                 content_hash="sha256:" + hashlib.sha256(b"a").hexdigest(), page_count=None)


def para(text: str, idx: int = 0) -> dict:
    loc = dict(FLOW, paragraph_index=idx)
    return {"kind": "paragraph", "text": text, "locator": loc, "confidence": 1.0, "state": "det", "text_source": "native"}


def tree(blocks, **kw) -> DocumentTree:
    return DocumentTree(document_id="doc-1", version=1, layer_state="det", source=SRC, blocks=blocks, **kw)


def simple_table(src_a="text_layer", src_b="text_layer") -> Table:
    return Table(n_rows=1, n_cols=2, cells=[Cell(row=0, col=0, text="가", text_source=src_a),
                                            Cell(row=0, col=1, text="1", text_source=src_b)])


def test_build_blocks_assigns_order_hash_and_id():
    blocks = build_blocks("doc-1", [para("첫 문단"), para("둘째", 1)])
    assert [b.order for b in blocks] == [0, 1]
    assert blocks[0].content_hash == compute_content_hash("paragraph", "첫 문단", None, None)
    assert blocks[0].block_id == compute_block_id("doc-1", blocks[0].content_hash, 0)


def test_duplicate_paragraphs_get_distinct_ids():
    blocks = build_blocks("doc-1", [para("같은 문단"), para("같은 문단", 1)])
    assert blocks[0].block_id != blocks[1].block_id
    tree(blocks)


def test_build_blocks_rejects_precomputed_fields():
    with pytest.raises(ValueError):
        build_blocks("doc-1", [dict(para("x"), order=3)])


def test_table_block_text_and_source_rules():
    t = simple_table()
    blocks = build_blocks("doc-1", [{"kind": "table", "table": t, "locator": FLOW, "confidence": 0.9,
                                     "state": "det", "text_source": "text_layer"}])
    assert blocks[0].text == "가\t1"
    with pytest.raises(ValidationError, match="mixed"):
        build_blocks("doc-1", [{"kind": "table", "table": simple_table("text_layer", "vlm"), "locator": FLOW,
                                "confidence": 0.9, "state": "vlm", "text_source": "vlm"}])
    ok = build_blocks("doc-1", [{"kind": "table", "table": simple_table("text_layer", "vlm"), "locator": FLOW,
                                 "confidence": 0.9, "state": "vlm", "text_source": "mixed"}])
    assert ok[0].text_source == "mixed"


def test_kind_field_combinations():
    with pytest.raises(ValidationError, match="level"):
        build_blocks("doc-1", [dict(para("제목"), kind="heading")])
    with pytest.raises(ValidationError, match="level"):
        build_blocks("doc-1", [dict(para("본문"), level=1)])
    with pytest.raises(ValidationError, match="table"):
        build_blocks("doc-1", [dict(para("표"), kind="table")])
    with pytest.raises(ValidationError, match="mixed"):
        build_blocks("doc-1", [dict(para("본문"), text_source="mixed")])


def test_content_hash_mismatch_rejected():
    b = build_blocks("doc-1", [para("원문")])[0]
    with pytest.raises(ValidationError, match="content_hash"):
        Block.model_validate(dict(b.model_dump(), text="바뀐 글"))


def test_block_rejects_nan_confidence():
    with pytest.raises(ValidationError):
        build_blocks("doc-1", [dict(para("x"), confidence=math.nan)])
    with pytest.raises(ValidationError):
        build_blocks("doc-1", [dict(para("x"), confidence=1.5)])


def test_tree_rejects_wrong_block_id_and_unsorted_order():
    blocks = build_blocks("doc-1", [para("a"), para("b", 1)])
    with pytest.raises(ValidationError, match="block_id"):
        DocumentTree(document_id="doc-OTHER", version=1, layer_state="det", source=SRC, blocks=blocks)
    with pytest.raises(ValidationError, match="order"):
        tree(tuple(reversed(blocks)))


def test_tree_checks_pages():
    pblock = build_blocks("doc-1", [dict(para("p"), locator=PAGE1, text_source="text_layer")])
    with pytest.raises(ValidationError, match="unknown page"):
        tree(pblock)
    pages = [{"page": 1, "width_pt": 595.0, "height_pt": 842.0, "render_dpi": 144}]
    assert tree(pblock, pages=pages).pages[0].page == 1
    with pytest.raises(ValidationError, match="duplicate page"):
        tree(pblock, pages=pages * 2)


def test_empty_document_and_schema_version():
    assert tree(()).blocks == ()
    with pytest.raises(ValidationError):
        DocumentTree(schema_version="0.2", document_id="doc-1", version=1, layer_state="det", source=SRC)


def test_source_hash_format():
    with pytest.raises(ValidationError):
        SourceInfo(name="a", mime="x", content_hash="md5:abc")
