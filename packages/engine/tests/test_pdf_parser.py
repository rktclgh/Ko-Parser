import io

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from ko_parser import LocalEngine, MemoryStore
from ko_parser.errors import ParseError
from ko_parser.formats.detect import default_parsers, detect_parser
from ko_parser.formats.pdf import PdfParser

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))


def make_pdf(pages: list[list[tuple[float, str, int]]], **kw) -> bytes:
    """쪽마다 (기준선 y, 글자, 렌더 모드) 줄 목록. 11pt, x=72."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, 842.0), invariant=1, pageCompression=0, **kw)
    for lines in pages:
        for y, s, mode in lines:
            c.saveState()
            t = c.beginText(72, y)
            t.setFont(FONT, 11)
            t.setTextRenderMode(mode)
            t.textOut(s)
            c.drawText(t)
            c.restoreState()
        c.showPage()
    c.save()
    return buf.getvalue()


def write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


PARAS = [[(770, "첫째 문단이다.", 0), (730, "둘째 문단이다.", 0), (690, "셋째 문단이다.", 0)]]


def test_registered_for_pdf_extension_case_insensitive():
    parser = detect_parser("보고서.PDF", default_parsers())
    assert isinstance(parser, PdfParser) and parser.mimes == ("application/pdf",)


def test_pages_carry_state_stats_and_render_dpi():
    data = make_pdf([[(770, "보이는 쪽이다.", 0)], [(770, "숨은 글자층이다.", 3)]])
    parsed = PdfParser().parse(data, "a.pdf")
    assert parsed.mime == "application/pdf"
    assert [(p.page, p.text_layer, p.render_dpi, p.rotation) for p in parsed.pages] == [
        (1, "digital", 144, 0), (2, "scanned", 144, 0)]
    assert parsed.pages[0].text_stats.chars == 7 and parsed.pages[1].text_stats.invisible_ratio == 1.0
    assert [(b["text"], b["locator"]["page"]) for b in parsed.blocks] == [("보이는 쪽이다.", 1)]


def test_engine_ingest_pdf_source_and_page_count(tmp_path):
    engine = LocalEngine(MemoryStore())
    tree = engine.get_tree(engine.ingest(str(write(tmp_path / "보고서.pdf", make_pdf(PARAS * 2)))).document_id)
    assert (tree.source.name, tree.source.mime, tree.source.page_count) == ("보고서.pdf", "application/pdf", 2)
    assert len(tree.pages) == 2 and len(tree.blocks) == 6


@pytest.mark.parametrize("data,reason", [(make_pdf(PARAS, encrypt="secret"), "encrypted PDF"),
                                         (b"%PDF-1.7\n%%EOF\n", "invalid PDF")])
def test_encrypted_or_corrupt_pdf_leaves_store_untouched(tmp_path, data, reason):
    engine = LocalEngine(MemoryStore())
    with pytest.raises(ParseError, match=reason) as info:
        engine.ingest(str(write(tmp_path / "깨짐.pdf", data)))
    assert info.value.location == "깨짐.pdf"
    assert engine.documents() == () and engine.changes(None).changes == ()


def test_reingest_keeps_block_ids(tmp_path):
    path = write(tmp_path / "a.pdf", make_pdf(PARAS))
    engine = LocalEngine(MemoryStore())
    first = engine.ingest(str(path), document_id="d")
    v1 = engine.get_tree("d")
    assert engine.ingest(str(path), document_id="d") == first
    assert engine.ingest(str(path), document_id="d", force=True) == first  # 다시 파싱해도 블록이 같다
    write(path, make_pdf([[(770, "첫째 문단이다.", 0), (730, "고친 둘째 문단이다.", 0), (690, "셋째 문단이다.", 0)]]))
    assert engine.ingest(str(path), document_id="d").version == 2
    v2 = engine.get_tree("d")
    change = engine.changes(1).changes[0]
    assert (v2.blocks[0].block_id, v2.blocks[2].block_id) == (v1.blocks[0].block_id, v1.blocks[2].block_id)
    assert change.added == (v2.blocks[1].block_id,) and change.removed == (v1.blocks[1].block_id,)
    assert change.updated == ()


def test_page_state_change_without_block_change_is_a_new_version(tmp_path):
    path = write(tmp_path / "a.pdf", make_pdf([[(770, "숨은 글자층이다.", 3)]]))
    engine = LocalEngine(MemoryStore())
    engine.ingest(str(path), document_id="d")
    write(path, make_pdf([[(770, "다른 숨은 글자층이다.", 3), (40, "1", 0)]]))
    ref = engine.ingest(str(path), document_id="d")
    assert ref.version == 2 and engine.get_tree("d").pages[0].text_stats.chars == 1
    change = engine.changes(1).changes[0]
    assert (change.added, change.updated, change.removed) == ((), (), ())
