import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from hanji import LocalEngine, MemoryStore
from hanji.errors import ParseError
from hanji.formats.detect import default_parsers, detect_parser
from hanji.formats.pdf import PdfParser, extract
from hanji.formats.pdf import parser as pdf_parser

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
GRAY_JPEG = bytes.fromhex(  # 8×8 회색 JPEG
    "ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e1211101318281a181616183123251d283a333d"
    "3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffc0000b080008000801011100ffc40014000100"
    "000000000000000000000000000005ffc40014100100000000000000000000000000000000ffda0008010100003f0041ffd9")


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


def draw_pdf(draw, rotation: int = 0) -> bytes:
    """한 쪽짜리 PDF. rotation은 /Rotate."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, 842.0), invariant=1, pageCompression=0)
    c.setPageRotation(rotation)
    draw(c)
    c.showPage()
    c.save()
    return buf.getvalue()


def put(c: Canvas, x: float, y: float, size: float, s: str, mode: int = 0) -> None:
    c.saveState()
    t = c.beginText(x, y)
    t.setFont(FONT, size)
    t.setTextRenderMode(mode)
    t.textOut(s)
    c.drawText(t)
    c.restoreState()


def kinds_texts(parsed) -> list[tuple[str, str]]:
    return [(b["kind"], b["text"]) for b in parsed.blocks]


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
    engine = LocalEngine(MemoryStore(), default_parsers(layout=False))  # 레이아웃을 켜면 회색 사각형이 그림 블록이 된다
    engine.ingest(str(path), document_id="d")
    write(path, draw_pdf(lambda c: (put(c, 72, 770, 11, "다른 숨은 글자층이다.", 3),
                                    c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 72, 72, width=100, height=100))))
    ref = engine.ingest(str(path), document_id="d")
    assert ref.version == 2 and engine.get_tree("d").pages[0].text_stats.max_image_coverage > 0
    change = engine.changes(1).changes[0]
    assert (change.added, change.updated, change.removed) == ((), (), ())


def test_parsing_from_many_threads_at_once_is_safe():
    """PDFium은 문서가 달라도 동시에 부르면 프로세스가 죽는다. 패키지 잠금으로 한 번에 하나씩 부른다."""
    data = make_pdf(PARAS * 3)
    expected = PdfParser().parse(data, "a.pdf")
    start = threading.Barrier(8)

    def run(_):
        start.wait()
        return [PdfParser().parse(data, "a.pdf") for _ in range(30)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [r for rs in pool.map(run, range(8)) for r in rs]
    assert len(results) == 240 and all(r == expected for r in results)


def test_extract_waits_for_the_package_pdfium_lock():
    """잠금은 패키지에 하나(쪽 그림 렌더러도 같이 쓴다). 다른 스레드가 잡고 있으면 추출은 기다린다."""
    data = make_pdf(PARAS)
    done = threading.Event()
    worker = threading.Thread(target=lambda: (extract.extract_pages(data, "a.pdf"), done.set()))
    with extract.PDFIUM_LOCK:
        worker.start()
        assert not done.wait(0.3)
    worker.join(10)
    assert done.is_set()


def test_clipped_logo_does_not_make_a_titled_page_scanned():
    """쪽 크기 그림을 40×40으로 잘라 보이는 로고: 그림 면적은 보이는 부분(클리핑 영역)만 센다."""
    def draw(c):
        c.saveState()
        clip = c.beginPath()
        clip.rect(50, 700, 40, 40)
        c.clipPath(clip, stroke=0, fill=0)
        c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 0, 0, width=595, height=842)
        c.restoreState()
        put(c, 72, 600, 18, "디지털 문서 제목과 부제")

    parsed = PdfParser().parse(draw_pdf(draw), "logo.pdf")
    (page,) = parsed.pages
    assert page.text_layer == "digital"
    assert page.text_stats.max_image_coverage == pytest.approx(1600 / (595 * 842), abs=1e-4)
    assert kinds_texts(parsed) == [("paragraph", "디지털 문서 제목과 부제")]


def test_scanned_image_page_keeps_visible_text_as_blocks():
    """scanned는 '그림 속 글자는 OCR이 필요하다'는 뜻: 보이는 글자(쪽 번호)는 블록으로 남는다."""
    def draw(c):
        c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 97.5, 300, width=400, height=370)
        put(c, 282, 30, 9, "- 3 -")

    parsed = PdfParser(layout=False).parse(draw_pdf(draw), "scan.pdf")  # 레이아웃은 회색 사각형을 그림으로 본다
    assert parsed.pages[0].text_layer == "scanned"
    assert [(b["text"], b["text_source"], b["locator"]["page"]) for b in parsed.blocks] == [("- 3 -", "text_layer", 1)]


def test_scanned_page_drops_only_invisible_ocr_text():
    only_ocr = PdfParser().parse(make_pdf([[(770, "숨은 글자층이다.", 3), (750, "보이지 않는다.", 3)]]), "a.pdf")
    assert only_ocr.pages[0].text_layer == "scanned" and only_ocr.blocks == ()
    mixed = PdfParser().parse(make_pdf([[(770, "숨은 글자층이다.", 3), (750, "보이지 않는다.", 3),
                                         (730, "보이는 글자.", 0)]]), "a.pdf")
    assert mixed.pages[0].text_layer == "scanned" and kinds_texts(mixed) == [("paragraph", "보이는 글자.")]


@pytest.mark.parametrize("mode,lost", [(3, False), (0, True)])
def test_single_glyph_invisible_ocr_text_is_not_lost_text_without_hangul_font(monkeypatch, mode, lost):
    """한글 글꼴 없는 컴퓨터(아래는 그 흉내: '가'를 못 그리고 PDFium 텍스트 쪽이 한 글자 객체를 뺀다)의 스캔 쪽.
    숨은 OCR 글자(렌더 모드 3)는 어차피 버리므로 빠져도 잃은 글자가 아니다. 보이는 글자가 빠지면 여전히 실패."""
    def draw(c):
        c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 0, 0, width=595, height=842)
        put(c, 72, 770, 11, "가", mode=mode)

    monkeypatch.setattr(extract, "_draws_hangul", lambda pdf, font: False)
    monkeypatch.setattr(extract.pdfium_c, "FPDFText_CountChars", lambda textpage: 0)
    if lost:
        with pytest.raises(ParseError, match="no Hangul glyphs"):
            PdfParser().parse(draw_pdf(draw), "scan.pdf")
    else:
        parsed = PdfParser().parse(draw_pdf(draw), "scan.pdf")
        assert parsed.pages[0].text_layer == "scanned" and parsed.blocks == ()


def two_lines(c):
    """공백 글자 없이 4pt 띄운 두 낱말 + 아랫줄(11pt). reportlab은 90·270도면 MediaBox를 가로로 눕히므로 아래쪽에 쓴다."""
    put(c, 72, 500, 11, "회전")
    put(c, 72 + 22 + 4, 500, 11, "글자")
    put(c, 72, 484, 11, "둘째 줄")


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_page_text_comes_out_in_reading_order(rotation):
    parsed = PdfParser().parse(draw_pdf(two_lines, rotation), "rot.pdf")
    assert kinds_texts(parsed) == [("paragraph", "회전 글자\n둘째 줄")]
    box = parsed.blocks[0]["locator"]["bbox"]
    wide = (box["x1"] - box["x0"]) * parsed.pages[0].width_pt > (box["y1"] - box["y0"]) * parsed.pages[0].height_pt
    assert wide == (rotation in (0, 180))  # 상자는 보이는 쪽 기준


def test_negative_font_size_and_mirrored_text_keep_reading_order():
    """음수 Tf(180° 뒤집힘)와 거울 행렬(진행이 왼쪽)은 진행 방향을 따라 읽는다. 뒤집힌 글자는 쪽을 돌려 읽으므로
    PDF에서 아래에 있는 줄이 먼저다."""
    def negative(c):
        t = c.beginText(300, 400)
        t.setFont(FONT, -11)
        t.textOut("가나 다라")
        c.drawText(t)
        t = c.beginText(300, 300)
        t.setFont(FONT, -11)
        t.textOut("마바")
        c.drawText(t)
        t = c.beginText(300 - 22 - 4, 300)  # 공백 글자 없이 4pt 띄운 다음 낱말
        t.setFont(FONT, -11)
        t.textOut("사아")
        c.drawText(t)

    def mirrored(c):
        c.saveState()
        c.transform(-1, 0, 0, 1, 595, 0)
        put(c, 300, 400, 11, "가나 다라")
        c.restoreState()

    assert [t for _, t in kinds_texts(PdfParser().parse(draw_pdf(negative), "n.pdf"))] == ["마바 사아", "가나 다라"]
    assert [t for _, t in kinds_texts(PdfParser().parse(draw_pdf(mirrored), "m.pdf"))] == ["가나 다라"]


def test_parser_hands_each_page_mode_to_the_block_builder(monkeypatch):
    """파서는 쪽 상태와 함께 쪽마다 처리 모드를 블록 명세에 넘긴다(digital → layer, scanned → scan)."""
    seen = []
    build = pdf_parser.build_specs

    def spy(*args, **kw):
        seen.append(kw["modes"])
        return build(*args, **kw)

    monkeypatch.setattr(pdf_parser, "build_specs", spy)
    parsed = PdfParser(ocr=False, layout=False).parse(
        make_pdf([PARAS[0], [(770, "숨은 글자층이다.", 3), (30, "- 2 -", 0)]]), "a.pdf")
    assert [p.text_layer for p in parsed.pages] == ["digital", "scanned"] and seen == [["layer", "scan"]]
