import ctypes
import io

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from ko_parser.errors import ParseError
from ko_parser.formats.pdf.extract import PageText, decode_unicode, extract_pages, normalize_point

FONT = "HYGothic-Medium"  # reportlab 내장 CID 글꼴: ascent 752, descent -142, 한글 너비 1000
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
GRAY_JPEG = bytes.fromhex(  # 8×8 회색 JPEG
    "ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e1211101318281a181616183123251d283a333d"
    "3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffc0000b080008000801011100ffc40014000100"
    "000000000000000000000000000005ffc40014100100000000000000000000000000000000ffda0008010100003f0041ffd9")


def make_pdf(draw, size=(595.0, 842.0), **kw) -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=size, invariant=1, pageCompression=0, **kw)
    draw(c)
    c.showPage()
    c.save()
    return buf.getvalue()


def put(c, x, y, size, s, mode=0):
    c.saveState()
    t = c.beginText(x, y)
    t.setFont(FONT, size)
    t.setTextRenderMode(mode)
    t.textOut(s)
    c.drawText(t)
    c.restoreState()


def only_page(data: bytes):
    pages = extract_pages(data, "t.pdf")
    assert len(pages) == 1
    return pages[0]


def test_page_size_rotation_and_korean_text():
    page = only_page(make_pdf(lambda c: put(c, 72, 770, 18, "가나 다")))
    assert (page.page, page.width_pt, page.height_pt, page.rotation) == (1, 595.0, 842.0, 0)
    assert "".join(c.text for c in page.chars) == "가나 다"
    assert page.image_coverage == ()


def test_char_box_uses_font_metrics_not_glyph_outline():
    """글리프 외곽(대체 글꼴에 따라 OS마다 다름)이 아니라 글꼴 사전의 너비·ascent·descent로 계산한다."""
    first = only_page(make_pdf(lambda c: put(c, 72, 770, 18, "가"))).chars[0]
    assert first.x0 == pytest.approx(72 / 595) and first.x1 == pytest.approx(90 / 595)
    assert first.y0 == pytest.approx((842 - (770 + 0.752 * 18)) / 842)
    assert first.y1 == pytest.approx((842 - (770 - 0.142 * 18)) / 842)
    assert first.baseline == pytest.approx((842 - 770) / 842) and first.size == pytest.approx(18)


def test_effective_size_uses_text_matrix_scale():
    """한글 프로그램 PDF처럼 단위 크기 125로 그리고 행렬로 0.12배 줄이면 실제 크기는 15pt."""
    def draw(c):
        c.saveState()
        c.scale(0.12, 0.12)
        put(c, 600, 6000, 125, "가나")
        c.restoreState()
        c.saveState()
        c.scale(0.9, 1.0)  # 장평 90%: 세로 배율만 크기에 쓴다
        put(c, 80, 600, 10, "다")
        c.restoreState()

    chars = only_page(make_pdf(draw)).chars
    assert [c.size for c in chars] == pytest.approx([15.0, 15.0, 10.0])
    assert (chars[0].x1 - chars[0].x0) * 595 == pytest.approx(15.0)
    assert (chars[2].x1 - chars[2].x0) * 595 == pytest.approx(9.0)


def test_render_modes_invisible_and_fill_stroke_bold():
    def draw(c):
        put(c, 72, 770, 11, "숨은", mode=3)
        put(c, 72, 750, 11, "굵게", mode=2)
        put(c, 72, 730, 11, "보통")

    chars = only_page(make_pdf(draw)).chars
    assert [(c.text, c.invisible, c.bold) for c in chars] == [
        ("숨", True, False), ("은", True, False), ("굵", False, True), ("게", False, True),
        ("보", False, False), ("통", False, False)]


def test_render_mode_is_inherited_across_text_objects():
    """Tr은 그래픽 상태라 다음 BT로 이어진다(reportlab은 0 Tr을 생략). PDFium이 보는 대로 따른다."""
    def draw(c):
        put(c, 72, 770, 11, "가", mode=0)
        t = c.beginText(72, 750)
        t.setFont(FONT, 11)
        t.setTextRenderMode(3)
        t.textOut("나")
        c.drawText(t)
        t = c.beginText(72, 730)
        t.setFont(FONT, 11)
        t.textOut("다")  # Tr 없음 → 3을 물려받는다
        c.drawText(t)

    assert [(c.text, c.invisible) for c in only_page(make_pdf(draw)).chars] == [
        ("가", False), ("나", True), ("다", True)]


def test_image_coverage_top_level_and_clipped_to_page():
    def draw(c):
        c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 0, 0, width=595, height=421)  # 쪽 절반
        c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), -100, 800, width=200, height=100)  # 쪽 밖으로 잘림

    coverage = only_page(make_pdf(draw)).image_coverage
    assert coverage[0] == pytest.approx(0.5, abs=1e-3)
    assert coverage[1] == pytest.approx(100 * 42 / (595 * 842), abs=1e-4)


def test_large_form_with_small_image_counts_only_the_image():
    """쪽 전체 서식 폼(테두리) 안의 작은 로고: 폼 상자가 아니라 그림 상자를, 폼 행렬(이동·축소)을 거쳐 잰다."""
    def draw(c):
        c.beginForm("template")
        c.rect(10, 10, 575, 822)
        c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 50, 700, width=60, height=60)
        c.endForm()
        c.saveState()
        c.translate(20, -30)
        c.scale(0.5, 0.5)
        c.doForm("template")
        c.restoreState()

    assert only_page(make_pdf(draw)).image_coverage == pytest.approx((30 * 30 / (595 * 842),))


def test_negative_font_size_gives_positive_size():
    def draw(c):
        t = c.beginText(300, 400)
        t.setFont(FONT, -11)  # 글자가 뒤집혀 왼쪽으로 진행한다
        t.textOut("가나")
        c.drawText(t)

    chars = only_page(make_pdf(draw)).chars
    assert [c.size for c in chars] == pytest.approx([11.0, 11.0])
    assert all(c.x0 < c.x1 and c.y0 < c.y1 for c in chars)


def test_chars_outside_page_are_dropped():
    page = only_page(make_pdf(lambda c: (put(c, -300, 770, 11, "밖"), put(c, 72, 770, 11, "안"))))
    assert [c.text for c in page.chars] == ["안"]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_normalize_point_matches_pdfium_device_mapping(rotation):
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(300, 500)
    page.set_mediabox(10, 20, 310, 520)
    page.set_rotation(rotation)
    width, height = page.get_size()
    for x, y in [(10, 20), (310, 520), (60, 120), (200, 400)]:
        dx, dy = ctypes.c_int(), ctypes.c_int()
        pdfium_c.FPDF_PageToDevice(page, 0, 0, int(width * 1000), int(height * 1000), 0, x, y, dx, dy)
        expected = (dx.value / 1000 / width, dy.value / 1000 / height)
        assert normalize_point(x, y, page.get_bbox(), rotation) == pytest.approx(expected, abs=1e-5)
    doc.close()


def test_decode_unicode_pairs_and_unmapped():
    assert decode_unicode(ord("가"), None) == ("가", False, False)
    assert decode_unicode(0xD835, 0xDC00) == ("\U0001D400", False, True)
    assert decode_unicode(0xD835, ord("a")) == ("\ufffd", True, False)
    assert decode_unicode(0, None) == ("\ufffd", True, False)
    assert decode_unicode(0xFFFD, None) == ("\ufffd", True, False)


def test_encrypted_pdf_is_parse_error():
    data = make_pdf(lambda c: put(c, 72, 770, 11, "비밀"), encrypt="secret")
    with pytest.raises(ParseError) as info:
        extract_pages(data, "암호.pdf")
    assert (info.value.reason, info.value.location) == ("encrypted PDF", "암호.pdf")


@pytest.mark.parametrize("data", [b"", b"%PDF-1.4\n", b"not a pdf at all"])
def test_corrupt_pdf_is_parse_error(data):
    with pytest.raises(ParseError, match="invalid PDF") as info:
        extract_pages(data, "깨짐.pdf")
    assert info.value.location == "깨짐.pdf"


@pytest.mark.parametrize("cut", ["xref", "stream", "header"])
def test_truncated_pdf_is_parse_error_or_readable(cut):
    """잘린 PDF는 PDFium이 고쳐 읽거나(쪽이 나온다) ParseError다. 다른 예외는 없다(문서·쪽 단계 모두)."""
    data = make_pdf(lambda c: put(c, 72, 770, 11, "가나다"))
    end = {"xref": data.index(b"xref"), "stream": data.index(b"stream") + 10, "header": 20}[cut]
    try:
        pages = extract_pages(data[:end], "잘림.pdf")
    except ParseError as exc:
        assert exc.location.startswith("잘림.pdf")
    else:
        assert pages and all(isinstance(p, PageText) for p in pages)


def zero_page_pdf() -> bytes:
    """쪽 트리가 비어 있는(/Count 0) 최소 PDF."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [] /Count 0 >>"]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 3\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref


def test_pdf_without_pages_is_parse_error():
    with pytest.raises(ParseError) as info:
        extract_pages(zero_page_pdf(), "빈.pdf")
    assert info.value.location == "빈.pdf"
