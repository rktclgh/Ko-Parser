"""PdfParser의 스캔 쪽 OCR: scanned 쪽만 읽고, 텍스트 레이어가 우선이며, 블록은 윗변 순서로 합쳐진다.
그림은 OS 글꼴에 기대지 않으려고 ko-parser-fonts 글꼴(Pillow)로 그려 reportlab PDF에 넣는다."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

import ko_parser_fonts
from ko_parser.errors import OcrUnavailable
from ko_parser.formats.pdf import PdfParser, ocr, scan

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
NOTO = str(ko_parser_fonts.font_dir() / ko_parser_fonts.FONT_FILE)
PX = 200 / 72  # 그림 화소 / pt
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pdf" / "inputs"
BODY = [(40, 60, 14, "스캔한 쪽의 글자를 읽는다."), (40, 82, 14, "두 줄로 된 문단이다."),
        (40, 140, 14, "다음 문단은 한 줄이다.")]


def text_image(width: float, height: float, lines) -> Image.Image:
    """보이는 쪽 크기(pt)의 흰 그림에 (왼쪽 pt, 윗변 pt, 크기 pt, 글자) 줄을 그린다(200 DPI)."""
    img = Image.new("L", (round(width * PX), round(height * PX)), 255)
    draw = ImageDraw.Draw(img)
    for x, top, size, s in lines:
        draw.text((x * PX, top * PX), s, font=ImageFont.truetype(NOTO, round(size * PX)), fill=0)
    return img


def put(c: Canvas, x: float, y: float, size: float, s: str) -> None:
    c.setFont(FONT, size)
    c.drawString(x, y, s)


def scanned_page(c: Canvas, width: float, height: float, lines, visible=()) -> None:
    """쪽 전체 그림 + 보이는 글자(왼쪽, 기준선 y(PDF 좌표), 크기, 글자)."""
    c.drawImage(ImageReader(text_image(width, height, lines)), 0, 0, width=width, height=height)
    for x, y, size, s in visible:
        put(c, x, y, size, s)


def pdf(*pages, size=(300.0, 400.0)) -> bytes:
    """pages: Canvas를 받아 한 쪽을 그리는 함수들."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=size, invariant=1, pageCompression=0)
    for draw in pages:
        draw(c)
        c.showPage()
    c.save()
    return buf.getvalue()


def blocks(data: bytes, parser: PdfParser | None = None) -> list[tuple[str, str, str]]:
    return [(b["kind"], b["text_source"], b["text"]) for b in (parser or PdfParser()).parse(data, "s.pdf").blocks]


def test_scanned_page_text_becomes_ocr_paragraph_blocks():
    pytest.importorskip("onnxruntime")
    parsed = PdfParser().parse(pdf(lambda c: scanned_page(c, 300, 400, BODY)), "s.pdf")
    assert parsed.pages[0].text_layer == "scanned"
    assert [(b["kind"], b["text_source"], b["text"]) for b in parsed.blocks] == [
        ("paragraph", "ocr", "스캔한 쪽의 글자를 읽는다.\n두 줄로 된 문단이다."),
        ("paragraph", "ocr", "다음 문단은 한 줄이다.")]
    first = parsed.blocks[0]
    assert first["state"] == "det" and first["section_path"] == () and 0.4 <= first["confidence"] <= 0.5
    box = first["locator"]["bbox"]  # 그린 자리: 왼쪽 40pt(0.133), 윗변 60pt(0.15), 두 줄 아래 약 100pt(0.25)
    assert abs(box["x0"] - 40 / 300) <= 0.02 and abs(box["y0"] - 60 / 400) <= 0.02
    assert abs(box["y1"] - 100 / 400) <= 0.02


def test_only_scanned_pages_are_read(monkeypatch):
    pytest.importorskip("onnxruntime")
    calls = []
    real = ocr.read_lines
    monkeypatch.setattr(ocr, "read_lines", lambda image: calls.append(image.size) or real(image))

    def digital(c):
        put(c, 40, 340, 11, "디지털 쪽의 글자는 텍스트 레이어로 읽는다.")

    data = pdf(digital, lambda c: scanned_page(c, 300, 400, BODY[:1]), digital)
    assert blocks(data) == [("paragraph", "text_layer", "디지털 쪽의 글자는 텍스트 레이어로 읽는다."),
                            ("paragraph", "ocr", "스캔한 쪽의 글자를 읽는다."),
                            ("paragraph", "text_layer", "디지털 쪽의 글자는 텍스트 레이어로 읽는다.")]
    assert calls == [(834, 1112)]  # 둘째 쪽(300×400pt를 200 DPI로, 올림)만


def test_ocr_off_or_not_installed_keeps_text_layer_only(monkeypatch):
    data = pdf(lambda c: scanned_page(c, 300, 400, BODY, visible=[(148, 20, 9, "1")]))
    expected = [("paragraph", "text_layer", "1")]
    monkeypatch.setattr(scan, "render", lambda *a: pytest.fail("OCR을 끄면 쪽을 그리지 않는다"))
    assert blocks(data, PdfParser(ocr=False)) == expected
    monkeypatch.setattr(ocr, "available", lambda: False)
    assert blocks(data, PdfParser()) == expected


def test_ocr_true_without_the_install_fails_at_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(ocr, "_reader", None)
    with pytest.raises(OcrUnavailable, match=r"ko-parser-engine\[ocr\]"):
        PdfParser(ocr=True)
    assert PdfParser().ocr is None and PdfParser(ocr=False).ocr is False  # 자동·끔은 만들 때 확인하지 않는다


def test_visible_text_is_not_read_twice_and_blocks_merge_by_top():
    """보이는 글자(쪽 위 줄·아래 쪽 번호)도 렌더 그림에 그려져 OCR이 읽지만 텍스트 레이어가 우선이라 버린다.
    블록은 윗변 순서: 위 텍스트 레이어 → OCR 문단 → 아래 쪽 번호."""
    pytest.importorskip("onnxruntime")
    visible = [(40, 370, 12, "보이는 글자 줄"), (148, 20, 9, "1")]
    data = pdf(lambda c: scanned_page(c, 300, 400, BODY, visible))
    assert blocks(data) == [("paragraph", "text_layer", "보이는 글자 줄"),
                            ("paragraph", "ocr", "스캔한 쪽의 글자를 읽는다.\n두 줄로 된 문단이다."),
                            ("paragraph", "ocr", "다음 문단은 한 줄이다."),
                            ("paragraph", "text_layer", "1")]


def test_ocr_blocks_follow_the_previous_heading():
    pytest.importorskip("onnxruntime")
    def digital(c):
        put(c, 40, 340, 18, "1. 추진 배경")
        for i in range(4):
            put(c, 40, 300 - 16 * i, 11, "본문 크기를 정하는 디지털 쪽의 글자다.")

    parsed = PdfParser().parse(pdf(digital, lambda c: scanned_page(c, 300, 400, BODY[2:])), "s.pdf")
    ocr_blocks = [b for b in parsed.blocks if b["text_source"] == "ocr"]
    assert [(b["text"], b["section_path"]) for b in ocr_blocks] == [("다음 문단은 한 줄이다.", ("1. 추진 배경",))]


def test_rotated_scanned_page_has_visible_page_coordinates():
    """/Rotate 90 쪽: 그림을 PDF 좌표에서 돌려 넣어 보이는 쪽에서 바로 선다. 결과는 돌리지 않은 쪽과 같다."""
    pytest.importorskip("onnxruntime")
    image = text_image(300, 400, BODY)

    def upright(c):
        c.drawImage(ImageReader(image), 0, 0, width=300, height=400)

    def rotated(c):
        c.setPageRotation(90)  # reportlab은 MediaBox를 400×300으로 눕힌다. PDF 점 (x, y)는 보이는 (y, x)
        c.translate(400, 0)
        c.rotate(90)
        c.drawImage(ImageReader(image), 0, 0, width=300, height=400)

    plain = PdfParser().parse(pdf(upright), "u.pdf")
    turned = PdfParser().parse(pdf(rotated), "r.pdf")
    assert turned.pages[0].rotation == 90 and (turned.pages[0].width_pt, turned.pages[0].height_pt) == (300, 400)
    assert [b["text"] for b in turned.blocks] == [b["text"] for b in plain.blocks] and len(plain.blocks) == 2
    for a, b in zip(plain.blocks, turned.blocks):
        assert all(abs(a["locator"]["bbox"][k] - b["locator"]["bbox"][k]) <= 0.01 for k in ("x0", "y0", "x1", "y1"))


def test_blank_scanned_page_has_no_ocr_blocks():
    data = pdf(lambda c: scanned_page(c, 300, 400, [], visible=[(148, 20, 9, "1")]))
    assert blocks(data) == [("paragraph", "text_layer", "1")]


@pytest.mark.parametrize("name", ["image_page.pdf", "scanned_invisible.pdf"])
def test_scanned_golden_inputs_are_the_same_with_ocr_on(name):
    """기존 scanned 골든 예제(보이는 쪽 번호만, 숨은 글자층): OCR을 켜도 쪽 번호를 두 번 내지 않는다."""
    pytest.importorskip("onnxruntime")
    data = (FIXTURES / name).read_bytes()
    assert PdfParser(ocr=True).parse(data, name) == PdfParser(ocr=False).parse(data, name)


def test_parallel_parses_give_the_same_ocr_blocks():
    """여러 스레드가 동시에 스캔 쪽을 파싱해도(렌더는 PDFIUM_LOCK, OCR 세션은 공유) 결과가 같다."""
    pytest.importorskip("onnxruntime")
    from concurrent.futures import ThreadPoolExecutor

    data = pdf(lambda c: scanned_page(c, 300, 400, BODY))
    expected = PdfParser().parse(data, "s.pdf")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: PdfParser().parse(data, "s.pdf"), range(8)))
    assert all(r == expected for r in results) and len(expected.blocks) == 2
