"""스캔 쪽 OCR 연결(scan.py): 그림 화소 → 보이는 쪽 pt, 점수 거르기, 텍스트 레이어 우선, XY 분할, 문단, 렌더 배율."""

import io
import os

import pytest
from reportlab.pdfgen.canvas import Canvas

from ko_parser import models
from ko_parser.formats.pdf import scan
from ko_parser.formats.pdf.extract import Char, PageText
from ko_parser.formats.pdf.ocr import MODEL_NAMES, OcrLine

W, H = 600.0, 800.0


def page(chars=()) -> PageText:
    return PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=tuple(chars), image_coverage=(1.0,))


def char(text: str, x0: float, y0: float, x1: float, y1: float, invisible: bool = False) -> Char:
    """pt 상자 → 0~1 글자."""
    return Char(text=text, x0=x0 / W, y0=y0 / H, x1=x1 / W, y1=y1 / H, baseline=y1 / H, size=y1 - y0,
                invisible=invisible)


def line(x0: float, y0: float, x1: float, y1: float, text: str = "가나다라", score: float = 0.95) -> scan.OcrText:
    return scan.OcrText(text=text, score=score, x0=x0, y0=y0, x1=x1, y1=y1)


def texts(lines) -> list[str]:
    return [t.text for t in lines]


def test_image_pixels_map_to_visible_page_points():
    """렌더 그림(1200×1600px)은 보이는 쪽(600×800pt)과 같은 틀: 화소 ÷ 2 = pt. 쪽 밖은 자른다."""
    raw = [OcrLine(box=((100, 40), (500, 44), (500, 80), (100, 76)), text="가나", score=0.9),
           OcrLine(box=((-6, 1590), (1210, 1590), (1210, 1620), (-6, 1620)), text="다라", score=0.8)]
    a, b = scan.to_page(raw, (1200, 1600), page())
    assert (a.text, a.score, a.x0, a.y0, a.x1, a.y1) == ("가나", 0.9, 50.0, 20.0, 250.0, 40.0)
    assert (b.x0, b.y0, b.x1, b.y1) == (0.0, 795.0, 600.0, 800.0)


@pytest.mark.parametrize("text,score,kept", [
    ("가나다", 0.49, False), ("가나다", 0.5, True), ("가나다", 0.55, True),
    ("가나", 0.69, False), ("가나", 0.7, True), ("가 나", 0.69, False), ("가", 0.95, True)])
def test_score_filters_with_a_higher_bar_for_short_lines(text, score, kept):
    """점수 < OCR_MIN_SCORE는 버리고, 공백 뺀 2글자 이하는 OCR_SHORT_SCORE 이상만 남긴다(그림 속 작은 이름표)."""
    assert scan.keep(line(100, 100, 200, 120, text, score), page()) is kept


def test_line_over_visible_text_is_dropped_but_not_over_hidden_text():
    """텍스트 레이어가 OCR보다 우선: 보이는 글자가 줄 길이의 절반 이상을 덮으면 버린다. 숨은 글자(렌더 모드 3)는 세지 않는다."""
    ocr_line = line(100, 100, 300, 120)
    visible = [char(c, 105 + 40 * i, 103, 140 + 40 * i, 117) for i, c in enumerate("가나다라")]
    hidden = [char(c, 105 + 40 * i, 103, 140 + 40 * i, 117, invisible=True) for i, c in enumerate("가나다라")]
    assert scan.keep(ocr_line, page(visible)) is False
    assert scan.keep(ocr_line, page(hidden)) is True
    assert scan.keep(ocr_line, page(visible[:2])) is True  # 70pt / 200pt만 덮는다
    assert scan.keep(line(100, 300, 300, 320), page(visible)) is True  # 다른 줄 높이


def test_one_char_line_over_its_visible_char_is_dropped():
    """OCR 줄 상자는 글자보다 여백이 크다(실측: 글자 11×10pt, OCR 18×20pt, 넓이로는 0.30만 덮임). 길이로 잰다."""
    assert scan.keep(line(76.5, 103.1, 94.7, 122.7, "가", 0.99), page([char("가", 80, 107.7, 91, 117.6)])) is False


def test_tall_line_is_measured_along_its_height():
    tall = line(100, 100, 120, 300)
    stacked = [char(c, 102, 105 + 40 * i, 118, 140 + 40 * i) for i, c in enumerate("가나다라마")]
    assert scan.keep(tall, page(stacked)) is False
    assert scan.keep(tall, page(stacked[:1])) is True


def test_two_columns_under_a_title_read_column_by_column():
    title = line(50, 40, 550, 60, "제목")
    left = [line(50, 100 + 25 * i, 280, 115 + 25 * i, f"왼{i}") for i in range(4)]
    right = [line(320, 100 + 25 * i, 550, 115 + 25 * i, f"오{i}") for i in range(4)]
    order = scan.reading_order([*right, title, *left])
    assert texts(order) == ["제목", "왼0", "왼1", "왼2", "왼3", "오0", "오1", "오2", "오3"]


def test_narrow_table_like_columns_are_read_row_by_row():
    """빈 세로 띠가 있어도 영역 너비의 25%보다 좁은 단이 생기면 표 배치로 보고 행 순서로 읽는다."""
    rows = [[line(50, 100 + 25 * i, 100, 115 + 25 * i, f"{i}a"), line(180, 100 + 25 * i, 260, 115 + 25 * i, f"{i}b"),
             line(320, 100 + 25 * i, 550, 115 + 25 * i, f"{i}c")] for i in range(3)]
    order = scan.reading_order([t for row in reversed(rows) for t in row])
    assert texts(order) == ["0a", "0b", "0c", "1a", "1b", "1c", "2a", "2b", "2c"]


def test_same_row_is_left_to_right_even_when_slightly_higher():
    """세로 중심 차 ≤ 줄 높이 × 0.7이면 같은 줄(기울어진 스캔에서 오른쪽 칸이 조금 위에 있어도)."""
    order = scan.reading_order([line(220, 96, 400, 116, "오른"), line(50, 103, 200, 123, "왼")])  # 사이 20pt: 단 아님
    assert texts(order) == ["왼", "오른"]
    order = scan.reading_order([line(150, 80, 400, 100, "위"), line(50, 103, 200, 123, "아래")])  # 중심 차 23 > 14
    assert texts(order) == ["위", "아래"]


def test_paragraphs_join_close_lines_and_split_on_gap_indent_or_size():
    lines = [line(50, 100, 400, 115, "첫 줄", 0.9), line(52, 120, 380, 135, "둘째 줄", 0.8),
             line(50, 170, 400, 185, "간격이 넓다"), line(100, 190, 400, 205, "들여 씀"),
             line(100, 210, 400, 240, "큰 글자")]
    paras = scan.paragraphs(lines, page())
    assert [p.text for p in paras] == ["첫 줄\n둘째 줄", "간격이 넓다", "들여 씀", "큰 글자"]
    first = paras[0]
    assert first.bbox == (50 / W, 100 / H, 400 / W, 135 / H) and first.confidence == 0.425  # 평균 0.85 × 0.5


def test_empty_page_has_no_paragraphs():
    assert scan.reading_order([]) == [] and scan.paragraphs([], page()) == []


def one_page_pdf(width: float, height: float, rotation: int = 0) -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(width, height), invariant=1, pageCompression=0)
    c.setPageRotation(rotation)
    c.rect(10, 10, 50, 30, stroke=1, fill=0)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_render_is_200_dpi_in_the_visible_frame():
    assert scan.render(one_page_pdf(595, 842), "a.pdf", 0).size == (1653, 2339)
    # reportlab은 회전 쪽의 MediaBox를 눕혀(842×595) 보이는 크기를 595×842로 둔다. 렌더는 보이는 쪽 그대로
    assert scan.render(one_page_pdf(595, 842, rotation=90), "r.pdf", 0).size == (1653, 2339)


def require_models(*names: str) -> None:
    """실제 모델 파일이 필요한 테스트: 찾을 수 없으면 건너뛰고, KO_PARSER_CI_REQUIRE_MODELS=1이면 실패한다."""
    missing = [name for name in names if models.find(name) is None]
    if missing:
        if os.environ.get("KO_PARSER_CI_REQUIRE_MODELS") == "1":
            pytest.fail(f"model files not found: {missing}; run `ko-parser models fetch`")
        pytest.skip(f"model files not found: {missing} (ko-parser models fetch)")


def test_huge_page_render_is_capped():
    """긴 변이 MAX_RENDER_SIDE(4000px)를 넘지 않게 배율을 낮춘다(A0 두 배 쪽도 메모리 수십 MB)."""
    image = scan.render(one_page_pdf(4768, 6741), "big.pdf", 0)
    assert max(image.size) <= scan.MAX_RENDER_SIDE and image.mode == "RGB"


def test_ocr_boxes_on_a_page_with_offset_cropbox_are_in_visible_page_coordinates():
    """CropBox 원점이 0이 아닌 스캔 쪽(MediaBox 600×800, CropBox 50 60 550 760 → 보이는 쪽 500×700pt): 렌더가 보이는
    쪽만 그리므로 OCR 블록 상자는 보이는 쪽 0~1에서 글자가 실제로 있는 자리에 온다."""
    pytest.importorskip("onnxruntime")
    require_models(*MODEL_NAMES)
    ko_parser_fonts = pytest.importorskip("ko_parser_fonts")
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.utils import ImageReader

    from ko_parser.formats.pdf.extract import extract_pages

    font = ImageFont.truetype(str(ko_parser_fonts.font_dir() / ko_parser_fonts.FONT_FILE), 40)
    image = Image.new("RGB", (1000, 200), "white")
    ImageDraw.Draw(image).text((60, 60), "스캔한 쪽의 글자를 읽는다.", font=font, fill="black")
    ink = Image.eval(image.convert("L"), lambda v: 255 - v).getbbox()  # 글자 잉크 상자(화소)
    left, bottom, pt = 120.0, 500.0, 72 / 200  # 그림 화소 하나 = 0.36pt(200 DPI 렌더에서 원래 크기)
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(600, 800), invariant=1, pageCompression=0)
    c.setCropBox((50, 60, 550, 760))
    c.drawImage(ImageReader(image), left, bottom, width=1000 * pt, height=200 * pt)
    c.showPage()
    c.save()
    data = buf.getvalue()
    pages = extract_pages(data, "crop.pdf")
    assert (pages[0].width_pt, pages[0].height_pt) == (500, 700) and pages[0].chars == ()
    (para,), = scan.ocr_pages(data, "crop.pdf", pages, ["scanned"])
    top = bottom + 200 * pt  # 그림 윗변(PDF 좌표)
    expected = ((left + ink[0] * pt - 50) / 500, (760 - (top - ink[1] * pt)) / 700,
                (left + ink[2] * pt - 50) / 500, (760 - (top - ink[3] * pt)) / 700)
    assert para.text == "스캔한 쪽의 글자를 읽는다."
    assert all(abs(got - want) <= 0.02 for got, want in zip(para.bbox, expected)), (para.bbox, expected)


def test_ocr_boxes_on_a_rotated_scanned_page_are_in_visible_page_coordinates():
    """/Rotate 90 스캔 쪽(MediaBox 800×600 → 보이는 쪽 600×800pt): 스캐너가 눕혀 담은 그림을 쪽 회전으로 세운 실제 경우.
    그림은 PDF 좌표에서 반시계로 누워 있어 보이는 쪽에서 바로 선다. 렌더가 회전을 반영하므로 OCR이 글자를 읽고,
    블록 상자는 보이는 쪽 0~1에서 글자가 실제로 있는 자리에 온다(PDF 점 (x, y)는 보이는 (y, x))."""
    pytest.importorskip("onnxruntime")
    require_models(*MODEL_NAMES)
    ko_parser_fonts = pytest.importorskip("ko_parser_fonts")
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.utils import ImageReader

    from ko_parser.formats.pdf.extract import extract_pages

    font = ImageFont.truetype(str(ko_parser_fonts.font_dir() / ko_parser_fonts.FONT_FILE), 40)
    upright = Image.new("RGB", (1000, 200), "white")
    ImageDraw.Draw(upright).text((60, 60), "돌린 쪽의 글자도 바로 읽는다.", font=font, fill="black")
    image = upright.rotate(90, expand=True)  # 200×1000, 반시계로 눕힘(쪽 회전 90°가 시계 방향으로 세운다)
    ink = Image.eval(image.convert("L"), lambda v: 255 - v).getbbox()  # 누운 그림의 글자 잉크 상자(화소)
    left, bottom, pt = 150.0, 120.0, 72 / 200
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(600, 800), invariant=1, pageCompression=0)
    c.setPageRotation(90)  # reportlab은 MediaBox를 800×600으로 눕힌다
    c.drawImage(ImageReader(image), left, bottom, width=200 * pt, height=1000 * pt)
    c.showPage()
    c.save()
    data = buf.getvalue()
    pages = extract_pages(data, "rot.pdf")
    assert (pages[0].rotation, pages[0].width_pt, pages[0].height_pt) == (90, 600, 800) and pages[0].chars == ()
    (para,), = scan.ocr_pages(data, "rot.pdf", pages, ["scanned"])
    top = bottom + 1000 * pt  # 그림 윗변(PDF y)
    expected = ((top - ink[3] * pt) / 600, (left + ink[0] * pt) / 800,
                (top - ink[1] * pt) / 600, (left + ink[2] * pt) / 800)
    assert para.text == "돌린 쪽의 글자도 바로 읽는다."
    assert all(abs(got - want) <= 0.02 for got, want in zip(para.bbox, expected)), (para.bbox, expected)
