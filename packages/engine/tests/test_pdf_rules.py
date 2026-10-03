import io

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from ko_parser.formats.pdf.extract import MIN_RULE, PageText, Rule, extract_pages

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
W, H = 595.0, 842.0


def page_of(draw, rotation: int = 0, crop=None) -> PageText:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(W, H), invariant=1, pageCompression=0)
    if rotation:
        c.setPageRotation(rotation)
    if crop:
        c.setCropBox(crop)
    draw(c)
    c.showPage()
    c.save()
    (page,) = extract_pages(buf.getvalue(), "t.pdf")
    return page


def rules_of(draw) -> list[Rule]:
    return list(page_of(draw).rules)


def test_stroked_lines_and_rectangle_sides_in_top_left_points():
    """PDF 좌표(원점 왼쪽 아래)의 선이 글자와 같은 보이는 쪽 틀(원점 왼쪽 위, pt)로 온다. 사각형은 네 변."""
    def draw(c):
        c.line(100, 700, 300, 700)
        c.line(150, 600, 150, 650.1234)
        c.rect(100, 400, 200, 50, stroke=1, fill=0)

    assert sorted(rules_of(draw), key=lambda r: (r.axis, r.pos, r.start)) == [
        Rule("h", 142.0, 100.0, 300.0), Rule("h", 392.0, 100.0, 300.0), Rule("h", 442.0, 100.0, 300.0),
        Rule("v", 100.0, 392.0, 442.0), Rule("v", 150.0, 191.877, 242.0), Rule("v", 300.0, 392.0, 442.0)]


def test_thin_filled_rect_is_a_center_line_and_wide_filled_rect_gives_fill_sides():
    def draw(c):
        c.setFillColorRGB(0, 0, 0)
        c.rect(100, 700, 200, 1.0, stroke=0, fill=1)  # 두께 1pt 선(채운 사각형)
        c.setFillColorRGB(0.85, 0.85, 0.85)
        c.rect(100, 500, 200, 20, stroke=0, fill=1)  # 칸 배경

    assert sorted(rules_of(draw), key=lambda r: (r.axis, r.pos)) == [
        Rule("h", 141.5, 100.0, 300.0),
        Rule("h", 322.0, 100.0, 300.0, "fill"), Rule("h", 342.0, 100.0, 300.0, "fill"),
        Rule("v", 100.0, 322.0, 342.0, "fill"), Rule("v", 300.0, 322.0, 342.0, "fill")]


def test_white_diagonal_curved_and_invisible_paths_are_dropped():
    def draw(c):
        c.setStrokeColorRGB(1, 1, 1)
        c.line(100, 700, 300, 700)  # 하얀 선
        c.setFillColorRGB(1, 1, 1)
        c.rect(100, 600, 200, 30, stroke=0, fill=1)  # 하얀 채움
        c.setStrokeColorRGB(0, 0, 0)
        c.line(100, 500, 300, 520)  # 사선
        c.circle(200, 300, 50, stroke=1, fill=0)  # 곡선
        c.rect(100, 100, 200, 30, stroke=0, fill=0)  # 그리지 않는 경로

    assert rules_of(draw) == []


def test_short_segment_is_dropped_but_dotted_line_is_joined():
    """MIN_RULE(2pt)보다 짧은 선분 하나는 버리고, 0.48pt 점이 1.2pt 간격으로 이어진 점선은 한 선(한글 프로그램 PDF 실측
    모양)."""
    def draw(c):
        c.line(100, 700, 100 + MIN_RULE - 0.5, 700)
        c.setLineWidth(0.36)
        for i in range(100):
            x = 100 + 1.2 * i
            c.line(x, 500, x + 0.48, 500)

    assert rules_of(draw) == [Rule("h", 342.0, 100.0, 219.28)]


def test_rules_inside_form_xobject_use_the_form_matrix():
    def draw(c):
        c.beginForm("grid")
        c.line(0, 0, 100, 0)
        c.line(0, 0, 0, 50)
        c.endForm()
        c.saveState()
        c.translate(100, 400)
        c.scale(2, 2)
        c.doForm("grid")
        c.restoreState()

    assert sorted(rules_of(draw), key=lambda r: r.axis) == [Rule("h", 442.0, 100.0, 300.0), Rule("v", 100.0, 342.0, 442.0)]


@pytest.mark.parametrize("rotation,crop,expected", [
    (0, None, Rule("h", 542.0, 100.0, 200.0)), (90, None, Rule("v", 300.0, 100.0, 200.0)),
    (180, None, Rule("h", 300.0, 395.0, 495.0)), (270, None, Rule("v", 295.0, 642.0, 742.0)),
    (0, (20, 30, 575, 812), Rule("h", 512.0, 80.0, 180.0))])
def test_rules_follow_page_rotation_like_chars(rotation, crop, expected):
    """선과 같은 기준선에 쓴 글자: 선의 위치가 그 글자의 기준선(보이는 쪽 pt)과 같다. 회전(reportlab은 90·270도면
    MediaBox를 842×595로 눕혀 보이는 크기를 595×842로 둔다)과 원점이 0이 아닌 CropBox에서도."""
    def draw(c):
        c.line(100, 300, 200, 300)
        t = c.beginText(100, 300)
        t.setFont(FONT, 10)
        t.textOut("가")
        c.drawText(t)

    page = page_of(draw, rotation, crop)
    (rule,), (char,) = page.rules, page.chars
    assert rule == expected
    down = char.axes[1]  # 줄 아래 방향: 보이는 쪽 +x·+y·−x·−y = 0·1·2·3
    span = page.width_pt if down % 2 == 0 else page.height_pt
    assert rule.axis == ("v" if down % 2 == 0 else "h")
    assert rule.pos == pytest.approx(char.baseline * span if down < 2 else span - char.baseline * span, abs=1e-3)


def test_rules_are_clipped_to_the_page():
    def draw(c):
        c.line(-50, 700, 100, 700)  # 왼쪽이 쪽 밖
        c.line(100, 900, 300, 900)  # 쪽 위 밖

    assert rules_of(draw) == [Rule("h", 142.0, 0.0, 100.0)]
