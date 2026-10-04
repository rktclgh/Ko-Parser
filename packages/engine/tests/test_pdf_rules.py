import io

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from ko_parser.formats.pdf import extract
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


def assert_rules(found, expected) -> None:
    """축·종류는 같고 좌표는 ± 0.002pt."""
    assert [(r.axis, r.kind) for r in found] == [(r.axis, r.kind) for r in expected]
    for got, want in zip(found, expected, strict=True):
        assert (got.pos, got.start, got.end) == pytest.approx((want.pos, want.start, want.end), abs=2e-3)


@pytest.mark.parametrize("size,count,horizontal", [(0.48, 100, True), (0.48, 100, False), (1.0, 30, True)])
def test_dotted_line_of_small_filled_squares_is_one_rule(size, count, horizontal):
    """작은 채운 네모 점(가로·세로 길이가 같다)이 줄지은 점선은 위치와 상관없이 한 선(좌표가 둥근 수가 아니어도)."""
    x0, y0 = (100.37, 500.13) if horizontal else (200.29, 300.41)

    def draw(c):
        c.setFillColorRGB(0, 0, 0)
        for i in range(count):
            dx, dy = (1.2 * i, 0.0) if horizontal else (0.0, 1.2 * i)
            c.rect(x0 + dx, y0 + dy, size, size, stroke=0, fill=1)

    run = 1.2 * (count - 1) + size
    if horizontal:
        expected = Rule("h", H - (y0 + size / 2), x0, x0 + run)
    else:
        expected = Rule("v", x0 + size / 2, H - (y0 + run), H - y0)
    assert_rules(rules_of(draw), [expected])


def test_slowly_drifting_dotted_line_is_one_rule():
    """점마다 위치가 0.02pt씩 흘러도(40개, 모두 0.78pt) 한 선."""
    def draw(c):
        c.setLineWidth(0.36)
        for i in range(40):
            x, y = 100 + 1.2 * i, 500 + 0.02 * i
            c.line(x, y, x + 0.48, y)

    (rule,) = rules_of(draw)
    assert (rule.axis, rule.start, rule.end) == ("h", 100.0, pytest.approx(147.28, abs=1e-3))


def test_vertical_dotted_line_is_joined():
    def draw(c):
        c.setLineWidth(0.36)
        for i in range(100):
            y = 400 + 1.2 * i
            c.line(100, y, 100, y + 0.48)

    assert_rules(rules_of(draw), [Rule("v", 100.0, 322.72, 442.0)])


def test_filled_triangle_and_zero_area_fill_give_no_rules():
    """채운 직각삼각형(꼭짓점이 모두 외접 상자 위)과 넓이 0인 채운 사각형은 선이 아니다."""
    def draw(c):
        c.setFillColorRGB(0, 0, 0)
        for x, y, w, h in ((100, 600, 200, 100), (100, 400, 200, 1.0)):
            p = c.beginPath()
            p.moveTo(x, y)
            p.lineTo(x + w, y)
            p.lineTo(x, y + h)
            p.close()
            c.drawPath(p, stroke=0, fill=1)
        c.rect(100, 300, 200, 0, stroke=0, fill=1)
        c.rect(100, 200, 0, 50, stroke=0, fill=1)

    assert rules_of(draw) == []


def test_rules_inside_nested_form_xobjects_compose_both_matrices():
    def draw(c):
        c.beginForm("inner")
        c.line(0, 0, 10, 0)
        c.endForm()
        c.beginForm("outer")
        c.saveState()
        c.translate(5, 5)
        c.scale(2, 2)
        c.doForm("inner")
        c.restoreState()
        c.endForm()
        c.saveState()
        c.translate(100, 400)
        c.scale(3, 3)
        c.doForm("outer")
        c.restoreState()

    assert rules_of(draw) == [Rule("h", 427.0, 115.0, 175.0)]


def test_stroked_and_filled_rectangle_gives_stroke_and_fill_sides():
    """그리고 채운 넓은 사각형은 같은 네 변을 stroke와 fill로 둘 다 낸다(표 검출이 같은 선으로 합친다)."""
    def draw(c):
        c.setFillColorRGB(0.85, 0.85, 0.85)
        c.rect(100, 400, 200, 50, stroke=1, fill=1)

    sides = [("h", 392.0, 100.0, 300.0), ("h", 442.0, 100.0, 300.0), ("v", 100.0, 392.0, 442.0),
             ("v", 300.0, 392.0, 442.0)]
    assert sorted(rules_of(draw), key=lambda r: (r.kind, r.axis, r.pos)) == (
        [Rule(*s, "fill") for s in sides] + [Rule(*s) for s in sides])


def test_slanted_dotted_line_is_not_a_rule():
    """점마다 0.375pt씩 오르는 점선(60개)은 기울어진 선이다: 이웃 점끼리는 가까워도 이어 붙이면 사선."""
    def draw(c):
        c.setLineWidth(0.36)
        for i in range(60):
            x, y = 100 + 1.2 * i, 500 + 0.375 * i
            c.line(x, y, x + 0.48, y)

    assert rules_of(draw) == []


def test_filled_near_square_above_dash_size_gives_one_direction():
    """2.2 × 2.0pt 채운 사각형은 점선 조각(두 변 < MIN_RULE)이 아니므로 두 변을 그대로 비교해 긴 쪽(가로) 한 방향만."""
    def draw(c):
        c.setFillColorRGB(0, 0, 0)
        c.rect(100, 500, 2.2, 2.0, stroke=0, fill=1)

    assert_rules(rules_of(draw), [Rule("h", 341.0, 100.0, 102.2)])


@pytest.mark.parametrize("x", [100.0, 100.37, 333.3, 501.13])
@pytest.mark.parametrize("length,kept", [(1.99, False), (2.0, True), (2.01, True)])
def test_rule_of_exactly_min_length_survives_float_error(x, length, kept):
    """MIN_RULE(2pt) 길이 선은 좌표 변환의 부동소수 오차로 2pt를 살짝 밑돌아도 남는다."""
    def draw(c):
        c.line(x, 400, x + length, 400)
        c.line(200, x, 200, x + length)

    assert [r.axis for r in rules_of(draw)] == (["h", "v"] if kept else [])


def extract_time(draw) -> tuple[float, PageText]:
    """draw(canvas)로 만든 PDF의 extract_pages CPU 시간(세 번 중 가장 짧은 것)과 그 쪽."""
    from time import process_time

    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(W, H), invariant=1, pageCompression=0)
    draw(c)
    c.showPage()
    c.save()
    data = buf.getvalue()
    times = []
    for _ in range(3):
        start = process_time()
        (page,) = extract_pages(data, "t.pdf")
        times.append(process_time() - start)
    return min(times), page


def test_one_fill_path_of_many_squares_is_fast_and_still_dotted_lines():
    """채운 path 하나에 1pt 네모 점 10,000개(가로 2pt 간격 100개씩 100줄): 부분 경로끼리 비교하지 않고, 점은 얇은
    사각형 처리(점선 조각)를 그대로 받아 줄마다 한 선으로 이어진다."""
    def draw(c):
        path = c.beginPath()
        for i in range(10_000):
            path.rect(20 + 2 * (i % 100), 20 + 8 * (i // 100), 1, 1)
        c.drawPath(path, stroke=0, fill=1)

    seconds, page = extract_time(draw)
    assert seconds < 0.5
    assert len(page.rules) == 100 and {(r.axis, r.kind, r.end - r.start) for r in page.rules} == {("h", "stroke", 199.0)}


def test_one_fill_path_of_many_nested_rects_is_fast():
    """채운 path 하나에 겹겹이 든 사각형 4,000개: 겹침을 따지지 않으니 선형."""
    def draw(c):
        path = c.beginPath()
        for i in range(4_000):
            d = 0.07 * i
            path.rect(10 + d, 10 + d, 575 - 2 * d, 822 - 2 * d)
        c.drawPath(path, stroke=0, fill=1, fillMode=0)

    seconds, page = extract_time(draw)
    assert seconds < 0.5
    assert {r.kind for r in page.rules} == {"stroke"}


def test_rects_in_a_fill_path_with_other_shapes_are_still_borders():
    """채운 path에 사각형 둘과 삼각형 하나: 삼각형은 무시하고 사각형 둘은 사각형만 있을 때와 같은 테두리(stroke 8변)."""
    def rects(c, triangle):
        path = c.beginPath()
        path.rect(100, 500, 100, 50)
        path.rect(300, 500, 100, 50)
        if triangle:
            path.moveTo(100, 300)
            path.lineTo(200, 300)
            path.lineTo(150, 350)
            path.close()
        c.setFillColorRGB(0.85, 0.85, 0.85)
        c.drawPath(path, stroke=0, fill=1)

    def key(rule):
        return rule.axis, rule.pos, rule.start, rule.end

    alone = sorted(rules_of(lambda c: rects(c, False)), key=key)
    assert len(alone) == 8 and {r.kind for r in alone} == {"stroke"}
    assert sorted(rules_of(lambda c: rects(c, True)), key=key) == alone


def test_bowtie_through_the_four_corners_is_not_a_rect():
    """네 모서리를 대각선으로 지나는 나비 모양(꼭짓점은 모두 외접 상자 모서리)은 사각형이 아니다."""
    def draw(c):
        path = c.beginPath()
        path.moveTo(100, 500)
        path.lineTo(200, 550)
        path.lineTo(200, 500)
        path.lineTo(100, 550)
        path.close()
        c.drawPath(path, stroke=0, fill=1)

    assert rules_of(draw) == []


def walk(c, points):
    """points를 차례로 이은 닫힌 path 하나를 채운다(PDF 좌표)."""
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    c.setFillColorRGB(0.85, 0.85, 0.85)
    c.drawPath(path, stroke=0, fill=1)


def test_retracing_walk_over_rect_corners_is_not_a_rect():
    """A→B→C→D→C→B→A: 꼭짓점·변은 모두 사각형 둘레 위지만 넓이가 0이다(없는 왼쪽 변을 내지 않는다)."""
    a, b, cc, d = (100, 500), (200, 500), (200, 550), (100, 550)
    assert rules_of(lambda c: walk(c, [a, b, cc, d, cc, b, a])) == []


@pytest.mark.parametrize("backwards", [False, True])
def test_filled_rect_of_either_winding_is_a_rect(backwards):
    corners = [(100, 500), (200, 500), (200, 550), (100, 550)]
    found = rules_of(lambda c: walk(c, corners[::-1] if backwards else corners))
    assert sorted((r.axis, r.pos, r.start, r.end, r.kind) for r in found) == [
        ("h", 292.0, 100.0, 200.0, "fill"), ("h", 342.0, 100.0, 200.0, "fill"),
        ("v", 100.0, 292.0, 342.0, "fill"), ("v", 200.0, 292.0, 342.0, "fill")]


def draw_budget_page(c):
    """path 구간 52개(작은 네모 10개 × 5 + 선 1개 × 2), 점선 조각 60개(네모마다 그은 변 4 + 가운데 선 2)와 글자 하나."""
    for i in range(10):
        c.rect(100 + 20 * i, 500, 0.48, 0.48, stroke=1, fill=1)
    c.line(100, 300, 300, 300)
    t = c.beginText(100, 200)
    t.setFont(FONT, 10)
    t.textOut("가")
    c.drawText(t)


@pytest.mark.parametrize("budget,has_rules", [(51, False), (55, False), (60, True), (1000, True)])
def test_rule_collection_stops_when_the_page_exceeds_the_segment_budget(monkeypatch, budget, has_rules):
    """쪽의 path 구간(51 < 52) 또는 점선 조각(55 < 60)이 MAX_RULE_SEGMENTS를 넘으면 그 쪽 선은 () — 글자는 그대로."""
    monkeypatch.setattr(extract, "MAX_RULE_SEGMENTS", budget)
    page = page_of(draw_budget_page)
    assert page.rules == ((Rule("h", 542.0, 100.0, 300.0),) if has_rules else ())
    assert [ch.text for ch in page.chars] == ["가"]
