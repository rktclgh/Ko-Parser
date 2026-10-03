import io

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from ko_parser.formats.pdf.extract import UPRIGHT, Char, PageText, Rule, extract_pages
from ko_parser.formats.pdf.tables import TableSpec, find_tables

FONT = "HYGothic-Medium"  # 한글 너비 = 크기, ASCII = 크기/2, 상자는 기준선 위 0.752·아래 0.142 × 크기
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
W, H = 595.0, 842.0


class Pen:
    """보이는 쪽 좌표(원점 왼쪽 위, pt)로 그린다. height는 MediaBox 높이(회전 쪽은 눕힌 MediaBox)."""

    def __init__(self, c: Canvas, height: float = H):
        self.c, self.height = c, height

    def h(self, y: float, x0: float, x1: float) -> None:
        self.c.line(x0, self.height - y, x1, self.height - y)

    def v(self, x: float, y0: float, y1: float) -> None:
        self.c.line(x, self.height - y0, x, self.height - y1)

    def text(self, x: float, baseline: float, s: str, size: float = 10) -> None:
        t = self.c.beginText(x, self.height - baseline)
        t.setFont(FONT, size)
        t.textOut(s)
        self.c.drawText(t)

    def fill(self, x0: float, y0: float, x1: float, y1: float, gray: float = 0.85) -> None:
        self.c.saveState()
        self.c.setFillGray(gray)
        self.c.rect(x0, self.height - y1, x1 - x0, y1 - y0, stroke=0, fill=1)
        self.c.restoreState()


def page_of(draw, rotation: int = 0) -> PageText:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(W, H), invariant=1, pageCompression=0)
    if rotation:
        c.setPageRotation(rotation)
    draw(Pen(c, W if rotation in (90, 270) else H))
    c.showPage()
    c.save()
    (page,) = extract_pages(buf.getvalue(), "t.pdf")
    return page


def cells(spec: TableSpec) -> list[tuple[int, int, int, int, str]]:
    return [(c.row, c.col, c.rowspan, c.colspan, c.text) for c in spec.table.cells]


def grid(p: Pen, xs, ys, skip_v=(), skip_h=()) -> None:
    """xs·ys 격자. skip_v = {(x 순번, 행 순번)}: 그 행에서 그 세로선을 긋지 않는다. skip_h = {(y 순번, 열 순번)}."""
    for j, y in enumerate(ys):
        for k in range(len(xs) - 1):
            if (j, k) not in skip_h:
                p.h(y, xs[k], xs[k + 1])
    for i, x in enumerate(xs):
        for r in range(len(ys) - 1):
            if (i, r) not in skip_v:
                p.v(x, ys[r], ys[r + 1])


XS, YS = [100, 200, 300, 400], [100, 130, 160, 190]
TEXTS = [["구분", "2025", "2026"], ["수출", "11", "22"], ["수입", "33", "44"]]


def fill_texts(p: Pen, texts=TEXTS, xs=XS, ys=YS) -> None:
    for r, row in enumerate(texts):
        for k, s in enumerate(row):
            if s:
                p.text(xs[k] + 10, ys[r] + 20, s)


def full_table(p: Pen) -> None:
    grid(p, XS, YS)
    fill_texts(p)


def test_fully_ruled_3x3_table():
    page = page_of(full_table)
    (spec,) = find_tables(page)
    assert (spec.table.n_rows, spec.table.n_cols) == (3, 3)
    assert cells(spec) == [(r, k, 1, 1, TEXTS[r][k]) for r in range(3) for k in range(3)]
    assert {c.header for c in spec.table.cells} == {"none"}
    assert {c.text_source for c in spec.table.cells} == {"text_layer"}
    assert spec.bbox == (0.168, 0.119, 0.672, 0.226) and spec.axes == UPRIGHT
    assert spec.char_ids == frozenset(range(len(page.chars)))


def test_open_left_and_right_borders_are_closed_with_virtual_lines():
    def draw(p):
        grid(p, XS, YS, skip_v={(0, r) for r in range(3)} | {(3, r) for r in range(3)})
        fill_texts(p)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec) == [(r, k, 1, 1, TEXTS[r][k]) for r in range(3) for k in range(3)]
    assert spec.bbox == (0.168, 0.119, 0.672, 0.226)


def test_cells_without_a_line_between_them_are_merged():
    """맨 윗행 2·3열(세로선 없음)과 1열 2·3행(가로선 없음)이 합쳐진 칸."""
    def draw(p):
        grid(p, XS, YS, skip_v={(2, 0)}, skip_h={(2, 0)})
        p.text(110, 120, "구분")
        p.text(280, 120, "실적")
        p.text(110, 164, "합계")
        for r, k, s in [(1, 1, "1월"), (1, 2, "2월"), (2, 1, "3월"), (2, 2, "4월")]:
            p.text(XS[k] + 10, YS[r] + 20, s)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec) == [(0, 0, 1, 1, "구분"), (0, 1, 1, 2, "실적"), (1, 0, 2, 1, "합계"), (1, 1, 1, 1, "1월"),
                           (1, 2, 1, 1, "2월"), (2, 1, 1, 1, "3월"), (2, 2, 1, 1, "4월")]


def test_cell_lines_join_with_newline():
    def draw(p):
        grid(p, XS, [100, 140, 170])
        p.text(110, 115, "첫 줄")
        p.text(110, 130, "둘째 줄")
        for k, s in [(1, "가"), (2, "나")]:
            p.text(XS[k] + 10, 120, s)
        for k, s in enumerate(["다", "라", "마"]):
            p.text(XS[k] + 10, 160, s)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec)[0] == (0, 0, 1, 1, "첫 줄\n둘째 줄")


@pytest.mark.parametrize("filled_rows,header", [((0,), True), ((1,), False), ((0, 1, 2), False)])
def test_header_row_is_the_top_row_with_a_background(filled_rows, header):
    def draw(p):
        for r in filled_rows:
            p.fill(XS[0], YS[r], XS[-1], YS[r + 1])
        full_table(p)

    (spec,) = find_tables(page_of(draw))
    assert [c.header for c in spec.table.cells if c.row == 0] == ["column" if header else "none"] * 3
    assert {c.header for c in spec.table.cells if c.row > 0} == {"none"}


def box_with_text(p):
    grid(p, [100, 400], [100, 160])
    p.text(110, 135, "요약: 상자 안 글은 문단이다")


def strip_1xn(p):
    grid(p, XS, [100, 130])
    fill_texts(p, [TEXTS[0]])


def bar_chart(p):
    """가로 눈금선 11개와 막대(채운 사각형). 글자는 축 이름 하나뿐이라 칸 대부분이 비었다."""
    for i in range(11):
        p.h(100 + 20 * i, 100, 400)
    p.v(100, 100, 300)
    for i, height in enumerate([80, 140, 60, 180]):
        p.fill(130 + 60 * i, 300 - height, 160 + 60 * i, 300, gray=0.3)
    p.text(105, 115, "건수")


def underline(p):
    p.text(100, 120, "밑줄 친 글자")
    p.h(123, 100, 160)


@pytest.mark.parametrize("draw", [box_with_text, strip_1xn, bar_chart, underline])
def test_not_tables(draw):
    assert find_tables(page_of(draw)) == []


def test_nearby_lines_of_another_table_do_not_move_this_grid():
    """다른 표의 세로선이 1.2pt 옆(같은 위치로 모으는 거리)에 있어도 끝이 이어지지 않으면 섞지 않는다."""
    def draw(p):
        full_table(p)
        xs, ys = [100, 200, 300, 401.2], [y + 300 for y in YS]
        grid(p, xs, ys)
        fill_texts(p, xs=xs, ys=ys)

    first, second = find_tables(page_of(draw))
    assert (first.bbox[2], second.bbox[2]) == (0.672, 0.674)


def test_table_inside_a_box_only_the_inner_table():
    def draw(p):
        grid(p, [80, 420], [80, 210])
        p.text(90, 95, "참고")
        full_table(p)

    (spec,) = find_tables(page_of(draw))
    assert spec.bbox == (0.168, 0.119, 0.672, 0.226)


def test_two_tables_on_a_page_in_reading_order():
    def draw(p):
        ys = [y + 300 for y in YS]
        grid(p, XS, ys)
        fill_texts(p, [["다", "음", "표"], ["가", "나", "다"], ["라", "마", "바"]], ys=ys)
        full_table(p)

    first, second = find_tables(page_of(draw))
    assert cells(first)[0][4] == "구분" and cells(second)[0][4] == "다"


def test_rotated_page_reads_the_table_in_text_direction():
    """/Rotate 90 쪽: 글자가 보이는 쪽에서 세로로 흐른다. 읽기 방향으로 같은 표, bbox는 보이는 쪽 기준."""
    (spec,) = find_tables(page_of(full_table, rotation=90))
    assert cells(spec) == [(r, k, 1, 1, TEXTS[r][k]) for r in range(3) for k in range(3)]
    assert spec.bbox == (0.681, 0.119, 0.832, 0.475) and spec.axes == (1, 2)


def test_table_drawn_inside_a_form_xobject():
    def draw(p):
        p.c.beginForm("table")
        grid(p, XS, YS)
        p.c.endForm()
        p.c.doForm("table")
        fill_texts(p)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec) == [(r, k, 1, 1, TEXTS[r][k]) for r in range(3) for k in range(3)]


def test_white_lines_make_no_table():
    def draw(p):
        p.c.setStrokeColorRGB(1, 1, 1)
        full_table(p)

    assert find_tables(page_of(draw)) == []


def test_grid_over_the_contract_cell_limit_is_not_a_table():
    """317 × 317 칸 > MAX_TABLE_CELLS(100,000). 선만으로 만든 쪽(글자 없음)."""
    n = 318
    rules = tuple([Rule("h", 10 + i * 2.5, 10, 10 + (n - 1) * 2.5) for i in range(n)]
                  + [Rule("v", 10 + i * 2.5, 10, 10 + (n - 1) * 2.5) for i in range(n)])
    page = PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=(), image_coverage=(), rules=rules)
    assert find_tables(page) == []


def glyph(x: float, y: float) -> Char:
    """상자 중심 (x, y) pt, 10pt 한글 한 글자(HYGothic 비율)."""
    baseline = y + 3.05
    return Char(text="가", x0=(x - 5) / W, y0=(baseline - 7.52) / H, x1=(x + 5) / W, y1=(baseline + 1.42) / H,
                baseline=baseline / H, size=10.0)


def test_overlapping_tables_do_not_share_chars():
    """서로 닿지 않는 두 선 묶음의 바깥 닫기 상자가 겹치면(가로선만 길게 뻗은 열린 표) 겹친 곳 글자가 두 표에 들어갈
    수 있다. 앞(위) 표만 남긴다: 한 글자는 한 블록에만."""
    rules = (*(Rule("h", y, 100, 300) for y in (100, 130, 160)), Rule("v", 200, 100, 160),
             *(Rule("h", y, 250, 450) for y in (140, 170, 200)), Rule("v", 350, 140, 200))
    chars = (glyph(150, 115), glyph(250, 115), glyph(150, 145), glyph(275, 150),
             glyph(400, 155), glyph(300, 185), glyph(400, 185))
    page = PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=chars, image_coverage=(), rules=rules)
    (spec,) = find_tables(page)
    assert spec.bbox == (0.168, 0.119, 0.504, 0.19) and 3 in spec.char_ids


def test_chars_of_another_direction_stay_out_of_the_table():
    page = page_of(full_table)
    stray = Char(text="세", x0=0.3, y0=0.15, x1=0.31, y1=0.16, baseline=0.3, size=10.0, axes=(1, 2))
    (spec,) = find_tables(PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=(*page.chars, stray),
                                   image_coverage=(), rules=page.rules))
    assert len(page.chars) not in spec.char_ids
