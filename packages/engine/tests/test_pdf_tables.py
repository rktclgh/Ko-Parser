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


def framed_two_column_page(p, footer=False):
    """쪽 테두리 상자 + 제목 밑줄 + 단 나눔 세로선(+ 꼬리말 위 가로선): 쪽 꾸밈이지 표가 아니다."""
    left, right, top, bottom = 40, 555, 40, 802
    grid(p, [left, right], [top, bottom])
    p.h(100, left, right)
    p.v(297, 100, 760 if footer else bottom)
    if footer:
        p.h(760, left, right)
        p.text(60, 785, "- 1 -")
    p.text(60, 80, "보도자료 제목")
    for i in range(30):
        p.text(50, 130 + 20 * i, f"왼쪽 단 본문 {i}줄")
        p.text(307, 130 + 20 * i, f"오른쪽 단 본문 {i}줄")


@pytest.mark.parametrize("draw", [box_with_text, strip_1xn, bar_chart, underline, framed_two_column_page,
                                  lambda p: framed_two_column_page(p, footer=True)],
                         ids=["box", "strip", "chart", "underline", "page-frame", "page-frame-footer"])
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


def test_invisible_inner_vertical_line_column_from_text_alignment():
    """2·3열 사이 세로선이 안 보인다(한글 프로그램 안쪽 테두리 '없음'). 모든 행에서 글자가 비켜 가는 넓은 빈틈."""
    def draw(p):
        grid(p, XS, YS, skip_v={(2, r) for r in range(3)})
        fill_texts(p)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec) == [(r, k, 1, 1, TEXTS[r][k]) for r in range(3) for k in range(3)]


CONTACTS = [("담당 부서", "총괄과", "과장", "김가나", "(044-000-0001)"),
            ("", "지원과", "사무관", "이다라", "(044-000-0002)"),
            ("", "지원과", "주무관", "박마바", "(044-000-0003)")]


def contact_table(p, rows=CONTACTS):
    """보도자료 끝 담당자 표: 이름과 전화 사이에 선이 없고 공백 하나 남짓(3pt) 떨어져 있다."""
    xs = [60, 150, 240, 300, 540]
    ys = [100 + 20 * i for i in range(len(rows) + 1)]
    grid(p, xs, ys)
    for r, row in enumerate(rows):
        for k, s in enumerate(row[:3]):
            if s:
                p.text(xs[k] + 5, ys[r] + 14, s)
        p.text(305, ys[r] + 14, row[3])
        p.text(338, ys[r] + 14, row[4])


def test_narrow_aligned_gap_in_every_row_is_a_column():
    (spec,) = find_tables(page_of(contact_table))
    assert spec.table.n_cols == 5
    assert [(c.col, c.text) for c in spec.table.cells if c.row == 0][-2:] == [(3, "김가나"), (4, "(044-000-0001)")]


def test_narrow_gap_needs_three_rows():
    (spec,) = find_tables(page_of(lambda p: contact_table(p, CONTACTS[:2])))
    assert spec.table.n_cols == 4
    assert [c.text for c in spec.table.cells if c.row == 0][-1] == "김가나 (044-000-0001)"


@pytest.mark.parametrize("second", [157, 162], ids=["110%", "160%"])
def test_two_line_paragraph_in_cells_is_not_split_into_rows(second):
    """안쪽 가로선이 없는 행 띠에 두 열 모두 두 줄 문단(줄 간격이 좁다). 한 줄짜리 행보다 줄 간격이 좁으니 나누지 않는다.
    160%는 한글 프로그램 기본 줄 간격(10pt 글자, 기준선 16pt 간격)."""
    def draw(p):
        ys = [100, 130, 160]
        grid(p, XS, ys)
        fill_texts(p, [TEXTS[0]], ys=ys)
        p.text(110, 146, "가나다")
        p.text(110, second, "라마")
        p.text(210, 146, "바사아")
        p.text(210, second, "자차")
        p.text(310, 151, "카")

    (spec,) = find_tables(page_of(draw))
    assert spec.table.n_rows == 2
    assert [c.text for c in spec.table.cells if c.row == 1] == ["가나다\n라마", "바사아\n자차", "카"]


def test_aligned_lines_with_wide_spacing_split_into_rows():
    """가로선 없이 값 줄과 비율 줄이 열마다 같은 높이로 넉넉히 떨어져 있다(1열 이름은 두 줄 가운데: 병합 칸)."""
    def draw(p):
        ys = [100, 130, 190]
        grid(p, XS, ys)
        fill_texts(p, [TEXTS[0]], ys=ys)
        p.text(110, 164, "수출")
        for k, (value, ratio) in enumerate([("11", "(1.5)"), ("22", "(3.5)")], 1):
            p.text(XS[k] + 10, 150, value)
            p.text(XS[k] + 10, 178, ratio)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec)[3:] == [(1, 0, 2, 1, "수출"), (1, 1, 1, 1, "11"), (1, 2, 1, 1, "22"),
                               (2, 1, 1, 1, "(1.5)"), (2, 2, 1, 1, "(3.5)")]


def test_one_char_per_row_in_a_column_is_a_merged_cell():
    """세로로 한 글자씩 쓴 칸: 1열에는 행 사이 가로선이 없고 행마다 한 글자뿐이다."""
    def draw(p):
        ys = [100, 120, 140, 160, 180]
        grid(p, XS, ys, skip_h={(j, 0) for j in (2, 3)})
        p.text(110, 115, "구분")
        for r, s in [(1, "조"), (2, "사"), (3, "치")]:
            p.text(145, ys[r] + 15, s)
        for r in range(4):
            p.text(210, ys[r] + 15, f"{r}건")
            p.text(310, ys[r] + 15, f"{r}명")

    (spec,) = find_tables(page_of(draw))
    assert cells(spec)[3] == (1, 0, 3, 1, "조\n사\n치")


@pytest.mark.parametrize("baseline,merged", [(119, False), (129, True)])
def test_one_sided_text_splits_only_when_centered_in_its_own_row(baseline, merged):
    """1열 1·2행 사이 가로선이 없고 위 칸에만 글자. 글자가 제 행 가운데면 나뉜 칸(아래는 빈 칸), 경계 쪽으로 치우쳐
    있으면(병합 칸 가운데 정렬) 합친 칸."""
    def draw(p):
        grid(p, XS, YS, skip_h={(2, 0)})
        fill_texts(p, [TEXTS[0], ["", "1", "2"], ["", "3", "4"]])
        p.text(110, baseline + 30, "합계")

    (spec,) = find_tables(page_of(draw))
    col0 = [(c.row, c.rowspan, c.text) for c in spec.table.cells if c.col == 0]
    assert col0 == ([(0, 1, "구분"), (1, 2, "합계")] if merged else [(0, 1, "구분"), (1, 1, "합계"), (2, 1, "")])


def test_words_across_a_partial_line_merge_the_header_cell():
    """머리행에만 2·3열 사이 세로선이 없고, 낱말 사이 공백이 그 자리에 걸친 머리글은 한 칸(2열 병합)."""
    def draw(p):
        grid(p, XS, YS, skip_v={(2, 0)})
        p.text(110, 120, "구분")
        p.text(270, 120, "가나다 라마")
        for r in (1, 2):
            for k in range(3):
                p.text(XS[k] + 10, YS[r] + 20, TEXTS[r][k])

    (spec,) = find_tables(page_of(draw))
    assert cells(spec)[:2] == [(0, 0, 1, 1, "구분"), (0, 1, 1, 2, "가나다 라마")]


def ruled_grid(rows: int, cols: int, cw: float, ch: float, per_cell: int, every: int = 1) -> PageText:
    """선을 모두 그은 rows × cols 표. every번째 칸마다 per_cell 글자(크기는 칸에 맞춘다). PDF 없이 PageText로."""
    x0, y0 = 20.0, 20.0
    rules = tuple([Rule("h", y0 + r * ch, x0, x0 + cols * cw) for r in range(rows + 1)]
                  + [Rule("v", x0 + k * cw, y0, y0 + rows * ch) for k in range(cols + 1)])
    size = min(cw / (per_cell + 1), ch * 0.6)
    chars = []
    for n, (r, k) in enumerate((r, k) for r in range(rows) for k in range(cols)):
        if n % every:
            continue
        base = y0 + r * ch + ch * 0.7
        for i in range(per_cell):
            x = x0 + k * cw + 0.5 + i * size
            chars.append(Char(text="가", x0=x / W, y0=(base - 0.752 * size) / H, x1=(x + size) / W,
                              y1=(base + 0.142 * size) / H, baseline=base / H, size=size))
    return PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=tuple(chars), image_coverage=(), rules=rules)


@pytest.mark.parametrize("page,shape,limit", [
    (ruled_grid(30, 60, 9.0, 26.0, 3), (30, 60), 0.5),  # 5,400자(칸마다 비교하면 1.25초 걸리던 크기)
    (ruled_grid(199, 199, 2.8, 4.0, 1, every=20), None, 3.0)],  # 칸 39,601개·1,980자(10.6초 걸리던 크기)
    ids=["30x60", "199x199"])
def test_large_grid_stays_fast(page, shape, limit):
    """글자 정렬·경계 판정이 격자 칸 수 × 글자 수로 커지지 않는다(글자를 격자 칸별로 한 번 나눈다). CPU 시간 세 번 중
    가장 짧은 것으로 잰다. 한도는 실측(0.04초·0.15초)의 10배 이상 여유(느린 CI)."""
    from time import process_time

    times = []
    for _ in range(3):
        start = process_time()
        found = find_tables(page)
        times.append(process_time() - start)
    assert min(times) < limit
    assert [(t.table.n_rows, t.table.n_cols) for t in found] == ([shape] if shape else [])


def merged_table(p):
    """맨 윗행 2·3열과 1열 2·3행이 합쳐진 표."""
    grid(p, XS, YS, skip_v={(2, 0)}, skip_h={(2, 0)})
    p.text(110, 120, "구분")
    p.text(280, 120, "실적")
    p.text(110, 164, "합계")
    for r, k, s in [(1, 1, "1월"), (1, 2, "2월"), (2, 1, "3월"), (2, 2, "4월")]:
        p.text(XS[k] + 10, YS[r] + 20, s)


def test_table_over_the_expanded_text_limit_is_not_a_table(monkeypatch):
    """병합 칸 글자를 덮인 칸마다 펼친 글자 수가 계약 상한을 넘으면 표로 내지 않는다(글자는 문단에 남는다)."""
    from ko_parser.formats.pdf import tables

    page = page_of(merged_table)
    (spec,) = find_tables(page)
    expanded = sum(len(c.text) * c.rowspan * c.colspan for c in spec.table.cells)
    monkeypatch.setattr(tables, "MAX_TABLE_EXPANDED_CHARS", expanded - 1)
    assert find_tables(page) == []


@pytest.mark.parametrize("shaded,header", [([0], False), ([2], False), ([0, 1, 2], True)])
def test_header_needs_every_top_cell_shaded(shaded, header):
    """칸마다 따로 칠한 배경: 맨 윗행 칸이 모두 칠해져야 머리행(선과 이어진 한 칸의 배경이 행 전체로 번지지 않는다)."""
    def draw(p):
        for k in shaded:
            p.fill(XS[k], YS[0], XS[k + 1], YS[1])
        full_table(p)

    (spec,) = find_tables(page_of(draw))
    assert {c.header for c in spec.table.cells if c.row == 0} == {"column" if header else "none"}


def test_zero_length_rule_is_not_a_segment():
    from ko_parser.formats.pdf.tables import _reading_segs

    assert _reading_segs([Rule("h", 10, 5, 5), Rule("v", 20, 7, 7)], UPRIGHT, W, H) == []


def test_non_rectangular_merge_takes_the_largest_rectangle_then_single_cells():
    """ㄱ자로 이어진 칸 묶음: 가장 큰 직사각형부터, 남은 칸은 1×1."""
    from ko_parser.formats.pdf.tables import _cells

    joined = {(0, 0, True), (0, 0, False), (0, 1, True), (1, 2, False)}  # (0,0)-(0,1)-(0,2), (0,0)-(1,0), (1,2)-(2,2)
    out = _cells(3, 3, lambda r, c, across: (r, c, across) not in joined)
    assert out == [(0, 0, 1, 3), (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 2, 1), (2, 0, 1, 1), (2, 1, 1, 1)]
    staircase = {(0, 0, False), (1, 0, True), (1, 1, False), (2, 1, True)}  # (0,0)-(1,0)-(1,1)-(2,1)-(2,2)
    out = _cells(3, 3, lambda r, c, across: (r, c, across) not in staircase)
    covered = sorted((r + i, c + j) for r, c, h, w in out for i in range(h) for j in range(w))
    assert covered == [(r, c) for r in range(3) for c in range(3)]
    assert (0, 0, 2, 1) in out and (2, 2, 1, 1) in out


def test_random_separations_always_make_a_valid_contract_table():
    """임의의 칸 경계 판정 200가지: _cells → _compact 결과가 늘 계약 Table(칸이 겹치지 않고 격자를 다 덮음)이다."""
    import random

    from ko_parser.formats.pdf.tables import _cells, _compact
    from ko_parser_contracts import Cell, Table

    rng = random.Random(20261004)
    for _ in range(200):
        n_rows, n_cols = rng.randint(1, 7), rng.randint(1, 7)
        p = rng.random()
        seps = {(r, c, a): rng.random() < p for r in range(n_rows) for c in range(n_cols) for a in (True, False)}
        out = _cells(n_rows, n_cols, lambda r, c, across, seps=seps: seps[(r, c, across)])
        xs, ys, out = _compact([float(i) for i in range(n_cols + 1)], [float(i) for i in range(n_rows + 1)], out)
        Table(n_rows=len(ys) - 1, n_cols=len(xs) - 1, cells=tuple(
            Cell(row=r, col=c, rowspan=h, colspan=w, text="", text_source="text_layer", header="none")
            for r, c, h, w in out))


def test_one_glyph_values_in_text_aligned_rows_stay_separate():
    """머리행 아래 가로선과 맨 아래 선만 있고 본문 세 행은 글자 줄로만 나뉜다. 숫자 열·표시 열은 행마다 한 글자지만
    세로로 한 글자씩 쓴 병합 칸이 아니다(글자 정렬 행 경계에서 묶음이 끊긴다)."""
    rows = [["구분", "건수", "여부"], ["수출", "1", "○"], ["수입", "2", "-"], ["합계", "3", "○"]]

    def draw(p):
        grid(p, XS, [100, 130, 220])
        for r, row in enumerate(rows):
            for k, s in enumerate(row):
                p.text(XS[k] + 10, 120 + 30 * r, s)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec) == [(r, k, 1, 1, rows[r][k]) for r in range(4) for k in range(3)]


def test_spaced_two_glyph_labels_are_not_split_into_columns():
    """'구 분'처럼 두 글자 사이를 띄운 이름표가 모든 행에서 같은 자리: 한 글자|한 글자 빈틈은 열 경계가 아니다."""
    rows = [["구 분", "2025", "2026"], ["수 출", "11", "22"], ["수 입", "33", "44"]]

    def draw(p):
        grid(p, XS, YS)
        fill_texts(p, rows)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec) == [(r, k, 1, 1, rows[r][k]) for r in range(3) for k in range(3)]


def test_phone_numbers_with_a_space_are_not_split_into_columns():
    """'(044) 215-0001'의 공백이 모든 행에서 같은 자리: 한 칸 안의 낱말 사이(공백 글자가 있다)."""
    rows = [("총괄", "(044) 215-0001"), ("지원", "(044) 215-0002"), ("운영", "(044) 215-0003")]

    def draw(p):
        ys = [100, 120, 140, 160]
        grid(p, [100, 200, 400], ys)
        for r, row in enumerate(rows):
            for k, s in enumerate(row):
                p.text([110, 210][k], ys[r] + 14, s)

    (spec,) = find_tables(page_of(draw))
    assert [c.text for c in spec.table.cells] == [s for row in rows for s in row]


def test_distributed_names_are_not_split_into_columns():
    """배분 정렬한 이름(글자마다 12pt 띄움)이 모든 행에서 같은 자리: 빈틈 양쪽이 모두 한 글자뿐이다."""
    names = ["홍길동", "김철수", "이영희"]

    def draw(p):
        ys = [100, 120, 140, 160]
        grid(p, [100, 200, 400], ys)
        for r, name in enumerate(names):
            p.text(110, ys[r] + 14, "담당")
            for i, ch in enumerate(name):
                p.text(210 + 22 * i, ys[r] + 14, ch)

    (spec,) = find_tables(page_of(draw))
    assert spec.table.n_cols == 2
    assert [c.text.replace(" ", "") for c in spec.table.cells if c.col == 1] == names


def test_narrow_gap_missing_in_one_row_is_not_a_column():
    """네 행 중 세 행에는 이름|전화 좁은 빈틈이 있고 한 행은 이름만 있다(빈틈이 없다): 좁은 빈틈 열은 글자 있는
    모든 행에 있어야 한다."""
    def draw(p):
        contact_table(p)
        ys = [160, 180]
        grid(p, [60, 150, 240, 300, 540], ys)
        p.text(65, ys[1] - 6, "지원과")
        p.text(305, ys[1] - 6, "박마바")

    (spec,) = find_tables(page_of(draw))
    assert spec.table.n_cols == 4


def test_box_centre_on_a_grid_line_goes_to_the_right_or_lower_cell():
    """상자 중심이 안쪽 격자선 위면 오른쪽(아래) 칸, 바깥 오른쪽·아래 변 위면 마지막 칸(_build의 글자 배정과 같다)."""
    from ko_parser.formats.pdf.tables import _buckets

    cells_ = _buckets([0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [(5.0, 0.0, 15.0, 10.0), (15.0, 15.0, 25.0, 25.0)])
    assert cells_[0][1] == [(5.0, 0.0, 15.0, 10.0)] and cells_[1][1] == [(15.0, 15.0, 25.0, 25.0)]


def test_large_non_rectangular_merge_stays_fast():
    """199 × 199 격자(바깥 테두리와 짧은 눈금만, 맨 왼쪽 위 칸만 선으로 나뉨): 나머지 ㄱ자 칸 묶음에서 가장 큰
    직사각형을 찾는 일이 칸 수의 제곱으로 커지지 않는다(이전 16.4초)."""
    from time import process_time

    n, cw, ch, x0, y0 = 199, 2.8, 4.0, 20.0, 20.0
    right, bottom = x0 + n * cw, y0 + n * ch
    rules = (Rule("h", y0, x0, right), Rule("h", bottom, x0, right), Rule("v", x0, y0, bottom), Rule("v", right, y0, bottom),
             *(Rule("v", x0 + k * cw, y0, y0 + 1) for k in range(1, n)),
             *(Rule("h", y0 + r * ch, x0, x0 + 1) for r in range(1, n)),
             Rule("v", x0 + cw, y0, y0 + ch), Rule("h", y0 + ch, x0, x0 + cw))
    size = 1.0
    base = y0 + ch * 0.7
    char = Char(text="가", x0=(x0 + 0.5) / W, y0=(base - 0.752 * size) / H, x1=(x0 + 1.5) / W,
                y1=(base + 0.142 * size) / H, baseline=base / H, size=size)
    page = PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=(char,), image_coverage=(), rules=rules)
    times = []
    for _ in range(3):
        start = process_time()
        found = find_tables(page)
        times.append(process_time() - start)
    assert min(times) < 2.0
    (spec,) = found
    assert [(c.row, c.col, c.rowspan, c.colspan) for c in spec.table.cells] == [(0, 0, 1, 1), (0, 1, 2, 1), (1, 0, 1, 1)]


def framed_header_table(p, fill_mode=0, insets=(1,), reverse=False, one_path=False):
    """3 × 3 표의 맨 윗행 칸을 '바깥 사각형 + 안쪽 사각형(insets만큼 들임)'으로 채워 그린다. one_path면 세 칸을 한
    path로, reverse면 안쪽 사각형을 반대 방향으로 감는다. 나머지 선은 그은 선."""
    def rect(path, x, y, w, h, backwards):
        corners = [(x, y), (x, y + h), (x + w, y + h), (x + w, y)] if backwards else \
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        path.moveTo(*corners[0])
        for corner in corners[1:]:
            path.lineTo(*corner)
        path.close()

    path = p.c.beginPath()
    for k in range(3):
        x, y = XS[k], p.height - YS[1]
        rect(path, x, y, 100, 30, False)
        for d in insets:
            rect(path, x + d, y + d, 100 - 2 * d, 30 - 2 * d, reverse)
        if not one_path or k == 2:
            p.c.drawPath(path, stroke=0, fill=1, fillMode=fill_mode)
            path = p.c.beginPath()
    grid(p, XS, YS[1:])
    fill_texts(p)


FRAME_CASES = {"even-odd": {}, "even-odd-one-path": {"one_path": True},
               "nonzero-same": {"fill_mode": 1}, "nonzero-opposite": {"fill_mode": 1, "reverse": True},
               "triple": {"insets": (0.5, 1)}, "triple-one-path": {"insets": (0.5, 1), "one_path": True},
               "duplicate-inner": {"insets": (1, 1)}}


@pytest.mark.parametrize("kwargs", FRAME_CASES.values(), ids=FRAME_CASES.keys())
def test_multi_rect_filled_paths_are_borders_never_a_header_background(kwargs):
    """채운 path 하나에 사각형이 여럿이면(속이 빈 틀, 칸 여러 개의 틀, 셋을 겹친 사각형, 같은 안쪽 사각형 둘) 채움
    규칙을 따지지 않고 모두 테두리로 본다. 맞바꾼 것: nonzero 같은 방향·세 겹처럼 실제로 속까지 칠한 배경도 머리행
    표시를 잃는다(머리행은 덧붙인 정보일 뿐이다)."""
    (spec,) = find_tables(page_of(lambda p: framed_header_table(p, **kwargs)))
    assert cells(spec) == [(r, k, 1, 1, TEXTS[r][k]) for r in range(3) for k in range(3)]
    assert {c.header for c in spec.table.cells} == {"none"}


@pytest.mark.parametrize("kwargs", [{}, {"fill_mode": 1, "reverse": True}, {"insets": (0.5, 1)}],
                         ids=["even-odd", "nonzero-opposite", "triple"])
def test_frames_in_one_path_give_the_same_rules_as_separate_paths(kwargs):
    separate = page_of(lambda p: framed_header_table(p, **kwargs))
    merged = page_of(lambda p: framed_header_table(p, one_path=True, **kwargs))

    def key(rule):
        return rule.axis, rule.pos, rule.start, rule.end, rule.kind

    assert sorted(merged.rules, key=key) == sorted(separate.rules, key=key)
    assert {r.kind for r in merged.rules} == {"stroke"}


def test_shaded_rows_under_a_tall_top_left_cell_are_not_a_column_header():
    """맨 왼쪽 위 칸이 두 행을 차지하고(rowspan 2) 위 두 행이 칠해졌다: 머리글이 한 행이 아니니 열 머리행으로 보지
    않는다(그 아래 셋째 행과 비교해 머리행이라 하지 않는다)."""
    def draw(p):
        p.fill(XS[0], YS[0], XS[1], YS[2])  # 칸마다 따로 칠한 배경
        for r in (0, 1):
            p.fill(XS[1], YS[r], XS[-1], YS[r + 1])
        grid(p, XS, YS, skip_h={(1, 0)})
        p.text(110, 134, "구분")
        for r, row in enumerate([["", "상반기", "하반기"], ["", "1월", "7월"], ["수출", "11", "22"]]):
            for k, s in enumerate(row):
                if s:
                    p.text(XS[k] + 10, YS[r] + 20, s)

    (spec,) = find_tables(page_of(draw))
    assert cells(spec)[0] == (0, 0, 2, 1, "구분")
    assert {c.header for c in spec.table.cells} == {"none"}


def test_large_ruled_table_with_a_paragraph_cell_is_a_table():
    """쪽의 75%를 덮는 4 × 4 선 격자에 6줄 문단 칸이 있어도 표다(쪽 꾸밈 선 거르기는 3 × 3 이하 선 격자에만)."""
    xs = [40 + 128.75 * k for k in range(5)]
    ys = [60 + 182.5 * r for r in range(5)]

    def draw(p):
        grid(p, xs, ys)
        for r in range(4):
            for k in range(4):
                if (r, k) != (1, 1):
                    p.text(xs[k] + 10, ys[r] + 30, f"칸{r}{k}")
        for i in range(6):
            p.text(xs[1] + 10, ys[1] + 30 + 14 * i, f"문단 {i}줄")

    (spec,) = find_tables(page_of(draw))
    assert (spec.table.n_rows, spec.table.n_cols) == (4, 4)
    assert [c.text.count("\n") for c in spec.table.cells if (c.row, c.col) == (1, 1)] == [5]
