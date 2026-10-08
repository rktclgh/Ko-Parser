"""선 없는 표 모듈(borderless.recover·candidates): 합성 쪽(reportlab, 지어낸 글자)으로 칸 배치·칸 글자·글자 소유·실패
사유를 고정한다. 글꼴은 미임베드 한국어 CID 글꼴(한글 너비 = 크기, ASCII = 크기/2)."""

import io
import unicodedata
from collections import Counter

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from hanji.formats.pdf import borderless
from hanji.formats.pdf.borderless import Checks, Failure, Recovery, recover
from hanji.formats.pdf.extract import UPRIGHT, PageText, extract_pages

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
W, H = 595.0, 842.0


def page_of(draw) -> PageText:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(W, H), invariant=1, pageCompression=0)
    draw(c)
    c.showPage()
    c.save()
    (page,) = extract_pages(buf.getvalue(), "t.pdf")
    return page


def put(c: Canvas, x: float, baseline: float, s: str, size: float = 10.0) -> None:
    """보이는 쪽 좌표(원점 왼쪽 위)의 기준선에 쓴다."""
    c.setFont(FONT, size)
    c.drawString(x, H - baseline, s)


def rule(c: Canvas, y: float, x0: float, x1: float) -> None:
    c.line(x0, H - y, x1, H - y)


def rows_at(c: Canvas, rows, xs, top: float, pitch: float = 18.0) -> None:
    for i, row in enumerate(rows):
        for x, s in zip(xs, row):
            if s:
                put(c, x, top + pitch * i, s)


def everything(page: PageText) -> frozenset[int]:
    return frozenset(range(len(page.chars)))


def cells(rec: Recovery) -> list[tuple[int, int, int, int, str]]:
    return [(c.row, c.col, c.rowspan, c.colspan, c.text) for c in rec.table.cells]


def glyphs(text: str) -> Counter:
    return Counter(ch for ch in unicodedata.normalize("NFC", text) if not ch.isspace())


def assert_owns_exactly_its_cells(page: PageText, rec: Recovery) -> None:
    """표 char_ids의 글자 다중집합(공백 제외, NFC) = 칸 글자의 합. 칸 배치·칸 글자는 각 테스트가 칸마다 본다."""
    owned = sum((glyphs(page.chars[i].text) for i in rec.char_ids), Counter())
    assert owned == sum((glyphs(c.text) for c in rec.table.cells), Counter())
    assert all(c.header == "none" and c.text_source == "text_layer" for c in rec.table.cells)


TABLE_4X3 = [["구분", "2024", "2025"], ["수입", "120", "135"], ["지출", "98", "110"], ["잔액", "22", "25"]]
TABLE_5X2 = [["인건비", "1,200"], ["운영비", "850"], ["장비비", "2,400"], ["여비", "310"], ["합계", "4,760"]]


def table_4x3(c: Canvas) -> None:
    rows_at(c, TABLE_4X3, (72, 202, 332), 120)


def table_5x2(c: Canvas) -> None:
    rows_at(c, TABLE_5X2, (72, 300), 120)


@pytest.mark.parametrize("draw,rows", [(table_4x3, TABLE_4X3), (table_5x2, TABLE_5X2)])
def test_aligned_numbers_recover_every_cell_and_own_exactly_their_chars(draw, rows):
    page = page_of(draw)
    rec = recover(page, (60, 100, 400, 200), everything(page))
    assert isinstance(rec, Recovery)
    assert cells(rec) == [(r, k, 1, 1, s) for r, row in enumerate(rows) for k, s in enumerate(row)]
    assert rec.checks == Checks(chars=sum(len(s) for row in rows for s in row), rows=len(rows), cols=len(rows[0]),
                                cells=len(rows) * len(rows[0]), filled=1.0,
                                expanded_chars=sum(len(s) for row in rows for s in row))
    assert rec.bbox == pytest.approx((72.0, 112.5, 352.0 if len(rows[0]) == 3 else 325.0, 175.4 if len(rows) == 4
                                      else 193.4), abs=0.1)  # 칸에 넣은 글자의 바깥 상자
    assert_owns_exactly_its_cells(page, rec)


def table_with_side_borders(c: Canvas) -> None:
    """왼쪽·오른쪽 테두리 세로선만 있다(글자 밖)."""
    table_4x3(c)
    for x in (64, 372):
        c.line(x, H - 105, x, H - 182)


def test_outer_border_verticals_do_not_add_empty_outer_columns():
    page = page_of(table_with_side_borders)
    rec = recover(page, (55, 100, 380, 185), everything(page))
    assert isinstance(rec, Recovery)
    assert (rec.table.n_rows, rec.table.n_cols, rec.checks.filled) == (4, 3, 1.0)
    assert cells(rec) == [(r, k, 1, 1, s) for r, row in enumerate(TABLE_4X3) for k, s in enumerate(row)]
    assert_owns_exactly_its_cells(page, rec)


def wrapped_labels(c: Canvas) -> None:
    """행 간격 20pt, 둘째 줄로 넘어간 라벨은 11pt 아래."""
    for x, s in zip((72, 250, 350, 450), ("항목", "1분기", "2분기", "3분기")):
        put(c, x, 100, s)
    y = 120.0
    for k, (label, more) in enumerate([("인건비", ""), ("시설 유지", "보수비"), ("운영비", ""), ("교육 훈련", "참가비"),
                                       ("여비", ""), ("장비비", ""), ("합계", "")]):
        for x, s in zip((72, 250, 350, 450), (label, str(10 + k), str(20 + k), str(30 + k))):
            put(c, x, y, s)
        if more:
            put(c, 72, y + 11, more)
            y += 11
        y += 20


def test_label_wrapped_to_a_second_line_stays_in_its_row():
    page = page_of(wrapped_labels)
    rec = recover(page, (60, 85, 500, 270), everything(page))
    assert isinstance(rec, Recovery) and (rec.table.n_rows, rec.table.n_cols) == (8, 4)
    assert [c[4] for c in cells(rec) if c[1] == 0] == ["항목", "인건비", "시설 유지\n보수비", "운영비",
                                                       "교육 훈련\n참가비", "여비", "장비비", "합계"]
    assert [c[4] for c in cells(rec) if c[0] == 4] == ["교육 훈련\n참가비", "13", "23", "33"]
    assert_owns_exactly_its_cells(page, rec)


def booktabs(c: Canvas) -> None:
    """위·가운데·아래 가로선, 두 줄 머리(위 줄 '실적' 아래 짧은 가로선 cmidrule이 두 열을 덮는다)."""
    rule(c, 100, 72, 400)
    put(c, 245, 114, "실적")
    rule(c, 118, 200, 400)
    for x, s in zip((72, 210, 320), ("구분", "2024년", "2025년")):
        put(c, x, 132, s)
    rule(c, 137, 72, 400)
    rows_at(c, TABLE_4X3[1:], (72, 210, 320), 152)
    rule(c, 192, 72, 400)


def test_booktabs_header_spans_by_its_cmidrule_and_blank_header_cells_merge_down():
    page = page_of(booktabs)
    rec = recover(page, (72, 95, 400, 195), everything(page))
    assert isinstance(rec, Recovery)
    assert cells(rec) == [(0, 0, 2, 1, "구분"), (0, 1, 1, 2, "실적"), (1, 1, 1, 1, "2024년"), (1, 2, 1, 1, "2025년"),
                          (2, 0, 1, 1, "수입"), (2, 1, 1, 1, "120"), (2, 2, 1, 1, "135"), (3, 0, 1, 1, "지출"),
                          (3, 1, 1, 1, "98"), (3, 2, 1, 1, "110"), (4, 0, 1, 1, "잔액"), (4, 1, 1, 1, "22"),
                          (4, 2, 1, 1, "25")]
    assert_owns_exactly_its_cells(page, rec)


def money(c: Canvas) -> None:
    """통화 기호와 %가 숫자에서 반 글자 넘게 떨어져 있다(따로 덩이)."""
    for x, s in zip((72, 220, 340), ("항목", "금액", "비율")):
        put(c, x, 100, s)
    for i, (label, amount, rate) in enumerate([("가", "1,200", "15"), ("나", "850", "8"), ("다", "2,400", "30")]):
        y = 118 + 18 * i
        put(c, 72, y, label)
        put(c, 200, y, "$")
        put(c, 220, y, amount)
        put(c, 340, y, rate)
        put(c, 360, y, "%")


def test_currency_sign_and_percent_join_their_number():
    page = page_of(money)
    rec = recover(page, (60, 85, 400, 160), everything(page))
    assert isinstance(rec, Recovery) and (rec.table.n_rows, rec.table.n_cols) == (4, 3)
    assert [c[4] for c in cells(rec) if c[0] > 0 and c[1] > 0] == ["$ 1,200", "15 %", "$ 850", "8 %", "$ 2,400",
                                                                   "30 %"]
    assert_owns_exactly_its_cells(page, rec)


def centred_label(c: Canvas) -> None:
    """0열 '사업'이 두 하위 행 가운데에 있다."""
    for x, s in zip((72, 200, 320), ("구분", "항목", "금액")):
        put(c, x, 100, s)
    put(c, 200, 120, "가")
    put(c, 320, 120, "10")
    put(c, 72, 129, "사업")
    put(c, 200, 138, "나")
    put(c, 320, 138, "20")
    for x, s in zip((72, 200, 320), ("합계", "다", "30")):
        put(c, x, 156, s)


def test_vertically_centred_row_label_spans_its_sub_rows():
    page = page_of(centred_label)
    rec = recover(page, (60, 85, 400, 165), everything(page))
    assert isinstance(rec, Recovery)
    assert cells(rec) == [(0, 0, 1, 1, "구분"), (0, 1, 1, 1, "항목"), (0, 2, 1, 1, "금액"), (1, 0, 2, 1, "사업"),
                          (1, 1, 1, 1, "가"), (1, 2, 1, 1, "10"), (2, 1, 1, 1, "나"), (2, 2, 1, 1, "20"),
                          (3, 0, 1, 1, "합계"), (3, 1, 1, 1, "다"), (3, 2, 1, 1, "30")]
    assert_owns_exactly_its_cells(page, rec)


def section_rows(c: Canvas) -> None:
    rows_at(c, [["지역", "2024", "2025"], ["국내", "", ""], ["서울특별시", "10", "12"], ["부산", "7", "9"],
                ["해외", "", ""], ["도쿄", "3", "4"]], (72, 220, 340), 100)


def test_row_with_only_a_first_column_label_spans_all_columns():
    page = page_of(section_rows)
    rec = recover(page, (60, 85, 400, 200), everything(page))
    assert isinstance(rec, Recovery)
    assert [c for c in cells(rec) if c[0] in (1, 4)] == [(1, 0, 1, 3, "국내"), (4, 0, 1, 3, "해외")]
    assert (rec.table.n_rows, rec.table.n_cols, len(rec.table.cells)) == (6, 3, 14)
    assert_owns_exactly_its_cells(page, rec)


def spread_label(c: Canvas) -> None:
    """한글 균등 배분 라벨: 글자 사이가 글자 폭만큼 벌어졌다."""
    put(c, 72, 100, "구")
    put(c, 92, 100, "분")
    put(c, 220, 100, "금액")
    put(c, 72, 118, "인건비")
    put(c, 220, 118, "100")
    put(c, 72, 136, "합")
    put(c, 92, 136, "계")
    put(c, 220, 136, "100")


def test_letter_spaced_korean_label_stays_one_cell():
    page = page_of(spread_label)
    rec = recover(page, (60, 85, 300, 145), everything(page))
    assert isinstance(rec, Recovery)
    assert cells(rec) == [(0, 0, 1, 1, "구 분"), (0, 1, 1, 1, "금액"), (1, 0, 1, 1, "인건비"), (1, 1, 1, 1, "100"),
                          (2, 0, 1, 1, "합 계"), (2, 1, 1, 1, "100")]
    assert_owns_exactly_its_cells(page, rec)


def table_with_turned_note(c: Canvas) -> None:
    table_4x3(c)
    c.saveState()
    c.translate(380, H - 175)
    c.rotate(90)
    c.setFont(FONT, 10)
    c.drawString(0, 0, "세로메모")
    c.restoreState()


def test_turned_chars_and_chars_outside_free_stay_out_of_the_table():
    page = page_of(table_with_turned_note)
    turned = {i for i, c in enumerate(page.chars) if c.axes != UPRIGHT}
    assert len(turned) == 4
    rec = recover(page, (60, 100, 400, 200), everything(page))
    assert isinstance(rec, Recovery) and not rec.char_ids & turned
    assert cells(rec) == [(r, k, 1, 1, s) for r, row in enumerate(TABLE_4X3) for k, s in enumerate(row)]
    assert_owns_exactly_its_cells(page, rec)
    last_row = {i for i, c in enumerate(page.chars) if c.axes == UPRIGHT and c.y0 * H > 160}
    rec = recover(page, (60, 100, 400, 200), everything(page) - last_row)
    assert isinstance(rec, Recovery) and not rec.char_ids & last_row
    assert (rec.table.n_rows, rec.table.n_cols) == (3, 3)
    assert_owns_exactly_its_cells(page, rec)


@pytest.mark.parametrize("draw,box,expected", [
    (lambda c: put(c, 72, 100, "가"), (60, 85, 200, 110), Failure("few_chars", Checks(chars=1))),
    (lambda c: None, (60, 85, 200, 110), Failure("few_chars", Checks(chars=0))),
    (lambda c: rows_at(c, [["가나다", "라마바", "사아자"]], (72, 200, 330), 100), (60, 85, 400, 110),
     Failure("under_2x2", Checks(chars=9, rows=1, cols=3, cells=3))),
    (lambda c: rows_at(c, [["가나다"], ["라마바"], ["사아자"]], (72,), 100), (60, 85, 400, 150),
     Failure("under_2x2", Checks(chars=9, rows=3, cols=1, cells=3))),
])
def test_too_little_to_be_a_table_fails_with_its_reason_and_measured_values(draw, box, expected):
    page = page_of(draw)
    assert recover(page, box, everything(page)) == expected


def test_grid_over_the_cell_cap_fails(monkeypatch):
    monkeypatch.setattr(borderless, "MAX_TABLE_CELLS", 11)
    page = page_of(table_4x3)
    assert recover(page, (60, 100, 400, 200), everything(page)) == Failure(
        "cells_cap", Checks(chars=31, rows=4, cols=3, cells=12))


def test_expanded_text_over_the_char_cap_fails(monkeypatch):
    monkeypatch.setattr(borderless, "MAX_TABLE_EXPANDED_CHARS", 30)
    page = page_of(table_4x3)
    assert recover(page, (60, 100, 400, 200), everything(page)) == Failure(
        "chars_cap", Checks(chars=31, rows=4, cols=3, cells=12, filled=1.0, expanded_chars=31))


def test_grid_the_contract_rejects_fails_without_raising(monkeypatch):
    """계약 Table이 격자를 거부하면(ValueError) 예외 대신 bad_grid: 표 하나가 문서 파싱 전체를 멈추지 않는다."""
    def reject(**_):
        raise ValueError("bad grid")

    monkeypatch.setattr(borderless, "Table", reject)
    page = page_of(table_4x3)
    assert recover(page, (60, 100, 400, 200), everything(page)) == Failure(
        "bad_grid", Checks(chars=31, rows=4, cols=3, cells=12, filled=1.0, expanded_chars=31))


def test_sparse_grid_is_recovered_with_its_filled_ratio():
    """빈 칸 비율은 recover가 거르지 않는다(검출기 후보 거르기는 settle)."""
    page = page_of(lambda c: rows_at(c, [["가", "1", "", ""], ["나", "", "2", ""], ["다", "", "", "3"]],
                                     (72, 200, 300, 400), 100))
    rec = recover(page, (60, 85, 450, 150), everything(page))
    assert isinstance(rec, Recovery)
    assert (rec.table.n_rows, rec.table.n_cols, rec.checks.filled) == (3, 4, 0.5)
    assert_owns_exactly_its_cells(page, rec)


def test_merging_a_centred_row_shrinks_merged_cells_that_covered_it():
    """가운데 놓인 행을 위 행으로 병합하며 그 행을 지우면, 그 행을 덮던 위 병합 칸도 한 행 줄어든다(칸 겹침 없음)."""
    def line(y0: float) -> borderless._Line:
        return borderless._Line([], y0=y0, y1=y0 + 10)

    lc = [[((0, 0), [])], [((2, 2), [])], [((1, 1), [])], [((0, 0), []), ((2, 2), [])]]
    grid = borderless._grid([[0], [1], [2], [3]], lc, [line(0), line(10), line(20), line(40)], 10.0)
    assert [[(c0, c1, rs) for c0, c1, _, rs in spans] for spans in grid] == [
        [(0, 0, 1), (1, 1, 2), (2, 2, 1)], [(0, 0, 1), (2, 2, 1)]]


def test_first_column_label_row_does_not_widen_over_a_cell_merged_from_above():
    cell = borderless._Cell
    cells = [cell(0, 0, 1, 1, "가", []), cell(0, 1, 2, 1, "나", []), cell(0, 2, 1, 1, "다", []),
             cell(1, 0, 1, 1, "라", []), cell(1, 2, 1, 1, "", [])]
    out = borderless._projected_row_headers(cells, 2, 3)
    assert [(c.row, c.col, c.rowspan, c.colspan) for c in out] == [(0, 0, 1, 1), (0, 1, 2, 1), (0, 2, 1, 1),
                                                                   (1, 0, 1, 1), (1, 2, 1, 1)]


def page_glyphs(page: PageText) -> Counter:
    return sum((glyphs(c.text) for c in page.chars), Counter())


@pytest.mark.parametrize("draw,shape", [(table_4x3, (4, 3)), (table_5x2, (5, 2)), (wrapped_labels, (8, 4)),
                                        (money, (4, 3)), (centred_label, (4, 3))])
def test_detector_finds_an_aligned_table_and_recover_rebuilds_it(draw, shape):
    page = page_of(draw)
    (box,) = borderless.candidates(page, everything(page))
    rec = recover(page, box, everything(page))
    assert isinstance(rec, Recovery) and (rec.table.n_rows, rec.table.n_cols) == shape
    assert sum((glyphs(page.chars[i].text) for i in rec.char_ids), Counter()) == page_glyphs(page)


def test_detector_starts_a_booktabs_table_at_its_first_multi_column_line():
    """booktabs 위 줄의 걸친 머리('실적' 한 덩이)는 묶음을 시작하지 않는다: 표는 '구분' 줄부터다."""
    page = page_of(booktabs)
    (box,) = borderless.candidates(page, everything(page))
    rec = recover(page, box, everything(page))
    assert isinstance(rec, Recovery)
    assert cells(rec) == [(r, k, 1, 1, s) for r, row in enumerate([["구분", "2024년", "2025년"], *TABLE_4X3[1:]])
                          for k, s in enumerate(row)]


def test_detector_sees_only_free_chars():
    page = page_of(table_4x3)
    assert borderless.candidates(page, frozenset()) == []
    top_rows = frozenset(i for i, c in enumerate(page.chars) if c.y0 * H < 140)  # 위 두 줄만
    (box,) = borderless.candidates(page, top_rows)
    assert box[3] < 140


def bullets(c: Canvas) -> None:
    for i in range(6):
        put(c, 72, 120 + 18 * i, "•")
        put(c, 90, 120 + 18 * i, f"가나다 라마바 사아자 차카타 {i}")


def two_column_prose(c: Canvas) -> None:
    line = "가나다라 마바사아 자차카타 파하가나 다라마바 사아"
    for i in range(12):
        put(c, 50, 120 + 15 * i, line, 10.5)
        put(c, 310, 120 + 15 * i, line, 10.5)


def bar_chart(c: Canvas) -> None:
    """채운 막대 6개와 아래 연도·위 값 이름표."""
    for k in range(6):
        c.rect(100 + 70 * k, H - 450, 20, 50 + 30 * k, stroke=0, fill=1)
        put(c, 92 + 70 * k, 465, f"{2020 + k}", 8)
        put(c, 95 + 70 * k, 387 - 30 * k, f"{10 + 5 * k}", 8)


def heading_with_unit(c: Canvas) -> None:
    put(c, 72, 120, "1. 사업 개요", 14)
    put(c, 450, 120, "(단위: 백만원)", 9)
    put(c, 72, 150, "본문 한 줄이다.")


CONTENTS = ["1. 서론", "2. 추진 배경", "3. 세부 과제", "4. 예산 계획", "5. 일정", "6. 맺음말"]


def dotted_contents(c: Canvas) -> None:
    for i, s in enumerate(CONTENTS):
        put(c, 72, 120 + 18 * i, f"{s} {'.' * (60 - 2 * len(s))} {3 + 4 * i}", 10.5)


@pytest.mark.parametrize("draw", [bullets, two_column_prose, bar_chart, heading_with_unit, dotted_contents])
def test_detector_leaves_lists_prose_charts_headings_and_dotted_contents(draw):
    page = page_of(draw)
    assert borderless.candidates(page, everything(page)) == []


def unmarked_pairs(c: Canvas) -> None:
    items = ["사과 상자", "배 상자", "포도 상자", "감 상자", "귤 상자", "밤 상자"]
    for i in range(6):
        put(c, 72, 120 + 18 * i, items[i], 10.5)
        put(c, 320, 120 + 18 * i, items[(i + 3) % 6] + " 묶음", 10.5)


def short_three_columns(c: Canvas) -> None:
    words = ["가나다 라마바 사아", "자차카 타파하 가나", "다라마 바사아 자차", "카타파 하가나 다라"]
    for k in range(3):
        for i in range(12):
            put(c, 50 + 175 * k, 120 + 15 * i, words[(i + k) % 4], 10.5)


def form_fields(c: Canvas) -> None:
    for i, (a, b) in enumerate([("성명", "생년월일"), ("주소", "우편번호"), ("연락처", "전자우편"), ("소속", "직위")]):
        put(c, 72, 120 + 22 * i, a + ":", 10.5)
        put(c, 150, 120 + 22 * i, "__________", 10.5)
        put(c, 320, 120 + 22 * i, b + ":", 10.5)
        put(c, 410, 120 + 22 * i, "__________", 10.5)


def plain_contents(c: Canvas) -> None:
    for i, s in enumerate(CONTENTS):
        put(c, 72, 120 + 18 * i, s, 10.5)
        put(c, 500, 120 + 18 * i, str(3 + 4 * i), 10.5)


@pytest.mark.parametrize("draw,shape", [(unmarked_pairs, (6, 2)), (short_three_columns, (12, 6)),
                                        (form_fields, (4, 4)), (plain_contents, (6, 3))])
def test_aligned_text_that_is_not_a_table_becomes_one_out_of_scope(draw, shape):
    """지원 범위 밖(스펙 §1.3): 표지 없는 2열 목록·짧은 3단 글·정렬된 양식 칸·점선 없는 차례는 표가 된다. 글자는
    그대로다(표가 쪽 글자를 모두 칸에 담는다). 범위를 넓히면 이 기대를 바꾼다."""
    page = page_of(draw)
    (box,) = borderless.candidates(page, everything(page))
    rec = recover(page, box, everything(page))
    assert isinstance(rec, Recovery) and (rec.table.n_rows, rec.table.n_cols) == shape
    assert sum((glyphs(page.chars[i].text) for i in rec.char_ids), Counter()) == page_glyphs(page)
