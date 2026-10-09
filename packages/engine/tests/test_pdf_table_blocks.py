import io
from collections import Counter
from dataclasses import replace

import pytest
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from hanji.formats.pdf import PdfParser
from hanji.formats.pdf import parser as pdf_parser
from hanji.formats.pdf.extract import extract_pages
from hanji.formats.pdf.group import build_specs
from hanji.formats.pdf.tables import find_tables
from hanji_contracts import Table, build_blocks

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
H = 842.0
GRAY_JPEG = bytes.fromhex(  # 8×8 회색 JPEG
    "ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e1211101318281a181616183123251d283a333d"
    "3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffc0000b080008000801011100ffc40014000100"
    "000000000000000000000000000005ffc40014100100000000000000000000000000000000ffda0008010100003f0041ffd9")


def put(c: Canvas, x: float, baseline: float, s: str, size: float = 11) -> None:
    """보이는 쪽 좌표(원점 왼쪽 위)."""
    t = c.beginText(x, H - baseline)
    t.setFont(FONT, size)
    t.textOut(s)
    c.drawText(t)


def table(c: Canvas, top: float, rows: list[list[str]], size: float = 11, xs=(72, 222, 372, 522)) -> None:
    """칸 높이 24pt, 선 모두 그은 표."""
    ys = [top + 24 * i for i in range(len(rows) + 1)]
    for y in ys:
        c.line(xs[0], H - y, xs[-1], H - y)
    for x in xs:
        c.line(x, H - ys[0], x, H - ys[-1])
    for r, row in enumerate(rows):
        for k, s in enumerate(row):
            put(c, xs[k] + 8, ys[r] + 16, s, size)


def parse(draw) -> list[dict]:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, H), invariant=1, pageCompression=0)
    draw(c)
    c.showPage()
    c.save()
    return list(PdfParser().parse(buf.getvalue(), "t.pdf").blocks)


ROWS = [["구분", "2025년", "2026년"], ["수출", "12,345", "23,456"], ["수입", "34,567", "45,678"]]


def report(c: Canvas) -> None:
    put(c, 72, 80, "1. 수출입 현황", 16)
    put(c, 72, 120, "올해 수출입 실적은 아래 표와 같다.")
    table(c, 140, ROWS)
    put(c, 72, 250, "수출은 전년보다 늘었다.")


def test_table_block_in_reading_order_and_its_text_is_not_repeated():
    specs = parse(report)
    assert [s["kind"] for s in specs] == ["heading", "paragraph", "table", "paragraph"]
    block = specs[2]
    assert isinstance(block["table"], Table)
    assert [[c.text for c in block["table"].cells if c.row == r] for r in range(3)] == ROWS
    assert block["text"] == block["table"].plain_text() == "구분\t2025년\t2026년\n수출\t12,345\t23,456\n수입\t34,567\t45,678"
    assert (block["text_source"], block["state"], block["confidence"]) == ("text_layer", "det", 0.6)
    assert block["section_path"] == ("1. 수출입 현황",)
    assert block["locator"] == {"kind": "page", "page": 1, "bbox": {"x0": 0.121, "y0": 0.166, "x1": 0.877, "y1": 0.252}}
    others = "".join(s["text"] for s in specs if s["kind"] != "table")
    assert "12,345" not in others and "수입" not in others
    build_blocks("d", specs)  # 계약 검사(표 블록 text = plain_text, text_source 일치)


def test_body_size_ignores_table_chars():
    """표 글자(9pt)가 본문(11pt)보다 훨씬 많아도 본문 크기는 표 밖 글자로 정한다: 11pt 줄은 제목이 아니다."""
    def draw(c):
        put(c, 72, 80, "표 앞 문단이다.")
        table(c, 100, [[f"{r}{k}가나다라마바" for k in range(3)] for r in range(8)], size=9)
        put(c, 72, 320, "표 뒤 문단이다.")

    assert [s["kind"] for s in parse(draw)] == ["paragraph", "table", "paragraph"]


def test_paragraph_lines_do_not_join_across_a_table():
    """표 위·아래 줄이 같은 왼쪽·크기여도 표를 건너 한 문단으로 잇지 않는다."""
    def draw(c):
        put(c, 72, 80, "위 문단이다.")
        table(c, 92, [["가", "나", "다"], ["라", "마", "바"]])
        put(c, 72, 156, "아래 문단이다.")

    assert [(s["kind"], s["text"]) for s in parse(draw) if s["kind"] != "table"] == [
        ("paragraph", "위 문단이다."), ("paragraph", "아래 문단이다.")]


def test_scanned_page_gets_tables_from_its_visible_text():
    """그림이 쪽의 15% 이상이고 보이는 글자가 50자 미만인 scanned 쪽도 보이는 글자로 표를 찾는다."""
    def draw(c):
        c.drawImage(ImageReader(io.BytesIO(GRAY_JPEG)), 72, 300, width=450, height=300)
        table(c, 100, [["가", "나"], ["다", "라"]], xs=(72, 222, 372))

    buf = io.BytesIO()
    canvas = Canvas(buf, pagesize=(595.0, H), invariant=1, pageCompression=0)
    draw(canvas)
    canvas.showPage()
    canvas.save()
    parsed = PdfParser(layout=False).parse(buf.getvalue(), "t.pdf")  # 레이아웃은 회색 사각형을 그림으로 본다
    assert parsed.pages[0].text_layer == "scanned"
    assert [s["kind"] for s in parsed.blocks] == ["table"]


def test_unreliable_page_keeps_its_tables_with_low_confidence(monkeypatch):
    """OCR 추가 설치가 없어 깨진 글자층으로 읽는 unreliable 쪽도 digital처럼 표를 찾는다. 그 쪽 블록은 신뢰도 0.2
    이하이고 제목을 만들지 않는다("1. 수출입 현황"은 앞머리가 있어 목록 항목)."""
    monkeypatch.setattr(pdf_parser.ocr_runtime, "available", lambda: False)
    monkeypatch.setattr(pdf_parser, "classify", lambda stats: "unreliable")
    specs = parse(report)
    assert [(s["kind"], s["confidence"]) for s in specs] == [
        ("list_item", 0.2), ("paragraph", 0.2), ("table", 0.2), ("paragraph", 0.2)]
    assert [[c.text for c in specs[2]["table"].cells if c.row == r] for r in range(3)] == ROWS
    build_blocks("d", specs)


def test_unreliable_page_without_ocr_text_falls_back_and_keeps_its_tables(monkeypatch):
    """OCR을 쓸 수 있어도 받아들인 OCR 글자가 없어 깨진 글자층으로 돌아온 unreliable 쪽은 표 찾기도 그 최종 모드
    (layer)로 한다: 선 있는 표가 OCR을 끈 TC-A 유지 경로와 똑같이 신뢰도 0.2 table 블록으로 남고, 처리 이력은
    unreliable_text_layer_kept + 실패한 ocr_text 검사다."""
    monkeypatch.setattr(pdf_parser, "classify", lambda stats: "unreliable")
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, H), invariant=1, pageCompression=0)
    report(c)
    c.showPage()
    c.save()
    kept = PdfParser(ocr=False, layout=False).parse(buf.getvalue(), "t.pdf")
    monkeypatch.setattr(pdf_parser.ocr_runtime, "available", lambda: True)
    monkeypatch.setattr(pdf_parser.ocr_runtime, "get_reader", lambda: None)
    monkeypatch.setattr(pdf_parser.ocr_runtime, "read_lines", lambda image: [])
    parsed = PdfParser(layout=False).parse(buf.getvalue(), "t.pdf")
    assert [(s["kind"], s["confidence"]) for s in parsed.blocks] == [
        ("list_item", 0.2), ("paragraph", 0.2), ("table", 0.2), ("paragraph", 0.2)]
    assert [[c.text for c in parsed.blocks[2]["table"].cells if c.row == r] for r in range(3)] == ROWS
    assert parsed.blocks == kept.blocks
    (note,) = parsed.regions
    assert (note.fallback_reason, [(k.name, k.passed) for k in note.gate.checks]) == (
        "unreliable_text_layer_kept", [("ocr_text", False)])


def rotated_page(own_direction: bool) -> list[dict]:
    """/Rotate 90 쪽(MediaBox 842×595): 보통으로 그린 글(표 포함)은 보이는 쪽에서 세로로 흐르고(쪽에서 가장 많은
    방향), 90° 돌려 그린 줄은 보이는 쪽에서 바로 선다. own_direction이면 표와 같은 방향 문단이 표 위·아래에도 있다."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, H), invariant=1, pageCompression=0)
    c.setPageRotation(90)
    top = 595.0  # 눕힌 MediaBox 높이: 표 방향 읽기 좌표의 y = 595 − PDF y
    xs, ys = (100, 250, 400), [100 + 24 * i for i in range(4)]
    for y in ys:
        c.line(xs[0], top - y, xs[-1], top - y)
    for x in xs:
        c.line(x, top - ys[0], x, top - ys[-1])
    for r, row in enumerate([["구분", "실적"], ["수출", "12,345"], ["수입", "34,567"]]):
        for k, s in enumerate(row):
            t = c.beginText(xs[k] + 8, top - ys[r] - 16)
            t.setFont(FONT, 11)
            t.textOut(s)
            c.drawText(t)
    if own_direction:
        for y, s in [(70, "표 위 문단."), (130 + 72, "표 아래 문단.")]:
            t = c.beginText(100, top - y)
            t.setFont(FONT, 11)
            t.textOut(s)
            c.drawText(t)
    for x, s in [(50 if not own_direction else 600, "바로 선 첫 줄"), (650, "바로 선 둘째 줄")]:
        c.saveState()
        c.translate(x, 40)
        c.rotate(90)  # 보이는 쪽에서 바로 선 줄: 보이는 y = PDF x
        t = c.beginText(0, 0)
        t.setFont(FONT, 11)
        t.textOut(s)
        c.drawText(t)
        c.restoreState()
    c.showPage()
    c.save()
    return list(PdfParser().parse(buf.getvalue(), "t.pdf").blocks)


@pytest.mark.parametrize("own_direction,expected", [
    (False, ["바로 선 첫 줄", "table", "바로 선 둘째 줄"]),
    (True, ["바로 선 첫 줄", "바로 선 둘째 줄", "표 위 문단.", "table", "표 아래 문단."])])
def test_table_is_placed_by_its_top_among_text_of_its_direction(own_direction, expected):
    """표 글자가 쪽의 주된 방향이고 남은 글은 다른 방향일 때도 표는 쪽 끝으로 밀리지 않는다: 같은 방향 글이 있으면
    그 글 사이에 표 방향 읽기 좌표로, 없으면 남은 글 사이에 그 글 방향 읽기 좌표로 윗변 위치에 끼운다."""
    specs = rotated_page(own_direction)
    assert [s["text"] if s["kind"] != "table" else "table" for s in specs] == expected


def test_every_char_lands_in_exactly_one_block():
    """표 칸 글자와 나머지 블록 글자를 합치면 쪽의 보이는 글자와 같다(빠지거나 두 번 나오는 글자 없음)."""
    specs = parse(report)
    seen = Counter(ch for s in specs for ch in s["text"] if not ch.isspace())
    expected = Counter(ch for s in ["1. 수출입 현황", "올해 수출입 실적은 아래 표와 같다.", "수출은 전년보다 늘었다.",
                                    *(x for row in ROWS for x in row)] for ch in s if not ch.isspace())
    assert seen == expected


def side_label_page(table_top: float) -> list[dict]:
    """바로 선 두 문단과 표, 그리고 90° 돌린 짧은 옆 글(바로 선 글보다 글자가 적어 조각 순서가 뒤다)."""
    def draw(c):
        put(c, 72, 80, "첫 문단이다.")
        put(c, 72, 200, "둘째 문단이다.")
        table(c, table_top, [["가", "나", "다"], ["라", "마", "바"]])
        c.saveState()
        c.translate(40, 300)
        c.rotate(90)
        t = c.beginText(0, 0)
        t.setFont(FONT, 11)
        t.textOut("옆 글")
        c.drawText(t)
        c.restoreState()

    return parse(draw)


@pytest.mark.parametrize("table_top,expected", [
    (400, ["첫 문단이다.", "둘째 문단이다.", "table", "옆 글"]),
    (120, ["첫 문단이다.", "table", "둘째 문단이다.", "옆 글"])])
def test_table_below_its_direction_text_comes_before_text_of_other_directions(table_top, expected):
    """표 윗변이 같은 방향 조각보다 모두 아래여도 다른 방향 조각 뒤(쪽 끝)로 밀리지 않는다."""
    assert [s["text"] if s["kind"] != "table" else "table" for s in side_label_page(table_top)] == expected


def test_table_bbox_keeps_nonzero_width_and_height():
    """반올림한 표 bbox의 폭·높이가 0이면 _box처럼 0.001 넓힌다(쪽 끝이면 안쪽으로)."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, H), invariant=1, pageCompression=0)
    report(c)
    c.showPage()
    c.save()
    page = extract_pages(buf.getvalue(), "t.pdf")[0]
    spec = find_tables(page)[0]
    boxes = []
    for bbox in [(0.2, 0.3, 0.2, 0.3), (1.0, 1.0, 1.0, 1.0)]:
        specs = build_specs([page], ["digital"], [[replace(spec, bbox=bbox)]])
        boxes += [s["locator"]["bbox"] for s in specs if s["kind"] == "table"]
    assert boxes == [{"x0": 0.2, "y0": 0.3, "x1": 0.201, "y1": 0.301},
                     {"x0": 0.999, "y0": 0.999, "x1": 1.0, "y1": 1.0}]


def test_tables_in_margin_zones_are_not_page_headers_or_footers():
    """여러 쪽 같은 위치(위·아래 8% 안)에 같은 글자의 표가 있어도 표 글자는 머리말·꼬리말이 되지 않고 표로 남는다."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, H), invariant=1, pageCompression=0)
    for n in range(3):
        table(c, 10, [["기관", "부서"], ["담당", "연락"]], size=9, xs=(72, 222, 372))
        put(c, 72, 400, f"{n + 1}쪽 본문이다.")
        table(c, 784, [["작성", "검토"], ["승인", "배포"]], size=9, xs=(72, 222, 372))
        c.showPage()
    c.save()
    specs = list(PdfParser().parse(buf.getvalue(), "t.pdf").blocks)
    assert [(s["locator"]["page"], s["kind"]) for s in specs] == [
        (p, k) for p in (1, 2, 3) for k in ("table", "paragraph", "table")]


def rotated_parse(draw, rotation: int) -> list[dict]:
    """MediaBox에 보통으로 그리고 /Rotate만 바꾼 쪽: 읽기 좌표(그린 좌표)의 순서는 회전과 상관없이 같아야 한다.
    reportlab은 90°·270°면 MediaBox를 842×595로 눕히므로 put·table의 H 기준 좌표를 그 높이로 옮긴다."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, H), invariant=1, pageCompression=0)
    if rotation:
        c.setPageRotation(rotation)
    c.translate(0, (595.0 if rotation in (90, 270) else H) - H)
    draw(c)
    c.showPage()
    c.save()
    return list(PdfParser().parse(buf.getvalue(), "t.pdf").blocks)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_tables_on_a_table_only_page_keep_reading_order_on_rotated_pages(rotation):
    """표 밖 글자가 없는 쪽: 표는 자기 방향(TableSpec.axes) 읽기 좌표로 위→아래 순서다."""
    def draw(c):
        table(c, 80, [["가", "나", "다"], ["라", "마", "바"]])
        table(c, 260, [["사", "아", "자"], ["차", "카", "타"]])

    specs = rotated_parse(draw, rotation)
    assert [s["text"].split("\t")[0] for s in specs] == ["가", "사"]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_tables_at_the_same_top_are_ordered_left_to_right_in_reading_coordinates(rotation):
    """윗변이 같은 두 표는 그 방향 읽기 좌표의 왼쪽 표가 먼저다."""
    def draw(c):
        put(c, 72, 60, "표 앞 문단이다.")
        table(c, 100, [["가", "나"], ["다", "라"]], xs=(72, 172, 272))
        table(c, 100, [["사", "아"], ["자", "차"]], xs=(322, 422, 522))

    specs = rotated_parse(draw, rotation)
    assert [s["text"].split("\t")[0] for s in specs] == ["표 앞 문단이다.", "가", "사"]
