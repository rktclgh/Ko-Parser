"""그림·캡션 정리(figures.py): 분류·기준값, 감싸는 상자, 이미지 객체 합치기, 장식·배경, 그림 우선, 캡션 짝, 글자 이동,
PNG, 바이트 상한. 합성 PageText(보이는 쪽 pt)로 본다."""

import hashlib
import io
import math
import time

from PIL import Image

from hanji.formats.pdf import figures
from hanji.formats.pdf.extract import Char, PageText
from hanji.formats.pdf.figures import Region
from hanji.formats.pdf.scan import OcrText
from hanji.formats.pdf.tables import TableSpec
from hanji_contracts import Cell, Table

W, H = 595.0, 842.0


def line(text: str, x: float, baseline: float, size: float = 10.0) -> list[Char]:
    """x·baseline은 pt(원점 왼쪽 위). 한글 너비 = size, ASCII = size/2(공백 포함)."""
    out = []
    for ch in text:
        width = size / 2 if ord(ch) < 0x80 else size
        out.append(Char(text=ch, x0=x / W, y0=(baseline - 0.752 * size) / H, x1=(x + width) / W,
                        y1=(baseline + 0.142 * size) / H, baseline=baseline / H, size=size))
        x += width
    return out


def page(*lines: list[Char], images=(), paths: int = 0) -> PageText:
    """images: 이미지 객체 상자(보이는 쪽 pt)."""
    return PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=tuple(c for chars in lines for c in chars),
                    image_coverage=(), images=tuple((x0 / W, y0 / H, x1 / W, y1 / H) for x0, y0, x1, y1 in images),
                    paths=paths)


def region(cls: str, box, score: float = 0.9) -> Region:
    return Region(cls, score, tuple(box))


def table_at(box, char_ids=()) -> TableSpec:
    x0, y0, x1, y1 = box
    cells = [Cell(row=r, col=k, text="가", text_source="text_layer") for r in range(2) for k in range(2)]
    return TableSpec(bbox=(x0 / W, y0 / H, x1 / W, y1 / H), table=Table(n_rows=2, n_cols=2, cells=cells),
                     char_ids=frozenset(char_ids))


def boxes(plan) -> list[tuple]:
    return [(f.category, tuple(round(v, 1) for v in f.box)) for f in plan.figures]


def test_classes_and_score_thresholds():
    regions = [region("image", (50, 100, 250, 300), 0.51), region("chart", (300, 100, 500, 300), 0.49),
               region("seal", (50, 400, 150, 500)), region("header", (50, 20, 300, 40)),
               region("table", (50, 600, 500, 700))]
    plan = figures.arrange(page(), "scan", regions)
    assert boxes(plan) == [("image", (50.0, 100.0, 250.0, 300.0))]  # 0.49 차트·도장·머리말은 그림이 아니다
    assert [r.box for r in plan.layout_tables] == [(50, 600, 500, 700)]


def test_box_around_two_figures_is_dropped_but_one_inside_another_is_a_piece():
    pair = [region("chart", (50, 100, 545, 300)), region("chart", (55, 105, 295, 295)),
            region("chart", (300, 105, 540, 295))]
    assert boxes(figures.arrange(page(), "scan", pair)) == [
        ("chart", (55.0, 105.0, 295.0, 295.0)), ("chart", (300.0, 105.0, 540.0, 295.0))]
    uneven = [region("chart", (50, 100, 545, 300)), region("chart", (55, 105, 345, 295)),
              region("chart", (350, 105, 540, 295))]  # 60/40 줄: 큰 차트가 묶음 상자의 절반을 넘는다
    assert boxes(figures.arrange(page(), "scan", uneven)) == [
        ("chart", (55.0, 105.0, 345.0, 295.0)), ("chart", (350.0, 105.0, 540.0, 295.0))]
    packed = [region("chart", (50, 100, 550, 300)), region("chart", (50, 100, 300, 300)),
              region("chart", (300, 100, 550, 300))]  # 여백 없이 붙은 같은 크기 둘
    assert boxes(figures.arrange(page(), "scan", packed)) == [
        ("chart", (50.0, 100.0, 300.0, 300.0)), ("chart", (300.0, 100.0, 550.0, 300.0))]
    nested = [region("chart", (50, 100, 545, 300)), region("image", (60, 110, 200, 200), 0.8)]
    assert boxes(figures.arrange(page(), "scan", nested)) == [("chart", (50.0, 100.0, 545.0, 300.0))]


def test_digital_candidates_inside_an_image_object_take_its_box_and_uncovered_photos_become_figures():
    p = page(images=[(100, 100, 400, 300), (100, 400, 400, 600)])  # 사진 둘(장식 기준을 넘는다)
    regions = [region("chart", (120, 120, 250, 200), 0.6), region("image", (110, 110, 390, 290), 0.95)]
    assert boxes(figures.arrange(p, "layer", regions)) == [
        ("image", (100.0, 100.0, 400.0, 300.0)), ("image", (100.0, 400.0, 400.0, 600.0))]
    assert boxes(figures.arrange(p, "layer", [region("chart", (110, 110, 390, 290))]))[0] == (
        "chart", (100.0, 100.0, 400.0, 300.0))  # 분류는 합친 후보(점수 높은 것)를 따른다


def test_a_chart_found_as_both_chart_and_image_is_not_a_box_around_its_pieces():
    triple = [region("chart", (90, 100, 510, 300)), region("image", (91, 101, 509, 299), 0.7),
              region("image", (100, 110, 200, 200), 0.8)]  # 같은 차트를 두 분류로 + 안쪽 범례 조각
    assert boxes(figures.arrange(page(), "scan", triple)) == [("chart", (90.0, 100.0, 510.0, 300.0))]
    twice = [region("chart", (50, 100, 545, 300)), region("chart", (60, 110, 200, 200)),
             region("image", (61, 111, 199, 199), 0.7)]  # 감싼 것은 두 분류로 찾은 조각 하나
    assert boxes(figures.arrange(page(), "scan", twice)) == [("chart", (50.0, 100.0, 545.0, 300.0))]


def test_same_class_boxes_of_equal_area_keep_the_higher_score():
    regions = [region("chart", (100, 100, 300, 300), 0.5), region("chart", (110, 110, 310, 310), 0.9)]
    assert boxes(figures.arrange(page(), "scan", regions)) == [("chart", (110.0, 110.0, 310.0, 310.0))]


def test_logo_and_background_image_objects_are_not_figures():
    text = [line("배경 위에 얹힌 본문 글자가 쉰 자를 넘는다.", 60, 120 + 14 * i) for i in range(3)]  # 보이는 글자 54자
    p = page(*text, images=[(520, 22, 550, 52), (0, 0, W, H)])  # 로고(작다)·쪽 배경
    assert figures.photo_boxes(p) == []
    assert boxes(figures.arrange(p, "layer", [region("image", (521, 23, 549, 51), 0.8)])) == []  # 로고 안 후보도
    assert not figures.wants_layout(p, "layer")


def test_overlapping_same_class_keeps_the_larger_and_pieces_inside_are_dropped():
    regions = [region("chart", (100, 100, 400, 300)), region("chart", (130, 120, 440, 330), 0.5),
               region("image", (150, 150, 200, 200), 0.95), region("image", (100, 500, 300, 700))]
    assert boxes(figures.arrange(page(), "scan", regions)) == [
        ("chart", (130.0, 120.0, 440.0, 330.0)), ("image", (100.0, 500.0, 300.0, 700.0))]


def test_figure_wins_over_a_grid_table_inside_it_but_not_over_a_confirmed_table():
    chart = region("chart", (90, 100, 510, 300))
    grid = table_at((100, 110, 500, 280), char_ids={0})  # 차트 눈금 격자를 표로 찾은 것
    real = table_at((60, 350, 540, 450))  # 진짜 표: 모델 table 상자와 겹친다
    regions = [chart, region("table", (58, 348, 541, 452)), region("image", (70, 360, 530, 440), 0.6)]
    plan = figures.arrange(page(), "layer", regions, [grid, real])
    assert plan.dropped == (grid,) and plan.tables == (real,)
    assert boxes(plan) == [("chart", (90.0, 100.0, 510.0, 300.0))]  # 진짜 표 안의 image 후보는 버린다


def test_an_image_object_the_model_did_not_find_keeps_the_ruled_table_inside_it():
    """리뷰 I1: 모델이 찾지 않은 이미지 객체(점수 0: 모델 없음·모델이 놓침)는 그 위에 그린 선 있는 표를 버리지 않는다
    (음영 위 결재란 등: 이미지 객체 안 선 있는 표는 대개 진짜 표). 모델이 찾아 합친 이미지 객체는 그림이 이긴다."""
    shading = page(images=[(90, 100, 510, 300)])
    grid = table_at((100, 110, 500, 280))
    plan = figures.arrange(shading, "layer", (), [grid])
    assert plan.tables == (grid,) and plan.dropped == ()
    assert boxes(plan) == [("image", (90.0, 100.0, 510.0, 300.0))]
    found = figures.arrange(shading, "layer", [region("image", (95, 105, 505, 295))], [grid])
    assert found.dropped == (grid,) and found.tables == ()


def test_caption_pairs_with_the_nearest_figure_above_or_below():
    p = page(line("그림 1. 위 차트", 200, 330), line("그림 2. 아래 사진", 200, 405))
    regions = [region("chart", (90, 100, 510, 300)), region("image", (90, 420, 510, 600)),
               region("figure_title", (195, 320, 300, 334)), region("figure_title", (195, 395, 310, 409))]
    first, second = figures.arrange(p, "layer", regions).figures
    assert (first.caption.text, first.caption.above) == ("그림 1. 위 차트", False)
    assert (second.caption.text, second.caption.above) == ("그림 2. 아래 사진", True)
    assert first.text == second.text == "" and first.caption.char_ids and not first.char_ids


def test_table_title_is_not_a_figure_caption():
    p = page(line("표 1. 예산", 60, 340))
    regions = [region("chart", (90, 100, 510, 300)), region("figure_title", (58, 330, 120, 343)),
               region("table", (58, 348, 541, 452))]
    (chart,) = figures.arrange(p, "layer", regions, [table_at((60, 350, 540, 450))]).figures
    assert chart.caption is None and chart.text == ""


def test_caption_shared_by_two_side_by_side_figures_goes_to_one_figure_only():
    p = page(line("그림 3. 두 차트", 260, 330))
    regions = [region("chart", (60, 100, 290, 300)), region("chart", (305, 100, 535, 300)),
               region("figure_title", (255, 320, 340, 334))]
    left, right = figures.arrange(p, "layer", regions).figures
    assert left.caption is not None and right.caption is None  # 간격이 같으면 앞(왼쪽) 그림


def test_digital_text_inside_a_figure_moves_into_it_with_a_small_pad():
    labels = line("1Q", 100, 290) + line("2Q", 150, 290)
    outside = line("본문", 100, 400)
    edge = line("80", 504, 150)  # 상자 오른쪽 끝(510)에 걸친 눈금: "0"의 중심 511.5는 2pt 넓힌 상자 안
    (fig,) = figures.arrange(page(labels, outside, edge), "layer", [region("chart", (90, 100, 510, 300))]).figures
    assert fig.text == "80\n1Q\n2Q" and fig.char_ids == frozenset({0, 1, 2, 3, 6, 7})  # 본문(4, 5)은 남는다


def test_scanned_page_takes_ocr_lines_for_the_figure_and_its_caption():
    lines = [OcrText("가로축", 0.9, 100, 280, 160, 292), OcrText("그림 1. 스캔 차트", 0.9, 200, 320, 300, 332),
             OcrText("본문 줄", 0.9, 100, 400, 300, 412)]
    regions = [region("chart", (90, 100, 510, 300)), region("figure_title", (195, 318, 305, 334))]
    plan = figures.arrange(page(), "scan", regions, (), lines)
    (fig,) = plan.figures
    assert (fig.text, fig.caption.text) == ("가로축", "그림 1. 스캔 차트")
    assert (fig.line_ids, fig.caption.line_ids, plan.used_lines) == (frozenset({0}), frozenset({1}), frozenset({0, 1}))


def test_layout_runs_on_scanned_pages_and_on_digital_pages_with_photos_or_many_paths():
    assert figures.wants_layout(page(), "scan")
    assert not figures.wants_layout(page(paths=9), "layer") and figures.wants_layout(page(paths=10), "layer")
    assert figures.wants_layout(page(images=[(100, 100, 300, 250)]), "layer")


def test_crop_png_is_deterministic_and_has_no_metadata():
    image = Image.new("RGB", (1190, 1684), "white")  # 595×842pt를 144 DPI(×2)로 그린 쪽
    image.paste((200, 30, 30), (300, 300, 900, 700))
    png, width, height, dpi = figures.crop_png(image, (150, 150, 450, 350), page())
    assert (width, height, dpi) == (600, 400, 144)
    assert png == figures.crop_png(image, (150, 150, 450, 350), page())[0]
    chunks, pos = [], 8
    while pos < len(png):
        length = int.from_bytes(png[pos:pos + 4], "big")
        chunks.append(png[pos + 4:pos + 8])
        pos += 12 + length
    assert chunks[0] == b"IHDR" and chunks[-1] == b"IEND" and set(chunks[1:-1]) == {b"IDAT"}
    decoded = Image.open(io.BytesIO(png))
    assert decoded.mode == "RGB" and decoded.getpixel((10, 10)) == (200, 30, 30)


def test_crop_is_clamped_to_the_image_and_long_side_capped():
    image = Image.new("RGB", (1190, 1684), (0, 0, 255))
    assert figures.crop_png(image, (-50, 800, 700, 900), page())[1:3] == (1190, 84)  # 쪽 밖은 자른다
    assert figures.crop_png(image, (100, 100, 100, 300), page()) is None  # 폭 0
    wide = Image.new("RGB", (10000, 50), (0, 0, 255))
    strip = PageText(page=1, width_pt=5000.0, height_pt=25.0, rotation=0, chars=(), image_coverage=())
    assert figures.crop_png(wide, (0, 0, 5000, 25), strip)[1:] == (4000, 20, 58)  # 긴 변 4000px, dpi도 줄인다


def test_asset_budget_dedupes_identical_bytes_and_stops_at_the_limit():
    budget = figures.AssetBudget(limit=10)
    first = budget.add(b"12345")
    assert first == "sha256:" + hashlib.sha256(b"12345").hexdigest()
    assert budget.add(b"12345") == first and budget.used == 5  # 같은 바이트는 한 번만 센다
    assert budget.add(b"123456") is None  # 5 + 6 > 10: 이 그림은 이미지 없이
    assert budget.add(b"abcde") is not None and budget.used == 10
    assert list(budget.assets) == [first, "sha256:" + hashlib.sha256(b"abcde").hexdigest()]
    assert figures.AssetBudget().limit == figures.ASSET_LIMIT == 512 * 1024 * 1024


def test_hundreds_of_image_objects_icons_drop_and_stacked_copies_merge():
    icons = [(10 + 1.4 * k, 700, 20 + 1.4 * k, 710) for k in range(400)]  # 10×10pt 아이콘 400개(장식)
    stacked = [(100, 100, 400, 300)] * 300  # 같은 사진 300겹
    start = time.perf_counter()
    plan = figures.arrange(page(images=icons + stacked), "layer", [region("image", (101, 101, 399, 299))])
    assert boxes(plan) == [("image", (100.0, 100.0, 400.0, 300.0))]
    assert figures.photo_boxes(page(images=icons)) == [] and not figures.wants_layout(page(images=icons), "layer")
    assert time.perf_counter() - start < 10.0  # 느린 CI 러너에도 넉넉히(사전 리뷰 6)


def test_non_finite_image_objects_are_not_photos():
    p = page(images=[(math.nan, 100, 400, 300), (100, 100, math.inf, 300)])
    assert figures.photo_boxes(p) == [] and figures.arrange(p, "layer").figures == ()
    assert not figures.wants_layout(p, "layer")


def test_background_check_with_many_chars_and_stacked_images_is_fast():
    text = [line("0123456789" * 10, 40, 20 + 7 * r, 6.0) for r in range(100)]  # 10,000자(사진 밖)
    stacked = [(100, 760, 500, 835)] * 10_000
    clear = page(*text, images=stacked)
    covered = page(*text, line("가" * 50, 110, 800, 6.0), images=stacked)  # 사진 위에 50자: 배경
    start = time.perf_counter()
    assert figures.wants_layout(clear, "layer")
    assert [tuple(round(v, 1) for v in b) for b in figures.photo_boxes(clear)] == [(100.0, 760.0, 500.0, 835.0)]
    assert boxes(figures.arrange(clear, "layer")) == [("image", (100.0, 760.0, 500.0, 835.0))]
    assert figures.photo_boxes(covered) == [] and boxes(figures.arrange(covered, "layer")) == []
    assert time.perf_counter() - start < 10.0  # 느린 CI 러너에도 넉넉히(사전 리뷰 6)


def test_a_model_table_box_at_the_same_place_as_a_chart_does_not_take_its_caption():
    """사전 리뷰 1: 모델이 같은 차트를 chart와 table로 둘 다 내면(선 있는 표 없음) 그 table 상자는 그림과 같은 자리를
    두 분류로 찾은 것이라 표 제목 판정에 쓰지 않는다: 차트 아래 캡션은 차트와 짝이다."""
    p = page(line("그림 1. 막대 차트", 200, 330))
    for table in ((90, 100, 510, 300), (87, 97, 513, 303)):  # 같은 상자, 3pt 큰 상자
        regions = [region("chart", (90, 100, 510, 300)), region("table", table, 0.55),
                   region("figure_title", (195, 320, 300, 334), 0.8)]
        (chart,) = figures.arrange(p, "layer", regions).figures
        assert chart.caption is not None and chart.caption.text == "그림 1. 막대 차트", table


def test_photos_under_a_model_decor_box_are_not_figures():
    """사전 리뷰 2: 모델이 장식(seal·header_image·footer_image, 점수 ≥ FIGURE_MIN_SCORE)으로 본 영역 안의 이미지
    객체(기관 머리띠·직인)는 그림이 아니다. 모델이 없거나 점수가 낮으면 지금처럼 사진 그림이다."""
    p = page(line("기관 이름 머리말", 72, 820, 9), images=[(0, 2, 595, 52), (420, 600, 560, 740)])
    regions = [region("header_image", (0, 2, 595, 52), 0.95), region("seal", (420, 600, 560, 740), 0.93)]
    assert figures.arrange(p, "layer", regions).figures == ()
    assert len(figures.arrange(p, "layer").figures) == 2
    assert len(figures.arrange(p, "layer", [region("seal", (420, 600, 560, 740), 0.49)]).figures) == 2
