import pytest

from hanji.formats.pdf.extract import Char, PageText
from hanji.formats.pdf.triage import classify, hidden_chars, page_mode, page_stats
from hanji_contracts import TextLayerStats


def char(text="가", invisible=False, unmapped=False) -> Char:
    return Char(text=text, x0=0.1, y0=0.1, x1=0.12, y1=0.11, baseline=0.11, size=11.0, invisible=invisible,
                unmapped=unmapped)


def page(chars=(), coverage=()) -> PageText:
    return PageText(page=1, width_pt=595.0, height_pt=842.0, rotation=0, chars=tuple(chars),
                    image_coverage=tuple(coverage))


def stats(chars=100, invisible=0.0, unmapped=0.0, pua=0.0, coverage=0.0) -> TextLayerStats:
    return TextLayerStats(chars=chars, invisible_ratio=invisible, unmapped_ratio=unmapped, pua_ratio=pua,
                          max_image_coverage=coverage)


def test_stats_count_visible_non_space_and_ratios_over_all_non_space():
    chars = [char("가"), char(" "), char("\n"), char("나", invisible=True), char("\ufffd", unmapped=True),
             char("\ue000"), char("\U000f0001")]
    s = page_stats(page(chars, coverage=(0.2, 0.61234)))
    assert s.chars == 4  # 공백 둘·숨은 글자 하나 제외
    assert (s.invisible_ratio, s.unmapped_ratio, s.pua_ratio) == (0.2, 0.2, 0.4)
    assert s.max_image_coverage == 0.6123  # 소수 넷째 자리


def test_stats_of_empty_page_are_zero():
    assert page_stats(page()) == stats(chars=0)
    assert page_stats(page([char(" ")])) == stats(chars=0)


def test_ratios_are_rounded_to_four_places():
    s = page_stats(page([char("가")] * 2 + [char("나", invisible=True)]))
    assert s.invisible_ratio == 0.3333


@pytest.mark.parametrize("s,expected", [
    (stats(invisible=0.5), "scanned"),
    (stats(invisible=0.4999), "digital"),
    (stats(chars=49, coverage=0.15), "scanned"),
    (stats(chars=50, coverage=0.15), "digital"),
    (stats(chars=0, coverage=0.1499), "digital"),
    (stats(chars=3, coverage=0.6), "scanned"),
    (stats(chars=0, coverage=1.0), "scanned"),  # 그림만 있는 쪽
    (stats(unmapped=0.1), "unreliable"),
    (stats(unmapped=0.0999), "digital"),
    (stats(pua=0.1), "unreliable"),
    (stats(pua=0.0999), "digital"),
    (stats(chars=0), "digital"),  # 빈 쪽
])
def test_classify_boundaries(s, expected):
    assert classify(s) == expected


def test_scanned_rules_come_before_unreliable():
    assert classify(stats(invisible=0.9, unmapped=0.9, pua=0.9)) == "scanned"
    assert classify(stats(chars=10, coverage=0.5, unmapped=0.5)) == "scanned"
    assert classify(stats(unmapped=0.05, pua=0.2)) == "unreliable"


def test_page_mode_reads_unreliable_pages_from_the_text_layer():
    """처리 모드는 쪽 상태와 따로다: scanned만 scan, digital과 unreliable(OCR 없이 둔 쪽)은 layer."""
    assert [page_mode(s) for s in ("digital", "scanned", "unreliable")] == ["layer", "scan", "layer"]


def test_hidden_chars_count_invisible_non_space_on_any_page():
    chars = [char("가"), char("나", invisible=True), char(" ", invisible=True), char("다", invisible=True)]
    assert hidden_chars(page(chars)) == 2 and hidden_chars(page()) == 0
