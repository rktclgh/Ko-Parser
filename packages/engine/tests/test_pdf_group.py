import unicodedata

import pytest

from ko_parser.formats.pdf.extract import Char, PageText
from ko_parser.formats.pdf.group import LIST_MARKER, body_size, build_specs, fragments
from ko_parser_contracts import build_blocks

W, H = 595.0, 842.0


def line(text: str, x: float, baseline: float, size: float = 11.0, bold: bool = False, gap: float = 0.0,
         invisible: bool = False) -> list[Char]:
    """x·baseline은 pt(원점 왼쪽 위). 한글·기호 너비 = size, ASCII = size/2, 글자 사이 gap(pt)."""
    out = []
    for ch in text:
        width = size / 2 if ord(ch) < 0x80 else size
        out.append(Char(text=ch, x0=x / W, y0=(baseline - 0.752 * size) / H, x1=(x + width) / W,
                        y1=(baseline + 0.142 * size) / H, baseline=baseline / H, size=size, bold=bold,
                        invisible=invisible))
        x += width + gap
    return out


def page(*lines: list[Char], number: int = 1) -> PageText:
    return PageText(page=number, width_pt=W, height_pt=H, rotation=0,
                    chars=tuple(c for chars in lines for c in chars), image_coverage=())


def specs(*pages: PageText, states=None):
    return build_specs(pages, states or ["digital"] * len(pages))


def kinds_texts(result) -> list[tuple[str, str]]:
    return [(s["kind"], s["text"]) for s in result]


def test_fragments_split_on_wide_gap_and_order_left_to_right():
    p = page(line("금액", 300, 100), line("구분", 72, 100), line("다음 줄", 72, 116))
    assert [f.text for f in fragments(p)] == ["구분", "금액", "다음 줄"]


def test_same_line_tolerates_half_size_baseline_shift():
    p = page(line("가나", 72, 100), line("다", 94, 105.4), line("라", 72, 106))
    assert [f.text for f in fragments(p)] == ["가나다", "라"]


def test_space_inserted_where_gap_exceeds_tracking():
    """자간을 -3pt로 좁히고 공백 글자 없이 낱말 사이만 1.4pt 벌린 줄(한글 프로그램 PDF 실측 모양)."""
    chars = line("상장사", 72, 100, size=12, gap=-3) + line("임직원", 72 + 2 * 9 + 12 + 1.4, 100, size=12, gap=-3)
    assert [f.text for f in fragments(page(chars))] == ["상장사 임직원"]
    assert [f.text for f in fragments(page(line("가 나", 72, 100)))] == ["가 나"]  # 공백 글자는 그대로


def test_space_rule_threshold_and_symbol_gaps():
    """공백 = 간격 − 보통 자간 > 크기 × 0.2. 보통 자간은 글자·숫자끼리 간격의 중앙값(음수일 때만)."""
    near = line("가나", 72, 100, size=10) + line("다라", 72 + 20 + 1.9, 100, size=10)  # 0.19 × 10pt
    far = line("가나", 72, 100, size=10) + line("다라", 72 + 20 + 2.1, 100, size=10)  # 0.21 × 10pt
    assert [f.text for f in fragments(page(near))] == ["가나다라"]
    assert [f.text for f in fragments(page(far))] == ["가나 다라"]
    # 목차: 글자는 붙어 있고 점선 기호끼리는 겹친다(음수 간격). 기호 간격이 자간 추정을 끌어내리면 안 된다
    toc = line("국채시장", 72, 100, size=10) + line("······", 112, 100, size=10, gap=-6)
    assert [f.text for f in fragments(page(toc))] == ["국채시장······"]
    # 넓은 양수 자간(글자 사이를 띄운 제목)은 보통 자간으로 보지 않는다: 띄운 그대로 공백
    assert [f.text for f in fragments(page(line("보도자료", 72, 100, size=10, gap=4)))] == ["보 도 자 료"]


def test_invisible_and_whitespace_only_chars_make_no_fragment():
    p = page(line("숨은 글자", 72, 100, invisible=True), line("   ", 72, 120))
    assert fragments(p) == [] and specs(p) == []


def test_body_size_is_most_common_char_size_ties_to_smaller():
    p = page(line("가나다라", 72, 100, size=11), line("마바사", 72, 130, size=16), line("아자차카", 72, 160, size=13))
    assert body_size([fragments(p)]) == 11.0
    assert body_size([fragments(page(line("가.", 72, 100, size=11.2)))]) == 11.0  # 0.5pt 단위
    assert body_size([[]]) is None


def test_headings_levels_and_section_path():
    p = page(line("사업 계획", 72, 60, 20), line("1. 추진 배경", 72, 100, 16), line("본문 첫째", 72, 130),
             line("가. 세부 목표", 72, 170, 13), line("본문 둘째", 72, 200), line("2. 예산", 72, 240, 16),
             line("본문 셋째", 72, 270))
    result = specs(p)
    assert [(s["kind"], s.get("level"), s["section_path"]) for s in result] == [
        ("heading", 1, ()), ("heading", 2, ("사업 계획",)), ("paragraph", None, ("사업 계획", "1. 추진 배경")),
        ("heading", 3, ("사업 계획", "1. 추진 배경")),
        ("paragraph", None, ("사업 계획", "1. 추진 배경", "가. 세부 목표")),
        ("heading", 2, ("사업 계획",)), ("paragraph", None, ("사업 계획", "2. 예산"))]
    assert {s["confidence"] for s in result if s["kind"] == "heading"} == {0.6}


def test_heading_ratio_and_bold_ratio():
    plain = page(line("본문 글자 크기", 72, 100), line("작은 제목", 72, 130, 12.5), line("본문 글자 크기", 72, 160))
    assert kinds_texts(specs(plain))[1] == ("paragraph", "작은 제목")  # 12.5 < 11 × 1.15
    bold = page(line("본문 글자 크기", 72, 100), line("굵은 제목", 72, 130, 12, bold=True), line("본문 글자 크기", 72, 160))
    assert kinds_texts(specs(bold))[1] == ("heading", "굵은 제목")  # 12 ≥ 11 × 1.05


def test_two_heading_lines_join_three_become_paragraph():
    two = page(line("본문 글자가 가장 많다", 72, 60), line("길게 이어지는", 72, 100, 20), line("제목", 72, 124, 20),
               line("본문 글자가 가장 많다", 72, 160))
    assert kinds_texts(specs(two))[1] == ("heading", "길게 이어지는 제목")
    three = page(line("본문 글자가 가장 많은 줄이다", 72, 60), line("큰 글자", 72, 100, 16), line("세 줄이면", 72, 120, 16),
                 line("문단", 72, 140, 16))
    assert kinds_texts(specs(three))[1] == ("paragraph", "큰 글자\n세 줄이면\n문단")


def test_paragraph_lines_join_with_newline_and_split_on_each_rule():
    joined = page(line("첫째 줄", 72, 100), line("둘째 줄", 72, 116))
    assert kinds_texts(specs(joined)) == [("paragraph", "첫째 줄\n둘째 줄")]
    far = page(line("첫째 줄", 72, 100), line("멀리 떨어진 줄", 72, 130))  # 빈 간격 > 줄 높이 × 0.8
    sized = page(line("첫째 줄", 72, 100), line("조금 큰 줄", 72, 116, 12))  # 크기 차 > 0.5pt
    shifted = page(line("첫째 줄", 72, 100), line("들여 쓴 줄", 84, 116))  # 왼쪽 시작 차 > 본문 크기
    for p in (far, sized, shifted):
        assert len(specs(p)) == 2


@pytest.mark.parametrize("text", ["1. 항목", "12) 항목", "(3) 항목", "가. 항목", "하) 항목", "① 항목", "⑳ 항목",
                                  "□ 항목", "■ 항목", "○ 항목", "● 항목", "◦ 항목", "◆ 항목", "◇ 항목", "▶ 항목",
                                  "▷ 항목", "- 항목", "– 항목", "• 항목", "※ 항목", "  □ 항목",
                                  "ㅇ 항목", "ㆍ 항목", "· 항목", "∙ 항목", "‣ 항목", "▸ 항목", "▪ 항목", "⇨ 항목", "→ 항목"])
def test_list_marker_matches(text):
    assert LIST_MARKER.match(text)


@pytest.mark.parametrize("text", ["1.5배 증가", "□항목", "ㅇ항목", "가나다", "(가) 항목", "2026년 계획", "* 주석", "→결과"])
def test_list_marker_rejects(text):
    assert not LIST_MARKER.match(text)


def test_list_items_split_and_keep_marker_with_hanging_indent_continuation():
    p = page(line("□ 첫째 항목은", 72, 100), line("이어지는 줄이다", 88.5, 116), line("○ 둘째 항목", 72, 132),
             line("- 셋째 항목", 72, 148))
    assert kinds_texts(specs(p)) == [("list_item", "□ 첫째 항목은\n이어지는 줄이다"), ("list_item", "○ 둘째 항목"),
                                     ("list_item", "- 셋째 항목")]
    assert {s["confidence"] for s in specs(p)} == {0.7}


def test_hanging_indent_after_any_leading_marker():
    """첫 줄 앞머리(목록 표지 또는 글자·숫자가 아닌 한 글자) 다음 글자에 맞춘 줄은 이어진다."""
    for marker in ("ㅇ", "✅", "*"):
        p = page(line(f"{marker} 첫째 줄은", 72, 100), line("이어지는 줄", 88.5, 116))
        assert kinds_texts(specs(p))[0][1] == f"{marker} 첫째 줄은\n이어지는 줄"
    p = page(line("가나 첫째 줄", 72, 100), line("둘째 줄", 88.5, 116))  # 두 글자 낱말은 앞머리가 아니다
    assert len(specs(p)) == 2


def footer_pages(n: int, header_y: float = 30.0) -> list[PageText]:
    return [page(line("2026년 사업 계획 보고", 72, header_y, 9), line(f"{i}쪽 본문은 머리말보다 글자가 많다", 72, 300),
                 line(f"- {i} -", 282, 812, 9), number=i) for i in range(1, n + 1)]


def test_header_footer_repeat_on_half_of_pages():
    result = specs(*footer_pages(3))
    assert [s["kind"] for s in result] == ["page_header", "paragraph", "page_footer"] * 3
    assert [s["text"] for s in result if s["kind"] == "page_footer"] == ["- 1 -", "- 2 -", "- 3 -"]
    assert {(s["confidence"], s["section_path"]) for s in result if s["kind"].startswith("page_")} == {(0.8, ())}


def test_header_footer_needs_three_pages():
    assert "page_footer" not in [s["kind"] for s in specs(*footer_pages(2))]


def test_header_footer_needs_same_position_on_half_the_pages():
    pages = footer_pages(4)
    moved = [page(line("2026년 사업 계획 보고", 72, 12 + 18 * i, 9), line("본문은 머리말보다 글자가 많다", 72, 300),
                  number=i + 1)
             for i in range(4)]  # 위 여백 안이지만 쪽마다 2% 넘게 움직인다
    assert "page_header" not in [s["kind"] for s in specs(*moved)]
    states = ["digital", "scanned", "scanned", "digital"]  # 반복 2/4쪽 = 절반
    assert [s["kind"] for s in specs(*pages, states=states)].count("page_footer") == 2


def test_scanned_and_unreliable_pages_make_no_blocks():
    p1, p2 = page(line("보이는 쪽", 72, 100)), page(line("스캔 쪽", 72, 100), number=2)
    result = specs(p1, p2, states=["digital", "scanned"])
    assert [s["locator"]["page"] for s in result] == [1]
    assert specs(p1, p2, states=["unreliable", "scanned"]) == []


def test_text_is_nfc_and_block_fields():
    nfd = unicodedata.normalize("NFD", "한글 문단")
    result = specs(page(line(nfd, 72, 100)))
    assert result[0]["text"] == "한글 문단" and unicodedata.is_normalized("NFC", result[0]["text"])
    assert {k: result[0][k] for k in ("state", "text_source", "confidence")} == {
        "state": "det", "text_source": "text_layer", "confidence": 0.7}


def test_bbox_rounded_to_three_places_and_never_degenerate():
    tiny = Char(text=".", x0=0.50001, y0=0.40001, x1=0.50004, y1=0.40004, baseline=0.40004, size=11)
    result = specs(PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=(tiny,), image_coverage=()))
    assert result[0]["locator"]["bbox"] == {"x0": 0.5, "y0": 0.4, "x1": 0.501, "y1": 0.401}
    edge = Char(text="끝", x0=0.99996, y0=0.99996, x1=1.0, y1=1.0, baseline=1.0, size=11)
    box = specs(PageText(page=1, width_pt=W, height_pt=H, rotation=0, chars=(edge,), image_coverage=()))[0]
    assert box["locator"]["bbox"] == {"x0": 0.999, "y0": 0.999, "x1": 1.0, "y1": 1.0}
    p = page(line("가나다", 72, 100))
    assert specs(p)[0]["locator"]["bbox"] == {"x0": 0.121, "y0": 0.109, "x1": 0.176, "y1": 0.121}


def test_specs_are_valid_contract_blocks_in_reading_order():
    p1 = page(line("둘째", 72, 200), line("첫째", 72, 100), line("오른쪽", 300, 100))
    p2 = page(line("다음 쪽", 72, 100), number=2)
    blocks = build_blocks("doc", specs(p1, p2))
    assert [b.text for b in blocks] == ["첫째", "오른쪽", "둘째", "다음 쪽"]


@pytest.mark.parametrize("opener", ["(", "〈", "《", "「", "『", "[", "［", "（", '"', "“", "'", "‘"])
def test_opening_bracket_is_not_a_leading_marker(opener):
    """여는 괄호·따옴표는 앞머리가 아니다: 둘째 글자에 맞춘 다음 줄도 내어쓰기로 잇지 않는다."""
    p = page(line(opener, 72, 100) + line("단위: 백만원 )", 88.5, 100), line("이어지는 줄", 88.5, 116))
    assert kinds_texts(specs(p)) == [("paragraph", f"{opener} 단위: 백만원 )"), ("paragraph", "이어지는 줄")]
