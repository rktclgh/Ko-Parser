"""쪽 판정: digital·scanned·unreliable과 그 근거(TextLayerStats). 기준값은 공공누리 8건으로 조정한 상수다."""

from hanji_contracts import TextLayerState, TextLayerStats

from .extract import PageText

INVISIBLE_MIN = 0.5  # 숨은 글자층(OCR 결과 등) 비율이 이 이상이면 scanned
IMAGE_PAGE_MIN_COVERAGE = 0.15  # 가장 큰 그림이 쪽의 이 비율 이상을 덮고
IMAGE_PAGE_MAX_CHARS = 50  # 보이는 글자가 이보다 적으면 scanned(본문이 그림 한 장인 쪽)
UNMAPPED_MIN = 0.1  # 유니코드로 못 읽는 글자 비율이 이 이상이면 unreliable
PUA_MIN = 0.1  # 사용자 정의 영역 글자 비율이 이 이상이면 unreliable


def _is_pua(text: str) -> bool:
    code = ord(text[0])
    return 0xE000 <= code <= 0xF8FF or code >= 0xF0000


def page_stats(page: PageText) -> TextLayerStats:
    """비율의 분모는 공백이 아닌 글자 전체(숨은 글자 포함), 글자가 없으면 0. 비율은 소수 넷째 자리로 반올림."""
    chars = [c for c in page.chars if not c.text.isspace()]
    total = len(chars)

    def ratio(n: int) -> float:
        return round(n / total, 4) if total else 0.0

    return TextLayerStats(
        chars=sum(1 for c in chars if not c.invisible),
        invisible_ratio=ratio(sum(1 for c in chars if c.invisible)),
        unmapped_ratio=ratio(sum(1 for c in chars if c.unmapped)),
        pua_ratio=ratio(sum(1 for c in chars if _is_pua(c.text))),
        max_image_coverage=round(max(page.image_coverage, default=0.0), 4),
    )


def classify(stats: TextLayerStats) -> TextLayerState:
    """순서대로: scanned → unreliable → digital(글자 0인 빈 쪽 포함). 저장되는 반올림 값으로 판정해 근거와 판정이 어긋나지 않는다."""
    if stats.invisible_ratio >= INVISIBLE_MIN or (
            stats.max_image_coverage >= IMAGE_PAGE_MIN_COVERAGE and stats.chars < IMAGE_PAGE_MAX_CHARS):
        return "scanned"
    if stats.unmapped_ratio >= UNMAPPED_MIN or stats.pua_ratio >= PUA_MIN:
        return "unreliable"
    return "digital"
