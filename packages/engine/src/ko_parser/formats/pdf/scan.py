"""스캔 쪽 OCR 연결: 쪽 그림(200 DPI) → OCR 줄 → 보이는 쪽 좌표(pt) → 텍스트 레이어와 겹친 줄·낮은 점수 줄 버리기
→ 읽기 순서(XY 분할) → 문단. 렌더는 PDFIUM_LOCK 안(open_pdf 경유), OCR은 잠금 밖에서 한다."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ko_parser_contracts import TextLayerState
from PIL import Image

from ...errors import ParseError
from . import ocr
from .extract import PDFIUM_LOCK, PageText, open_pdf, pdfium

OCR_DPI = 200  # 300 DPI는 이득이 없다(스펙 §2)
MAX_RENDER_SIDE = 4000  # 큰 쪽은 긴 변이 이 화소를 넘지 않게 배율을 낮춘다(실행부가 어차피 2000px로 줄인다)
OCR_MIN_SCORE = 0.5  # 점수가 이보다 낮은 줄은 버린다
OCR_SHORT_SCORE = 0.7  # 공백 뺀 글자 수가 SHORT_TEXT 이하인 줄은 점수가 이 이상이어야 남는다(그림 속 작은 이름표).
# 채점(T5): 0.6 → 0.7에서 DIG F1 0.9750 → 0.9752(aCER 0.0367 그대로), SYN F1 0.9722 → 0.9723·aCER 0.0585 → 0.0580
SHORT_TEXT = 2
OVERLAP = 0.5  # 보이는 글자가 OCR 줄 상자 길이의 이 비율 이상을 덮으면 그 줄은 버린다(텍스트 레이어 우선)
TALL = 1.5  # 줄 상자 세로가 가로의 1.5배 이상이면 세로줄(실행부 crop_quad와 같은 기준)
COLUMN_GAP = 2.0  # 단 나누기: 줄을 가로지르지 않는 빈 세로 띠 ≥ 본문 줄 높이 × 2
MIN_COLUMN = 0.25  # 단 너비 하한(나눌 영역 너비 대비). 더 좁은 단이 생기면 표로 보고 줄 순서로 읽는다.
# 채점(T5): 이 하한이 없으면 표 열을 단으로 읽어 aCER DIG 0.0375 → 0.0460, SYN 0.0602 → 0.0694(0.2·0.3은 같음)
BAND_GAP = 1.0  # 단을 못 나누면 빈 가로 띠 ≥ 본문 줄 높이 × 1에서 위아래로 나눈 뒤 다시 본다(제목 아래 2단)
SAME_ROW = 0.7  # 세로 중심 차 ≤ 줄 높이 × 0.7이면 같은 줄(왼쪽→오른쪽).
# 채점(T5): 0.5 → 0.7에서 aCER DIG 0.0375 → 0.0367, SYN 0.0602 → 0.0585(0.8은 0.0442·0.0607로 나빠짐)
PARA_GAP = 0.8  # 같은 문단: 줄 사이 빈 간격 ≤ 앞 줄 높이 × 0.8
PARA_INDENT = 1.0  # 왼쪽 시작 차 ≤ 앞 줄 높이 × 1
PARA_HEIGHT = 0.5  # 줄 높이 차 ≤ 앞 줄 높이 × 0.5
CONFIDENCE_SCALE = 0.5  # 블록 신뢰도 = 평균 점수 × 0.5(검증 전, VLM 확인 대상)


@dataclass(frozen=True, slots=True)
class OcrText:
    """OCR 줄 하나. 좌표는 보이는 쪽 pt(원점 왼쪽 위, 회전 보정: 렌더한 그림과 같은 틀)."""

    text: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class OcrParagraph:
    """OCR 문단 블록 하나. bbox는 보이는 쪽 0~1(x0, y0, x1, y1), text는 줄을 \\n으로 이은 것."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


def render(data: bytes, name: str, index: int) -> Image.Image:
    """쪽 하나를 OCR_DPI로 그린다(쪽 회전 반영, 긴 변 MAX_RENDER_SIDE 이하). PDFIUM_LOCK 안에서 열고 닫는다."""
    with PDFIUM_LOCK:
        pdf = open_pdf(data, name)
        page = bitmap = None
        try:
            page = pdf[index]
            width, height = page.get_size()
            scale = min(OCR_DPI / 72, MAX_RENDER_SIDE / max(width, height))
            bitmap = page.render(scale=scale)
            return bitmap.to_pil().convert("RGB")  # 복사본: 비트맵을 닫아도 남는다
        except pdfium.PdfiumError as exc:
            raise ParseError(f"invalid PDF page: {exc}", f"{name}:{index + 1}") from None
        finally:
            try:
                if bitmap is not None:
                    bitmap.close()
                if page is not None:
                    page.close()
            finally:
                pdf.close()


def to_page(lines: Sequence[ocr.OcrLine], size: tuple[int, int], page: PageText) -> list[OcrText]:
    """그림 화소 → 보이는 쪽 pt(외접 상자). 렌더 그림은 보이는 쪽을 같은 배율로 그렸다."""
    sx, sy = page.width_pt / size[0], page.height_pt / size[1]
    out = []
    for line in lines:
        xs, ys = [x for x, _ in line.box], [y for _, y in line.box]
        out.append(OcrText(text=line.text, score=line.score,
                           x0=max(min(xs) * sx, 0.0), y0=max(min(ys) * sy, 0.0),
                           x1=min(max(xs) * sx, page.width_pt), y1=min(max(ys) * sy, page.height_pt)))
    return out


def _covered(line: OcrText, page: PageText) -> float:
    """보이는 글자가 줄 상자의 긴 축(가로줄이면 x)을 덮는 비율(0~1). 글자 상자 중심이 줄 상자의 짧은 축 구간 안에
    든 글자만 센다. 넓이 비율을 쓰지 않는 까닭: OCR 줄 상자는 글자보다 여백이 커서 같은 글자를 다시 읽은 한 글자
    줄도 넓이로는 0.30~0.37만 덮인다(실측)."""
    w, h = page.width_pt, page.height_pt
    wide = line.y1 - line.y0 < TALL * (line.x1 - line.x0)  # 실행부처럼 세로가 1.5배 이상이면 세로줄
    lo, hi = (line.x0, line.x1) if wide else (line.y0, line.y1)
    if hi <= lo:
        return 1.0
    spans = []
    for c in page.chars:
        if c.invisible:
            continue
        cx0, cy0, cx1, cy1 = c.x0 * w, c.y0 * h, c.x1 * w, c.y1 * h
        if wide and not line.y0 <= (cy0 + cy1) / 2 <= line.y1:
            continue
        if not wide and not line.x0 <= (cx0 + cx1) / 2 <= line.x1:
            continue
        a, b = (cx0, cx1) if wide else (cy0, cy1)
        a, b = max(a, lo), min(b, hi)
        if b > a:
            spans.append((a, b))
    union, end = 0.0, -math.inf
    for a, b in sorted(spans):
        if b > end:
            union += b - max(a, end)
            end = b
    return union / (hi - lo)


def keep(line: OcrText, page: PageText) -> bool:
    """점수 거르기(짧은 줄은 더 높게)와 텍스트 레이어 우선(보이는 글자와 OVERLAP 이상 겹친 줄은 버린다)."""
    if line.score < OCR_MIN_SCORE:
        return False
    if len("".join(line.text.split())) <= SHORT_TEXT and line.score < OCR_SHORT_SCORE:
        return False
    return _covered(line, page) < OVERLAP


def _split(lines: Sequence[OcrText], vertical: bool, gap: float) -> list[list[OcrText]]:
    """vertical이면 x 구간, 아니면 y 구간을 투영해 빈 띠가 gap 이상인 곳마다 나눈다(나눈 조각은 그 축 순서)."""
    def span(t: OcrText) -> tuple[float, float]:
        return (t.x0, t.x1) if vertical else (t.y0, t.y1)

    groups: list[list[OcrText]] = []
    end = -math.inf
    for t in sorted(lines, key=span):
        lo, hi = span(t)
        if not groups or lo - end >= gap:
            groups.append([])
        groups[-1].append(t)
        end = max(end, hi)
    return groups


def _rows(lines: Sequence[OcrText]) -> list[OcrText]:
    """위→아래, 세로 중심이 가까운(줄 높이 × SAME_ROW) 줄끼리는 왼쪽→오른쪽."""
    rows: list[list[OcrText]] = []
    for t in sorted(lines, key=lambda t: (t.y0, t.x0)):
        if rows:
            anchor = rows[-1][0]
            if abs((t.y0 + t.y1) - (anchor.y0 + anchor.y1)) / 2 <= SAME_ROW * min(t.height, anchor.height):
                rows[-1].append(t)
                continue
        rows.append([t])
    return [t for row in rows for t in sorted(row, key=lambda t: t.x0)]


def _columns(lines: list[OcrText], body: float) -> list[list[OcrText]]:
    """빈 세로 띠로 나눈 단. 단 하나라도 영역 너비의 MIN_COLUMN보다 좁으면 단이 아니라 표·이름표 배치로 보고 나누지 않는다."""
    columns = _split(lines, True, COLUMN_GAP * body)
    width = max(t.x1 for t in lines) - min(t.x0 for t in lines)
    if any(max(t.x1 for t in c) - min(t.x0 for t in c) < MIN_COLUMN * width for c in columns):
        return [lines]
    return columns


def _cut(lines: list[OcrText], body: float) -> list[OcrText]:
    columns = _columns(lines, body)
    if len(columns) > 1:
        return [t for column in columns for t in _cut(column, body)]
    bands = _split(lines, False, BAND_GAP * body)
    if len(bands) > 1:
        return [t for band in bands for t in _cut(band, body)]
    return _rows(lines)


def reading_order(lines: Sequence[OcrText]) -> list[OcrText]:
    """XY 분할: 줄을 가로지르지 않는 빈 세로 띠(본문 줄 높이 × 2 이상)로 단을 나누고, 못 나누면 빈 가로 띠
    (줄 높이 × 1 이상)로 위아래를 나눠 다시 본다. 더 못 나누면 위→아래, 같은 줄은 왼쪽→오른쪽.
    본문 줄 높이는 쪽 줄 높이의 중앙값."""
    if not lines:
        return []
    return _cut(list(lines), statistics.median(t.height for t in lines))


def _joins(prev: OcrText, cur: OcrText) -> bool:
    h = prev.height
    return (cur.y0 > prev.y0 + 0.5 * h
            and cur.y0 - prev.y1 <= PARA_GAP * h
            and abs(cur.x0 - prev.x0) <= PARA_INDENT * h
            and abs(cur.height - h) <= PARA_HEIGHT * h)


def paragraphs(lines: Sequence[OcrText], page: PageText) -> list[OcrParagraph]:
    """읽기 순서의 줄을 문단으로 묶는다(앞 줄 바로 아래, 간격·왼쪽 시작·높이가 가까우면 같은 문단)."""
    groups: list[list[OcrText]] = []
    for t in lines:
        if groups and _joins(groups[-1][-1], t):
            groups[-1].append(t)
        else:
            groups.append([t])
    w, h = page.width_pt, page.height_pt
    return [OcrParagraph(text="\n".join(t.text for t in g),
                         bbox=(min(t.x0 for t in g) / w, min(t.y0 for t in g) / h,
                               max(t.x1 for t in g) / w, max(t.y1 for t in g) / h),
                         confidence=round(statistics.fmean(t.score for t in g) * CONFIDENCE_SCALE, 3))
            for g in groups]


def page_paragraphs(data: bytes, name: str, index: int, page: PageText) -> list[OcrParagraph]:
    """쪽 하나: 렌더(잠금 안) → OCR(잠금 밖) → 거르기 → 순서 → 문단."""
    image = render(data, name, index)
    lines = [t for t in to_page(ocr.read_lines(image), image.size, page) if keep(t, page)]
    return paragraphs(reading_order(lines), page)


def ocr_pages(data: bytes, name: str, pages: Sequence[PageText],
              states: Sequence[TextLayerState]) -> list[list[OcrParagraph]]:
    """쪽마다 OCR 문단. scanned 쪽만 읽고 나머지는 빈 목록."""
    return [page_paragraphs(data, name, i, page) if state == "scanned" else []
            for i, (page, state) in enumerate(zip(pages, states, strict=True))]
