"""그림·캡션 정리(순수 함수: PDFium·onnxruntime을 부르지 않는다). 레이아웃 상자·이미지 객체·표·글자(텍스트 레이어
또는 OCR 줄) → 그림과 짝지은 캡션(스펙 §4.2~4.5). 그림 PNG 잘라내기(§4.6)와 문서 바이트 상한도 여기 둔다.
길이 단위는 보이는 쪽 pt(원점 왼쪽 위, 회전 보정: 글자 상자 × 쪽 너비·높이와 같은 틀)."""

import bisect
import hashlib
import io
import math
import unicodedata
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from hanji_contracts import MAX_DOCUMENT_ASSET_BYTES, MAX_FIGURE_SIDE
from PIL import Image

from .extract import UPRIGHT, PageText
from .group import fragments
from .scan import OcrText, reading_order
from .tables import TableSpec, _merge, _positions, _reading_segs
from .triage import PageMode

if TYPE_CHECKING:
    from .layout import LayoutBox

FIGURE_MIN_SCORE = 0.5  # image·chart 상자 점수 기준
# 채점: 0.4 → 0.5에서 공공누리 R/P/캡션 0.895/0.968/2 → 0.895/1.0/2(04 잘못 찾은 쪽 6 → 3), 선·도형 R/P/캡션
# 0.95/0.964/162 → 0.95/0.979/162(0.3·0.35는 공공누리 P, 0.45는 선·도형 R이 나빠짐)
CAPTION_MIN_SCORE = 0.4  # figure_title 상자 점수 기준
# 채점: 0.5 → 0.4에서 공공누리 R/P/캡션 0.895/1.0/2 → 0.895/1.0/2(04 잘못 찾은 쪽 3 그대로), 선·도형 R/P/캡션
# 0.95/0.979/162 → 0.95/0.979/167(0.6은 선·도형 캡션 145로 나빠짐)
TABLE_MIN_SCORE = 0.5  # 모델 table 상자: 표 확인·표 제목 판정·처리 이력에 쓴다
CAPTION_GAP = 30.0  # 캡션은 그림 바로 위·아래 이 간격(pt) 안(스펙 시작값)
CAPTION_OVERLAP = 0.3  # 캡션과 그림(표)의 가로 겹침 ≥ 좁은 쪽 너비 × 0.3
LAYOUT_MIN_PATHS = 10  # digital 쪽은 path 객체가 이만큼 있거나 그림이 될 이미지 객체가 있을 때만 모델을 돌린다.
# 시작값: 선·도형 세트 정답 61쪽의 path 수 최솟값 14보다 작게(글자만 있는 쪽은 건너뛴다)
TABLE_RULES = 3  # layer 모드 쪽은 선 있는 표 밖 긴 가로선이 이만큼 있어도 모델을 돌린다(선 없는 표 상자, 스펙 D1)
TABLE_RULE_WIDTH = 0.3  # 긴 가로선: 쪽 너비의 30% 이상
MIN_IMAGE_SIDE = 24.0  # 장식이 아닌 이미지 객체: 양변 ≥ 24pt이고
MIN_IMAGE_AREA = 0.025  # 쪽 면적의 2.5% 이상(스펙 §4.2)
BACKGROUND_CHARS = 50  # 보이는 글자가 이만큼 얹힌 이미지 객체는 배경(쪽 배경·워터마크): 그림도 합치기 대상도 아니다
CONTAIN = 0.9  # 감싼다: 안쪽 상자 넓이의 90% 이상이 바깥 상자 안(스펙 §4.3-3)
INSIDE = 0.8  # 대부분 든다: 이미지 객체 합치기·그림 안 표 버리기·진짜 표 안 그림 버리기
SAME_IOU = 0.5  # 같은 분류 그림끼리 IoU ≥ 0.5면 큰 쪽 하나(스펙 §4.3-5)
TABLE_CONFIRM_IOU = 0.5  # 모델 table 상자와 IoU ≥ 0.5인 선 있는 표는 진짜 표(그림이 이기지 않는다)
CAPTURE_PAD = 2.0  # 그림·캡션 상자 안 글자를 셀 때 상자를 이만큼(pt) 넓힌다(렌더 글꼴과 글꼴 너비 정보의 차이)
PNG_LEVEL = 6  # PNG 압축 단계(최적화 끔·메타데이터 없음: 같은 컴퓨터에서 같은 바이트)
ASSET_LIMIT = MAX_DOCUMENT_ASSET_BYTES  # 문서 하나의 그림 바이트 총합 상한(테스트가 바꿔 끼운다)
FIGURE_CLASSES = {"image": "image", "chart": "chart"}  # 모델 분류 → 그림 분류. seal 등 장식·표·글자는 그림이 아니다
CAPTION_CLASS = "figure_title"
DECOR_CLASSES = frozenset({"seal", "header_image", "footer_image"})  # 모델 장식 분류: 그 안 이미지 객체는 그림이 아니다
TABLE_CLASS = "table"

Box = tuple[float, float, float, float]  # 보이는 쪽 pt (x0, y0, x1, y1)
_Candidate = tuple[Box, str, float]  # (상자, 그림 분류, 점수)


@dataclass(frozen=True, slots=True)
class Region:
    """레이아웃 상자 하나(보이는 쪽 pt)."""

    cls: str
    score: float
    box: Box


@dataclass(frozen=True, slots=True)
class Caption:
    """그림과 짝지은 캡션. char_ids는 가져간 텍스트 레이어 글자(page.chars 순번), line_ids는 OCR 줄 순번."""

    box: Box
    text: str
    char_ids: frozenset[int]
    line_ids: frozenset[int]
    above: bool


@dataclass(frozen=True, slots=True)
class Figure:
    """그림 하나. text는 상자 안 글자(줄은 \\n), category는 image·chart. model은 레이아웃 모델이 찾은 그림인지(모델
    상자, 또는 모델 상자를 합친 이미지 객체. 모델이 찾지 않은 이미지 객체는 거짓): 처리 이력에 쓴다."""

    box: Box
    category: str
    text: str = ""
    char_ids: frozenset[int] = frozenset()
    line_ids: frozenset[int] = frozenset()
    caption: Caption | None = None
    model: bool = False


@dataclass(frozen=True, slots=True)
class PagePlan:
    """한 쪽의 정리 결과: 그림(윗변 순), 남긴 표, 그림이 이겨 버린 표, 모델 table 상자(처리 이력용)."""

    figures: tuple[Figure, ...] = ()
    tables: tuple[TableSpec, ...] = ()
    dropped: tuple[TableSpec, ...] = ()
    layout_tables: tuple[Region, ...] = ()

    @property
    def used_lines(self) -> frozenset[int]:
        """그림·캡션이 가져간 OCR 줄 순번(나머지가 OCR 문단이 된다)."""
        return frozenset(i for f in self.figures for i in (*f.line_ids, *(f.caption.line_ids if f.caption else ())))


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _iou(a: Box, b: Box) -> float:
    inter = _inter(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _inside(inner: Box, outer: Box) -> float:
    """inner 넓이 중 outer 안에 든 비율."""
    size = area(inner)
    return _inter(inner, outer) / size if size > 0 else 0.0


def _pt(page: PageText, box: tuple[float, float, float, float]) -> Box:
    """보이는 쪽 0~1 → pt."""
    return (box[0] * page.width_pt, box[1] * page.height_pt, box[2] * page.width_pt, box[3] * page.height_pt)


def _decor(box: Box, page: PageText) -> bool:
    """장식 이미지(로고·아이콘·글머리): 한 변이 MIN_IMAGE_SIDE보다 짧거나 쪽 면적의 MIN_IMAGE_AREA보다 작다."""
    width, height = box[2] - box[0], box[3] - box[1]
    return min(width, height) < MIN_IMAGE_SIDE or width * height < MIN_IMAGE_AREA * page.width_pt * page.height_pt


def _centres(page: PageText) -> list[tuple[float, float]]:
    """보이는 글자(공백 제외)의 상자 중심(pt). 쪽마다 한 번 구한다."""
    w, h = page.width_pt, page.height_pt
    out = [((c.x0 + c.x1) / 2 * w, (c.y0 + c.y1) / 2 * h) for c in page.chars if not c.invisible and not c.text.isspace()]
    return [p for p in out if math.isfinite(p[0]) and math.isfinite(p[1])]


def _counts(points: Sequence[tuple[float, float]], boxes: Sequence[Box]) -> list[int]:
    """상자마다 안(경계 포함)에 든 점 수. 점을 x 순으로 넣으며 y 펜윅 트리로 센다: 한 번 훑기,
    O((점 + 상자) log 점). 상자 수 × 글자 수로 커지지 않는다."""
    ys = sorted({y for _, y in points})
    tree = [0] * (len(ys) + 1)

    def upto(i: int) -> int:  # y 순번 i개(작은 쪽부터)에 든 점 수
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    # (x, 종류, 값): 종류 0 = x < x0 상자(같은 x 점보다 먼저), 1 = 점, 2 = x ≤ x1 상자(같은 x 점 다음)
    events: list[tuple[float, int, float]] = [(x, 1, y) for x, y in points]
    events += [(b[0], 0, k) for k, b in enumerate(boxes)] + [(b[2], 2, k) for k, b in enumerate(boxes)]
    events.sort(key=lambda e: (e[0], e[1]))
    counts = [0] * len(boxes)
    for _, kind, value in events:
        if kind == 1:
            i = bisect.bisect_left(ys, value) + 1
            while i <= len(ys):
                tree[i] += 1
                i += i & -i
            continue
        b = boxes[int(value)]
        inside = upto(bisect.bisect_right(ys, b[3])) - upto(bisect.bisect_left(ys, b[1]))
        counts[int(value)] += inside if kind == 2 else -inside
    return counts


# 마지막으로 본 쪽의 (쪽, 사진, 장식). 쪽은 약한 참조라 파싱이 끝난 쪽을 붙잡지 않는다(리뷰 M2)
_last: tuple["weakref.ref[PageText]", tuple[Box, ...], tuple[Box, ...]] | None = None


def _image_boxes(page: PageText) -> tuple[tuple[Box, ...], tuple[Box, ...]]:
    """이미지 객체 상자(보이는 쪽 pt)를 (그림이 될 사진, 장식)으로 나눈다. 같은 상자(겹쳐 넣은 사본)는 처음 하나만,
    유한하지 않은 상자는 뺀다. 배경(보이는 글자(공백 제외) BACKGROUND_CHARS개 이상이 중심을 두고 얹힌 쪽 배경·
    워터마크)은 어느 쪽도 아니다. 마지막 쪽 결과를 기억해 wants_layout·photo_boxes·arrange가 같은 쪽을 다시
    훑지 않는다(쪽은 바뀌지 않는 값이고 같은 객체인지로 본다. 약한 참조라 쪽을 붙잡지 않는다)."""
    global _last
    last = _last
    if last is not None and last[0]() is page:
        return last[1], last[2]
    rects = list(dict.fromkeys(box for box in (_pt(page, rect) for rect in page.images)
                               if all(math.isfinite(v) for v in box)))
    decor = tuple(box for box in rects if _decor(box, page))
    rest = [box for box in rects if not _decor(box, page)]
    photos = tuple(box for box, n in zip(rest, _counts(_centres(page), rest), strict=True) if n < BACKGROUND_CHARS)
    _last = (weakref.ref(page), photos, decor)
    return photos, decor


def photo_boxes(page: PageText) -> list[Box]:
    """그림이 될 이미지 객체(보이는 쪽 pt, 같은 상자는 한 번): 장식·배경이 아닌 것. 모델 없이도 그림 블록이 된다
    (digital 쪽)."""
    return list(_image_boxes(page)[0])


def table_rules(page: PageText, tables: Sequence[TableSpec] = ()) -> int:
    """선 있는 표(tables) 상자(±2pt) 밖의 긴 가로선 수: 같은 위치·끝이 이어진 조각을 모은(tables._merge) 가로선 가운데
    길이 ≥ 쪽 너비 × TABLE_RULE_WIDTH인 것. 같은 위치(± SNAP)의 선은 하나로 센다."""
    w, h = page.width_pt, page.height_pt
    boxes = [_pt(page, t.bbox) for t in tables]
    long = [s.pos for s in _merge(_reading_segs(page.rules, UPRIGHT, w, h))
            if s.axis == "h" and s.end - s.start >= TABLE_RULE_WIDTH * w
            and not any(b[1] - 2 <= s.pos <= b[3] + 2 and s.start >= b[0] - 2 and s.end <= b[2] + 2 for b in boxes)]
    return len(_positions(long))


def wants_layout(page: PageText, mode: PageMode, tables: Sequence[TableSpec] = ()) -> bool:
    """레이아웃 모델을 돌릴 쪽(§4.2): scan 모드 쪽, 그리고 그림이 될 이미지 객체가 있거나 path 객체가
    LAYOUT_MIN_PATHS 이상이거나 선 있는 표(tables) 밖 긴 가로선이 TABLE_RULES 이상인 layer 모드 쪽(digital, 글자층으로
    읽는 unreliable)."""
    if mode == "scan":
        return True
    return page.paths >= LAYOUT_MIN_PATHS or bool(photo_boxes(page)) or table_rules(page, tables) >= TABLE_RULES


def page_regions(boxes: Sequence["LayoutBox"], size: tuple[int, int], page: PageText) -> list[Region]:
    """레이아웃 상자(쪽 렌더 화소) → 보이는 쪽 pt. 렌더는 보이는 쪽을 같은 배율로 그렸다."""
    sx, sy = page.width_pt / size[0], page.height_pt / size[1]
    return [Region(b.cls, b.score, (b.box[0] * sx, b.box[1] * sy, b.box[2] * sx, b.box[3] * sy)) for b in boxes]


def _same(a: Box, b: Box) -> bool:
    """거의 같은 상자: 서로 상대 넓이의 CONTAIN 이상을 품는다(IoU ≥ 약 0.81)."""
    return _inside(a, b) >= CONTAIN and _inside(b, a) >= CONTAIN


def _drop_containers(cands: list[_Candidate]) -> list[_Candidate]:
    """다른 후보 둘 이상을 대부분(≥ CONTAIN) 감싸는 후보는 버린다(차트 묶음 위에 덧붙은 상자, §4.3-3). 거의 같은
    상자(서로 상대 넓이의 CONTAIN 이상을 품는다: 같은 곳을 chart·image로 둘 다 찾은 것)는 감싼 것으로 세지 않고,
    감싼 것끼리 거의 같으면 하나로 센다. 묶음 상자의 절반을 넘는 진짜 조각은 거의 같은 상자가 아니다."""
    out: list[_Candidate] = []
    for i, c in enumerate(cands):
        wrapped: list[Box] = []
        for j, o in enumerate(cands):
            if (j != i and len(wrapped) < 2 and _inside(o[0], c[0]) >= CONTAIN and not _same(o[0], c[0])
                    and not any(_same(o[0], w) for w in wrapped)):
                wrapped.append(o[0])
        if len(wrapped) < 2:
            out.append(c)
    return out


def _merge_images(cands: list[_Candidate], page: PageText, marks: Sequence[Box] = ()) -> list[_Candidate]:
    """digital 쪽(§4.3-4): 그림이 될 이미지 객체 하나 안에 대부분(≥ INSIDE) 드는 후보는 그 이미지 객체 상자 하나로
    합치고(분류는 점수 높은 후보, 여러 객체면 가장 작은 객체), 모델이 찾지 못한 그런 객체도 그림(image)이 된다.
    장식 이미지(로고, 그리고 모델 장식 상자 marks 안에 대부분 드는 이미지 객체: 머리띠·직인) 안에 대부분 드는 후보는
    버린다(장식 거르기)."""
    photos, decor = _image_boxes(page)
    if marks:
        under = [photo for photo in photos if any(_inside(photo, m) >= INSIDE for m in marks)]
        photos, decor = tuple(p for p in photos if p not in under), (*decor, *under)
    merged: dict[int, tuple[str, float]] = {}
    out: list[_Candidate] = []
    for box, category, score in cands:
        hosts = [k for k, photo in enumerate(photos) if _inside(box, photo) >= INSIDE]
        if hosts:
            k = min(hosts, key=lambda k: area(photos[k]))
            if k not in merged or score > merged[k][1]:
                merged[k] = (category, score)
            continue
        if any(_inside(box, logo) >= INSIDE for logo in decor):
            continue
        out.append((box, category, score))
    return out + [(photo, *merged.get(k, ("image", 0.0))) for k, photo in enumerate(photos)]


def _dedupe(cands: list[_Candidate]) -> list[_Candidate]:
    """같은 분류 IoU ≥ SAME_IOU면 큰 쪽 하나(§4.3-5), 다른 그림 안에 대부분(≥ CONTAIN) 드는 조각은 버린다.
    넓이가 같으면 점수 높은 쪽, 그것도 같으면 앞 후보."""
    kept: list[_Candidate] = []
    for c in sorted(cands, key=lambda c: (-area(c[0]), -c[2])):
        if any((o[1] == c[1] and _iou(c[0], o[0]) >= SAME_IOU) or _inside(c[0], o[0]) >= CONTAIN for o in kept):
            continue
        kept.append(c)
    return kept


def _settle_tables(cands: list[_Candidate], tables: Sequence[TableSpec], layout_tables: Sequence[Region],
                   page: PageText) -> tuple[list[TableSpec], list[TableSpec], list[_Candidate]]:
    """그림이 이긴다(§4.4): 모델이 찾은 그림 상자(점수 ≥ FIGURE_MIN_SCORE: 모델 후보와 그것을 합친 이미지 객체) 안에
    넓이의 INSIDE 이상이 드는 표는 버린다. 모델이 찾지 않은 이미지 객체(점수 0)는 표를 버리지 않는다(리뷰 I1: 음영·
    그림 채우기 위에 그린 선 있는 표는 대개 진짜 표다). 모델 table 상자와 IoU ≥ TABLE_CONFIRM_IOU인 표는 진짜 표로
    남기고, 그 표 안에 대부분 드는 그림 후보를 버린다(표를 그림으로 본 것). 반환: (남긴 표, 버린 표, 남은 후보)."""
    boxes = [_pt(page, t.bbox) for t in tables]
    sure = {k for k, box in enumerate(boxes) if any(_iou(box, r.box) >= TABLE_CONFIRM_IOU for r in layout_tables)}
    cands = [c for c in cands if not any(_inside(c[0], boxes[k]) >= INSIDE for k in sure)]
    found = [c[0] for c in cands if c[2] >= FIGURE_MIN_SCORE]
    kept: list[TableSpec] = []
    dropped: list[TableSpec] = []
    for k, (table, box) in enumerate(zip(tables, boxes, strict=True)):
        if k not in sure and any(_inside(box, f) >= INSIDE for f in found):
            dropped.append(table)
        else:
            kept.append(table)
    return kept, dropped, cands


def _gap(caption: Box, other: Box) -> tuple[float, bool] | None:
    """caption이 other 바로 위(above=True)나 아래에 있을 때 (세로 빈틈(겹치면 0), 위인지). 가로로 좁은 쪽 너비의
    CAPTION_OVERLAP 이상 겹치고 빈틈이 CAPTION_GAP 이하일 때만, 아니면 None."""
    overlap = min(caption[2], other[2]) - max(caption[0], other[0])
    width = min(caption[2] - caption[0], other[2] - other[0])
    if width <= 0 or overlap < CAPTION_OVERLAP * width:
        return None
    above = (caption[1] + caption[3]) / 2 < (other[1] + other[3]) / 2
    gap = max(0.0, other[1] - caption[3]) if above else max(0.0, caption[1] - other[3])
    return (gap, above) if gap <= CAPTION_GAP else None


def _table_title(caption: Box, tables: Sequence[Box], figure_boxes: Sequence[Box]) -> bool:
    """표 바로 위·아래에 붙은 캡션 후보(표 제목, §4.5): 그림보다 표에 가깝거나 같으면 그림과 짝짓지 않는다."""
    near_table = min((g[0] for t in tables if (g := _gap(caption, t)) is not None), default=None)
    if near_table is None:
        return False
    near_figure = min((g[0] for f in figure_boxes if (g := _gap(caption, f)) is not None), default=None)
    return near_figure is None or near_table <= near_figure


def _pair(figure_boxes: Sequence[Box], captions: Sequence[Box]) -> dict[int, tuple[int, bool]]:
    """그림마다 가장 가까운 캡션 하나(§4.5). 가까운 짝부터 정한다: 한 캡션은 그림 하나에만(두 그림 사이 캡션은 더
    가까운 그림, 같으면 앞 그림). 반환: 그림 순번 → (캡션 순번, 위 캡션인지)."""
    found = sorted((g[0], fi, ci, g[1]) for fi, f in enumerate(figure_boxes) for ci, c in enumerate(captions)
                   if (g := _gap(c, f)) is not None)
    pairs: dict[int, tuple[int, bool]] = {}
    used: set[int] = set()
    for _, fi, ci, above in found:
        if fi in pairs or ci in used:
            continue
        pairs[fi] = (ci, above)
        used.add(ci)
    return pairs


def _capture(page: PageText, mode: PageMode, box: Box, taken: set[int], lines: Sequence[OcrText],
             used: set[int]) -> tuple[str, frozenset[int], frozenset[int]]:
    """상자(CAPTURE_PAD만큼 넓혀) 안에 중심이 드는 글자: scan 모드 쪽은 OCR 줄(읽기 순서, used는 이미 가져간 줄),
    아니면 보이는 텍스트 레이어 글자(taken은 이미 표·캡션이 가져간 글자). 반환: (글자(NFC, 줄은 \\n), 글자 순번,
    줄 순번)."""
    x0, y0, x1, y1 = box[0] - CAPTURE_PAD, box[1] - CAPTURE_PAD, box[2] + CAPTURE_PAD, box[3] + CAPTURE_PAD
    if mode == "scan":
        ids = [i for i, t in enumerate(lines) if i not in used
               and x0 <= (t.x0 + t.x1) / 2 <= x1 and y0 <= (t.y0 + t.y1) / 2 <= y1]
        text = "\n".join(t.text for t in reading_order([lines[i] for i in ids]))
        return unicodedata.normalize("NFC", text).strip(), frozenset(), frozenset(ids)
    w, h = page.width_pt, page.height_pt
    ids = [i for i, c in enumerate(page.chars) if not c.invisible and i not in taken
           and x0 <= (c.x0 + c.x1) / 2 * w <= x1 and y0 <= (c.y0 + c.y1) / 2 * h <= y1]
    text = "\n".join(f.text for f in fragments(replace(page, chars=tuple(page.chars[i] for i in ids))))
    return unicodedata.normalize("NFC", text).strip(), frozenset(ids), frozenset()


def arrange(page: PageText, mode: PageMode, regions: Sequence[Region] = (), tables: Sequence[TableSpec] = (),
            lines: Sequence[OcrText] = ()) -> PagePlan:
    """한 쪽의 그림·캡션(§4.3~4.5). regions는 레이아웃 상자(보이는 쪽 pt), tables는 find_tables 결과, lines는
    scan 모드 쪽의 거른 OCR 줄. layer 모드 쪽은 이미지 객체(page.images)도 본다. 순서: 분류·점수 기준 → 감싸는 상자
    버리기 → (layer) 이미지 객체 합치기 → 겹침·조각 정리 → 그림 우선(표) → 표 제목 거르기 → 캡션 짝 → 캡션 글자 →
    그림 글자. scan 모드 쪽은 OCR 줄만 그림·캡션으로 옮긴다: 상자 안 텍스트 레이어 글자는 문단으로 남는다."""
    cands: list[_Candidate] = [(r.box, FIGURE_CLASSES[r.cls], r.score) for r in regions
                               if r.cls in FIGURE_CLASSES and r.score >= FIGURE_MIN_SCORE and area(r.box) > 0]
    caption_boxes = [r.box for r in regions if r.cls == CAPTION_CLASS and r.score >= CAPTION_MIN_SCORE and area(r.box) > 0]
    layout_tables = tuple(r for r in regions if r.cls == TABLE_CLASS and r.score >= TABLE_MIN_SCORE)
    marks = [r.box for r in regions if r.cls in DECOR_CLASSES and r.score >= FIGURE_MIN_SCORE]
    cands = _drop_containers(cands)
    if mode == "layer":
        cands = _merge_images(cands, page, marks)
    cands = _dedupe(cands)
    kept, dropped, cands = _settle_tables(cands, tables, layout_tables, page)
    cands.sort(key=lambda c: (c[0][1], c[0][0]))
    figure_boxes = [c[0] for c in cands]
    taken = {i for t in kept for i in t.char_ids}
    used: set[int] = set()
    # 그림 후보와 같은 자리의 모델 table 상자(같은 곳을 두 분류로 찾은 것)는 표 제목 판정에 쓰지 않는다(사전 리뷰 1)
    table_boxes = [_pt(page, t.bbox) for t in kept] + [
        r.box for r in layout_tables
        if not any(_inside(r.box, f) >= INSIDE or _iou(r.box, f) >= TABLE_CONFIRM_IOU for f in figure_boxes)]
    captions = [box for box in caption_boxes if not _table_title(box, table_boxes, figure_boxes)
                and _capture(page, mode, box, taken, lines, used)[0]]
    linked: dict[int, Caption] = {}
    for fi, (ci, above) in sorted(_pair(figure_boxes, captions).items()):
        text, chars, line_ids = _capture(page, mode, captions[ci], taken, lines, used)
        if text:  # 앞 캡션이 글자를 먼저 가져가 비면 짝 없음
            taken |= chars
            used |= line_ids
            linked[fi] = Caption(captions[ci], text, chars, line_ids, above)
    found = []
    for fi, (box, category, score) in enumerate(cands):
        text, chars, line_ids = _capture(page, mode, box, taken, lines, used)
        taken |= chars
        used |= line_ids
        found.append(Figure(box, category, text, chars, line_ids, linked.get(fi), score >= FIGURE_MIN_SCORE))
    return PagePlan(tuple(found), tuple(kept), tuple(dropped), layout_tables)


def crop_png(image: Image.Image, box: Box, page: PageText) -> tuple[bytes, int, int, int] | None:
    """쪽 렌더(보이는 쪽 틀)에서 상자를 잘라 PNG로: (바이트, 너비, 높이, dpi). 상자 바깥쪽 화소까지(내림·올림) 자르고
    그림 밖은 뺀다. 긴 변이 MAX_FIGURE_SIDE를 넘으면 비율을 지켜 줄인다(dpi도). 메타데이터 없이 최적화를 끄고
    저장한다(같은 입력이면 같은 바이트). 넓이가 없으면 None."""
    sx, sy = image.width / page.width_pt, image.height / page.height_pt
    left, top = max(0, math.floor(box[0] * sx)), max(0, math.floor(box[1] * sy))
    right, bottom = min(image.width, math.ceil(box[2] * sx)), min(image.height, math.ceil(box[3] * sy))
    if right <= left or bottom <= top:
        return None
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    dpi = 72 * sx
    if max(crop.size) > MAX_FIGURE_SIDE:
        scale = MAX_FIGURE_SIDE / max(crop.size)
        size = (max(1, min(MAX_FIGURE_SIDE, round(crop.width * scale))),
                max(1, min(MAX_FIGURE_SIDE, round(crop.height * scale))))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        dpi *= scale
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=False, compress_level=PNG_LEVEL)
    return buf.getvalue(), crop.width, crop.height, max(1, round(dpi))


class AssetBudget:
    """문서 하나의 그림 이미지("sha256:…" → PNG). 같은 바이트는 한 번만 담고 한 번만 센다. 총합이 limit을 넘을 그림은
    담지 않는다(그 그림 블록은 이미지 없이 위치·글자만, 처리 이력에 남긴다)."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = ASSET_LIMIT if limit is None else limit
        self.assets: dict[str, bytes] = {}
        self.used = 0

    def add(self, png: bytes) -> str | None:
        """담은(또는 이미 담긴) 이미지의 키. 상한을 넘으면 None."""
        key = "sha256:" + hashlib.sha256(png).hexdigest()
        if key in self.assets:
            return key
        if self.used + len(png) > self.limit:
            return None
        self.assets[key] = png
        self.used += len(png)
        return key
