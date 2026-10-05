"""보이는 글자 → 줄 조각 → 블록 명세(제목·문단·목록·머리말·꼬리말). 길이 단위는 pt.

줄·순서·간격은 글자마다 읽기 좌표(x = 진행 방향, y = 줄 아래 방향, 원점은 그 방향으로 읽을 때의 왼쪽 위)에서
잰다. 바로 선 글자면 보이는 쪽 좌표와 같고, 회전한 쪽·음수 Tf·거울 글자도 진행 방향 순서대로 읽는다."""

import re
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ko_parser_contracts import TextLayerState

from .extract import UPRIGHT, Axes, Char, PageText
from .scan import OcrParagraph

if TYPE_CHECKING:  # tables.py가 이 모듈을 import하므로 실행 중에는 가져오지 않는다
    from .tables import TableSpec

SAME_LINE = 0.5  # 기준선 차이 ≤ 실제 크기 × 0.5면 같은 줄
SPLIT_GAP = 3.0  # 줄 안 글자 간격 > 실제 크기 × 3이면 줄 조각을 나눈다(표 칸·다단)
SPACE_GAP = 0.2  # 글자 간격 − 보통 자간 > 실제 크기 × 0.2면 공백을 끼운다(공백 글자가 없는 PDF, 공공누리로 조정)
SIZE_STEP = 0.5  # 크기는 0.5pt 단위로 묶는다
HEADING_RATIO = 1.15  # 제목: 본문 크기 × 1.15 이상
BOLD_HEADING_RATIO = 1.05  # 굵으면 × 1.05 이상
HEADING_MAX_LINES = 2  # 이어지는 제목 크기 줄이 이보다 많으면 문단
MAX_LEVEL = 6
PARA_GAP = 0.8  # 같은 블록: 줄 사이 빈 간격 ≤ 줄 높이 × 0.8
PARA_SIZE_DIFF = 0.5  # 크기 차 ≤ 0.5pt
PARA_INDENT = 1.0  # 왼쪽 시작 차 ≤ 본문 크기 × 1
MARGIN = 0.08  # 머리말·꼬리말 영역: 쪽 높이의 위·아래 8%(줄의 세로 중심 기준)
SAME_POSITION = 0.02  # 같은 위치: 세로 중심 차 ≤ 쪽 높이의 2%
MIN_PAGES_FOR_REPEAT = 3
CONFIDENCE = {"paragraph": 0.7, "list_item": 0.7, "heading": 0.6, "page_header": 0.8, "page_footer": 0.8,
              "table": 0.6}
# 스펙 §5.2-5의 앞머리 + 공공누리에서 본 글머리표(ㅇ ㆍ · ∙ ‣ ▸ ▪ ⇨ →). 차례 글자는 가~하 열네 글자뿐
# (유니코드 범위 가-하가 아니다). 숫자 뒤에 "10. "·"10.03"처럼 숫자 차례가 또 오면 날짜("2026. 10. 3.",
# "2026. 10.03.")라 표지가 아니다. "1.5배"처럼 숫자 차례 뒤가 숫자·공백이 아니면 표지다("2. 1.5배 증가").
LIST_MARKER = re.compile(
    r"^\s*(\d+[.)](?!\s*\d+[.)](?:\s|$|\d+[.)]))|\(\d+\)|[가나다라마바사아자차카타파하][.)]|[①-⑳]"
    r"|[□■○●◦◆◇▶▷\-–•※ㅇㆍ·∙‣▸▪⇨→])\s")
_OPENERS = frozenset("(〈《「『[［（\"“'‘")  # 여는 괄호·따옴표는 앞머리가 아니다("( 단위: 백만원 )")
_DIGITS = re.compile(r"\d")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Fragment:
    """줄 조각. 좌표는 읽기 좌표 pt(axes 방향, 바로 선 글자면 보이는 쪽 좌표), size는 0.5pt 단위 대표 크기
    (가장 많은 크기, 같으면 큰 쪽)."""

    page: int
    text: str
    chars: tuple[Char, ...]  # 공백이 아닌 글자(읽기 좌표 0~1)
    x0: float
    y0: float
    x1: float
    y1: float
    baseline: float
    size: float
    bold: bool  # 글자 과반이 굵다
    content_x0: float  # 앞머리(목록 표지 또는 글자·숫자가 아닌 한 글자) 다음 글자의 시작. 앞머리가 없으면 x0
    axes: Axes = UPRIGHT


def step(size: float) -> float:
    return round(size / SIZE_STEP) * SIZE_STEP


def frame_size(page: PageText, axes: Axes) -> tuple[float, float]:
    """읽기 좌표의 (너비, 높이) pt. 진행 방향이 보이는 쪽의 세로면 너비·높이가 바뀐다."""
    w, h = page.width_pt, page.height_pt
    return (w, h) if axes[0] % 2 == 0 else (h, w)


def _span(lo: float, hi: float, axis: int) -> tuple[float, float]:
    """보이는 쪽 축 위 구간(0~1) → axis 방향(+x·+y·−x·−y = 0·1·2·3)으로 잰 구간. 같은 식이 역변환이다."""
    return (lo, hi) if axis < 2 else (1 - hi, 1 - lo)


def _turn(c: Char) -> Char:
    """글자 상자를 읽기 좌표(0~1)로 옮긴다. baseline은 이미 읽기 좌표다."""
    if c.axes == UPRIGHT:
        return c
    (x0, x1), (y0, y1) = (_span(*((c.x0, c.x1) if axis % 2 == 0 else (c.y0, c.y1)), axis) for axis in c.axes)
    return replace(c, x0=x0, y0=y0, x1=x1, y1=y1)


def _fragment(page: PageText, chars: Sequence[Char], axes: Axes) -> Fragment | None:
    """공백 글자 없이 벌어진 곳에 공백을 끼운다. 간격에서 보통 자간을 뺀 값으로 본다: 보통 자간은 글자·숫자끼리
    간격의 중앙값(음수일 때만, 목차 점선 같은 기호는 빼고). 공백만 있으면 None. chars는 읽기 좌표."""
    w, h = frame_size(page, axes)
    gaps = sorted((b.x0 - a.x1) * w for a, b in zip(chars, chars[1:]) if a.text.isalnum() and b.text.isalnum())
    # 중앙값. 작은 쪽 중앙값은 실제 문서에서 남는 공백을 늘려 쓰지 않음. 자간을 좁힌 문서(한글 -25% 등)
    tracking = min(gaps[len(gaps) // 2], 0.0) if gaps else 0.0
    parts = [chars[0].text]
    for a, b in zip(chars, chars[1:]):
        if (not a.text.isspace() and not b.text.isspace()
                and (b.x0 - a.x1) * w - tracking > SPACE_GAP * max(a.size, b.size)):
            parts.append(" ")
        parts.append(b.text)
    text = "".join(parts).strip()
    ink = tuple(c for c in chars if not c.text.isspace())
    if not ink:
        return None
    sizes = Counter(step(c.size) for c in ink)
    skip = _marker_length(text)
    return Fragment(page=page.page, text=text, chars=ink,
                    x0=min(c.x0 for c in ink) * w, y0=min(c.y0 for c in ink) * h,
                    x1=max(c.x1 for c in ink) * w, y1=max(c.y1 for c in ink) * h,
                    baseline=ink[0].baseline * h, size=max(sizes, key=lambda s: (sizes[s], s)),
                    bold=sum(c.bold for c in ink) * 2 > len(ink),
                    content_x0=ink[skip if skip < len(ink) else 0].x0 * w, axes=axes)


def _marker_length(text: str) -> int:
    """줄 앞머리의 (공백 아닌) 글자 수: 목록 표지, 또는 공백이 뒤따르는 글자·숫자가 아닌 한 글자(✅ ➊ * 등,
    여는 괄호·따옴표는 빼고). 없으면 0."""
    marker = LIST_MARKER.match(text)
    if marker:
        return len(_SPACES.sub("", marker.group(0)))
    head = text.split(maxsplit=1)
    return 1 if (len(head) == 2 and len(head[0]) == 1 and not head[0].isalnum()
                 and head[0] not in _OPENERS) else 0


def fragments(page: PageText) -> list[Fragment]:
    """보이는 글자를 읽는 방향(axes)별로 나눠 읽기 좌표에서 기준선으로 줄에 묶고, 줄 안의 큰 간격에서 나눈다.
    순서는 방향마다 위→아래, 왼쪽→오른쪽(읽기 좌표). 방향은 글자가 많은 것부터(같으면 axes 순)."""
    by_axes: dict[Axes, list[Char]] = defaultdict(list)
    for c in page.chars:
        if not c.invisible:
            by_axes[c.axes].append(_turn(c))
    out: list[Fragment] = []
    for axes in sorted(by_axes, key=lambda a: (-len(by_axes[a]), a)):
        w, h = frame_size(page, axes)
        lines: list[list[Char]] = []
        for c in sorted(by_axes[axes], key=lambda c: (c.baseline, c.x0)):
            anchor = lines[-1][0] if lines else None
            if anchor is None or (c.baseline - anchor.baseline) * h > SAME_LINE * max(c.size, anchor.size):
                lines.append([])
            lines[-1].append(c)
        for line in lines:
            line.sort(key=lambda c: c.x0)
            start = 0
            for i in range(1, len(line) + 1):
                if i == len(line) or (line[i].x0 - line[i - 1].x1) * w > SPLIT_GAP * max(
                        line[i].size, line[i - 1].size):
                    fragment = _fragment(page, line[start:i], axes)
                    if fragment is not None:
                        out.append(fragment)
                    start = i
    return out


def body_size(pages: Sequence[Sequence[Fragment]]) -> float | None:
    """문서 전체에서 글자 수가 가장 많은 크기(0.5pt 단위). 같으면 작은 쪽. 글자가 없으면 None."""
    counts = Counter(step(c.size) for frags in pages for f in frags for c in f.chars)
    return max(counts, key=lambda s: (counts[s], -s)) if counts else None


def _center(f: Fragment, page: PageText) -> float:
    """읽기 좌표에서 조각의 세로 중심(0~1)."""
    return (f.y0 + f.y1) / 2 / frame_size(page, f.axes)[1]


def _zone(f: Fragment, page: PageText) -> str | None:
    center = _center(f, page)
    if center <= MARGIN:
        return "page_header"
    if center >= 1 - MARGIN:
        return "page_footer"
    return None


def repeated_margins(pages: Sequence[PageText], frags: Sequence[Sequence[Fragment]]) -> dict[tuple[int, int], str]:
    """{(쪽 순번, 조각 순번): page_header|page_footer}. 위·아래 8% 안의 줄이 숫자를 지운 글자 기준으로
    같은 위치(±2%)에 문서 쪽 수의 절반 이상 반복되면 머리말·꼬리말. 숫자를 지우면 글자(문자)가 남지 않는 줄은
    그 쪽의 같은 영역에 그런 줄이 하나뿐일 때만 센다(쪽 번호. 표의 숫자 칸은 여럿). 3쪽 미만 문서는 없음."""
    if len(pages) < MIN_PAGES_FOR_REPEAT:
        return {}
    candidates: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
    for p, (page, page_frags) in enumerate(zip(pages, frags, strict=True)):
        for i, f in enumerate(page_frags):
            zone = _zone(f, page)
            if zone is not None:
                key = _SPACES.sub("", _DIGITS.sub("", unicodedata.normalize("NFC", f.text)))
                candidates[(zone, key)].append((p, i, _center(f, page)))
    letterless = Counter((zone, p) for (zone, key), items in candidates.items()
                         if not any(ch.isalpha() for ch in key) for p, _, _ in items)
    found: dict[tuple[int, int], str] = {}
    for (zone, key), items in candidates.items():
        if not any(ch.isalpha() for ch in key):
            items = [item for item in items if letterless[(zone, item[0])] == 1]
        for p, i, center in items:
            if len({q for q, _, c in items if abs(c - center) <= SAME_POSITION}) * 2 >= len(pages):
                found[(p, i)] = zone
    return found


def _widen(lo: float, hi: float) -> tuple[float, float]:
    """0~1 구간을 소수 셋째 자리로 반올림한다. 반올림으로 폭이 0이 되면 0.001 넓힌다(1이면 안쪽으로)."""
    a, b = (round(min(max(v, 0.0), 1.0), 3) for v in (lo, hi))
    if b > a:
        return a, b
    return (a, round(a + 0.001, 3)) if a < 1.0 else (round(b - 0.001, 3), b)


def _box(frags: Sequence[Fragment], page: PageText) -> dict[str, float]:
    """보이는 쪽 기준 0~1(읽기 좌표에서 되돌린다), 소수 셋째 자리 반올림(_widen). frags는 같은 axes."""
    axes = frags[0].axes
    w, h = frame_size(page, axes)
    spans = {axes[0] % 2: _span(min(f.x0 for f in frags) / w, max(f.x1 for f in frags) / w, axes[0]),
             axes[1] % 2: _span(min(f.y0 for f in frags) / h, max(f.y1 for f in frags) / h, axes[1])}
    (x0, x1), (y0, y1) = _widen(*spans[0]), _widen(*spans[1])
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _is_heading_size(f: Fragment, body: float) -> bool:
    return f.size >= body * (BOLD_HEADING_RATIO if f.bold else HEADING_RATIO)


def _continues(group: Sequence[Fragment], cur: Fragment, body: float) -> bool:
    """cur가 group에 이어지는가: 같은 읽는 방향, 아래 줄, 빈 간격 ≤ 줄 높이 × 0.8, 크기 차 ≤ 0.5pt, 왼쪽 시작 차
    ≤ 본문 크기(첫 줄에 앞머리가 있으면 앞머리 다음 글자의 시작도 기준: 내어쓰기), 새 목록 앞머리 아님, 제목 크기
    여부가 같음."""
    first, prev = group[0], group[-1]
    return (cur.axes == first.axes
            and cur.baseline > prev.baseline
            and cur.y0 - prev.y1 <= PARA_GAP * (prev.y1 - prev.y0)
            and abs(cur.size - prev.size) <= PARA_SIZE_DIFF
            and min(abs(cur.x0 - first.x0), abs(cur.x0 - first.content_x0)) <= PARA_INDENT * body
            and not LIST_MARKER.match(cur.text)
            and _is_heading_size(cur, body) == _is_heading_size(first, body))


def _table_corner(table: "TableSpec", page: PageText, axes: Axes) -> tuple[float, float]:
    """표 bbox(보이는 쪽 0~1)의 읽기 좌표(axes) (윗변, 왼변) pt."""
    x0, y0, x1, y1 = table.bbox
    left, _ = _span(*((x0, x1) if axes[0] % 2 == 0 else (y0, y1)), axes[0])
    top, _ = _span(*((x0, x1) if axes[1] % 2 == 0 else (y0, y1)), axes[1])
    w, h = frame_size(page, axes)
    return top * h, left * w


def _top(item: "list[Fragment] | TableSpec | OcrParagraph", page: PageText) -> float:
    """블록의 보이는 쪽 윗변(0~1)."""
    if isinstance(item, list):
        return _box(item, page)["y0"]
    return item.bbox[1]


def _merge_ocr(page_items: list[tuple[PageText, str | None, Any]], paras: Sequence[OcrParagraph],
               page: PageText) -> list[tuple[PageText, str | None, Any]]:
    """같은 쪽의 텍스트 레이어 블록(읽기 순서)과 OCR 문단(XY 분할 순서)을 윗변 기준으로 합친다. 두 목록 안의
    순서는 그대로 두고, 윗변이 같으면 텍스트 레이어 블록이 먼저다."""
    queue = deque(paras)
    out: list[tuple[PageText, str | None, Any]] = []
    for item in page_items:
        top = _top(item[2], page)
        while queue and queue[0].bbox[1] < top:
            out.append((page, None, queue.popleft()))
        out.append(item)
    return out + [(page, None, para) for para in queue]


def build_specs(pages: Sequence[PageText], states: Sequence[TextLayerState],
                tables: Sequence[Sequence["TableSpec"]] | None = None,
                ocr: Sequence[Sequence[OcrParagraph]] | None = None) -> list[dict[str, Any]]:
    """블록 명세(계약 build_blocks 입력). digital·scanned 쪽은 보이는 글자로 블록을 만들고(숨은 글자는 fragments가
    버린다), unreliable 쪽은 블록이 없다. tables는 쪽마다 표(tables.find_tables): 표 글자(char_ids)는 줄·조각에서
    빼고(본문 크기·머리말 판정에도 쓰지 않는다), 표마다 table 블록 하나를 표 윗변 위치에 끼운다(앞 문단과 잇지
    않는다). 표는 같은 읽기 방향(TableSpec.axes) 조각 사이에, 그 방향 조각이 없으면 쪽의 첫 방향 조각 사이에
    그 방향 읽기 좌표의 윗변으로 끼운다(조각이 없는 쪽은 표 자신의 방향). 윗변이 같으면 그 좌표의 왼쪽 표가
    먼저다. 그 방향 조각보다 아래인 표는 그 방향 조각 끝(다음 방향 조각 앞)에 둔다. 순서: 쪽 → 위→아래 → 왼→오.
    ocr는 쪽마다 OCR 문단(scan.ocr_pages): 문단 블록(text_source="ocr")으로 그 쪽 블록 사이에 윗변 기준으로 끼우고,
    section_path는 앞 블록을 따른다. 제목·목록·머리말 판정과 본문 크기에는 쓰지 않는다."""
    found = list(tables) if tables is not None else [[] for _ in pages]
    read = list(ocr) if ocr is not None else [[] for _ in pages]
    frags: list[list[Fragment]] = []
    for page, state, page_tables in zip(pages, states, found, strict=True):
        taken = {i for t in page_tables for i in t.char_ids}
        rest = replace(page, chars=tuple(c for i, c in enumerate(page.chars) if i not in taken)) if taken else page
        frags.append(fragments(rest) if state != "unreliable" else [])
    body = body_size(frags)
    margins = repeated_margins(pages, frags)
    # (쪽, 머리말·꼬리말 종류, 조각 묶음 또는 표 또는 OCR 문단)
    items: list[tuple[PageText, str | None, list[Fragment] | TableSpec | OcrParagraph]] = []
    for p, (page, page_frags, page_tables, state, paras) in enumerate(
            zip(pages, frags, found, states, read, strict=True)):
        start = len(items)
        present = {f.axes for f in page_frags}
        fallback = page_frags[0].axes if page_frags else UPRIGHT
        placed: dict[Axes, list[tuple[tuple[float, float], TableSpec]]] = defaultdict(list)  # 방향 → ((윗변, 왼변), 표)
        for t in page_tables if state != "unreliable" else []:
            axes = t.axes if t.axes in present or not page_frags else fallback
            placed[axes].append((_table_corner(t, page, axes), t))
        queues = {axes: deque(sorted(q, key=lambda item: item[0])) for axes, q in placed.items()}
        for i, f in enumerate(page_frags):
            if i and f.axes != page_frags[i - 1].axes:  # 앞 방향 조각이 끝났다: 그 방향에 남은 표를 먼저
                items += [(page, None, t) for _, t in queues.pop(page_frags[i - 1].axes, ())]
            queue = queues.get(f.axes)
            while queue and queue[0][0][0] <= f.y0:
                items.append((page, None, queue.popleft()[1]))
            margin = margins.get((p, i))
            last = items[-1] if items else None
            if (margin is None and last is not None and last[0] is page and last[1] is None
                    and isinstance(last[2], list) and body is not None and _continues(last[2], f, body)):
                last[2].append(f)
            else:
                items.append((page, margin, [f]))
        items += [(page, None, t) for queue in queues.values() for _, t in queue]
        if paras and state != "unreliable":
            items[start:] = _merge_ocr(items[start:], paras, page)

    def is_heading(group: Sequence[Fragment]) -> bool:
        return body is not None and _is_heading_size(group[0], body) and len(group) <= HEADING_MAX_LINES

    sizes = sorted({g[0].size for _, m, g in items if isinstance(g, list) and m is None and is_heading(g)},
                   reverse=True)
    specs: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []  # (단계, 제목 글자)
    for page, margin, group in items:
        extra: dict[str, Any] = {}
        if isinstance(group, OcrParagraph):
            (x0, x1), (y0, y1) = _widen(group.bbox[0], group.bbox[2]), _widen(group.bbox[1], group.bbox[3])
            specs.append({"kind": "paragraph", "text": unicodedata.normalize("NFC", group.text),
                          "section_path": tuple(t for _, t in stack), "confidence": group.confidence,
                          "state": "det", "text_source": "ocr",
                          "locator": {"kind": "page", "page": page.page,
                                      "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}})
            continue
        if not isinstance(group, list):
            (x0, x1), (y0, y1) = _widen(group.bbox[0], group.bbox[2]), _widen(group.bbox[1], group.bbox[3])
            specs.append({"kind": "table", "table": group.table, "text": group.table.plain_text(),
                          "section_path": tuple(t for _, t in stack), "confidence": CONFIDENCE["table"],
                          "state": "det", "text_source": "text_layer",
                          "locator": {"kind": "page", "page": page.page,
                                      "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}})
            continue
        if margin is not None:
            kind, text, path = margin, group[0].text, ()
        elif is_heading(group):
            kind, text = "heading", unicodedata.normalize("NFC", " ".join(f.text for f in group))
            extra["level"] = min(sizes.index(group[0].size) + 1, MAX_LEVEL)
            while stack and stack[-1][0] >= extra["level"]:
                stack.pop()
            path = tuple(t for _, t in stack)
            stack.append((extra["level"], text))
        else:
            kind = "list_item" if LIST_MARKER.match(group[0].text) else "paragraph"
            text, path = "\n".join(f.text for f in group), tuple(t for _, t in stack)
        text = unicodedata.normalize("NFC", text)
        if not text.strip():
            continue
        specs.append({"kind": kind, "text": text, "section_path": path, "confidence": CONFIDENCE[kind],
                      "state": "det", "text_source": "text_layer",
                      "locator": {"kind": "page", "page": page.page, "bbox": _box(group, page)}, **extra})
    return specs
