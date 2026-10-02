"""digital 쪽의 글자 → 줄 조각 → 블록 명세(제목·문단·목록·머리말·꼬리말). 길이 단위는 pt(원점 왼쪽 위)."""

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ko_parser_contracts import TextLayerState

from .extract import Char, PageText

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
CONFIDENCE = {"paragraph": 0.7, "list_item": 0.7, "heading": 0.6, "page_header": 0.8, "page_footer": 0.8}
# 스펙 §5.2-5의 앞머리 + 공공누리에서 본 글머리표(ㅇ ㆍ · ∙ ‣ ▸ ▪ ⇨ →)
LIST_MARKER = re.compile(r"^\s*(\d+[.)]|\(\d+\)|[가-하][.)]|[①-⑳]|[□■○●◦◆◇▶▷\-–•※ㅇㆍ·∙‣▸▪⇨→])\s")
_OPENERS = frozenset("(〈《「『[［（\"“'‘")  # 여는 괄호·따옴표는 앞머리가 아니다("( 단위: 백만원 )")
_DIGITS = re.compile(r"\d")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Fragment:
    """줄 조각. 좌표는 pt, size는 0.5pt 단위 대표 크기(가장 많은 크기, 같으면 큰 쪽)."""

    page: int
    text: str
    chars: tuple[Char, ...]  # 공백이 아닌 글자
    x0: float
    y0: float
    x1: float
    y1: float
    baseline: float
    size: float
    bold: bool  # 글자 과반이 굵다
    content_x0: float  # 앞머리(목록 표지 또는 글자·숫자가 아닌 한 글자) 다음 글자의 시작. 앞머리가 없으면 x0


def step(size: float) -> float:
    return round(size / SIZE_STEP) * SIZE_STEP


def _fragment(page: PageText, chars: Sequence[Char]) -> Fragment | None:
    """공백 글자 없이 벌어진 곳에 공백을 끼운다. 간격에서 보통 자간을 뺀 값으로 본다: 보통 자간은 글자·숫자끼리
    간격의 중앙값(음수일 때만, 목차 점선 같은 기호는 빼고). 공백만 있으면 None."""
    w, h = page.width_pt, page.height_pt
    gaps = sorted((b.x0 - a.x1) * w for a, b in zip(chars, chars[1:]) if a.text.isalnum() and b.text.isalnum())
    tracking = min(gaps[len(gaps) // 2], 0.0) if gaps else 0.0  # 자간을 좁힌 문서(한글 -25% 등)
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
                    content_x0=ink[skip if skip < len(ink) else 0].x0 * w)


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
    """보이는 글자를 기준선으로 줄에 묶고, 줄 안의 큰 간격에서 나눈다. 순서는 위→아래, 왼쪽→오른쪽."""
    chars = sorted((c for c in page.chars if not c.invisible), key=lambda c: (c.baseline, c.x0))
    lines: list[list[Char]] = []
    for c in chars:
        anchor = lines[-1][0] if lines else None
        if anchor is None or (c.baseline - anchor.baseline) * page.height_pt > SAME_LINE * max(c.size, anchor.size):
            lines.append([])
        lines[-1].append(c)
    out: list[Fragment] = []
    for line in lines:
        line.sort(key=lambda c: c.x0)
        start = 0
        for i in range(1, len(line) + 1):
            if i == len(line) or (line[i].x0 - line[i - 1].x1) * page.width_pt > SPLIT_GAP * max(
                    line[i].size, line[i - 1].size):
                fragment = _fragment(page, line[start:i])
                if fragment is not None:
                    out.append(fragment)
                start = i
    return out


def body_size(pages: Sequence[Sequence[Fragment]]) -> float | None:
    """문서 전체에서 글자 수가 가장 많은 크기(0.5pt 단위). 같으면 작은 쪽. 글자가 없으면 None."""
    counts = Counter(step(c.size) for frags in pages for f in frags for c in f.chars)
    return max(counts, key=lambda s: (counts[s], -s)) if counts else None


def _zone(f: Fragment, height: float) -> str | None:
    center = (f.y0 + f.y1) / 2 / height
    if center <= MARGIN:
        return "page_header"
    if center >= 1 - MARGIN:
        return "page_footer"
    return None


def repeated_margins(pages: Sequence[PageText], frags: Sequence[Sequence[Fragment]]) -> dict[tuple[int, int], str]:
    """{(쪽 순번, 조각 순번): page_header|page_footer}. 위·아래 8% 안의 줄이 숫자를 지운 글자 기준으로
    같은 위치(±2%)에 문서 쪽 수의 절반 이상 반복되면 머리말·꼬리말. 3쪽 미만 문서는 없음."""
    if len(pages) < MIN_PAGES_FOR_REPEAT:
        return {}
    candidates: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
    for p, (page, page_frags) in enumerate(zip(pages, frags, strict=True)):
        for i, f in enumerate(page_frags):
            zone = _zone(f, page.height_pt)
            if zone is not None:
                key = _SPACES.sub("", _DIGITS.sub("", unicodedata.normalize("NFC", f.text)))
                candidates[(zone, key)].append((p, i, (f.y0 + f.y1) / 2 / page.height_pt))
    found: dict[tuple[int, int], str] = {}
    for (zone, _), items in candidates.items():
        for p, i, center in items:
            if len({q for q, _, c in items if abs(c - center) <= SAME_POSITION}) * 2 >= len(pages):
                found[(p, i)] = zone
    return found


def _box(frags: Sequence[Fragment], page: PageText) -> dict[str, float]:
    """쪽 기준 0~1, 소수 셋째 자리 반올림. 반올림으로 폭·높이가 0이 되면 0.001 넓힌다."""
    def axis(lo: float, hi: float, full: float) -> tuple[float, float]:
        a, b = (round(min(max(v / full, 0.0), 1.0), 3) for v in (lo, hi))
        if b > a:
            return a, b
        return (a, round(a + 0.001, 3)) if a < 1.0 else (round(b - 0.001, 3), b)

    x0, x1 = axis(min(f.x0 for f in frags), max(f.x1 for f in frags), page.width_pt)
    y0, y1 = axis(min(f.y0 for f in frags), max(f.y1 for f in frags), page.height_pt)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _is_heading_size(f: Fragment, body: float) -> bool:
    return f.size >= body * (BOLD_HEADING_RATIO if f.bold else HEADING_RATIO)


def _continues(group: Sequence[Fragment], cur: Fragment, body: float) -> bool:
    """cur가 group에 이어지는가: 아래 줄, 빈 간격 ≤ 줄 높이 × 0.8, 크기 차 ≤ 0.5pt, 왼쪽 시작 차 ≤ 본문 크기
    (첫 줄에 앞머리가 있으면 앞머리 다음 글자의 시작도 기준: 내어쓰기), 새 목록 앞머리 아님, 제목 크기 여부가 같음."""
    first, prev = group[0], group[-1]
    return (cur.baseline > prev.baseline
            and cur.y0 - prev.y1 <= PARA_GAP * (prev.y1 - prev.y0)
            and abs(cur.size - prev.size) <= PARA_SIZE_DIFF
            and min(abs(cur.x0 - first.x0), abs(cur.x0 - first.content_x0)) <= PARA_INDENT * body
            and not LIST_MARKER.match(cur.text)
            and _is_heading_size(cur, body) == _is_heading_size(first, body))


def build_specs(pages: Sequence[PageText], states: Sequence[TextLayerState]) -> list[dict[str, Any]]:
    """블록 명세(계약 build_blocks 입력). scanned·unreliable 쪽은 블록이 없다. 순서: 쪽 → 위→아래 → 왼→오."""
    frags = [fragments(page) if state == "digital" else [] for page, state in zip(pages, states, strict=True)]
    body = body_size(frags)
    if body is None:
        return []
    margins = repeated_margins(pages, frags)
    groups: list[tuple[PageText, str | None, list[Fragment]]] = []  # (쪽, 머리말·꼬리말 종류, 조각)
    for p, (page, page_frags) in enumerate(zip(pages, frags)):
        for i, f in enumerate(page_frags):
            margin = margins.get((p, i))
            last = groups[-1] if groups else None
            if margin is None and last is not None and last[0] is page and last[1] is None and _continues(
                    last[2], f, body):
                last[2].append(f)
            else:
                groups.append((page, margin, [f]))

    def is_heading(group: Sequence[Fragment]) -> bool:
        return _is_heading_size(group[0], body) and len(group) <= HEADING_MAX_LINES

    sizes = sorted({g[0].size for _, m, g in groups if m is None and is_heading(g)}, reverse=True)
    specs: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []  # (단계, 제목 글자)
    for page, margin, group in groups:
        extra: dict[str, Any] = {}
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
