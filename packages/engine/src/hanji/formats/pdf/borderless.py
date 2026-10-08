"""선 없는 표(순수 함수: PDFium·onnxruntime을 부르지 않는다). 바로 선(UPRIGHT) 보이는 글자의 정렬만으로 상자 안
격자를 복원하고(recover), 쪽에서 표 후보 상자를 찾는다(candidates). 길이 단위는 보이는 쪽 pt(원점 왼쪽 위).

스펙 §4.5·§4.6(스파이크 A 최종 설정, hybrid 없음)을 옮겼다. 기준값은 상자(쪽)의 본문 크기 s(공백 아닌 글자 크기의
최빈값, 같으면 작은 쪽)의 배수다. 표본 안 값이다(FinTabNet·PubTables·DART 정답 표로 정했다).
글자는 원본 (page.chars 순번, Char, 상자)로 끝까지 들고 다닌다: 표의 char_ids는 칸에 실제로 넣은 글자뿐이다.
recover: 글자 → 줄 → 낱말 → 덩이 → 열 → 행 → 칸 → 표준 정리. candidates: 줄을 덩이로 나눠 위에서 아래로 묶음을 키운다."""

import re
import weakref
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Literal

from hanji_contracts import (
    MAX_TABLE_CELLS, MAX_TABLE_EXPANDED_CHARS, Attempt, BBox, Cell, GateCheck, GateResult, PageLocator, RegionRecord,
    Table,
)

from .extract import UPRIGHT, Char, PageText
from .figures import TABLE_MIN_SCORE, Figure, Region, _inside, _inter, _iou, _pt
from .group import LIST_MARKER, step, unit_box
from .layout import MODEL_ID
from .tables import TableSpec, _cell_text, _dominant_axes, _merge, _reading_segs, _Seg
from .triage import PageMode

Box = tuple[float, float, float, float]  # 보이는 쪽 pt (x0, y0, x1, y1)
_Item = tuple[int, Char, Box]  # (page.chars 순번, 글자, 글자 상자 pt)

PAD = 1.0  # 상자를 이만큼(pt) 넓혀 글자 중심을 본다
SAME_LINE = 0.5  # 기준선 차 ≤ max(s, 글자 크기) × 0.5면 같은 줄
SCRIPT = 0.85  # 모든 낱말이 s × 0.85보다 작은 줄은 위·아래 첨자: 겹치는 줄에 붙인다
SCRIPT_OVERLAP = 0.3  # 첨자 줄 높이의 이 비율 이상 세로로 겹칠 때
WORD_GAP = 0.2  # 공백 글자나 s × 0.2보다 넓은 틈에서 낱말을 나눈다
LETTER_GAP = 1.2  # s × 1.2 안에 붙은 낱자 비ASCII 낱말 묶음은 잇는다(한글 균등 배분 라벨)
GUTTER = 0.5  # 덩이: s × 0.5보다 좁은 틈은 잇는다. 열 틈의 최소 폭도 같다
CROSS_FRAC = 0.15  # 열 틈은 덩이가 둘 이상인 줄의 15%까지만 가로지를 수 있다
MIN_SUPPORT = 2  # 열 틈 양쪽에 덩이가 있는 줄이 이만큼 있어야 한다
ALL_LINES_GAP = 1.0  # 2차 열 틈: 아무 줄도 덮지 않는 폭 s × 1.0 이상 틈의 양쪽 줄 묶음이 위아래로 포개질 때
NEST_TOL = 0.5  # 포개짐의 여유(× s)
VRULE_COVER = 0.5  # 상자 높이의 50% 이상인 안쪽 세로선은 열 경계로 강제한다
VRULE_INSET = 2.0  # 상자 왼·오른 변에서 이만큼(pt) 안쪽 세로선만
BIMODAL = 1.15  # 줄 간격이 중앙값/1.15 이하면 칸 안 줄바꿈 후보
LEAD_TOL = 1.15  # 그리고 쪽 줄 간격 × 1.15 이하
LEAD_CAP = 1.6  # 쪽 줄 간격 상한(× s): 160%보다 넓은 간격은 칸 안 줄바꿈이 아니다
LABEL_FULL = 0.75  # 라벨 줄바꿈: 앞 줄 그 열이 열 글자 폭의 75% 이상 찼을 때
OVERLAP = 0.2  # 세로로 낮은 줄 높이의 20%보다 겹치고 열이 겹치지 않는 줄은 한 행
HEAD_RULE = 0.6  # 머리 영역: 글 폭의 60% 이상인 긴 가로선(booktabs 가운데 선) 위
CMID_DROP = 1.2  # 덩이 바로 아래 s × 1.2 안의 짧은 가로선(cmidrule)이 덮는 열까지 넓힌다
CMID_WIDTH = 0.9  # 상자 너비의 90%보다 짧은 가로선만 cmidrule
VCENTER = 0.3  # 위아래 행의 그 열이 비고 가운데(s × 0.3 안)에 놓인 행은 행 병합
CURRENCY = frozenset("$€£¥₩")  # 통화 기호만 있는 덩이는 뒤 덩이에 붙인다
CLOSERS = frozenset("%)")  # 혼자 있는 %·)는 앞 덩이에 붙인다
EPS = 1e-6
# 검출기(candidates)
CHUNK_GAP = 1.0  # 쪽 줄을 덩이로 나누는 틈(× 줄 크기)
MAX_VGAP = 4.5  # 묶음 안 줄 사이 빈 간격 상한(× s)
ALIGN = 1.0  # 덩이가 둘 이상인 줄은 열 가장자리 하나가 앞 줄과 s × 1.0 안에서 맞아야 묶음에 든다
MIN_MULTI = 2  # 덩이가 둘 이상인 줄 수
MIN_LINES = 2  # 묶음 줄 수
PROSE_WIDTH = 0.30  # 2단 산문: 첫 틈 양쪽 중 좁은 쪽 중앙값 > 쪽 너비 × 0.3
LIST_FRAC = 0.6  # 목록: 2열 묶음 왼쪽 덩이의 60% 이상이 목록 표지
SENTENCE_FRAC = 0.6  # 긴 문장: 덩이가 둘 이상인 줄의 60% 이상이 양쪽 모두 25자 이상
SENTENCE_CHARS = 25
CHART_RULES = 4  # 차트: 열 틈 밖에 세로 채움 선 조각 4개 이상
WIDE_CHUNK = 0.25  # 쪽 단 틈: 쪽 너비 25% 이상인 덩이 둘 사이 틈이
PAGE_GUTTER_LINES = 3  # 3줄 이상에서 겹치면 쪽 단 틈
_MARK = re.compile(r"^\s*(\d+[.)]?|\(\d+\)|[가-하][.)]|[①-⑳]|[□■○●◦◆◇▶▷\-–•※ㅇㆍ·∙‣▸▪⇨→*†‡§]+)\s*$")

Reason = Literal["few_chars", "under_2x2", "cells_cap", "chars_cap", "bad_grid"]


@dataclass(frozen=True, slots=True)
class Checks:
    """recover가 잰 값(못 잰 것은 None). chars는 상자 안 공백 아닌 글자 수, cells는 n_rows × n_cols, filled는 저장된
    칸(병합 칸은 하나) 중 글자 있는 칸 비율, expanded_chars는 병합 칸 글자를 덮인 칸마다 펼친 총 글자 수."""

    chars: int
    rows: int | None = None
    cols: int | None = None
    cells: int | None = None
    filled: float | None = None
    expanded_chars: int | None = None


@dataclass(frozen=True, slots=True)
class Recovery:
    """복원한 표 하나. bbox는 칸에 넣은 글자 상자의 바깥 상자(보이는 쪽 pt), table의 header는 모두 none,
    char_ids는 칸에 넣은 page.chars 순번(공백 포함, 각 순번은 정확히 한 칸에만)."""

    bbox: Box
    table: Table
    char_ids: frozenset[int]
    checks: Checks


@dataclass(frozen=True, slots=True)
class Failure:
    """복원 실패: 글자 소유는 바뀌지 않는다. checks는 실패할 때까지 잰 값."""

    reason: Reason
    checks: Checks


@dataclass(slots=True)
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    base: float  # 줄의 기준선(pt)
    size: float
    ink: list[_Item]  # 공백 아닌 글자(x 순)
    spaces: list[_Item] = field(default_factory=list)  # 이 낱말 뒤(줄 첫 낱말이면 앞도) 공백 글자: 소유만


@dataclass(slots=True)
class _Line:
    words: list[_Word]
    y0: float = 0.0
    y1: float = 0.0
    base: float = 0.0
    size: float = 0.0

    def fix(self) -> "_Line":
        self.words.sort(key=lambda w: w.x0)
        self.y0 = min(w.y0 for w in self.words)
        self.y1 = max(w.y1 for w in self.words)
        self.base = Counter(round(w.base, 1) for w in self.words).most_common(1)[0][0]
        self.size = max(w.size for w in self.words)
        return self


_Chunk = list  # [x0, x1, [_Word, ...]] (덩이, 고쳐 쓴다)


@dataclass(frozen=True, slots=True)
class _Context:
    """쪽 하나의 바로 선 보이는 글자(공백 포함)·모은 선분·쪽 줄 간격(× 크기, 없으면 None)."""

    chars: tuple[_Item, ...]
    hrules: tuple[_Seg, ...]
    vrules: tuple[_Seg, ...]
    leading: float | None
    width: float
    height: float


_last: tuple["weakref.ref[PageText]", _Context] | None = None  # 마지막 쪽(약한 참조): 같은 쪽을 다시 훑지 않는다


def _context(page: PageText) -> _Context:
    global _last
    last = _last
    if last is not None and last[0]() is page:
        return last[1]
    w, h = page.width_pt, page.height_pt
    chars = tuple((i, c, (c.x0 * w, c.y0 * h, c.x1 * w, c.y1 * h)) for i, c in enumerate(page.chars)
                  if not c.invisible and c.axes == UPRIGHT)
    segs = _merge(_reading_segs(page.rules, UPRIGHT, w, h))
    ctx = _Context(chars, tuple(s for s in segs if s.axis == "h"), tuple(s for s in segs if s.axis == "v"),
                   _page_leading(chars, h), w, h)
    _last = (weakref.ref(page), ctx)
    return ctx


def _page_leading(chars: Sequence[_Item], h: float) -> float | None:
    """쪽 글자의 흔한 줄 간격(× 크기): 이어진 두 줄(같은 크기, 가로로 짧은 쪽 50% 이상 겹침)의 기준선 차/크기를
    0.05 단위로 센 최빈값(같으면 작은 쪽). 그런 줄 쌍이 3개 미만이면 None."""
    groups: list[list] = []  # [기준선, 첫 기준선, x0, x1, [크기]]
    for _, c, b in sorted(((i, c, b) for i, c, b in chars if not c.text.isspace()), key=lambda t: t[1].baseline):
        base = c.baseline * h
        if groups and base - groups[-1][0] <= 0.5 * c.size:
            g = groups[-1]
            g[2], g[3] = min(g[2], b[0]), max(g[3], b[2])
            g[4].append(step(c.size))
        else:
            groups.append([base, base, b[0], b[2], [step(c.size)]])
    ratios: Counter[float] = Counter()
    for a, b in zip(groups, groups[1:]):
        sa, sb = Counter(a[4]).most_common(1)[0][0], Counter(b[4]).most_common(1)[0][0]
        if sa != sb or sa <= 0:
            continue
        if min(a[3], b[3]) - max(a[2], b[2]) < 0.5 * min(a[3] - a[2], b[3] - b[2]):
            continue
        ratios[round((b[0] - a[0]) / sa / 0.05) * 0.05] += 1
    if sum(ratios.values()) < 3:
        return None
    return max(ratios, key=lambda k: (ratios[k], -k))


def _body(ink: Iterable[_Item]) -> float:
    """공백 아닌 글자의 0.5pt 단위 크기 최빈값(같으면 작은 쪽)."""
    sizes = Counter(step(c.size) for _, c, _ in ink)
    return max(sizes, key=lambda s: (sizes[s], -s))


def _letter(word: list[_Item]) -> bool:
    """한 글자이고 ASCII 글자·숫자가 아닌 낱말(한글·한자·기호)."""
    return len(word) == 1 and not (word[0][1].text.isascii() and word[0][1].text.isalnum())


def _words(chars: Sequence[_Item], size: float) -> list[_Word]:
    """한 줄의 글자(x 순) → 낱말. 공백 글자나 s × WORD_GAP보다 넓은 틈에서 나누고, LETTER_GAP 안에 붙은 낱자 낱말은
    잇는다. 공백 글자는 앞 낱말(줄 첫 낱말 앞이면 첫 낱말)이 가진다."""
    out: list[list[_Item]] = []
    cur: list[_Item] = []
    prev: Box | None = None
    for item in chars:
        b = item[2]
        if item[1].text.isspace():
            if cur:
                out.append(cur)
            cur, prev = [], None
            continue
        if prev is not None and b[0] - prev[2] > WORD_GAP * size:
            out.append(cur)
            cur = []
        cur.append(item)
        prev = b
    if cur:
        out.append(cur)
    joined: list[list[_Item]] = []
    run: list[list[_Item]] = []
    for word in out:
        if _letter(word) and run and word[0][2][0] - run[-1][-1][2][2] < LETTER_GAP * size:
            run.append(word)
            continue
        if run:
            joined.append([item for g in run for item in g])
        run = [word] if _letter(word) else []
        if not _letter(word):
            joined.append(word)
    if run:
        joined.append([item for g in run for item in g])
    words = [_Word(min(b[0] for _, _, b in w), min(b[1] for _, _, b in w), max(b[2] for _, _, b in w),
                   max(b[3] for _, _, b in w), w[0][1].baseline,
                   Counter(step(c.size) for _, c, _ in w).most_common(1)[0][0], w) for w in joined]
    for item in chars:
        if item[1].text.isspace() and words:
            x = (item[2][0] + item[2][2]) / 2
            owner = next((w for w in reversed(words) if w.x0 <= x), words[0])
            owner.spaces.append(item)
    return words


def _vover(a: _Line, b: _Line) -> float:
    return min(a.y1, b.y1) - max(a.y0, b.y0)


def _lines(items: Sequence[_Item], size: float, h: float) -> list[_Line]:
    """글자 → 기준선으로 묶은 줄(위→아래). 줄 기준선은 본문 크기(s × SCRIPT 이상) 글자 기준선의 중앙값. 작은 글자만
    있는 줄(위·아래 첨자)은 세로로 겹치는 앞 줄, 아니면 다음 줄에 붙인다."""
    groups: list[tuple[float, list[_Item]]] = []
    for item in sorted(items, key=lambda t: (t[1].baseline, t[2][0])):
        base = item[1].baseline * h
        if groups and base - groups[-1][0] <= SAME_LINE * max(size, item[1].size):
            groups[-1][1].append(item)
        else:
            groups.append((base, [item]))
    lines: list[_Line] = []
    for base, chars in groups:
        chars.sort(key=lambda t: t[2][0])
        words = _words(chars, size)
        if not words:
            continue
        body = sorted(c.baseline * h for _, c, _ in chars if not c.text.isspace() and c.size >= SCRIPT * size)
        if body:
            base = body[len(body) // 2]
        for w in words:
            w.base = base
        lines.append(_Line(words).fix())
    out: list[_Line] = []
    for ln in lines:
        if (all(w.size < SCRIPT * size for w in ln.words) and out
                and _vover(out[-1], ln) > SCRIPT_OVERLAP * (ln.y1 - ln.y0)):
            out[-1].words += ln.words
            out[-1].fix()
            continue
        out.append(ln)
    res: list[_Line] = []
    for i, ln in enumerate(out):
        if (all(w.size < SCRIPT * size for w in ln.words) and i + 1 < len(out)
                and _vover(out[i + 1], ln) > SCRIPT_OVERLAP * (ln.y1 - ln.y0)):
            out[i + 1].words += ln.words
            out[i + 1].fix()
            continue
        res.append(ln)
    return res


def _wtext(w: _Word) -> str:
    return "".join(c.text for _, c, _ in w.ink)


def _chunks(line: _Line, gap: float) -> list[_Chunk]:
    """줄의 낱말을 gap보다 좁은 틈에서 이은 덩이 [x0, x1, [낱말]]. 통화 기호만 있는 덩이는 뒤 덩이에, 혼자 있는
    %·)는 앞 덩이에 붙인다."""
    out: list[_Chunk] = []
    for w in line.words:
        if out and w.x0 - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], w.x1)
            out[-1][2].append(w)
        else:
            out.append([w.x0, w.x1, [w]])
    if len(out) < 2:
        return out
    res: list[_Chunk] = []
    pending: _Chunk | None = None
    for c in out:
        t = "".join(_wtext(w) for w in c[2])
        if pending is not None:
            c = [pending[0], c[1], pending[2] + c[2]]
            pending = None
        if t and set(t) <= CURRENCY:
            pending = c
            continue
        if res and t and set(t) <= CLOSERS:
            res[-1] = [res[-1][0], c[1], res[-1][2] + c[2]]
            continue
        res.append(c)
    if pending is not None:
        res.append(pending)
    return res


def _free_runs(edges: Sequence[float], lines: Sequence[Sequence[_Chunk]],
               allowed: int) -> list[tuple[float, float, tuple[float, float]]]:
    """이웃 가장자리 사이 구간 중 allowed줄 이하만 덮는 구간을 이은 것들. (a, b, (pa, pb)): (pa, pb)는 그 안에서 가장
    적게 덮인 가장 넓은 구간(열 경계는 그 가운데)."""
    spans = [(a, b) for a, b in zip(edges, edges[1:]) if b - a > EPS]
    runs: list[list] = []
    cur: list | None = None
    for (a, b), cover in zip(spans, _covers([(a + b) / 2 for a, b in spans], lines)):
        if cover <= allowed:
            if cur and abs(cur[1] - a) < EPS:
                cur[1] = b
                cur[2].append((a, b, cover))
            else:
                cur = [a, b, [(a, b, cover)]]
                runs.append(cur)
        else:
            cur = None
    out = []
    for a, b, parts in runs:
        low = min(p[2] for p in parts)
        best: list[float] | None = None
        span: list[float] | None = None
        for pa, pb, cv in parts:
            if cv == low:
                span = [span[0], pb] if span and abs(span[1] - pa) < EPS else [pa, pb]
                if best is None or span[1] - span[0] > best[1] - best[0]:
                    best = list(span)
            else:
                span = None
        out.append((a, b, (best[0], best[1])))
    return out


def _covers(mids: Sequence[float], lines: Sequence[Sequence[_Chunk]]) -> list[int]:
    """점마다 그 점을 열린 구간 안에 품는 덩이가 있는 줄 수. 줄마다 덩이를 합집합으로 이어(맞닿기만 한 끝은 잇지
    않는다) 차분 배열로 센다: 줄 수 × 덩이 수만큼 점을 다시 훑지 않는다."""
    order = sorted(range(len(mids)), key=mids.__getitem__)
    pts = [mids[i] for i in order]
    diff = [0] * (len(pts) + 1)
    for ch in lines:
        merged: list[list[float]] = []
        for lo, hi in sorted((c[0], c[1]) for c in ch if c[0] < c[1]):
            if merged and lo < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        for lo, hi in merged:
            i, j = bisect_right(pts, lo), bisect_left(pts, hi)
            if i < j:
                diff[i] += 1
                diff[j] -= 1
    out = [0] * len(pts)
    for k, n in zip(order, accumulate(diff[:-1])):
        out[k] = n
    return out


def _clear_span(a: float, b: float, chunked: Sequence[Sequence[_Chunk]]) -> tuple[float, float] | None:
    """[a, b] 안에서 어느 줄 덩이도 덮지 않는 가장 넓은 구간."""
    cov = sorted((max(c[0], a), min(c[1], b)) for ch in chunked for c in ch if c[0] < b and c[1] > a)
    best, x = None, a
    for lo, hi in [*cov, (b, b)]:
        if lo > x and (best is None or lo - x > best[1] - best[0]):
            best = (x, lo)
        x = max(x, hi)
    return best


def _nchars(chunk: _Chunk) -> int:
    return sum(len(w.ink) for w in chunk[2])


def _gutters(lines: Sequence[_Line], size: float, forced: Sequence[float]) -> list[float]:
    """열 경계(x). 1차: 덩이가 둘 이상인 줄의 x 투영에서 폭 ≥ s × GUTTER이고 그런 줄의 CROSS_FRAC 이하만 가로지르며
    양쪽에 덩이가 있는 줄(둘 다 한 글자 덩이만은 아닌)이 MIN_SUPPORT개 이상인 틈. 2차: 아무 줄도 덮지 않는 폭
    s × ALL_LINES_GAP 이상 틈의 왼쪽·오른쪽 줄 묶음이 위아래로 포개지면 경계(가운데 맞춘 행 라벨). forced는 세로선."""
    gap = GUTTER * size
    chunked = [_chunks(ln, gap) for ln in lines]
    multi = [ch for ch in chunked if len(ch) >= 2]
    xs = sorted({x for ch in chunked for c in ch for x in (c[0], c[1])})
    lo, hi = xs[0], xs[-1]
    out: list[float] = []
    if multi:
        edges = sorted({x for ch in multi for c in ch for w in c[2] for x in (w.x0, w.x1)} | {lo, hi})
        for a, b, (pa, pb) in _free_runs(edges, multi, int(CROSS_FRAC * len(multi))):
            if a <= lo + EPS or b >= hi - EPS or b - a < gap:
                continue
            support, letters = 0, 0
            for ch in multi:
                if any(c[0] < b - EPS and c[1] > a + EPS for c in ch):
                    continue
                left = [c for c in ch if c[1] <= a + EPS]
                right = [c for c in ch if c[0] >= b - EPS]
                if left and right:
                    support += 1
                    letters += _nchars(left[-1]) == 1 and _nchars(right[0]) == 1
            if support >= MIN_SUPPORT and letters < support:
                clear = _clear_span(a, b, chunked)
                out.append((clear[0] + clear[1]) / 2 if clear else (pa + pb) / 2)
    edges = sorted({x for ch in chunked for c in ch for w in c[2] for x in (w.x0, w.x1)})
    for a, b, _ in _free_runs(edges, chunked, 0):
        if a <= lo + EPS or b >= hi - EPS or b - a < ALL_LINES_GAP * size:
            continue
        if any(a - gap <= x <= b + gap for x in out):
            continue
        left = [ln for ln, ch in zip(lines, chunked) if any(c[1] <= a + EPS for c in ch)]
        right = [ln for ln, ch in zip(lines, chunked) if any(c[0] >= b - EPS for c in ch)]
        if not left or not right:
            continue
        tol = NEST_TOL * size
        ly0, ly1 = min(ln.y0 for ln in left), max(ln.y1 for ln in left)
        ry0, ry1 = min(ln.y0 for ln in right), max(ln.y1 for ln in right)
        if (ly0 >= ry0 - tol and ly1 <= ry1 + tol) or (ry0 >= ly0 - tol and ry1 <= ly1 + tol):
            out.append((a + b) / 2)
    for f in forced:
        if not any(abs(f - x) < gap for x in out):
            out.append(f)
    return sorted(out)


def _col_of(x: float, bounds: Sequence[float]) -> int:
    k = 0
    while k < len(bounds) and x >= bounds[k]:
        k += 1
    return k


def recover(page: PageText, box: Box, free: frozenset[int]) -> Recovery | Failure:
    """상자(보이는 쪽 pt, PAD만큼 넓혀 중심을 본다) 안 free 글자 가운데 바로 선 보이는 글자로 표 하나를 복원한다.
    실패 사유: 공백 아닌 글자 < 2(few_chars), 2×2 미만(under_2x2), 격자 칸 > MAX_TABLE_CELLS(cells_cap), 펼친 글자 >
    MAX_TABLE_EXPANDED_CHARS(chars_cap), 계약 Table이 거부한 격자(bad_grid: 예외 대신 실패, 문서 파싱은 이어 간다).
    빈 칸 비율은 거르지 않는다(checks.filled만 잰다)."""
    ctx = _context(page)
    x0, y0, x1, y1 = box[0] - PAD, box[1] - PAD, box[2] + PAD, box[3] + PAD
    inside = [t for t in ctx.chars if t[0] in free
              and x0 <= (t[2][0] + t[2][2]) / 2 <= x1 and y0 <= (t[2][1] + t[2][3]) / 2 <= y1]
    ink = [t for t in inside if not t[1].text.isspace()]
    if len(ink) < 2:
        return Failure("few_chars", Checks(chars=len(ink)))
    size = _body(ink)
    lines = _lines(inside, size, ctx.height)
    forced = [s.pos for s in ctx.vrules if x0 + VRULE_INSET < s.pos < x1 - VRULE_INSET
              and max(0.0, min(s.end, y1) - max(s.start, y0)) >= VRULE_COVER * (y1 - y0)]
    bounds = _gutters(lines, size, forced)
    gap = GUTTER * size
    if bounds:  # 덩이 중심이 하나도 없는 열은 버린다(강제한 세로선 열은 남긴다)
        centers = [(c[0] + c[1]) / 2 for ln in lines for c in _chunks(ln, gap)]
        changed = True
        while changed and bounds:
            changed = False
            edges = [-1e9, *bounds, 1e9]
            for k in range(1, len(edges) - 2):
                lo, hi = edges[k], edges[k + 1]
                if not any(lo <= x < hi for x in centers) and not any(lo <= f <= hi for f in forced):
                    bounds = [*bounds[:k - 1], (lo + hi) / 2, *bounds[k + 1:]]
                    changed = True
                    break
        # 첫·끝 열에 덩이 중심이 없으면 바깥 경계를 버린다(글자 밖 테두리 세로선도)
        while bounds and not any(x < bounds[0] for x in centers):
            bounds = bounds[1:]
        while bounds and not any(x >= bounds[-1] for x in centers):
            bounds = bounds[:-1]
    n_cols = len(bounds) + 1
    hr = [s for s in ctx.hrules if y0 <= s.pos <= y1 and s.end > x0 and s.start < x1]
    rows, lc = _rows(lines, bounds, hr, size, gap, (x0, x1), ctx.leading)
    grid = _grid(rows, lc, lines, size)
    n_rows = len(grid)
    checks = Checks(chars=len(ink), rows=n_rows, cols=n_cols, cells=n_rows * n_cols)
    if n_rows < 2 or n_cols < 2:
        return Failure("under_2x2", checks)
    if n_rows * n_cols > MAX_TABLE_CELLS:
        return Failure("cells_cap", checks)
    cells = _cells(page, grid, n_rows, n_cols)
    cells = _projected_row_headers(cells, n_rows, n_cols)
    cells = _stub_rowspan(cells, n_rows, n_cols)
    if len(rows) == n_rows:
        cells = _header_rowspan(cells, n_rows, n_cols, rows, lines, hr)
    expanded = sum(len(c.text) * c.rowspan * c.colspan for c in cells)
    filled = sum(1 for c in cells if c.text) / len(cells)
    checks = Checks(chars=len(ink), rows=n_rows, cols=n_cols, cells=n_rows * n_cols, filled=round(filled, 4),
                    expanded_chars=expanded)
    if expanded > MAX_TABLE_EXPANDED_CHARS:
        return Failure("chars_cap", checks)
    try:  # pydantic ValidationError도 ValueError다
        table = Table(n_rows=n_rows, n_cols=n_cols, cells=tuple(
            Cell(row=c.row, col=c.col, rowspan=c.rowspan, colspan=c.colspan, text=c.text, text_source="text_layer")
            for c in sorted(cells, key=lambda c: (c.row, c.col))))
    except ValueError:
        return Failure("bad_grid", checks)
    placed = [t for c in cells for w in c.words for t in w.ink]
    ids = frozenset(t[0] for c in cells for w in c.words for t in (*w.ink, *w.spaces))
    bbox = (min(t[2][0] for t in placed), min(t[2][1] for t in placed),
            max(t[2][2] for t in placed), max(t[2][3] for t in placed))
    return Recovery(bbox, table, ids, checks)


def _rows(lines: list[_Line], bounds: list[float], hr: list[_Seg], size: float, gap: float,
          span: tuple[float, float], leading: float | None) -> tuple[list[list[int]], list[list]]:
    """줄마다 행 하나로 시작해 합친다(가로선을 넘어서는 합치지 않는다): (a) 세로로 겹치고 열이 겹치지 않는 줄,
    (b) 줄 간격이 짧은 칸 안 줄바꿈(머리 영역은 머리줄로), (c) 라벨 줄바꿈. 반환: (행마다 줄 순번, 줄마다
    [((c0, c1), 낱말)])."""
    x0, x1 = span

    def line_cells(ln: _Line) -> list:
        """줄의 덩이 → 덩이가 가로지르는 열 범위. 덩이 바로 아래 짧은 가로선(cmidrule)이 덮는 열까지 넓힌다."""
        out: dict[tuple[int, int], list[_Word]] = {}
        for a, b, ws in _chunks(ln, gap):
            c0, c1 = _col_of(a + 1e-3, bounds), _col_of(b - 1e-3, bounds)
            c1 = max(c1, c0)
            for t in hr:
                if (ln.y1 - 0.5 <= t.pos <= ln.y1 + CMID_DROP * size and t.start <= a + 1 and t.end >= b - 1
                        and t.end - t.start < CMID_WIDTH * (x1 - x0)):
                    r0, r1 = _col_of(t.start + gap, bounds), _col_of(t.end - gap, bounds)
                    if r0 <= c0 and r1 >= c1 and (r0, r1) != (c0, c1):
                        c0, c1 = r0, r1
                    break
            out.setdefault((c0, c1), []).extend(ws)
        merged: list = []
        for k in sorted(out):
            if merged and k[0] <= merged[-1][0][1]:
                merged[-1] = ((merged[-1][0][0], max(merged[-1][0][1], k[1])), merged[-1][1] + out[k])
            else:
                merged.append((k, list(out[k])))
        return merged

    lc = [line_cells(ln) for ln in lines]
    occ = [{c for (c0, c1), _ in cells for c in range(c0, c1 + 1)} for cells in lc]
    order = sorted(range(len(lines)), key=lambda i: (lines[i].y0 + lines[i].y1) / 2)
    rows = [[order[0]]]

    def mid(i: int) -> float:
        return (lines[i].y0 + lines[i].y1) / 2

    def rule_between(a: int, b: int) -> bool:
        return any(min(mid(a), mid(b)) < s.pos < max(mid(a), mid(b)) for s in hr)

    pitches = [lines[order[k + 1]].base - lines[order[k]].base for k in range(len(order) - 1)]
    free = sorted(p for k, p in enumerate(pitches) if not rule_between(order[k], order[k + 1]) and p > 0.5 * size)
    wrap = free[len(free) // 2] / BIMODAL if len(free) >= 2 else None
    ext: dict[int, list[float]] = {}  # 열마다 한 열 덩이의 글자 가로 범위
    for cells in lc:
        for (c0, c1), ws in cells:
            if c0 == c1:
                lo, hi = min(w.x0 for w in ws), max(w.x1 for w in ws)
                e = ext.setdefault(c0, [lo, hi])
                e[0], e[1] = min(e[0], lo), max(e[1], hi)

    def chunk_in(cells: list, c: int) -> list[_Word] | None:
        return next((ws for (c0, c1), ws in cells if c0 == c1 == c), None)

    def full(cells: list, c: int) -> bool:
        ws = chunk_in(cells, c)
        if ws is None or c not in ext or ext[c][1] - ext[c][0] <= 0:
            return False
        return max(w.x1 for w in ws) - ext[c][0] >= LABEL_FULL * (ext[c][1] - ext[c][0])

    def continues(prev: list, cur: list, c: int) -> bool:
        a, b = chunk_in(prev, c), chunk_in(cur, c)
        return a is not None and b is not None and min(w.x0 for w in b) >= min(w.x0 for w in a) - 0.5

    lead = min(leading, LEAD_CAP) * size if leading else (min(free[0], LEAD_CAP * size) if free else None)
    lead_max = lead * LEAD_TOL if lead else None
    head_end = 0
    xl, xr = min(ln.words[0].x0 for ln in lines), max(ln.words[-1].x1 for ln in lines)
    for k in range(1, len(order)):
        ca, cb = mid(order[k - 1]), mid(order[k])
        if any(ca < t.pos < cb and min(t.end, xr) - max(t.start, xl) >= HEAD_RULE * (xr - xl) for t in hr):
            head_end = k - 1 if k <= len(order) // 2 else 0
            break
    for k in range(1, len(order)):
        i, prev = order[k], rows[-1]
        j = prev[-1]
        row_occ = set().union(*(occ[t] for t in prev))
        merge = False
        if not rule_between(j, i) and occ[i]:
            ov = min(lines[j].y1, lines[i].y1) - max(lines[j].y0, lines[i].y0)
            short = wrap is not None and pitches[k - 1] < wrap
            near = lead_max is None or pitches[k - 1] <= lead_max
            if ov > OVERLAP * min(lines[j].y1 - lines[j].y0, lines[i].y1 - lines[i].y0) and not (occ[i] & occ[j]):
                merge = True
            elif short and k <= head_end:
                merge = True  # booktabs 가운데 선 위의 머리줄
            elif (short and occ[i] <= row_occ and near
                  and all(continues(lc[j], lc[i], c) for c in occ[i] if c in occ[j])):
                merge = True  # 행 간격이 넓은 표에서 짧은 간격: 칸 안 줄바꿈
            elif len(bounds) > 0 and near:
                if occ[i] < row_occ and all(full(lc[j], c) and continues(lc[j], lc[i], c) for c in occ[i]):
                    merge = True  # 위 맞춤 줄바꿈: 앞 줄이 찬 열에서만 이어진다
                elif len(prev) == 1 and occ[j] < occ[i] and all(full(lc[j], c) and continues(lc[j], lc[i], c)
                                                                 for c in occ[j]):
                    merge = True  # 아래 맞춤 줄바꿈: 라벨이 넘어가고 값은 마지막 줄에
        if merge:
            prev.append(i)
        else:
            rows.append([i])
    return rows, lc


def _grid(rows: list[list[int]], lc: list[list], lines: list[_Line], size: float) -> list[list[list]]:
    """행마다 칸 범위 [c0, c1, 낱말, rowspan]. 위아래 행의 그 열이 비고 가운데 놓인 행은 행 병합으로 옮긴다."""
    grid: list[list[list]] = []
    for r in rows:
        spans = sorted(([c0, c1, list(ws), 1] for i in r for (c0, c1), ws in lc[i]), key=lambda s: (s[0], s[1]))
        merged: list[list] = []
        for s in spans:
            if merged and s[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], s[1])
                merged[-1][2] += s[2]
            else:
                merged.append(s)
        grid.append(merged)
    top = [min(lines[i].y0 for i in r) for r in rows]
    bot = [max(lines[i].y1 for i in r) for r in rows]
    def used(k: int) -> set[int]:
        """k행에서 칸 범위가 차지한 열(위 행에서 행 병합으로 내려온 칸 포함)."""
        return {c for j in range(k + 1) for s in grid[j] if j + s[3] > k for c in range(s[0], s[1] + 1)}

    r = 1
    while r < len(grid) - 1:
        cols = {c for s in grid[r] for c in range(s[0], s[1] + 1)}
        middle = (top[r] + bot[r]) / 2
        best, a = 0, 1
        while r - a >= 0 and r + a < len(grid):
            up, dn = used(r - a), used(r + a)
            if (cols & up) or (cols & dn) or not up or not dn:
                break
            if abs(middle - (top[r - a] + bot[r + a]) / 2) <= VCENTER * size:
                best = a
            a += 1
        if best and all(s[3] == 1 for s in grid[r]):
            lo = r - best
            for j in range(r):  # 행 r을 지우면 그 행을 덮던 위 행 병합 칸은 한 행 줄어든다
                for s in grid[j]:
                    if j + s[3] > r:
                        s[3] -= 1
            for s in grid[r]:
                s[3] = 2 * best  # 행 r을 지운 뒤 lo .. lo + 2 × best − 1 행
                grid[lo].append(s)
            grid[lo].sort(key=lambda s: s[0])
            del grid[r], top[r], bot[r]
            continue
        r += 1
    return grid


@dataclass(slots=True)
class _Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str
    words: list[_Word]


def _cells(page: PageText, grid: list[list[list]], n_rows: int, n_cols: int) -> list[_Cell]:
    """칸 범위 → 칸(글자는 tables._cell_text: 줄은 \\n, 같은 줄 조각은 공백, NFC). 덮이지 않은 자리는 빈 칸."""
    cells: list[_Cell] = []
    covered: set[tuple[int, int]] = set()
    for r, spans in enumerate(grid):
        for c0, c1, ws, rs in spans:
            rs = min(rs, n_rows - r)
            chars = [t[1] for w in ws for t in sorted((*w.ink, *w.spaces), key=lambda t: t[0])]
            cells.append(_Cell(r, c0, rs, c1 - c0 + 1, _cell_text(page, chars), ws))
            covered |= {(r + i, c) for i in range(rs) for c in range(c0, c1 + 1)}
    cells += [_Cell(r, c, 1, 1, "", []) for r in range(n_rows) for c in range(n_cols) if (r, c) not in covered]
    return cells


def _projected_row_headers(cells: list[_Cell], n_rows: int, n_cols: int) -> list[_Cell]:
    """0열에만 글이 있는 행(첫 행 제외)은 모든 열을 덮는다. 위 행의 병합 칸이 내려와 있는 행은 그대로 둔다(겹침)."""
    own = _owners(cells)
    out: list[_Cell] = []
    by_row: dict[int, list[_Cell]] = {}
    for c in cells:
        by_row.setdefault(c.row, []).append(c)
    for r in range(n_rows):
        row = by_row.get(r, [])
        texted = [c for c in row if c.text]
        if (r > 0 and len(texted) == 1 and texted[0].col == 0 and texted[0].rowspan == 1
                and all(own[(r, k)].row == r for k in range(n_cols))):
            texted[0].colspan = n_cols
            out.append(texted[0])
        else:
            out += row
    return out


def _owners(cells: Iterable[_Cell]) -> dict[tuple[int, int], _Cell]:
    return {(c.row + i, c.col + j): c for c in cells for i in range(c.rowspan) for j in range(c.colspan)}


def _stub_rowspan(cells: list[_Cell], n_rows: int, n_cols: int) -> list[_Cell]:
    """0열 라벨은 아래 행들(다른 열에 글이 있는)의 빈 1×1 0열 칸을 흡수한다. 0열 글자·모든 열을 덮는 행·글 없는
    행에서 멈춘다."""
    own = _owners(cells)
    drop: set[int] = set()
    r = 0
    while r < n_rows:
        top = own.get((r, 0))
        if top is None or not top.text or top.colspan != 1:
            r += 1
            continue
        k = r + top.rowspan
        while k < n_rows:
            c = own.get((k, 0))
            if c is None or c.text or c.colspan != 1 or c.rowspan != 1:
                break
            if not any(own[(k, j)].text for j in range(1, n_cols) if (k, j) in own):
                break
            drop.add(id(c))
            top.rowspan += 1
            own[(k, 0)] = top
            k += 1
        r = k
    return [c for c in cells if id(c) not in drop]


def _header_rowspan(cells: list[_Cell], n_rows: int, n_cols: int, rows: list[list[int]], lines: list[_Line],
                    hr: list[_Seg]) -> list[_Cell]:
    """머리 = 표 너비의 HEAD_RULE 이상인 첫 가로선(0행 아래) 위의 행들. 머리의 빈 칸은 같은 열 아래 머리 칸과 합친다."""
    if not hr:
        return cells
    top_y = [min(lines[i].y0 for i in r) for r in rows]
    bot_y = [max(lines[i].y1 for i in r) for r in rows]
    xs0, xs1 = min(ln.words[0].x0 for ln in lines), max(ln.words[-1].x1 for ln in lines)
    n_head = 0
    for r in range(1, n_rows):
        if any(bot_y[r - 1] - 1 <= s.pos <= top_y[r] + 1
               and min(s.end, xs1) - max(s.start, xs0) >= HEAD_RULE * (xs1 - xs0) for s in hr):
            n_head = r
            break
    if n_head < 2 or n_head > n_rows // 2 + 1:
        return cells
    grid = _owners(cells)
    drop: set[int] = set()
    for col in range(n_cols):
        r = 0
        while r < n_head:
            c = grid[(r, col)]
            if not c.text and c.colspan == 1 and c.rowspan == 1:
                k = r + 1
                while k < n_head and not grid[(k, col)].text and grid[(k, col)].colspan == 1:
                    k += 1
                if k < n_head and grid[(k, col)].colspan == 1 and grid[(k, col)].rowspan == 1:
                    below = grid[(k, col)]
                    drop |= {id(grid[(t, col)]) for t in range(r, k)}
                    below.row, below.rowspan = r, k - r + 1
                    for t in range(r, k + 1):
                        grid[(t, col)] = below
                    r = k + 1
                    continue
            r += 1
    return [c for c in cells if id(c) not in drop]


# ---- 검출기


def _text(chunk: _Chunk) -> str:
    return "".join(_wtext(w) for w in chunk[2])


def _line_chunks(line: _Line, size: float) -> list[_Chunk]:
    return _chunks(line, CHUNK_GAP * line.size if line.size else CHUNK_GAP * size)


def _intersect(gutters: list[dict], free: list[tuple[float, float, float, float, float]], min_w: float,
               tol: float) -> list[dict]:
    """묶음의 열 틈마다 이 줄의 빈틈과 겹치는 것(겹친 구간으로 좁힌다) 가운데, 틈 옆 글자가 묶음 앞 줄과 tol 안에서
    맞는 것(왼쪽 열 끝, 오른쪽 열 시작, 오른쪽 열 끝)만 남긴다."""
    out = []
    for g in gutters:
        for f in free:
            a, b = max(g["lo"], f[0]), min(g["hi"], f[1])
            if b - a < min_w:
                continue
            if not any(abs(f[2] - e[0]) <= tol or abs(f[3] - e[1]) <= tol or abs(f[4] - e[2]) <= tol
                       for e in g["feats"]):
                continue
            out.append({"lo": a, "hi": b, "feats": [*g["feats"], f[2:]]})
            break
    return out


def _page_gutter(lines: list[_Line], size: float, width: float) -> tuple[float, float] | None:
    """쪽의 두 단 사이 틈: 한 줄에서 쪽 너비 WIDE_CHUNK 이상 덩이 둘 사이 틈 가운데 가장 많이 겹치는 것(3줄 이상)."""
    wide = WIDE_CHUNK * width
    gaps = []
    for ln in lines:
        ch = _line_chunks(ln, size)
        gaps += [(a[1], b[0]) for a, b in zip(ch, ch[1:]) if a[1] - a[0] >= wide and b[1] - b[0] >= wide]
    if len(gaps) < PAGE_GUTTER_LINES:
        return None
    best, count = None, 0
    for g in gaps:
        n = sum(1 for h in gaps if h[0] < g[1] and h[1] > g[0])
        if n > count:
            best, count = g, n
    return best if count >= PAGE_GUTTER_LINES else None


def candidates(page: PageText, free: frozenset[int]) -> list[Box]:
    """free 글자(바로 선, 보이는) 가운데 선 없는 표로 보이는 영역 상자(보이는 쪽 pt, 위→아래). 줄을 s × CHUNK_GAP
    틈에서 덩이로 나누고 위에서 아래로 묶음을 키운 뒤, 끝의 0열 꼬리 줄을 떼고 산문·목록·긴 문장·차트·쪽 단을 버린다."""
    ctx = _context(page)
    chars = [t for t in ctx.chars if t[0] in free]
    ink = [t for t in chars if not t[1].text.isspace()]
    if not ink:
        return []
    size = _body(ink)
    lines = sorted(_lines(chars, size, ctx.height), key=lambda ln: (ln.y0 + ln.y1) / 2)
    min_w = 0.5 * CHUNK_GAP * size
    page_gutter = _page_gutter(lines, size, ctx.width)
    blocks: list[dict] = []
    cur: dict | None = None
    for ln in lines:
        ch = _line_chunks(ln, size)
        multi = len(ch) >= 2
        if cur is not None and ln.y0 - cur["y1"] > MAX_VGAP * size:
            blocks.append(cur)
            cur = None
        if cur is not None:
            if multi:
                g = _intersect(cur["gutters"], [(a[1], b[0], a[1], b[0], b[1]) for a, b in zip(ch, ch[1:])], min_w,
                               ALIGN * size)
                if g:
                    cur["gutters"] = g
                    cur["lines"].append((ln, ch, True))
                    cur["y1"] = ln.y1
                    continue
            elif not any(c[0] < g["hi"] and c[1] > g["lo"] for c in ch for g in cur["gutters"]):
                cur["lines"].append((ln, ch, False))
                cur["y1"] = ln.y1
                continue
            blocks.append(cur)
            cur = None
        if multi:
            gutters = [{"lo": a[1], "hi": b[0], "feats": [(a[1], b[0], b[1])]} for a, b in zip(ch, ch[1:])
                       if b[0] - a[1] >= min_w]
            cur = {"lines": [(ln, ch, True)], "gutters": gutters, "y1": ln.y1} if gutters else None
    if cur is not None:
        blocks.append(cur)
    out = []
    for b in blocks:
        ls = _trim_col0_tail(b["lines"], b["gutters"])
        if _accepted(ls, b["gutters"], size, ctx, page_gutter):
            out.append((min(c[0] for _, ch, _ in ls for c in ch), min(ln.y0 for ln, _, _ in ls),
                        max(c[1] for _, ch, _ in ls for c in ch), max(ln.y1 for ln, _, _ in ls)))
    return out


def _trim_col0_tail(ls: list, gutters: list[dict]) -> list:
    """묶음 끝에서 덩이 하나뿐이고 첫 열 틈보다 왼쪽에서 끝나는 줄(표 아래 이어지는 문단 첫 줄)을 차례로 떼어 낸다."""
    g0 = min(g["lo"] for g in gutters)
    while ls and not ls[-1][2] and ls[-1][1][-1][1] <= g0:
        ls = ls[:-1]
    return ls


def _accepted(ls: list, gutters: list[dict], size: float, ctx: _Context,
              page_gutter: tuple[float, float] | None) -> bool:
    """덩이가 둘 이상인 줄 ≥ MIN_MULTI, 줄 ≥ MIN_LINES이고 2단 산문·목록·긴 문장·차트·쪽 단 틈이 아닌 묶음."""
    multis = [x for x in ls if x[2]]
    if len(multis) < MIN_MULTI or len(ls) < MIN_LINES:
        return False
    g0, g1 = gutters[0]["lo"], gutters[0]["hi"]
    narrow: list[float] = []
    sentences = marks = 0
    for _, ch, _ in multis:
        left = [c for c in ch if c[1] <= g0 + EPS]
        right = [c for c in ch if c[0] >= g1 - EPS]
        if not left or not right:
            continue
        narrow.append(min(left[-1][1] - left[0][0], right[-1][1] - right[0][0]))
        sentences += min(len(_text(left[-1])), len(_text(right[0]))) >= SENTENCE_CHARS
        lt = _text(left[0])
        marks += bool(_MARK.match(lt) or LIST_MARKER.match(lt + " "))
    narrow.sort()
    if narrow and narrow[len(narrow) // 2] > PROSE_WIDTH * ctx.width:
        return False
    if len(gutters) == 1 and marks >= LIST_FRAC * len(multis):
        return False
    if sentences >= SENTENCE_FRAC * len(multis):
        return False
    bx0, bx1 = min(c[0] for _, ch, _ in ls for c in ch), max(c[1] for _, ch, _ in ls for c in ch)
    by0, by1 = min(ln.y0 for ln, _, _ in ls), max(ln.y1 for ln, _, _ in ls)
    inner = sum(1 for r in ctx.vrules if r.fill and bx0 + size < r.pos < bx1 - size and r.start > by0 - size
                and r.end < by1 + size and not any(g["lo"] - size <= r.pos <= g["hi"] + size for g in gutters))
    if inner >= CHART_RULES:
        return False
    if page_gutter is not None and g0 < page_gutter[1] and g1 > page_gutter[0]:
        wide = WIDE_CHUNK * ctx.width
        sides = sum(any(c[1] - c[0] >= wide for c in ch if c[1] <= g0 + EPS or c[0] >= g1 - EPS) for _, ch, _ in ls)
        if sides * 2 >= len(ls):
            return False
    return True


# ---- 쪽 하나의 소유와 우선순위(parser가 쪽마다 한 번 부른다)

BORDERLESS_MIN_FILLED = 0.7  # 검출기 후보는 저장된 칸 중 글자 있는 칸이 70% 이상일 때만 표(빈 칸 ≤ 30%: 차트 이름표)
CONTAINED = 0.5  # 안에 든다: 안쪽 상자 넓이의 50% 이상
RULED_IOU = 0.5  # 모델 상자와 IoU ≥ 0.5인 선 있는 표는 선 있는 표가 이긴다
BORDERLESS_TABLE = "borderless_table"  # 선 없는 표 블록 하나마다(표시의 정본)
BORDERLESS_FAILED = "borderless_table_failed"  # 모델 표 상자 복원 실패(소유는 바뀌지 않았다)


def applies(page: PageText, mode: PageMode) -> bool:
    """선 없는 표를 찾는 쪽: layer 모드이고 쪽의 주된 읽기 방향이 바로 선 방향(scan 모드·회전 쪽은 시도하지 않는다)."""
    return mode == "layer" and _dominant_axes(page) == UPRIGHT


def settle(page: PageText, mode: PageMode, tables: Sequence[TableSpec], layout_tables: Sequence[Region],
           figures: Sequence[Figure]) -> tuple[list[TableSpec], list[RegionRecord]]:
    """쪽 하나의 선 없는 표(스펙 §4.3). tables는 남긴 선 있는 표, layout_tables는 모델 table 상자(보이는 쪽 pt),
    figures는 그림 정리 결과. 그림·짝 캡션·선 있는 표·확정한 선 없는 표의 글자는 쓰지 않는다. 그림·캡션 상자와 겹치는
    후보는 버린다. 1) 모델 상자(점수 높은 순): 선 있는 표와 겹치면(IoU ≥ 0.5 또는 그 표 넓이의 50% 이상이 상자 안)
    선 있는 표가 이기고, 확정한 표 안에 넓이 50% 이상이 들면 건너뛴다. 실패하면 borderless_table_failed 기록만.
    2) 검출기 후보: 확정한 표 안에 50% 이상 들면 버리고, 빈 칸 비율 ≤ 0.3일 때만 확정한다(아니면 기록 없이 버린다).
    반환: (선 있는 표 + 선 없는 표(ruled=False), 처리 이력). 시도하지 않는 쪽(applies 거짓)은 입력 그대로."""
    if not applies(page, mode):
        return list(tables), []
    protected = [f.box for f in figures] + [f.caption.box for f in figures if f.caption is not None]
    taken = {i for t in tables for i in t.char_ids}
    for f in figures:
        taken |= f.char_ids | (f.caption.char_ids if f.caption is not None else frozenset())
    free = frozenset(i for i, c in enumerate(page.chars) if not c.invisible and i not in taken)
    ruled = [_pt(page, t.bbox) for t in tables]
    made: list[Box] = []
    out = list(tables)
    notes: list[RegionRecord] = []
    counts: Counter[str] = Counter()

    def clear(box: Box) -> bool:
        return not any(_inter(box, p) > 0 for p in protected) and not any(_inside(box, m) >= CONTAINED for m in made)

    def keep(rec: Recovery, tag: str, model: bool) -> None:
        nonlocal free
        spec = _spec(page, rec)
        made.append(rec.bbox)
        out.append(spec)
        free -= rec.char_ids
        counts[tag] += 1
        notes.append(_record(page, f"borderless-{tag}-{counts[tag]}", spec.bbox, BORDERLESS_TABLE, model,
                             _gate(rec.checks, not model)))

    for region in sorted(layout_tables, key=lambda r: (-r.score, r.box[1], r.box[0])):
        box = region.box
        if (region.score < TABLE_MIN_SCORE or not clear(box)
                or any(_iou(r, box) >= RULED_IOU or _inside(r, box) >= CONTAINED for r in ruled)):
            continue
        rec = recover(page, box, free)
        if isinstance(rec, Recovery):
            keep(rec, "layout", True)
        else:
            counts["failed"] += 1
            unit = (box[0] / page.width_pt, box[1] / page.height_pt, box[2] / page.width_pt, box[3] / page.height_pt)
            notes.append(_record(page, f"borderless-failed-{counts['failed']}", unit, BORDERLESS_FAILED, True,
                                 _gate(rec.checks, False)))
    for box in candidates(page, free):
        if not clear(box):
            continue
        rec = recover(page, box, free)
        if isinstance(rec, Recovery) and rec.checks.filled >= BORDERLESS_MIN_FILLED:
            keep(rec, "rule", False)
    return out, notes


def _spec(page: PageText, rec: Recovery) -> TableSpec:
    """복원 → 표 명세(bbox는 보이는 쪽 0~1, 소수 셋째 자리: 선 있는 표와 같다)."""
    w, h = page.width_pt, page.height_pt
    x0, y0, x1, y1 = (round(min(max(v, 0.0), 1.0), 3) for v in (
        rec.bbox[0] / w, rec.bbox[1] / h, rec.bbox[2] / w, rec.bbox[3] / h))
    return TableSpec(bbox=(x0, y0, x1, y1), table=rec.table, char_ids=rec.char_ids, ruled=False)


def _gate(checks: Checks, detector: bool) -> GateResult:
    """잰 값마다 검사 하나(못 잰 값은 뺀다). 검출기 표만 filled를 본다."""
    limits = [("chars", checks.chars, ">=2", checks.chars >= 2),
              ("rows", checks.rows, ">=2", checks.rows is not None and checks.rows >= 2),
              ("cols", checks.cols, ">=2", checks.cols is not None and checks.cols >= 2),
              ("cells", checks.cells, f"<={MAX_TABLE_CELLS}",
               checks.cells is not None and checks.cells <= MAX_TABLE_CELLS),
              ("expanded_chars", checks.expanded_chars, f"<={MAX_TABLE_EXPANDED_CHARS}",
               checks.expanded_chars is not None and checks.expanded_chars <= MAX_TABLE_EXPANDED_CHARS)]
    if detector:
        limits.append(("filled", checks.filled, f">={BORDERLESS_MIN_FILLED}",
                       checks.filled is not None and checks.filled >= BORDERLESS_MIN_FILLED))
    found = tuple(GateCheck(name=name, passed=ok, value=value, threshold=limit)
                  for name, value, limit, ok in limits if value is not None)
    return GateResult(passed=all(c.passed for c in found), checks=found)


def _record(page: PageText, tag: str, box: tuple[float, float, float, float], reason: str, model: bool,
            gate: GateResult) -> RegionRecord:
    """처리 이력 한 줄(parser._region과 같은 모양). box는 보이는 쪽 0~1."""
    return RegionRecord(region_id=f"p{page.page}-{tag}", kind="table", chosen="det", fallback_reason=reason, gate=gate,
                        locator=PageLocator(page=page.page, bbox=BBox(**unit_box(*box))),
                        attempts=(Attempt(layer="det", model_id=MODEL_ID if model else None),))
