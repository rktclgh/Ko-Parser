"""선 있는 표 검출(순수 함수): 한 쪽의 가로·세로 선분(PageText.rules)과 글자 → TableSpec 목록. 길이 단위는 pt.

좌표는 쪽의 주된 읽기 방향(보이는 글자가 가장 많은 axes)의 읽기 좌표(원점 왼쪽 위, pt)에서 다룬다. 바로 선 쪽이면
보이는 쪽 좌표와 같다. 그 방향이 아닌 글자는 표에 넣지 않는다(문단에 그대로 남는다).
순서: 선 정리(_merge) → 영역(_regions) → 바깥 닫기·격자선(_grid_lines) → 격자(_layout) → 칸 사이 경계 판정
(_separated) → 병합(_cells) → 글자 배정 → 거르기 → 머리행."""

import bisect
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from ko_parser_contracts import MAX_TABLE_CELLS, Cell, Table

from .extract import UPRIGHT, Axes, Char, PageText, Rule
from .group import SAME_LINE, fragments, step

SNAP = 1.5  # 같은 축에서 위치 차 ≤ 1.5pt이고 끝이 이어지는 선분은 한 선으로 모은다(스펙 초기값, 채점에서 1.0~2.0 같음)
JOIN = 3.0  # 같은 위치에서 끝 사이가 ≤ 3pt면 잇는다(스펙 초기값)
INTERSECT = 2.0  # 가로·세로 선분이 ± 2pt 안에서 닿거나 교차하면 같은 영역(스펙 초기값)
EDGE_COVER = 0.5  # 격자 칸 사이 경계 길이의 이 비율 이상을 선이 덮으면 실제 선 경계(채점에서 0.3~0.7 같음)
MIN_FILLED = 0.3  # 글자 있는 칸 비율이 이보다 작으면 표가 아니다(차트·그림. 스펙 초기값, 채점에서 0.2~0.4 같음)
HEADER_COVER = 0.9  # 칸의 위·아래 변을 채운 사각형 변이 이 비율 이상 덮으면 배경 있는 칸

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 (읽기 좌표 pt)
CellPos = tuple[int, int, int, int]  # row, col, rowspan, colspan


@dataclass(frozen=True, slots=True)
class TableSpec:
    """표 하나. bbox는 보이는 쪽 기준 0~1(x0, y0, x1, y1), 소수 셋째 자리. char_ids는 표 안 글자(보이는, 쪽의
    주된 읽기 방향 글자 중 상자 중심이 표 안인 것, 공백 포함)의 page.chars 순번: 블록 묶기에서 뺄 글자.
    axes는 표를 읽은 방향(쪽의 주된 읽기 방향): 블록 순서를 정할 때 그 방향의 읽기 좌표로 윗변을 잰다."""

    bbox: tuple[float, float, float, float]
    table: Table
    char_ids: frozenset[int]
    axes: Axes = UPRIGHT


@dataclass(slots=True)
class _Seg:
    """읽기 좌표 선분. h면 pos = y, start·end = x. stroke·fill은 모은 선분 중 그 종류가 있는지."""

    axis: str
    pos: float
    start: float
    end: float
    stroke: bool
    fill: bool


class _Lines:
    """영역 선분 색인: 축마다 위치순으로 정렬해 두고 pos ± SNAP 안의 선분만 bisect로 꺼낸다(경계마다 영역 선분을
    모두 훑지 않는다)."""

    def __init__(self, segs: Iterable[_Seg]) -> None:
        self._by: dict[str, tuple[list[float], list[_Seg]]] = {}
        for axis in ("h", "v"):
            items = sorted((s for s in segs if s.axis == axis), key=lambda s: s.pos)
            self._by[axis] = ([s.pos for s in items], items)

    def cover(self, axis: str, pos: float, lo: float, hi: float, fill_only: bool = False) -> float:
        """axis 방향 선분들이 pos(± SNAP) 위치에서 [lo, hi]를 덮는 비율(fill_only면 채운 사각형 변만)."""
        keys, items = self._by[axis]
        near = items[bisect.bisect_left(keys, pos - SNAP):bisect.bisect_right(keys, pos + SNAP)]
        spans = sorted((max(s.start, lo), min(s.end, hi)) for s in near if s.fill or not fill_only)
        covered, end = 0.0, lo
        for a, b in spans:
            a = max(a, end)
            if b > a:
                covered += b - a
                end = b
        return covered / (hi - lo) if hi > lo else 0.0


@dataclass(slots=True)
class _Layout:
    """한 영역의 격자선과 글자. boxes는 공백이 아닌 글자 상자(읽기 좌표 pt), size는 영역의 본문 크기(pt).
    text_xs·text_ys는 그중 글자 정렬로 더한 격자선."""

    lines: _Lines
    xs: list[float]
    ys: list[float]
    boxes: list[Box]
    size: float
    text_xs: frozenset[float] = frozenset()
    text_ys: frozenset[float] = frozenset()


def _to_reading(x: float, y: float, axes: Axes, vw: float, vh: float) -> tuple[float, float]:
    """보이는 쪽 pt 점 → 읽기 좌표 pt 점. 축 k(+x·+y·−x·−y = 0·1·2·3) 방향 좌표는 x, y, W−x, H−y."""
    values = (x, y, vw - x, vh - y)
    return values[axes[0]], values[axes[1]]


def _to_visible(rx: float, ry: float, axes: Axes, vw: float, vh: float) -> tuple[float, float]:
    """_to_reading의 역변환."""
    out = [0.0, 0.0]
    for value, axis in ((rx, axes[0]), (ry, axes[1])):
        out[axis % 2] = value if axis < 2 else (vw if axis == 2 else vh) - value
    return out[0], out[1]


def _reading_segs(rules: Iterable[Rule], axes: Axes, vw: float, vh: float) -> list[_Seg]:
    out = []
    for r in rules:
        a = (r.start, r.pos) if r.axis == "h" else (r.pos, r.start)
        b = (r.end, r.pos) if r.axis == "h" else (r.pos, r.end)
        (x0, y0), (x1, y1) = _to_reading(*a, axes, vw, vh), _to_reading(*b, axes, vw, vh)
        stroke = r.kind == "stroke"
        if abs(y1 - y0) < abs(x1 - x0):
            out.append(_Seg("h", y0, min(x0, x1), max(x0, x1), stroke, not stroke))
        else:
            out.append(_Seg("v", x0, min(y0, y1), max(y0, y1), stroke, not stroke))
    return out


def _merge(segs: Sequence[_Seg]) -> list[_Seg]:
    """같은 축에서 위치 차 ≤ SNAP(이웃끼리 이어서)인 선분 가운데 끝이 JOIN 안으로 이어지는 것끼리 한 선분으로 잇는다.
    위치는 이은 선분들의 평균이다(같은 위치 근처의 다른 표 선분과는 섞지 않는다)."""
    out: list[_Seg] = []
    for axis in ("h", "v"):
        groups: list[list[_Seg]] = []
        for s in sorted((s for s in segs if s.axis == axis), key=lambda s: s.pos):
            if groups and s.pos - groups[-1][-1].pos <= SNAP:
                groups[-1].append(s)
            else:
                groups.append([s])
        for group in groups:
            runs: list[list[_Seg]] = []
            end = 0.0
            for s in sorted(group, key=lambda s: (s.start, s.end)):
                if runs and s.start <= end + JOIN:
                    runs[-1].append(s)
                    end = max(end, s.end)
                else:
                    runs.append([s])
                    end = s.end
            out += [_Seg(axis, sum(s.pos for s in run) / len(run), min(s.start for s in run),
                         max(s.end for s in run), any(s.stroke for s in run), any(s.fill for s in run))
                    for run in runs]
    return out


def _regions(segs: Sequence[_Seg]) -> list[list[_Seg]]:
    """서로 닿거나(± INTERSECT) 교차하는 가로·세로 선분의 묶음 중 가로선과 세로선이 모두 있는 것."""
    parent = list(range(len(segs)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    vs = sorted((i for i, s in enumerate(segs) if s.axis == "v"), key=lambda i: segs[i].pos)
    keys = [segs[j].pos for j in vs]
    for i, h in enumerate(segs):
        if h.axis != "h":
            continue
        for j in vs[bisect.bisect_left(keys, h.start - INTERSECT):bisect.bisect_right(keys, h.end + INTERSECT)]:
            v = segs[j]
            if v.start - INTERSECT <= h.pos <= v.end + INTERSECT:
                parent[root(i)] = root(j)
    groups: dict[int, list[_Seg]] = {}
    for i, s in enumerate(segs):
        groups.setdefault(root(i), []).append(s)
    return [g for g in groups.values() if any(s.axis == "h" for s in g) and any(s.axis == "v" for s in g)]


def _positions(values: Iterable[float]) -> list[float]:
    """정렬해 이웃 차 ≤ SNAP인 값끼리 평균 하나로."""
    groups: list[list[float]] = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= SNAP:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _grid_lines(region: Sequence[_Seg]) -> tuple[list[float], list[float]] | None:
    """바깥 닫기: 위·아래는 맨 위·맨 아래 가로선, 왼쪽·오른쪽은 선분들의 가장 바깥 끝(세로선이 없는 가장자리는
    가상 세로선). 안쪽 격자선은 그 사이 선분 위치들. 영역이 너무 납작하면 None."""
    hs = [s for s in region if s.axis == "h"]
    vs = [s for s in region if s.axis == "v"]
    top, bottom = min(s.pos for s in hs), max(s.pos for s in hs)
    left = min(min(s.start for s in hs), min(s.pos for s in vs))
    right = max(max(s.end for s in hs), max(s.pos for s in vs))
    if bottom - top <= SNAP or right - left <= SNAP:
        return None
    xs = [x for x in _positions(s.pos for s in vs) if left + SNAP < x < right - SNAP]
    ys = [y for y in _positions(s.pos for s in hs) if top + SNAP < y < bottom - SNAP]
    return [left, *xs, right], [top, *ys, bottom]


def _line_edge(lay: _Layout, r: int, c: int, across: bool) -> bool:
    """격자 칸 (r, c)와 오른쪽(across) 또는 아래 칸 사이 경계를 선이 EDGE_COVER 이상 덮는가."""
    if across:
        return lay.lines.cover("v", lay.xs[c + 1], lay.ys[r], lay.ys[r + 1]) >= EDGE_COVER
    return lay.lines.cover("h", lay.ys[r + 1], lay.xs[c], lay.xs[c + 1]) >= EDGE_COVER


def _separated(lay: _Layout, r: int, c: int, across: bool) -> bool:
    """경계가 실제 선이면 나뉜 칸, 아니면 합친다."""
    return _line_edge(lay, r, c, across)


def _layout(lines: _Lines, grid: tuple[list[float], list[float]], boxes: list[Box], size: float) -> _Layout:
    """선으로 정한 격자."""
    return _Layout(lines, grid[0], grid[1], boxes, size)


def _largest_rect(group: set[tuple[int, int]]) -> CellPos:
    """칸 묶음 안에 완전히 든 가장 큰 직사각형(넓이가 같으면 위·왼쪽부터)."""
    best, best_area = (0, 0, 1, 1), 0
    for r, c in sorted(group):
        width = 0
        while (r, c + width) in group:
            width += 1
        height = 0
        while width > 0:
            w = 0
            while w < width and (r + height, c + w) in group:
                w += 1
            if w == 0:
                break
            width, height = min(width, w), height + 1
            if width * height > best_area:
                best, best_area = (r, c, height, width), width * height
    return best


def _cells(n_rows: int, n_cols: int, separated: Callable[[int, int, bool], bool]) -> list[CellPos]:
    """separated(r, c, across)가 False인 격자 칸 사이를 합친다. 직사각형이 아닌 묶음은 가장 큰 직사각형부터 잡고
    나머지는 1×1."""
    parent = {(r, c): (r, c) for r in range(n_rows) for c in range(n_cols)}

    def root(k: tuple[int, int]) -> tuple[int, int]:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols and not separated(r, c, True):
                parent[root((r, c + 1))] = root((r, c))
            if r + 1 < n_rows and not separated(r, c, False):
                parent[root((r + 1, c))] = root((r, c))
    groups: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for k in parent:
        groups.setdefault(root(k), set()).add(k)
    out: list[CellPos] = []
    for group in groups.values():
        r0, c0 = min(r for r, _ in group), min(c for _, c in group)
        r1, c1 = max(r for r, _ in group), max(c for _, c in group)
        if len(group) == (r1 - r0 + 1) * (c1 - c0 + 1):
            out.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
            continue
        left = set(group)
        while left:
            r, c, h, w = _largest_rect(left)
            if h * w == 1:
                out += [(r, c, 1, 1) for r, c in sorted(left)]
                break
            out.append((r, c, h, w))
            left -= {(r + i, c + j) for i in range(h) for j in range(w)}
    return sorted(out)


def _compact(xs: list[float], ys: list[float], cells: list[CellPos]) -> tuple[list[float], list[float], list[CellPos]]:
    """어느 칸의 시작도 아닌 안쪽 격자선(합친 칸만 가로지르는 선)을 지운다."""
    starts_x, starts_y = {c for _, c, _, _ in cells}, {r for r, _, _, _ in cells}
    keep_x = [i for i in range(len(xs)) if i in (0, len(xs) - 1) or i in starts_x]
    keep_y = [i for i in range(len(ys)) if i in (0, len(ys) - 1) or i in starts_y]
    cx = {old: new for new, old in enumerate(keep_x)}
    cy = {old: new for new, old in enumerate(keep_y)}
    out = [(cy[r], cx[c], cy[r + h] - cy[r], cx[c + w] - cx[c]) for r, c, h, w in cells]
    return [xs[i] for i in keep_x], [ys[i] for i in keep_y], out


def _dominant_axes(page: PageText) -> Axes:
    counts = Counter(c.axes for c in page.chars if not c.invisible and not c.text.isspace())
    return max(counts, key=lambda a: (counts[a], a == UPRIGHT)) if counts else UPRIGHT


def _cell_text(page: PageText, chars: Sequence[Char]) -> str:
    """칸 글자를 E1-2a의 줄·조각·공백 규칙(group.fragments)으로 묶고 줄은 \\n, 같은 줄의 조각은 공백으로 잇는다. NFC."""
    parts: list[str] = []
    prev = None
    for f in fragments(replace(page, chars=tuple(chars))):
        if prev is not None:
            parts.append(" " if abs(f.baseline - prev.baseline) <= SAME_LINE * max(f.size, prev.size) else "\n")
        parts.append(f.text)
        prev = f
    return unicodedata.normalize("NFC", "".join(parts))


def _header(xs: Sequence[float], ys: Sequence[float], cells: Sequence[CellPos], lines: _Lines) -> bool:
    """맨 윗행 칸이 모두 채운 사각형(칸의 위·아래 변을 HEADER_COVER 이상 덮는 fill 변) 안에 있고, 그 아래 행 칸이
    모두 그렇지는 않으면 True."""
    def filled(r: int, c: int, h: int, w: int) -> bool:
        return (lines.cover("h", ys[r], xs[c], xs[c + w], fill_only=True) >= HEADER_COVER
                and lines.cover("h", ys[r + h], xs[c], xs[c + w], fill_only=True) >= HEADER_COVER)

    top = [cell for cell in cells if cell[0] == 0]
    below = [cell for cell in cells if cell[0] == max(h for _, _, h, _ in top)]
    return all(filled(*cell) for cell in top) and not all(filled(*cell) for cell in below)


def _build(page: PageText, region: list[_Seg], chars: Sequence[tuple[int, Char, Box]], axes: Axes) -> TableSpec | None:
    """영역 하나 → 표(2×2 이상, 글자 있는 칸 ≥ MIN_FILLED, 계약 칸 수 상한 이하) 또는 None."""
    grid = _grid_lines(region)
    if grid is None or (len(grid[0]) - 1) * (len(grid[1]) - 1) > MAX_TABLE_CELLS:
        return None
    (left, *_, right), (top, *_, bottom) = grid
    inside = [(i, c, b) for i, c, b in chars
              if left <= (b[0] + b[2]) / 2 <= right and top <= (b[1] + b[3]) / 2 <= bottom]
    ink = [(c, b) for _, c, b in inside if not c.text.isspace()]
    sizes = Counter(step(c.size) for c, _ in ink)
    size = max(sizes, key=lambda s: (sizes[s], -s)) if sizes else 10.0
    lines = _Lines(region)
    lay = _layout(lines, grid, [b for _, b in ink], size)
    if len(lay.xs) < 3 or len(lay.ys) < 3 or (len(lay.xs) - 1) * (len(lay.ys) - 1) > MAX_TABLE_CELLS:
        return None
    xs, ys, cells = _compact(lay.xs, lay.ys, _cells(len(lay.ys) - 1, len(lay.xs) - 1,
                                                    lambda r, c, across: _separated(lay, r, c, across)))
    n_rows, n_cols = len(ys) - 1, len(xs) - 1
    if n_rows < 2 or n_cols < 2:
        return None
    owner: dict[tuple[int, int], CellPos] = {}
    for cell in cells:
        r, c, h, w = cell
        owner.update({(r + i, c + j): cell for i in range(h) for j in range(w)})
    by_cell: dict[CellPos, list[Char]] = {cell: [] for cell in cells}
    for _, char, b in inside:
        col = min(max(bisect.bisect_right(xs, (b[0] + b[2]) / 2) - 1, 0), n_cols - 1)
        row = min(max(bisect.bisect_right(ys, (b[1] + b[3]) / 2) - 1, 0), n_rows - 1)
        by_cell[owner[(row, col)]].append(char)
    texts = {cell: _cell_text(page, cs) if any(not c.text.isspace() for c in cs) else ""
             for cell, cs in by_cell.items()}
    if sum(1 for t in texts.values() if t) < MIN_FILLED * len(cells):
        return None
    header = _header(xs, ys, cells, lines)
    table = Table(n_rows=n_rows, n_cols=n_cols, cells=tuple(
        Cell(row=r, col=c, rowspan=h, colspan=w, text=texts[(r, c, h, w)], text_source="text_layer",
             header="column" if header and r == 0 else "none") for r, c, h, w in cells))
    vw, vh = page.width_pt, page.height_pt
    corners = [_to_visible(x, y, axes, vw, vh) for x in (left, right) for y in (top, bottom)]
    x0, y0, x1, y1 = (round(min(max(v, 0.0), 1.0), 3) for v in (
        min(p[0] for p in corners) / vw, min(p[1] for p in corners) / vh,
        max(p[0] for p in corners) / vw, max(p[1] for p in corners) / vh))
    return TableSpec(bbox=(x0, y0, x1, y1), table=table, char_ids=frozenset(i for i, _, _ in inside), axes=axes)


def _contains(outer: tuple[float, ...], inner: tuple[float, ...]) -> bool:
    return (outer != inner and outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def find_tables(page: PageText) -> list[TableSpec]:
    """쪽의 선 있는 표(2×2 이상). 다른 표를 품은 영역(상자 안 표의 바깥 상자)은 표가 아니다. 영역이 겹쳐(서로 닿지
    않는 선끼리 바깥 닫기 상자만 겹침) 앞 표와 글자를 나눠 가지는 표는 버린다(한 글자는 한 블록에만). 순서는 보이는
    쪽에서 위→아래, 왼→오."""
    if not page.rules:
        return []
    axes = _dominant_axes(page)
    vw, vh = page.width_pt, page.height_pt
    segs = _merge(_reading_segs(page.rules, axes, vw, vh))
    chars = []
    for i, c in enumerate(page.chars):
        if c.invisible or c.axes != axes:
            continue
        (x0, y0), (x1, y1) = (_to_reading(c.x0 * vw, c.y0 * vh, axes, vw, vh),
                              _to_reading(c.x1 * vw, c.y1 * vh, axes, vw, vh))
        chars.append((i, c, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))
    found = [spec for region in _regions(segs) if (spec := _build(page, region, chars, axes)) is not None]
    found = [t for t in found if not any(_contains(t.bbox, o.bbox) for o in found)]
    kept: list[TableSpec] = []
    for t in sorted(found, key=lambda t: (t.bbox[1], t.bbox[0])):
        if not any(t.char_ids & k.char_ids for k in kept):
            kept.append(t)
    return kept
