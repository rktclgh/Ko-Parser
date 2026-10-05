"""PP-OCRv5 글자 검출: RapidOCR 3.9.2의 전처리(짧은 변 736 이상, 32 배수, 정규화)와 DB 후처리(thresh·box_thresh·
unclip_ratio·max_candidates, 2×2 팽창, 최소 넓이 사각형)를 numpy로 옮겼다. 값은 RapidOCR 기본값과 같다."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pyclipper

from .pixels import resize_linear

LIMIT_SIDE = 736  # 짧은 변이 이보다 작으면 키운다(RapidOCR limit_type=min)
DET_MAX_SIDE = 4000  # 검출 입력의 긴 변 상한(아주 길쭉한 그림에서 짧은 변을 키우다 메모리가 터지지 않게, RapidOCR에는 없다)
THRESH = 0.3  # 화소 확률 문턱
BOX_THRESH = 0.5  # 상자 안 평균 확률 문턱
UNCLIP_RATIO = 1.6
MAX_CANDIDATES = 1000  # 후보 상한: 짧은 변이 MIN_SIZE 이상인 덩어리만 센다(작은 점이 많은 쪽에서 줄을 잃지 않게)
MIN_SIZE = 3  # 상자의 짧은 변 하한(검출 출력 화소)
SAME_ROW = 10  # 정렬: 윗변 순으로 늘어놓고 이웃 윗변 차가 10px 미만이면 같은 줄로 묶어 왼쪽부터(RapidOCR sorted_boxes의
# 이웃 바꾸기와는 다르다. 쪽의 읽기 순서는 PR B scan.py가 다시 정한다)


def _hull(p: np.ndarray) -> np.ndarray:
    """볼록 껍질(Andrew monotone chain)."""
    p = np.unique(p, axis=0)
    if len(p) < 3:
        return p.astype(float)

    def half(seq: list[list[float]]) -> list[list[float]]:
        h: list[list[float]] = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]

    pl = p.tolist()
    return np.array(half(pl) + half(pl[::-1]), float)


def mini_box(points: np.ndarray) -> tuple[np.ndarray, float]:
    """cv2.minAreaRect + boxPoints + RapidOCR get_mini_boxes 꼭짓점 순서. (4×2 float32, 짧은 변)."""
    h = _hull(np.asarray(points, float).reshape(-1, 2))
    if len(h) == 1:
        return np.repeat(h, 4, 0).astype(np.float32), 0.0
    e = np.roll(h, -1, 0) - h
    e = e[np.linalg.norm(e, axis=1) > 0]
    u = e / np.linalg.norm(e, axis=1)[:, None]
    v = np.stack([-u[:, 1], u[:, 0]], 1)
    pu, pv = np.einsum("ij,kj->ik", h, u), np.einsum("ij,kj->ik", h, v)  # BLAS 행렬곱 대신(플랫폼 반올림·경고)
    area = (pu.max(0) - pu.min(0)) * (pv.max(0) - pv.min(0))
    k = int(np.argmin(area))
    p0, p1, q0, q1 = pu[:, k].min(), pu[:, k].max(), pv[:, k].min(), pv[:, k].max()
    corners = [u[k] * a + v[k] * b for a, b in ((p0, q0), (p1, q0), (p1, q1), (p0, q1))]
    pts = sorted(np.array(corners, np.float32).tolist(), key=lambda t: t[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return np.array([pts[i1], pts[i2], pts[i3], pts[i4]], np.float32), float(min(p1 - p0, q1 - q0))


def components(mask: np.ndarray) -> Iterator[np.ndarray]:
    """8-연결 덩어리마다 가로 구간 끝점들 [(x, y), ...](볼록 껍질은 윤곽선 껍질과 같다)."""
    _, w = mask.shape
    d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, xs = np.nonzero(d == 1)
    xe = np.nonzero(d == -1)[1] - 1
    n, k = len(rows), w + 2
    starts, ends = rows * k + xs, rows * k + xe
    lo = np.searchsorted(ends, (rows - 1) * k + xs - 1, "left")
    hi = np.searchsorted(starts, (rows - 1) * k + xe + 1, "right")
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in np.nonzero(hi > lo)[0].tolist():
        for j in range(lo[i], hi[i]):
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for members in groups.values():
        idx = np.array(members)
        yield np.concatenate([np.stack([xs[idx], rows[idx]], 1), np.stack([xe[idx], rows[idx]], 1)])


def _line(p: tuple[int, int], q: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """cv2.line(8-연결, 왼쪽→오른쪽 브레즌햄)의 화소 좌표."""
    (x1, y1), (x2, y2) = p, q
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx, dy, sy = x2 - x1, abs(y2 - y1), -1 if y2 < y1 else 1
    big, small = max(dx, dy), min(dx, dy)
    k = np.arange(big + 1)
    m = np.maximum(0, -((big - 2 * small * k) // (2 * big))) if big else k
    return (x1 + k, y1 + sy * m) if dx >= dy else (x1 + m, y1 + sy * k)


def fill_poly(shape: tuple[int, int], q: list[list[int]]) -> np.ndarray:
    """cv2.fillPoly(mask, [q], 1)(정수 다각형, shift=0, LINE_8). OpenCV 4.11 이상과 같다(4.14로 잼)."""
    mask = np.zeros(shape, bool)
    h, w = shape
    lo = np.full(h, 1 << 60)
    hi = np.full(h, -(1 << 60))
    for i in range(len(q)):
        (x0, y0), (x1, y1) = q[i - 1], q[i]
        xs, ys = _line((x0, y0), (x1, y1))
        k = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        mask[ys[k], xs[k]] = True
        if y0 == y1:
            continue
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        dxf = int(((x1 - x0) << 16) / (y1 - y0))  # C 정수 나눗셈처럼 0 쪽으로 자른다
        yy = np.arange(y0, y1)
        xx = (x0 << 16) + (yy - y0) * dxf
        k = (yy >= 0) & (yy < h)
        np.minimum.at(lo, yy[k], xx[k])
        np.maximum.at(hi, yy[k], xx[k])
    lo, hi = (lo + 65535) >> 16, hi >> 16  # 16.16 고정소수: 왼쪽 끝 올림, 오른쪽 끝 내림
    rows = np.nonzero(hi >= lo)[0]
    cols = np.arange(w)
    if len(rows):
        mask[rows] |= (cols[None] >= lo[rows, None]) & (cols[None] <= hi[rows, None])
    return mask


def _box_score(pred: np.ndarray, box: np.ndarray) -> float:
    """RapidOCR box_score_fast: fillPoly(box) 안 확률 평균."""
    h, w = pred.shape
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    q = (box - np.array([xmin, ymin], np.float32)).astype(np.int32).tolist()
    m = fill_poly((ymax - ymin + 1, xmax - xmin + 1), q)
    return float(pred[ymin:ymax + 1, xmin:xmax + 1][m].mean(dtype=np.float64)) if m.any() else 0.0


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray:
    x, y = box[:, 0].astype(float), box[:, 1].astype(float)
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    length = np.sum(np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y))
    off = pyclipper.PyclipperOffset()
    off.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return np.array(off.Execute(area * ratio / length)).reshape(-1, 2)


def _order_clockwise(pts: np.ndarray) -> np.ndarray:
    xs = pts[np.argsort(pts[:, 0], kind="stable"), :]
    left = xs[:2][np.argsort(xs[:2, 1], kind="stable")]
    right = xs[2:][np.argsort(xs[2:, 1], kind="stable")]
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def _input_size(h: int, w: int) -> tuple[int, int]:
    """검출 입력 (높이, 너비): 짧은 변을 736 이상으로 키우고 32 배수(최소 32). 키울 때 긴 변이 DET_MAX_SIDE를 넘지 않게
    덜 키운다. 이미 DET_MAX_SIDE보다 긴 그림은 줄이지 않는다(reader가 긴 변 reader.MAX_SIDE=2000px로 줄여 넘긴다). 빈 그림은 받지 않는다."""
    r = LIMIT_SIDE / min(h, w) if min(h, w) < LIMIT_SIDE else 1.0
    r = min(r, max(DET_MAX_SIDE / max(h, w), 1.0))
    return max(32, int(round(int(h * r) / 32) * 32)), max(32, int(round(int(w * r) / 32) * 32))


def detect(session: Any, img: np.ndarray) -> np.ndarray:
    """BGR uint8 그림 → 줄 상자 (N, 4, 2)(그림 화소, 왼쪽 위부터 시계 방향). 위→아래, 같은 줄은 왼쪽→오른쪽."""
    h, w = img.shape[:2]
    if not h or not w:
        return np.zeros((0, 4, 2), np.float32)
    rh, rw = _input_size(h, w)
    x = (resize_linear(img, rw, rh).astype(np.float32) * (1 / 255.0) - 0.5) / 0.5
    feed = {session.get_inputs()[0].name: np.ascontiguousarray(x.transpose(2, 0, 1)[None])}
    pred = session.run(None, feed)[0][0, 0]
    seg = pred > THRESH
    mask = seg.copy()
    mask[1:] |= seg[:-1]
    mask[:, 1:] |= mask[:, :-1].copy()  # cv2.dilate 2×2, 기준점 (1,1)
    bh, bw = pred.shape
    boxes = []
    candidates = 0
    for pts in components(mask):
        box, short = mini_box(pts)
        if short < MIN_SIZE:
            continue
        candidates += 1
        if candidates > MAX_CANDIDATES:
            break
        if BOX_THRESH > _box_score(pred, box):
            continue
        box, short = mini_box(_unclip(box, UNCLIP_RATIO))
        if short < MIN_SIZE + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] / bw * w), 0, w)
        box[:, 1] = np.clip(np.round(box[:, 1] / bh * h), 0, h)
        b = _order_clockwise(box.astype(np.int32))
        b[:, 0] = np.clip(b[:, 0], 0, w - 1).astype(int)
        b[:, 1] = np.clip(b[:, 1], 0, h - 1).astype(int)
        if int(np.linalg.norm(b[0] - b[1])) <= 3 or int(np.linalg.norm(b[0] - b[3])) <= 3:
            continue
        boxes.append(b)
    if not boxes:
        return np.zeros((0, 4, 2), np.float32)
    out = np.array(boxes)
    out = out[np.argsort(out[:, 0, 1], kind="stable")]
    row = np.concatenate([[0], np.cumsum(np.diff(out[:, 0, 1]) >= SAME_ROW)])
    return out[np.lexsort((out[:, 0, 0], row))]
