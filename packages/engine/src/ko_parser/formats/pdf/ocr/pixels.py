"""OCR 전처리 화소 연산(numpy). RapidOCR 3.9.2가 쓰는 OpenCV 연산과 같은 값을 내도록 옮겼다(opencv 없이 같은 결과).

그림은 uint8 H×W×C 배열이다."""

import numpy as np

_COEF = 2048  # OpenCV INTER_LINEAR 고정소수 계수(11비트)
_CHUNK = 1 << 16  # crop_quad가 한 번에 보간하는 화소 수(큰 상자의 메모리 상한)


def _linear_axis(n_out: int, n_in: int, clamp_frac: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fx = ((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5).astype(np.float32)
    sx = np.floor(fx).astype(np.int64)
    fx = (fx - sx).astype(np.float32)
    if clamp_frac:  # OpenCV는 가로 가장자리에서만 비율을 0으로 둔다(세로는 행 번호만 자른다)
        fx[(sx < 0) | (sx >= n_in - 1)] = 0
    c0 = np.rint((np.float32(1) - fx) * _COEF).astype(np.int64)
    c1 = np.rint(fx * np.float32(_COEF)).astype(np.int64)
    return np.clip(sx, 0, n_in - 1), np.clip(sx + 1, 0, n_in - 1), c0, c1


def resize_linear(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """cv2.resize(INTER_LINEAR)와 같은 값(일반 고정소수 경로). 크기가 같으면 복사하지 않고 입력 배열을 그대로 돌려준다."""
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    x0, x1, a0, a1 = _linear_axis(width, w, True)
    y0, y1, b0, b1 = _linear_axis(height, h, False)
    # 그림 전체를 int32로 바꾸지 않고 필요한 열만 모아서 바꾼다
    rows = (img[:, x0].astype(np.int32) * a0[None, :, None].astype(np.int32)
            + img[:, x1].astype(np.int32) * a1[None, :, None].astype(np.int32))
    b0, b1 = b0.astype(np.int32)[:, None, None], b1.astype(np.int32)[:, None, None]
    out = (((b0 * (rows[y0] >> 4)) >> 16) + ((b1 * (rows[y1] >> 4)) >> 16) + 2) >> 2
    return np.clip(out, 0, 255).astype(np.uint8)


def _cubic_weights(t: np.ndarray) -> list[np.ndarray]:
    a = np.float32(-0.75)
    t = np.float32(t)
    w0 = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
    w1 = ((a + 2) * t - (a + 3)) * t * t + 1
    w2 = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
    return [w0, w1, w2, np.float32(1) - w0 - w1 - w2]


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src 네 점 → dst 네 점의 3×3 원근 변환(cv2.getPerspectiveTransform)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    return np.append(np.linalg.solve(np.array(rows, float), np.array(rhs, float)), 1).reshape(3, 3)


def crop_quad(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """네 꼭짓점(왼쪽 위부터 시계 방향) 줄 상자를 원근 보정으로 잘라 바로 세운다(RapidOCR get_rotate_crop_image:
    양삼차 보간 A=-0.75, 가장자리 복제). 세로로 1.5배 이상 길면 90° 돌린다. 좌표를 부동소수로 계산하는 양삼차라
    OpenCV 4.x(고정소수 보간표)와 화소값이 ±2 안에서 다를 수 있다. 큰 상자도 메모리가 크게 늘지 않게 화소를 묶어 보간한다.
    넓이가 없는 상자(점·선)는 빈 그림(0×0)을 낸다."""
    c = img.shape[2]
    empty = np.zeros((0, 0, c), np.uint8)
    cw = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    ch = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if not cw or not ch:
        return empty
    std = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], float)
    try:
        m = _homography(std, pts.astype(float))
    except np.linalg.LinAlgError:  # 꼭짓점이 한 줄 위에 있다
        return empty
    n = ch * cw
    grid = np.empty((3, n))  # 출력 화소 (x, y, 1), 행 우선
    grid[0] = np.tile(np.arange(cw), ch)
    grid[1] = np.repeat(np.arange(ch), cw)
    grid[2] = 1
    p = m @ grid
    del grid
    sx, sy = (p[0] / p[2]).astype(np.float32), (p[1] / p[2]).astype(np.float32)
    del p
    h, w = img.shape[:2]
    flat = img.reshape(-1, c)
    out = np.empty((n, c), np.uint8)
    for s in range(0, n, _CHUNK):  # 화소 묶음마다, 16개 탭을 하나씩 더한다(쌓지 않는다)
        bx, by = sx[s:s + _CHUNK], sy[s:s + _CHUNK]
        ix, iy = np.floor(bx).astype(np.int64), np.floor(by).astype(np.int64)
        wx, wy = _cubic_weights(bx - ix), _cubic_weights(by - iy)
        acc = np.zeros((len(bx), c))
        for j in range(4):  # (y, x) 행 우선 탭 순서
            row = np.clip(iy + j - 1, 0, h - 1) * w
            for i in range(4):
                acc += np.take(flat, row + np.clip(ix + i - 1, 0, w - 1), axis=0) * (wy[j] * wx[i])[:, None]
        out[s:s + _CHUNK] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    out = out.reshape(ch, cw, c)
    return np.rot90(out) if ch / max(cw, 1) >= 1.5 else out
