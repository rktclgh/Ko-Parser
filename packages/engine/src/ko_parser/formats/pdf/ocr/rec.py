"""PP-OCRv5 한국어 줄 인식: 줄 그림을 높이 48로 맞추고(너비 비율이 비슷한 것끼리 6개 이하, 너비 예산 안) CTC 탐욕 디코드.
점수는 고른 글자 확률의 평균(소수 다섯째 자리, RapidOCR 3.9.2와 같다)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from .pixels import resize_linear

HEIGHT = 48
MIN_RATIO = 320 / 48  # 묶음 입력 너비 하한(너비/높이)
REC_BATCH = 6  # 한 번에 읽는 줄 수 상한(RapidOCR rec_batch_num)
REC_WIDTH_BUDGET = 6 * 3200  # 묶음 줄 수 × 입력 너비 상한: 작은 글자의 긴 줄이 모여도 출력 텐서가 수백 MB로 크지 않게


def _width(ratio: float) -> int:
    return int(HEIGHT * max(MIN_RATIO, ratio))


def _batches(order: np.ndarray, ratios: np.ndarray) -> list[list[int]]:
    """너비 순으로 늘어놓은 줄을 묶는다: REC_BATCH개 이하이고 줄 수 × 입력 너비(가장 넓은 줄) ≤ REC_WIDTH_BUDGET.
    예산보다 넓은 줄은 줄이지 않고 혼자 읽는다(줄이면 글자가 바뀐다)."""
    batches: list[list[int]] = []
    for i in order.tolist():
        last = batches[-1] if batches else None
        if last is not None and len(last) < REC_BATCH and (len(last) + 1) * _width(ratios[i]) <= REC_WIDTH_BUDGET:
            last.append(i)
        else:
            batches.append([i])
    return batches


def recognize(session: Any, symbols: Sequence[str], crops: Sequence[np.ndarray]) -> list[tuple[str, float]]:
    """줄 그림(BGR uint8)마다 (글자, 점수). symbols[0]은 CTC 빈칸, 나머지는 사전 순서 그대로."""
    ratios = np.array([c.shape[1] / float(c.shape[0]) for c in crops])
    order = np.argsort(ratios)
    out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
    name = session.get_inputs()[0].name
    for idx in _batches(order, ratios):
        width = _width(max(ratios[i] for i in idx))
        x = np.zeros((len(idx), 3, HEIGHT, width), np.float32)
        for k, i in enumerate(idx):
            rw = min(width, int(np.ceil(HEIGHT * ratios[i])))
            line = resize_linear(crops[i], rw, HEIGHT).astype(np.float32).transpose(2, 0, 1)
            x[k, :, :, :rw] = (line / 255 - 0.5) / 0.5
        probs = session.run(None, {name: x})[0]
        ids, best = probs.argmax(2), probs.max(2)
        for k, i in enumerate(idx):
            keep = np.ones(len(ids[k]), bool)
            keep[1:] = ids[k][1:] != ids[k][:-1]
            keep &= ids[k] != 0
            conf = [round(c, 5) for c in best[k][keep].tolist()] or [0]
            out[i] = ("".join(symbols[t] for t in ids[k][keep]), float(np.mean(conf).round(5)))
    return out
