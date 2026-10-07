"""레이아웃 모델 입출력 처리(numpy·Pillow만, onnxruntime 없음). 스파이크(.omc/research/layouteval/run_model.py의
Paddle 경로)와 같다: 800×800 bicubic(비율 무시)·/255·CHW float32, 점수 바닥 → PaddleX 방식 NMS(같은 분류 IoU 0.6,
다른 분류 0.98). 다른 점은 NMS 뒤 그림 안으로 자르고 넓이 0을 버리는 것, 그리고 스파이크는 0.1pt로 반올림한 상자로
NMS를 돌지만 여기는 화소 그대로 돈다는 것이다(IoU가 기준값에 딱 걸리는 상자만 갈릴 수 있고, 게이트 213쪽에는 없었다)."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from . import LayoutBox

LABELS = ("paragraph_title", "image", "text", "number", "abstract", "content", "figure_title", "formula",
          "table", "reference", "doc_title", "footnote", "header", "algorithm", "footer", "seal", "chart",
          "formula_number", "aside_text", "reference_content")  # PP-DocLayout_plus-L 분류 번호 순서(inference.yml)
SIZE = 800  # 모델 입력 한 변(화소)
MIN_SCORE = 0.2  # 이보다 낮은 상자는 NMS 전에 버린다(스파이크와 같은 바닥. 그림·캡션 기준값은 figures.py)
NMS_SAME = 0.6  # 같은 분류끼리 IoU가 이보다 크면 점수 낮은 쪽을 버린다
NMS_DIFF = 0.98  # 다른 분류끼리는 거의 같은 상자일 때만


def read_labels(config: Path) -> tuple[str, ...]:
    """inference.yml의 label_list(`label_list:` 다음 줄들의 `- 이름`)를 읽는다. PyYAML 없이 이 파일 모양만 다룬다."""
    labels: list[str] = []
    inside = False
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.startswith("label_list:"):
            inside = True
        elif inside and line.startswith("- "):
            labels.append(line[2:].strip())
        elif inside and line.strip():
            break
    return tuple(labels)


def preprocess(image: Image.Image) -> dict[str, np.ndarray]:
    """모델 입력: image(1×3×800×800, RGB/255), im_shape([[800, 800]]), scale_factor([[800/높이, 800/너비]])."""
    width, height = image.size
    rgb = image if image.mode == "RGB" else image.convert("RGB")  # RGB면 통째 복사하지 않는다
    pixels = np.asarray(rgb.resize((SIZE, SIZE), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return {"image": pixels.transpose(2, 0, 1)[None].astype(np.float32),
            "im_shape": np.array([[SIZE, SIZE]], np.float32),
            "scale_factor": np.array([[SIZE / height, SIZE / width]], np.float32)}


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(found: Iterable[LayoutBox]) -> list[LayoutBox]:
    """점수 높은 것부터 남긴다. 남긴 상자와 같은 분류면 IoU > NMS_SAME, 다른 분류면 > NMS_DIFF일 때 버린다
    (점수가 같으면 들어온 순서)."""
    keep: list[LayoutBox] = []
    for box in sorted(found, key=lambda b: -b.score):
        if all(_iou(box.box, k.box) <= (NMS_SAME if box.cls == k.cls else NMS_DIFF) for k in keep):
            keep.append(box)
    return keep


def postprocess(rows: np.ndarray, width: int, height: int) -> list[LayoutBox]:
    """모델 출력 행(분류, 점수, x0, y0, x1, y1: 입력 그림 화소. 7열째 이후는 무시) → 점수 바닥·NMS → 그림 안으로 자른
    상자. 모르는 분류(음수 포함)·유한하지 않은 값·자른 뒤 넓이 0은 버린다."""
    table = np.asarray(rows, dtype=np.float64)
    if table.size == 0:
        return []
    if table.ndim != 2:
        raise ValueError(f"layout output has shape {table.shape}, expected a 2-D table of rows")
    if table.shape[1] < 6:
        raise ValueError(f"layout output has {table.shape[1]} columns, expected at least 6")
    found = []
    for row in table[:, :6]:
        if not np.all(np.isfinite(row)):
            continue
        cls, score = int(row[0]), float(row[1])
        if not 0 <= cls < len(LABELS) or score < MIN_SCORE:
            continue
        found.append(LayoutBox(LABELS[cls], score, (float(row[2]), float(row[3]), float(row[4]), float(row[5]))))
    out = []
    for b in nms(found):
        x0, x1 = (min(max(v, 0.0), float(width)) for v in (b.box[0], b.box[2]))
        y0, y1 = (min(max(v, 0.0), float(height)) for v in (b.box[1], b.box[3]))
        if x1 > x0 and y1 > y0:
            out.append(LayoutBox(b.cls, b.score, (x0, y0, x1, y1)))
    return out
