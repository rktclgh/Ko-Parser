"""모델 패키지의 ONNX 세션 둘(검출·인식)과 사전으로 그림 한 장을 읽는다. 방향 판정은 하지 않는다(스펙 O2)."""

import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from . import OcrLine
from .det import detect
from .pixels import crop_quad, resize_linear
from .rec import recognize

MAX_SIDE = 2000  # 긴 변이 이보다 크면 줄여서 읽는다(RapidOCR max_side_len)


def _session(path: Path) -> ort.InferenceSession:
    """CPU, 스레드 수는 onnxruntime 기본값."""
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _shrunk_size(h: int, w: int) -> tuple[int, int]:
    s = MAX_SIDE / max(h, w)
    return max(32, int(round(int(h * s) / 32) * 32)), max(32, int(round(int(w * s) / 32) * 32))


class OcrReader:
    """세션은 만들 때 한 번 연다. 호출은 여러 스레드에서 동시에 해도 된다(onnxruntime run은 스레드 안전)."""

    def __init__(self, model_dir: Path, det_file: str, rec_file: str, dict_file: str) -> None:
        self.det = _session(model_dir / det_file)
        self.rec = _session(model_dir / rec_file)
        path = model_dir / dict_file
        symbols = path.read_text(encoding="utf-8").removesuffix("\n").split("\n")
        self.symbols = ["blank", *symbols, " "]  # CTC 빈칸 + 사전 + 공백(use_space_char)
        classes = self.rec.get_outputs()[0].shape[-1]
        if isinstance(classes, int) and classes != len(self.symbols):  # 글자 번호가 어긋난다
            raise ValueError(f"{path} gives {len(self.symbols)} classes but the recognition model has {classes}")
        if not isinstance(classes, int):  # 갈래 수가 정해지지 않은 모델: 모델 정보의 글자 목록이 있으면 그것과 맞춘다
            inner = self.rec.get_modelmeta().custom_metadata_map.get("character")
            if inner is not None and symbols != inner.removesuffix("\n").split("\n"):
                raise ValueError(f"{path} does not match the character list in the recognition model")

    def __call__(self, image: Image.Image) -> list[OcrLine]:
        """줄마다 OcrLine(상자는 입력 그림 화소, 글자는 NFC·앞뒤 공백 없음). 글자가 빈 줄은 버린다. 점수로는 거르지 않는다."""
        img = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])  # RapidOCR처럼 BGR
        h, w = img.shape[:2]
        if not h or not w:
            return []
        sy = sx = 1.0
        if max(h, w) > MAX_SIDE:
            nh, nw = _shrunk_size(h, w)
            img, sy, sx = resize_linear(img, nw, nh), h / nh, w / nw
        boxes = detect(self.det, img)
        if not len(boxes):
            return []
        lines = []
        for box, (text, score) in zip(boxes, recognize(self.rec, self.symbols, [crop_quad(img, b) for b in boxes])):
            text = unicodedata.normalize("NFC", text).strip()
            if not text:
                continue
            pts = np.minimum(np.maximum(box.astype(np.float64) * [sx, sy], 0), [w, h])
            lines.append(OcrLine(box=tuple((float(x), float(y)) for x, y in pts), text=text, score=score))
        return lines
