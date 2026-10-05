"""ko-parser 스캔 쪽 OCR 모델(PP-OCRv5 mobile 검출 + 한국어 mobile 인식, ONNX)과 인식 사전.

출처: RapidOCR v3.9.2 모델 저장소(PaddleOCR PP-OCRv5를 ONNX로 바꾼 것). 라이선스는 models/LICENSE(Apache-2.0).
방향 판정(180° 분류) 모델은 넣지 않는다: 긴 한국어 줄을 뒤집어 글자를 잃는다(스펙 §2)."""

from pathlib import Path

DET_FILE = "det.onnx"
REC_FILE = "rec.onnx"
DICT_FILE = "dict.txt"
LICENSE_FILE = "LICENSE"

_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2"
SOURCES = {
    DET_FILE: f"{_BASE}/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
    REC_FILE: f"{_BASE}/onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx",
    DICT_FILE: f"{_BASE}/paddle/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile/ppocrv5_korean_dict.txt",
    LICENSE_FILE: "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.0.0/LICENSE",
}
SHA256 = {
    DET_FILE: "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    REC_FILE: "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b",
    DICT_FILE: "a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7",
    LICENSE_FILE: "3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e",
}


def model_dir() -> Path:
    """모델·사전·라이선스가 든 폴더."""
    return Path(__file__).parent / "models"
