# ko-parser-ocr-models

ko-parser가 스캔 쪽(본문이 그림인 PDF 쪽)의 글자를 CPU로 읽을 때 쓰는 OCR 모델 패키지. 실행 중에 아무것도 내려받지 않는다.

- 설치: `pip install "ko-parser-engine[ocr]"` (이 패키지와 onnxruntime·numpy·pyclipper가 함께 설치된다)
- 모델: PaddleOCR PP-OCRv5 mobile 검출(`det.onnx`) + 한국어 mobile 인식(`rec.onnx`) + 인식 사전(`dict.txt`).
  방향 판정(180° 분류) 모델은 넣지 않는다(긴 한국어 줄을 뒤집어 글자를 잃는다)
- 출처(RapidOCR v3.9.2 모델 저장소, ModelScope `RapidAI/RapidOCR`)와 SHA-256:

| 파일 | 원본 | SHA-256 |
|---|---|---|
| `det.onnx` | `onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx` | `4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae` |
| `rec.onnx` | `onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx` | `cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b` |
| `dict.txt` | `paddle/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile/ppocrv5_korean_dict.txt` | `a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7` |
| `LICENSE` | PaddleOCR 태그 `v3.0.0`의 `LICENSE` | `3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e` |

  원본 주소의 앞부분은 `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/`이다(`SOURCES` 상수에 전체 주소).
- 라이선스: 모델은 Apache-2.0(PaddleOCR, `src/ko_parser_ocr_models/models/LICENSE`), 파이썬 코드도 Apache-2.0
- 고정 정책: 18MB 바이너리가 git에 들어가므로 모델 교체는 드물게 한다. 자주 바꿔야 하면 Git LFS를 검토한다
