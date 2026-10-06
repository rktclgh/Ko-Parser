# ko-parser

한국어 특화 문서 파싱 엔진. 다른 언어도 범용 처리한다.

- `packages/contracts` — 엔진·VLM 레이어·소비자가 주고받는 데이터 계약 (`ko-parser-contracts`)
- `packages/engine` — 결정론 엔진과 `ko-parser` 명령 (`ko-parser-engine`)
- `packages/fonts` — 리눅스용 한글 대체 글꼴 (`ko-parser-fonts`, `ko-parser-engine[fonts]`)
- 스캔 쪽 OCR(`ko-parser-engine[ocr]`): 개발용 `uv sync`는 OCR 런타임까지 설치한다(onnxruntime 휠이 없는 Intel macOS는 제외). 모델 파일은 저장소에 없고 `uv run ko-parser models fetch`가 Hugging Face 고정 커밋 주소에서 받는다

## 빠른 시작

```
uv sync
uv run ko-parser models fetch      # OCR 모델 파일(업스트림 고정 주소, SHA-256 확인)
uv run ko-parser parse 문서.md --format md
uv run ko-parser view 문서.pdf
uv run ko-parser changes
```

라이선스: Apache-2.0
