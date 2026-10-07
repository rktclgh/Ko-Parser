# hanji

한국어도 잘하는 범용 문서 파서.

- `packages/contracts` — 엔진·VLM 레이어·소비자가 주고받는 데이터 계약 (`hanji-contracts`)
- `packages/engine` — 결정론 엔진과 명령줄 도구 (`hanji`)
- `packages/fonts` — 리눅스용 한글 대체 글꼴 (`hanji-fonts`, `hanji[fonts]`)
- 스캔 쪽 OCR(`hanji[ocr]`)·그림 레이아웃(`hanji[layout]`, 한 번에 `[all]`): 개발용 `uv sync`는 OCR·레이아웃 런타임까지 설치한다(onnxruntime 휠이 없는 Intel macOS는 제외). 모델 파일은 저장소에 없고 `uv run hanji models fetch`가 Hugging Face 고정 커밋 주소에서 받는다(OCR 약 18MB, 레이아웃 약 130MB)

## 빠른 시작

```
uv sync
uv run hanji models fetch      # OCR·레이아웃 모델 파일(업스트림 고정 주소, SHA-256 확인)
uv run hanji parse 문서.md --format md
uv run hanji view 문서.pdf
uv run hanji changes
```

라이선스: Apache-2.0
