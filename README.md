# ko-parser

한국어 특화 문서 파싱 엔진. 다른 언어도 범용 처리한다.

- `packages/contracts` — 엔진·VLM 레이어·소비자가 주고받는 데이터 계약 (`ko-parser-contracts`)
- `packages/engine` — 결정론 엔진과 `ko-parser` 명령 (`ko-parser-engine`)

## 빠른 시작

```
uv sync
uv run ko-parser parse 문서.md --format md
uv run ko-parser view 문서.pdf
uv run ko-parser changes
```

라이선스: Apache-2.0
