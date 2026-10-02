# ko-parser-engine

ko-parser 결정론 엔진. 형식별 파서가 원문을 전사하고, core가 계약(`ko-parser-contracts`)의 문서 트리·변경 내역을 만들고, 저장 포트가 버전과 변경 피드를 보관한다.

- 형식: Markdown(`.md`, `.markdown`; UTF-8·UTF-8 BOM·cp949), PDF(`.pdf`; 텍스트 레이어. 쪽마다 digital·scanned·unreliable 판정과 근거, scanned 쪽은 보이는 글자만 블록(숨은 OCR 글자층은 버림), unreliable 쪽은 블록 없음)
- PDF 한계: 쪽 내용 스트림의 글자만 읽는다. 입력 양식(AcroForm) 필드 값과 주석(annotation) 모양의 글자는 뽑지 않는다. 표·다단 읽기 순서는 아직 규칙 기반이다
- 저장: `Store` 포트, 기본 구현 `MemoryStore`(테스트용)·`SqliteStore`(CLI 기본 상태 파일)
- 문서 ID: `--id`로 주거나, 없으면 `doc_` + 원본 sha256 앞 24자리. 같은 원본은 새 버전을 만들지 않는다(`--force`로 재파싱)

## CLI

```
ko-parser parse 문서.md [--id ID] [--force] [--format json|md] [--out PATH]
ko-parser export DOC_ID [--version N] [--format json|md] [--out PATH]
ko-parser documents
ko-parser changes [--cursor N] [--limit N]
ko-parser history DOC_ID [--version N]
```

- 상태 파일: `--db PATH` > 환경변수 `KO_PARSER_DB` > 사용자 데이터 폴더의 `ko-parser/state.db`
- 계약 0.2로 올라가며 이전 상태 파일은 지우고 다시 수집해야 한다
- 출력은 UTF-8, JSON은 계약 모델 그대로
- 종료 코드: 0 성공, 1 그 밖의 오류, 2 사용법, 3 지원하지 않는 형식, 4 파싱 실패, 5 문서·버전 없음

## 개발

- 골든 예제 재생성·확인: `uv run python packages/engine/tests/build_fixtures.py [--check]`
- PDF 골든 예제 재생성·확인: `uv run python packages/engine/tests/build_pdf_fixtures.py [--check]`(reportlab, 개발 의존성)
