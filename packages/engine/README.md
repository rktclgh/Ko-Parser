# ko-parser-engine

ko-parser 결정론 엔진. 형식별 파서가 원문을 전사하고, core가 계약(`ko-parser-contracts`)의 문서 트리·변경 내역을 만들고, 저장 포트가 버전과 변경 피드를 보관한다.

- 형식: Markdown(`.md`, `.markdown`; UTF-8·UTF-8 BOM·cp949), PDF(`.pdf`; 텍스트 레이어. 쪽마다 digital·scanned·unreliable 판정과 근거, scanned 쪽은 보이는 글자만 블록(숨은 OCR 글자층은 버림), unreliable 쪽은 블록 없음)
- 리눅스·Docker: PDF에 넣어 두지 않은(미임베드) 한글 글꼴은 시스템 글꼴로 대신 읽으므로 한글 글꼴이 필요하다. `pip install "ko-parser-engine[fonts]"`(번들 Noto Sans KR, OFL)를 권장한다. 시스템 한글 글꼴(`fonts-noto-cjk`·`fonts-nanum`)이 있으면 필요 없다. 둘 다 없으면 한글이 빠질 상황을 감지해 파싱 실패(종료 코드 4)로 알린다. 알려진 한계: 번들 글꼴에 없는 기호(∙‣▸)만 담긴 한 글자 객체는 리눅스에서 빠질 수 있다. Windows·macOS는 기본 글꼴로 충분하다
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
ko-parser view 문서.pdf [--id ID] [--out PATH] [--dpi N]
```

- `view`: 수집(원본이 같으면 저장된 버전) 후 인터넷 없이 열리는 HTML 한 장(원본이 그대로면 저장된 버전과 그때 저장한 파일 이름을 보여 준다). 쪽 이미지(JPEG, 기본 110 DPI) 위 블록 영역, 블록 목록, 쪽 판정 근거(스캔 쪽은 보이는 글자만 블록, 글자 깨짐 쪽은 블록 없이 근거), 이전 버전 대비 변경. 기본 출력은 현재 폴더의 `<파일 이름(확장자 제외)>.view.html`. `--dpi`는 기본 110(1~600), 300 이상이면 쪽 수가 많은 문서의 HTML이 수십~수백 MB가 된다(60쪽 110 DPI 약 8MB).
- 상태 파일: `--db PATH` > 환경변수 `KO_PARSER_DB` > 사용자 데이터 폴더의 `ko-parser/state.db`
- 계약 0.2로 올라가며 이전 상태 파일은 지우고 다시 수집해야 한다
- 출력은 UTF-8, JSON은 계약 모델 그대로
- 종료 코드: 0 성공, 1 그 밖의 오류, 2 사용법, 3 지원하지 않는 형식, 4 파싱 실패, 5 문서·버전 없음

## 개발

- 골든 예제 재생성·확인: `uv run python packages/engine/tests/build_fixtures.py [--check]`
- PDF 골든 예제 재생성·확인: `uv run python packages/engine/tests/build_pdf_fixtures.py [--check]`(reportlab, 개발 의존성)
