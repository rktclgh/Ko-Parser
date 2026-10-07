# ko-parser-engine

ko-parser 결정론 엔진. 형식별 파서가 원문을 전사하고, core가 계약(`ko-parser-contracts`)의 문서 트리·변경 내역을 만들고, 저장 포트가 버전과 변경 피드를 보관한다.

- 형식: Markdown(`.md`, `.markdown`; UTF-8·UTF-8 BOM·cp949), PDF(`.pdf`; 텍스트 레이어. 쪽마다 digital·scanned·unreliable 판정과 근거, scanned 쪽은 보이는 글자로 블록을 만들고(숨은 OCR 글자층은 버림) OCR 추가 설치가 있으면 그림 속 글자를 OCR 문단 블록으로 더함, unreliable 쪽은 블록 없음. 선 있는 표(2×2 이상 격자)는 `table` 블록: 병합 칸, 배경 있는 맨 윗행은 머리행, 안 보이는 안쪽 칸 경계는 글자 정렬로 찾는다)
- 리눅스·Docker: PDF에 넣어 두지 않은(미임베드) 한글 글꼴은 시스템 글꼴로 대신 읽으므로 한글 글꼴이 필요하다. `pip install "ko-parser-engine[fonts]"`(번들 Noto Sans KR, OFL)를 권장한다. 시스템 한글 글꼴(`fonts-noto-cjk`·`fonts-nanum`)이 있으면 필요 없다. 둘 다 없으면 한글이 빠질 상황을 감지해 파싱 실패(종료 코드 4)로 알린다. 알려진 한계: 번들 글꼴에 없는 기호(∙‣▸)만 담긴 한 글자 객체는 리눅스에서 빠질 수 있다. Windows·macOS는 기본 글꼴로 충분하다
- 스캔 쪽 OCR: `pip install "ko-parser-engine[ocr]"`(onnxruntime·numpy·pyclipper, 설치 약 100~130MB(OS마다 다름))와 모델 파일(`ko-parser models fetch ocr`, 약 18MB)이 있으면 `parse`·`view`가 scanned 쪽의 그림 속 글자를 200 DPI로 읽어 `paragraph` 블록(`text_source="ocr"`, `state="det"`, 신뢰도 = 평균 점수 × 0.5)으로 낸다. CPU에서 쪽당 약 1초(macOS M 계열 실측). 실행 중 아무것도 내려받지 않고 리눅스 슬림 Docker에서도 추가 시스템 라이브러리 없이 돈다. 모델: PaddleOCR PP-OCRv5 mobile 검출 + 한국어 mobile 인식(Apache-2.0, PaddlePaddle 공식 ONNX, 인식 글자 목록은 인식 설정 파일 안. 출처·라이선스는 `ko_parser/models/NOTICE`·`LICENSE`). 이전 판에서 함께 깔린 `ko-parser-ocr-models`(약 18MB)는 더 쓰지 않는다: `pip uninstall ko-parser-ocr-models`로 지운다(남아 있어도 해가 없다). 모델 파일을 찾을 수 없으면(받기 전) OCR은 설치가 없을 때처럼 꺼지고, 찾은 파일이 고정한 크기·SHA-256과 다르면 종료 코드 1로 알린다. `--no-ocr`로 끈다. 보이는 글자(쪽 번호 등)와 겹치는 OCR 줄은 버린다(텍스트 레이어 우선). 디지털 쪽은 OCR을 하지 않는다. onnxruntime 휠이 있는 플랫폼만(macOS 14+ arm64, 리눅스 x86_64·aarch64 glibc 2.28+, Windows x64·arm64). 메모리는 프로세스당 최대 약 1~1.2GiB(A4 200 DPI 실측)라 작은 컨테이너에서는 1.5GB 이상을 주거나 `--no-ocr`로 끈다
- OCR 한계: 표는 문단으로 읽는다(표 구조 없음), 세로 쓰기·뒤집힌 쪽(방향 판정 끔)·한자 정확도·디지털 쪽 안의 그림 속 글자(차트 이름표)는 다루지 않는다. 기호는 비슷한 모양으로 바뀔 수 있다(`•`→`·`, `「」`→`[]`). 읽기 순서는 빈 세로 띠로 단을 나누는 규칙 기반(XY 분할)이고 제목·목록 판정은 하지 않는다(OCR 블록은 모두 문단). 점수가 낮은 작은 이름표는 버린다. 스캔 쪽은 주석(도장·스탬프 등)도 그림에 그려 OCR로 읽는다(디지털 쪽은 주석 글자를 뽑지 않는다)
- OCR을 켜고 끈 결과를 바꾸려면 원본이 같아 저장된 버전을 그대로 쓰므로 `ko-parser parse 문서.pdf --force [--no-ocr]`로 다시 파싱한다. OCR을 새로 설치하거나 모델 파일을 받은 뒤에도 마찬가지다(받기 전에 파싱한 scanned 쪽은 OCR 없이 저장돼 있다). `view`에는 `--force`가 없으니 먼저 `ko-parser parse 문서.pdf --force [--no-ocr]`로 다시 파싱하거나 새 `--db`로 `view`한다
- 그림·캡션: digital 쪽의 사진(장식이 아닌 이미지 객체: 양변 24pt 이상·쪽 면적 2.5% 이상, 글자 50자 이상이 얹힌 배경은 제외)은 늘 `figure` 블록이 되고 잘라 낸 PNG(200 DPI)를 내용 해시로 가리킨다(`figure.asset`, `export --assets`로 꺼낸다). 레이아웃 추가 설치 `pip install "ko-parser-engine[layout]"`(onnxruntime·numpy)와 모델 파일(`ko-parser models fetch layout`, 공식 ONNX 약 130MB)이 있으면 PaddlePaddle PP-DocLayout_plus-L(Apache-2.0) 모델로 선·도형으로 그린 차트·도식, 스캔 쪽 그림, 그림 캡션(`caption` 블록, 그림 바로 위·아래 30pt 안, 그림 블록 `figure.caption_block_id`로 짝)을 찾는다. 모델은 스캔 쪽과, 사진이 있거나 선·도형이 10개 이상인 digital 쪽에서만 돈다. 그림 상자 안의 글자는 문단이 아니라 그림 블록 글자가 되고(글자는 그대로, 자리만 옮긴다), 그림 안에 잡힌 선 있는 표(차트 눈금 격자 등)는 표가 아니다(모델이 표라고 본 표는 남긴다). 표 바로 위·아래 캡션(표 제목)은 문단으로 둔다. 문서 하나의 그림 바이트가 512MiB를 넘으면 이후 그림은 이미지 없이 위치·글자만 낸다(처리 이력 `history`에 남는다). `--no-layout`으로 모델을 끈다(사진은 그대로 그림). 실행 중 아무것도 내려받지 않는다. 모델을 돌린 쪽은 쪽당 약 0.24초 더 걸리고(macOS M 계열 실측), 최대 RSS는 레이아웃만 약 0.8GiB, OCR과 함께 약 1.4GiB(작은 컨테이너에서는 2GB 이상을 주거나 `--no-layout`·`--no-ocr`)
- 그림 한계: 이미지를 담지 못한 그림(문서 그림 바이트 상한을 넘었거나 잘라 낼 화소가 없음)은 `figure` 필드 없이 위치·글자만 남고, 그 짝 캡션은 `caption` 블록으로 남지만 그림 쪽에서 가리키지 않는다(`caption_block_id`는 `figure` 안에 있다. 그 그림은 처리 이력에 남는다). 다단 읽기 순서, 디지털 쪽 래스터 그림 속 글자(OCR 안 함), 그림 해석(VLM)은 다루지 않는다. 쪽 전체 폭 100% 누적 막대 차트는 모델이 표로 본다(알려진 실패). 스캔 쪽의 인포그래픽은 조각으로 나올 수 있다. 레이아웃을 켜고 끈 결과를 바꾸려면 원본이 같아 저장된 버전을 그대로 쓰므로 `ko-parser parse 문서.pdf --force [--no-layout]`로 다시 파싱한다
- PDF 한계: 쪽 내용 스트림의 글자만 읽는다. 입력 양식(AcroForm) 필드 값과 주석(annotation) 모양의 글자는 뽑지 않는다. 선이 하나도 없는 표, 가로선만 있는 표(세로선 없는 삼선표 등), 쪽을 넘는 표 잇기(쪽마다 블록 하나), 왼쪽 열 행머리, 칸 안 그림은 다루지 않는다. 1×1 상자와 한 줄·한 칸짜리 띠는 표가 아니라 문단이다. 상자 안 표는 안쪽 표만 블록이 된다. 표 영역 안이라도 표와 다른 방향으로 쓴 글자(회전한 글자)는 표 칸에 넣지 않고 문단으로 남긴다. 다단 읽기 순서는 아직 규칙 기반이다
- 표 검출(E1-2c) 이전에 수집한 PDF는 원본이 같아 `parse`·`view`가 저장된 버전을 그대로 쓴다. `view`에는 `--force`가 없으니 먼저 `ko-parser parse 문서.pdf --force`로 다시 파싱한다. 그러면 표 자리의 문단·제목 블록이 사라지고(removed) 표 블록이 생기며(added), 뒤 블록은 순서(`order`)나 위치가 바뀌어 updated로 나올 수 있다. 정상 동작이다
- 저장: `Store` 포트, 기본 구현 `MemoryStore`(테스트용)·`SqliteStore`(CLI 기본 상태 파일)
- 문서 ID: `--id`로 주거나, 없으면 `doc_` + 원본 sha256 앞 24자리. 같은 원본은 새 버전을 만들지 않는다(`--force`로 재파싱)

## CLI

```
ko-parser parse 문서.md [--id ID] [--force] [--no-ocr] [--no-layout] [--format json|md] [--out PATH]
ko-parser export DOC_ID [--version N] [--format json|md] [--out PATH] [--assets DIR]
ko-parser documents
ko-parser changes [--cursor N] [--limit N]
ko-parser history DOC_ID [--version N]
ko-parser view 문서.pdf [--id ID] [--out PATH] [--dpi N] [--no-ocr] [--no-layout]
ko-parser models fetch [ocr|layout|all] [--to DIR]
```

- `view`: 수집(원본이 같으면 저장된 버전) 후 인터넷 없이 열리는 HTML 한 장(원본이 그대로면 저장된 버전과 그때 저장한 파일 이름을 보여 준다). 쪽 이미지(JPEG, 기본 110 DPI) 위 블록 영역, 블록 목록(표 블록은 병합 칸까지 격자로), 쪽 판정 근거(스캔 쪽은 OCR 블록이 있으면 "OCR로 읽음(검증 전)", 없으면 "OCR 필요(보이는 글자만 블록)", 글자 깨짐 쪽은 블록 없이 근거), 이전 버전 대비 변경. 기본 출력은 현재 폴더의 `<파일 이름(확장자 제외)>.view.html`. `--dpi`는 기본 110(1~600), 300 이상이면 쪽 수가 많은 문서의 HTML이 수십~수백 MB가 된다(60쪽 110 DPI 약 8MB).
- `export --assets DIR`: 그림 블록의 이미지(PNG)를 `DIR/<sha256 앞 16자>.png`로 쓴다. `--format md`면 이미지가 있는 그림 블록이 `![캡션](DIR/이름.png)`과 그 아래 그림 글자가 된다(캡션이 없으면 대체 글자가 비고, 이미지가 없는 그림 블록은 글자만). `--assets` 없이 `--format md`로 내보내면 그림 블록은 글자만 쓰고, 글자가 없는 그림 블록은 건너뛴다. 링크의 `DIR`은 `--out` 파일 폴더 기준 상대 경로(표준 출력이면 현재 폴더 기준 입력 그대로)를 `/`로 잇고 퍼센트 인코딩한 것이다(Windows에서 드라이브가 달라 상대 경로가 없으면 `file://` 주소). 이미지는 상태 파일에 내용 해시로 한 번만 저장된다
- `models fetch`: 모델 파일(OCR 검출·인식 모델과 인식 설정 약 18MB, 레이아웃 PP-DocLayout_plus-L 공식 ONNX·설정 약 130MB)을 업스트림이 공개한 고정 주소(Hugging Face 커밋이 든 URL, `ko_parser/models/models.toml`)에서 받아 크기·SHA-256을 확인한 뒤 사용자 캐시(`KO_PARSER_CACHE_DIR`로 바꿀 수 있다) 또는 `--to` 폴더에 둔다. 표준 라이브러리 urllib만 쓰고, 이미 맞는 파일은 다시 받지 않는다. 받은 경로를 한 줄씩 출력한다(상태 파일을 열지 않는다). 엔진은 `KO_PARSER_MODEL_DIR` → 사용자 캐시 순서로 찾고(두 곳 모두 `ocr/det.onnx`처럼 같은 상대 경로), 실행 중에는 아무것도 내려받지 않는다. 폐쇄망은 연결된 기계에서 `ko-parser models fetch --to 폴더`로 받아 그 폴더를 옮기고 `KO_PARSER_MODEL_DIR`로 가리킨다
- 상태 파일: `--db PATH` > 환경변수 `KO_PARSER_DB` > 사용자 데이터 폴더의 `ko-parser/state.db`
- 계약 0.3(그림 참조)으로 올라가며 이전 상태 파일은 지우고 다시 수집해야 한다(상태 파일 형식 3: 그림 이미지 자산을 함께 저장)
- 출력은 UTF-8, JSON은 계약 모델 그대로
- 종료 코드: 0 성공, 1 그 밖의 오류(OCR 등 추가 설치나 모델 파일이 깨졌거나 모델 받기에 실패한 경우 포함), 2 사용법, 3 지원하지 않는 형식, 4 파싱 실패, 5 문서·버전·그림 이미지 없음

## 개발

- 골든 예제 재생성·확인: `uv run python packages/engine/tests/build_fixtures.py [--check]`
- PDF 골든 예제 재생성·확인: `uv run python packages/engine/tests/build_pdf_fixtures.py [--check]`(reportlab, 개발 의존성). 골든 예제는 OCR을 끄고 만든다
- OCR 예제: `fixtures/ocr/scanned.pdf`는 `build_ocr_fixture.py`로 한 번 만들어 커밋한 고정 입력이다(미임베드 글꼴을 OS 글꼴로 그려 OS마다 바이트가 달라 `--check` 대상이 아니다). OCR 테스트는 글자 정확·상자 ±0.01로 본다
- 레이아웃 예제: `fixtures/layout/figures.pdf`는 `build_layout_fixture.py`로 한 번 만들어 커밋한 고정 입력이다(PDF 바이트는 결정적이지만 미임베드 한글 글꼴을 OS 글꼴로 그려 모델 입력이 OS마다 조금 다르다). 레이아웃 테스트는 그림 상자 IoU ≥ 0.8·캡션 글자로 본다. 골든 예제는 OCR·레이아웃을 끄고 만든다
- 뷰어: 그림 블록 상자는 분류별 색(사진·그림 주황, 차트 분홍), 블록 목록에 쪽 그림을 잘라 보인 썸네일과 캡션 짝(누르면 짝으로 간다). PDF인데 레이아웃 추가 설치가 없으면 머리에 "선·도형 그림·캡션은 레이아웃 추가 설치가 필요" 안내
