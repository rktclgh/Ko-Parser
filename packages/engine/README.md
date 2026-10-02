# ko-parser-engine

ko-parser 결정론 엔진. 형식별 파서가 원문을 전사하고, core가 계약(`ko-parser-contracts`)의 문서 트리·변경 내역을 만들고, 저장 포트가 버전과 변경 피드를 보관한다.

- 저장: `Store` 포트, 기본 구현 `MemoryStore`(테스트용)·`SqliteStore`(상태 파일)
- 문서 ID: 호출자가 주거나, 없으면 `doc_` + 원본 sha256 앞 24자리. 같은 원본은 새 버전을 만들지 않는다(`force=True`로 재파싱)
