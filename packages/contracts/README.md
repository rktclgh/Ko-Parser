# ko-parser-contracts

ko-parser의 데이터 계약(pydantic v2 모델, JSON Schema, 골든 예제, 테스트용 가짜 드라이버).

- 스키마 버전: 0.1 (0.x 동안 소비자는 같은 버전만 읽는다)
- 스키마 재생성: `uv run python -m ko_parser_contracts.schema`
- 스키마 확인: `uv run python -m ko_parser_contracts.schema --check`
- 골든 예제 재생성·확인: `uv run python packages/contracts/tests/build_fixtures.py [--check]`
