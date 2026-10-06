# ko-parser-contracts

ko-parser의 데이터 계약(pydantic v2 모델, JSON Schema, 골든 예제, 테스트용 가짜 드라이버).

- 스키마 버전: 0.3 (0.x 동안 소비자는 같은 버전만 읽는다)
- 0.3: 그림 블록의 이미지 참조 `Block.figure`(`FigureImage`: PNG 내용 해시 `asset`, 크기·dpi, 분류 image·chart, 짝 캡션 `caption_block_id`). 이미지 바이트는 트리 밖(저장소 `get_asset`). `content_hash`에 그림 해시가 들어간다(그림이 없는 블록의 해시는 0.2와 같다). 상수 `MAX_FIGURE_SIDE`(4000px)·`MAX_DOCUMENT_ASSET_BYTES`(512MiB). 한 캡션은 그림 하나에만
- 0.2: `PageInfo.text_layer`(digital·scanned·unreliable)와 판정 근거 `text_stats`(`TextLayerStats`)
- `previous_version`이 있는데 added·updated·removed가 모두 빈 `DocumentChange`는 "쪽 정보(판정 등)가 바뀌고 블록은 그대로"라는 뜻이다. 건너뛰지 말고 `get_tree`로 쪽 정보를 다시 읽는다
- 0.1.1: `Engine.ingest(path, document_id=None, force=False)`
- 스키마 재생성: `uv run python -m ko_parser_contracts.schema`
- 스키마 확인: `uv run python -m ko_parser_contracts.schema --check`
- 골든 예제 재생성·확인: `uv run python packages/contracts/tests/build_fixtures.py [--check]`
