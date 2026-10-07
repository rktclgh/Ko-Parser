"""파서 포트. 파서는 전사만 한다: 블록 종류·글자·출처·표·상태를 채우고 ID·버전은 모른다."""

from typing import Any, Protocol

from pydantic import Field

from hanji_contracts import ContractModel, PageInfo, RegionRecord


class ParsedSource(ContractModel):
    """파서 출력(엔진 내부 모델, 계약 아님). blocks는 계약 build_blocks에 넘길 블록 명세(그림 명세의 figure.caption_ref는
    짝 캡션 명세의 순번: 엔진이 caption_block_id로 바꾼다). assets는 그림 이미지("sha256:…" → PNG 바이트, 그림 블록
    figure.asset이 가리키는 것만), regions는 처리 이력에 남길 영역(블록이 되지 않은 표, 이미지 없이 낸 그림 등)."""

    mime: str = Field(min_length=1)
    pages: tuple[PageInfo, ...] = ()
    blocks: tuple[dict[str, Any], ...] = ()
    assets: dict[str, bytes] = Field(default_factory=dict)
    regions: tuple[RegionRecord, ...] = ()


class Parser(Protocol):
    mimes: tuple[str, ...]
    extensions: tuple[str, ...]  # 소문자, 점 포함(".md")

    def parse(self, data: bytes, name: str) -> ParsedSource:
        """실패는 ParseError(원인, 위치)로 알린다."""
        ...
