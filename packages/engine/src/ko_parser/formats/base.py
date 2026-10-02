"""파서 포트. 파서는 전사만 한다: 블록 종류·글자·출처·표·상태를 채우고 ID·버전은 모른다."""

from typing import Any, Protocol

from pydantic import Field

from ko_parser_contracts import ContractModel, PageInfo


class ParsedSource(ContractModel):
    """파서 출력(엔진 내부 모델, 계약 아님). blocks는 계약 build_blocks에 넘길 블록 명세."""

    mime: str = Field(min_length=1)
    pages: tuple[PageInfo, ...] = ()
    blocks: tuple[dict[str, Any], ...] = ()


class Parser(Protocol):
    mimes: tuple[str, ...]
    extensions: tuple[str, ...]  # 소문자, 점 포함(".md")

    def parse(self, data: bytes, name: str) -> ParsedSource:
        """실패는 ParseError(원인, 위치)로 알린다."""
        ...
