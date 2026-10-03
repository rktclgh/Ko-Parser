"""형식 판별: 확장자(소문자)로 파서를 고른다. 확장자가 모호한 형식이 붙으면 매직바이트 판별을 더한다."""

from collections.abc import Sequence
from pathlib import PurePath

from ..errors import UnsupportedFormat
from .base import Parser
from .markdown import MarkdownParser
from .pdf import PdfParser


def default_parsers() -> tuple[Parser, ...]:
    """기본 파서 묶음. 형식이 붙을 때마다 여기에 더한다."""
    return (MarkdownParser(), PdfParser())


def detect_parser(name: str, parsers: Sequence[Parser]) -> Parser:
    suffix = PurePath(name).suffix.lower()
    for parser in parsers:
        if suffix in parser.extensions:
            return parser
    raise UnsupportedFormat(f"unsupported format: {name}")
