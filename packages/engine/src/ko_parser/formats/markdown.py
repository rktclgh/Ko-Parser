"""Markdown 파서: markdown-it-py(commonmark + table) 토큰을 블록 명세로 옮긴다.

토큰은 평평한 목록 그대로 훑는다. 인라인 강조는 깊이 제한이 없어 재귀 트리(SyntaxTreeNode)를 쓰면
RecursionError가 날 수 있다.
"""

from collections.abc import Sequence
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import ValidationError

from ko_parser_contracts import Cell, LinesLocator, Table

from ..errors import ParseError
from .base import ParsedSource
from .text import decode_text

MIME = "text/markdown"
_BREAKS = frozenset({"softbreak", "hardbreak"})
_TEXT = frozenset({"text", "code_inline", "html_inline"})


def inline_text(tokens: Sequence[Token]) -> str:
    """인라인 표시를 지우고 글자만 남긴다. 링크는 링크 글자, 이미지는 대체 텍스트, 줄바꿈은 \n."""
    parts: list[str] = []
    stack = [iter(tokens)]
    while stack:
        token = next(stack[-1], None)
        if token is None:
            stack.pop()
        elif token.type in _TEXT:
            parts.append(token.content)
        elif token.type in _BREAKS:
            parts.append("\n")
        elif token.type == "image":
            stack.append(iter(token.children or ()))
    return "".join(parts)


def _is_figure(tokens: Sequence[Token]) -> bool:
    """이미지와 공백·줄바꿈만 있는 문단."""
    return any(t.type == "image" for t in tokens) and all(
        t.type == "image" or t.type in _BREAKS or (t.type == "text" and not t.content.strip()) for t in tokens)


class _Collector:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blocks: list[dict[str, Any]] = []
        self.headings: list[tuple[int, str]] = []  # (수준, 글자) 제목 스택
        self.last_line = 1

    def lines(self, token: Token) -> tuple[int, int]:
        """원문 줄 범위(1부터, 양 끝 포함). map이 없으면 직전 블록의 끝 줄."""
        if token.map is None:
            return self.last_line, self.last_line
        start = token.map[0] + 1
        end = max(token.map[1], start)
        self.last_line = end
        return start, end

    def add(self, kind: str, text: str, lines: tuple[int, int], **extra: Any) -> None:
        if not text.strip():  # 빈 글자 블록은 만들지 않는다
            return
        path = tuple(heading for _, heading in self.headings)
        self.blocks.append({
            "kind": kind, "text": text, "section_path": path,
            "locator": LinesLocator(section_path=path, line_start=lines[0], line_end=lines[1]),
            "confidence": 1.0, "state": "det", "text_source": "native", **extra,
        })

    def collect(self, tokens: Sequence[Token]) -> None:
        for i, token in enumerate(tokens):
            match token.type:
                case "heading_open":
                    self.heading(token, tokens[i + 1])
                case "paragraph_open":
                    self.paragraph(token, tokens[i + 1], tokens[i - 1] if i else None)
                case "fence" | "code_block" | "html_block":
                    self.add("paragraph", token.content.rstrip("\n"), self.lines(token))
                case "table_open":
                    self.table(token, tokens[i + 1:])
                case _:  # 구분선, 닫는 토큰, 목록·인용 여닫이, 표 안쪽 토큰
                    pass

    def heading(self, token: Token, inline: Token) -> None:
        level = int(token.tag[1:])
        text = inline_text(inline.children or ())
        lines = self.lines(token)
        if not text.strip():
            return
        while self.headings and self.headings[-1][0] >= level:
            self.headings.pop()
        self.add("heading", text, lines, level=level)
        self.headings.append((level, text))

    def paragraph(self, token: Token, inline: Token, before: Token | None) -> None:
        lines = self.lines(token)
        children = inline.children or []
        if before is not None and before.type == "list_item_open":  # 목록 항목의 직접 글자
            text = inline_text(children)
            if text.strip() and before.markup in (".", ")"):  # 순서 목록은 원문 번호 유지
                text = f"{before.info}{before.markup} {text}"
            self.add("list_item", text, lines)
        elif _is_figure(children):
            self.add("figure", "\n".join(inline_text(t.children or ()) for t in children if t.type == "image"),
                     lines)
        else:
            self.add("paragraph", inline_text(children), lines)

    def table(self, token: Token, rest: Sequence[Token]) -> None:
        lines = self.lines(token)
        rows: list[list[str]] = []
        for t in rest:
            if t.type == "table_close":
                break
            if t.type == "tr_open":
                rows.append([])
            elif t.type == "inline":
                rows[-1].append(inline_text(t.children or ()))
        n_cols = len(rows[0])
        try:
            table = Table(n_rows=len(rows), n_cols=n_cols, cells=[
                Cell(row=r, col=c, text=text, header="column" if r == 0 else "none", text_source="native")
                for r, row in enumerate(rows) for c, text in enumerate((row + [""] * n_cols)[:n_cols])
            ])
        except ValidationError as exc:
            raise ParseError(f"invalid table: {exc.errors()[0]['msg']}",
                             f"{self.name}:{lines[0]}-{lines[1]}") from exc
        self.add("table", table.plain_text(), lines, table=table)


class MarkdownParser:
    mimes: tuple[str, ...] = (MIME,)
    extensions: tuple[str, ...] = (".md", ".markdown")

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark").enable("table")

    def parse(self, data: bytes, name: str) -> ParsedSource:
        collector = _Collector(name)
        collector.collect(self._md.parse(decode_text(data, name)))
        return ParsedSource(mime=MIME, blocks=tuple(collector.blocks))
