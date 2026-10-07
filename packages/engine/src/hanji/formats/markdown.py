"""Markdown 파서: markdown-it-py(commonmark + table) 토큰을 블록 명세로 옮긴다.

토큰은 평평한 목록 그대로 훑는다. 인라인 강조는 깊이 제한이 없어 재귀 트리(SyntaxTreeNode)를 쓰면
RecursionError가 날 수 있다.
"""

from collections.abc import Sequence
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.token import Token
from pydantic import ValidationError

from hanji_contracts import Cell, LinesLocator, Table

from ..errors import ParseError
from .base import ParsedSource
from .text import decode_text

MIME = "text/markdown"
_BREAKS = frozenset({"softbreak", "hardbreak"})
_FIGURE_PARTS = frozenset({"image", "link_open", "link_close"}) | _BREAKS
_TEXT = frozenset({"text", "code_inline", "html_inline"})
_MAX_NESTING = 20  # markdown-it commonmark 기본 maxNesting. 넘으면 라이브러리가 내용을 조용히 버린다.


class _TooDeep(Exception):
    """블록 중첩이 한계를 넘어 내용이 버려질 상황."""


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
    """이미지와 공백·줄바꿈만 있는 문단. 이미지를 감싼 링크는 투명하게 본다."""
    return any(t.type == "image" for t in tokens) and all(
        t.type in _FIGURE_PARTS or (t.type == "text" and not t.content.strip()) for t in tokens)


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
        i = 0
        while i < len(tokens):
            token = tokens[i]
            i += 1
            match token.type:
                case "heading_open":
                    self.heading(token, tokens[i])
                case "paragraph_open":
                    self.paragraph(token, tokens[i], tokens[i - 2] if i > 1 else None)
                case "fence" | "code_block" | "html_block":
                    self.add("paragraph", token.content.rstrip("\n"), self.lines(token))
                case "table_open":
                    i = self.table(token, tokens, i)
                case _:  # 구분선, 닫는 토큰, 목록·인용 여닫이, 표 안쪽 토큰
                    pass

    def heading(self, token: Token, inline: Token) -> None:
        level = int(token.tag[1:])
        text = " ".join(line.strip() for line in inline_text(inline.children or ()).split("\n")).strip()  # 여러 줄은 한 줄로
        lines = self.lines(token)
        if not text:
            return
        while self.headings and self.headings[-1][0] >= level:
            self.headings.pop()
        self.add("heading", text, lines, level=level)
        self.headings.append((level, text))

    def paragraph(self, token: Token, inline: Token, before: Token | None) -> None:
        lines = self.lines(token)
        children = inline.children or []
        if before is not None and before.type == "list_item_open":  # 목록 항목의 직접 글자
            text = inline_text(children).strip()  # 이미지를 지운 가장자리 공백 제거
            if text and before.markup in (".", ")"):  # 순서 목록은 원문 번호 유지
                text = f"{before.info}{before.markup} {text}"
            self.add("list_item", text, lines)
        elif _is_figure(children):
            self.add("figure", "\n".join(inline_text(t.children or ()) for t in children if t.type == "image"),
                     lines)
        else:
            self.add("paragraph", inline_text(children).strip(), lines)

    def table(self, token: Token, tokens: Sequence[Token], start: int) -> int:
        """표를 블록으로 만들고 table_close 다음 위치를 돌려준다(복사 없이 인덱스로 훑는다)."""
        lines = self.lines(token)
        rows: list[list[str]] = []
        i = start
        while tokens[i].type != "table_close":
            t = tokens[i]
            i += 1
            if t.type == "tr_open":
                rows.append([])
            elif t.type == "inline":
                rows[-1].append(inline_text(t.children or ()).strip())
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
        return i + 1


def _guarded_tokenize(md: MarkdownIt) -> None:
    """한계 깊이에서 내용이 남아 있으면 버려지기 전에 알린다. 라이브러리 판정과 같은 조건이다."""
    tokenize = md.block.tokenize

    def guarded(state: StateBlock, start: int, end: int) -> None:
        if state.level >= _MAX_NESTING:
            line = state.skipEmptyLines(start)
            if line < end and state.sCount[line] >= state.blkIndent:
                raise _TooDeep
        tokenize(state, start, end)

    md.block.tokenize = guarded  # type: ignore[method-assign]


class MarkdownParser:
    mimes: tuple[str, ...] = (MIME,)
    extensions: tuple[str, ...] = (".md", ".markdown")

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark", {"maxNesting": _MAX_NESTING}).enable("table")
        _guarded_tokenize(self._md)

    def parse(self, data: bytes, name: str) -> ParsedSource:
        collector = _Collector(name)
        try:
            tokens = self._md.parse(decode_text(data, name))
        except _TooDeep:
            raise ParseError("block nesting too deep", name) from None
        collector.collect(tokens)
        return ParsedSource(mime=MIME, blocks=tuple(collector.blocks))
