"""PDF → ParsedSource. 쪽마다 판정(text_layer·text_stats)을 붙인다. digital·scanned 쪽은 보이는 글자(렌더 모드 3
제외)로 선 있는 표(table 블록)와 나머지 블록을 만들고(scanned는 '그림 속 글자는 OCR이 필요하다'는 뜻),
unreliable 쪽은 블록이 없다."""

from ko_parser_contracts import PageInfo

from ..base import ParsedSource
from .extract import extract_pages
from .group import build_specs
from .tables import find_tables
from .triage import classify, page_stats

MIME = "application/pdf"
RENDER_DPI = 144


class PdfParser:
    mimes: tuple[str, ...] = (MIME,)
    extensions: tuple[str, ...] = (".pdf",)

    def parse(self, data: bytes, name: str) -> ParsedSource:
        """암호화·손상 PDF는 ParseError."""
        pages = extract_pages(data, name)
        stats = [page_stats(page) for page in pages]
        states = [classify(s) for s in stats]
        infos = tuple(PageInfo(page=page.page, width_pt=page.width_pt, height_pt=page.height_pt,
                               rotation=page.rotation, render_dpi=RENDER_DPI, text_layer=state, text_stats=s)
                      for page, s, state in zip(pages, stats, states, strict=True))
        tables = [find_tables(page) if state != "unreliable" else [] for page, state in zip(pages, states, strict=True)]
        return ParsedSource(mime=MIME, pages=infos, blocks=tuple(build_specs(pages, states, tables)))
