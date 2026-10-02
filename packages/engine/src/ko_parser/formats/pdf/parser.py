"""PDF → ParsedSource. 쪽마다 판정(text_layer·text_stats)을 붙이고 digital 쪽만 블록을 만든다."""

from ko_parser_contracts import PageInfo

from ..base import ParsedSource
from .extract import extract_pages
from .group import build_specs
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
        return ParsedSource(mime=MIME, pages=infos, blocks=tuple(build_specs(pages, states)))
