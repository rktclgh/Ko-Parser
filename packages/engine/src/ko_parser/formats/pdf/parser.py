"""PDF → ParsedSource. 쪽마다 판정(text_layer·text_stats)을 붙인다. digital·scanned 쪽은 보이는 글자(렌더 모드 3
제외)로 선 있는 표(table 블록)와 나머지 블록을 만들고, scanned 쪽은 OCR을 켰으면 그림 속 글자를 OCR 문단 블록
(text_source="ocr")으로 더한다. unreliable 쪽은 블록이 없다."""

from ko_parser_contracts import PageInfo

from ..base import ParsedSource
from . import ocr as ocr_runtime
from .extract import extract_pages
from .group import build_specs
from .scan import ocr_pages
from .tables import find_tables
from .triage import classify, page_stats

MIME = "application/pdf"
RENDER_DPI = 144


class PdfParser:
    mimes: tuple[str, ...] = (MIME,)
    extensions: tuple[str, ...] = (".pdf",)

    def __init__(self, ocr: bool | None = None) -> None:
        """ocr: None이면 OCR 추가 설치가 있을 때 scanned 쪽을 OCR로 읽고, False면 읽지 않는다. True인데 추가 설치가
        없거나 깨졌으면 여기서 OcrUnavailable(문서 파싱 실패가 아니라 설정 오류)."""
        if ocr:
            ocr_runtime.get_reader()
        self.ocr = ocr

    def parse(self, data: bytes, name: str) -> ParsedSource:
        """암호화·손상 PDF는 ParseError."""
        pages = extract_pages(data, name)
        stats = [page_stats(page) for page in pages]
        states = [classify(s) for s in stats]
        infos = tuple(PageInfo(page=page.page, width_pt=page.width_pt, height_pt=page.height_pt,
                               rotation=page.rotation, render_dpi=RENDER_DPI, text_layer=state, text_stats=s)
                      for page, s, state in zip(pages, stats, states, strict=True))
        tables = [find_tables(page) if state != "unreliable" else [] for page, state in zip(pages, states, strict=True)]
        scanned = "scanned" in states
        use_ocr = scanned and (self.ocr or (self.ocr is None and ocr_runtime.available()))
        found = ocr_pages(data, name, pages, states) if use_ocr else None
        return ParsedSource(mime=MIME, pages=infos, blocks=tuple(build_specs(pages, states, tables, found)))
