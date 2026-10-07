"""PDF → ParsedSource. 쪽마다 판정(text_layer·text_stats)을 붙인다. digital·scanned 쪽은 보이는 글자(렌더 모드 3
제외)로 선 있는 표(table 블록)와 나머지 블록을 만들고, scanned 쪽은 OCR을 켰으면 그림 속 글자를 OCR 문단 블록
(text_source="ocr")으로 더한다. 그림은 digital 쪽 이미지 객체(사진)와 레이아웃 모델(선·도형 그림·스캔 쪽 그림·캡션)로
찾아 figure·caption 블록과 잘라 낸 PNG(ParsedSource.assets)로 낸다. unreliable 쪽은 블록이 없다.
쪽 렌더는 필요한 쪽만 한 번(PDFIUM_LOCK 안), OCR·모델·PNG 인코딩은 잠금 밖에서 한다."""

from dataclasses import dataclass, field
from typing import Any

from ko_parser_contracts import Attempt, BBox, PageInfo, PageLocator, RegionRecord, TextLayerState
from PIL import Image

from ..base import ParsedSource
from . import figures, scan
from . import layout as layout_runtime
from . import ocr as ocr_runtime
from .extract import PageText, extract_pages
from .group import FigureBlock, build_specs, unit_box
from .scan import OcrParagraph
from .tables import TableSpec, find_tables
from .triage import classify, page_stats

MIME = "application/pdf"
RENDER_DPI = 144
TABLE_IN_FIGURE = "ruled table inside a figure was dropped (figure wins)"
LAYOUT_TABLE = "layout table box (kept for borderless-table detection, not a block)"
NO_IMAGE_MODEL = ("layout-model figure image not stored "
                  "(empty crop or document figure bytes over MAX_DOCUMENT_ASSET_BYTES)")
NO_IMAGE_PHOTO = ("image-object figure image not stored "
                  "(empty crop or document figure bytes over MAX_DOCUMENT_ASSET_BYTES)")


@dataclass(slots=True)
class _PageResult:
    """쪽 하나의 남긴 표·OCR 문단·그림 블록·처리 이력."""

    tables: list[TableSpec]
    paras: list[OcrParagraph] = field(default_factory=list)
    figures: list[FigureBlock] = field(default_factory=list)
    regions: list[RegionRecord] = field(default_factory=list)


class PdfParser:
    mimes: tuple[str, ...] = (MIME,)
    extensions: tuple[str, ...] = (".pdf",)

    def __init__(self, ocr: bool | None = None, layout: bool | None = None) -> None:
        """ocr·layout: None이면 추가 설치가 있을 때 쓰고(깔렸는데 깨졌으면 쓸 쪽을 만날 때 OcrUnavailable·
        LayoutUnavailable), False면 쓰지 않는다. True인데 추가 설치가 없거나 깨졌으면 여기서 그 오류(문서 파싱 실패가
        아니라 설정 오류). layout이 False이거나 설치가 없어도 digital 쪽 사진(이미지 객체)은 그림 블록이 된다."""
        if ocr:
            ocr_runtime.get_reader()
        if layout:
            layout_runtime.get_detector()
        self.ocr = ocr
        self.layout = layout

    def parse(self, data: bytes, name: str) -> ParsedSource:
        """암호화·손상 PDF는 ParseError."""
        pages = extract_pages(data, name)
        stats = [page_stats(page) for page in pages]
        states = [classify(s) for s in stats]
        infos = tuple(PageInfo(page=page.page, width_pt=page.width_pt, height_pt=page.height_pt,
                               rotation=page.rotation, render_dpi=RENDER_DPI, text_layer=state, text_stats=s)
                      for page, s, state in zip(pages, stats, states, strict=True))
        tables = [find_tables(page) if state != "unreliable" else [] for page, state in zip(pages, states, strict=True)]
        use_ocr = "scanned" in states and bool(self.ocr or (self.ocr is None and ocr_runtime.available()))
        if use_ocr:
            ocr_runtime.get_reader()  # 깨진 설치는 쪽을 그리기 전에 알린다
        use_layout: bool | None = None  # 모델을 돌릴 첫 쪽에서 정한다(그런 쪽이 없으면 설치를 확인하지 않는다)
        budget = figures.AssetBudget()
        done: list[_PageResult] = []
        for index, (page, state, page_tables) in enumerate(zip(pages, states, tables, strict=True)):
            want = figures.wants_layout(page, state)  # 쪽 루프 안에서: 이미지 객체 상자를 쪽마다 한 번만 구한다
            if want and use_layout is None:
                use_layout = bool(self.layout or (self.layout is None and layout_runtime.available()))
                if use_layout:
                    layout_runtime.get_detector()  # 깨진 설치는 쪽을 그리기 전에 알린다(렌더하는 쪽은 모두 want)
            done.append(_page(data, name, index, page, state, page_tables, use_ocr and state == "scanned",
                              bool(use_layout) and want, budget))
        blocks = build_specs(pages, states, [d.tables for d in done], [d.paras for d in done],
                             [d.figures for d in done])
        return ParsedSource(mime=MIME, pages=infos, blocks=tuple(blocks), assets=budget.assets,
                            regions=tuple(r for d in done for r in d.regions))


def _page(data: bytes, name: str, index: int, page: PageText, state: TextLayerState, tables: list[TableSpec],
          ocr_here: bool, layout_here: bool, budget: figures.AssetBudget) -> _PageResult:
    """쪽 하나: (필요할 때만) 렌더(잠금 안) → OCR 줄·레이아웃 상자(잠금 밖) → 그림 정리 → PNG → 남은 OCR 줄은 문단.
    렌더가 필요 없는 쪽(글자만 있는 digital 쪽 등)은 지금과 같은 경로(표만)."""
    if state == "unreliable":
        return _PageResult(tables=[])
    if not (ocr_here or layout_here or (state == "digital" and figures.photo_boxes(page))):
        return _PageResult(tables=list(tables))
    image = scan.render(data, name, index)
    lines = scan.page_lines(image, page) if ocr_here else []
    regions = figures.page_regions(layout_runtime.detect(image), image.size, page) if layout_here else []
    plan = figures.arrange(page, state, regions, tables, lines)
    out = _PageResult(tables=list(plan.tables))
    source = "ocr" if ocr_here else "text_layer"
    for k, figure in enumerate(plan.figures, 1):
        fields = _store(image, figure, page, budget)
        out.figures.append(FigureBlock(figure, source, fields))
        if fields is None:
            reason = NO_IMAGE_MODEL if figure.model else NO_IMAGE_PHOTO  # 모델 이름은 모델이 찾은 그림에만
            out.regions.append(_region(page, f"figure-{k}", _unit(figure.box, page), "figure", reason, figure.model))
    rest = [line for i, line in enumerate(lines) if i not in plan.used_lines]
    out.paras = scan.paragraphs(scan.reading_order(rest), page)
    out.regions += [_region(page, f"table-in-figure-{k}", t.bbox, "table", TABLE_IN_FIGURE, layout_here)
                    for k, t in enumerate(plan.dropped, 1)]  # 표를 버리는 것은 모델이 찾은 그림뿐(리뷰 I1)
    out.regions += [_region(page, f"layout-table-{k}", _unit(r.box, page), "table", LAYOUT_TABLE, True)
                    for k, r in enumerate(plan.layout_tables, 1)]
    return out


def _store(image: Image.Image, figure: figures.Figure, page: PageText,
           budget: figures.AssetBudget) -> dict[str, Any] | None:
    """그림 PNG를 잘라 문서 자산에 담고 FigureImage 필드를 돌려준다. 자를 것이 없거나 바이트 상한을 넘으면 None."""
    crop = figures.crop_png(image, figure.box, page)
    if crop is None:
        return None
    png, width, height, dpi = crop
    asset = budget.add(png)
    if asset is None:
        return None
    return {"asset": asset, "mime": "image/png", "width_px": width, "height_px": height, "dpi": dpi,
            "category": figure.category}


def _unit(box: figures.Box, page: PageText) -> tuple[float, float, float, float]:
    """보이는 쪽 pt → 0~1."""
    return (box[0] / page.width_pt, box[1] / page.height_pt, box[2] / page.width_pt, box[3] / page.height_pt)


def _region(page: PageText, tag: str, box: tuple[float, float, float, float], kind: str, reason: str,
            model: bool) -> RegionRecord:
    """처리 이력 한 줄(블록이 되지 않았거나 이미지 없이 된 영역). box는 보이는 쪽 0~1."""
    return RegionRecord(region_id=f"p{page.page}-{tag}", kind=kind, chosen="det", fallback_reason=reason,
                        locator=PageLocator(page=page.page, bbox=BBox(**unit_box(*box))),
                        attempts=(Attempt(layer="det", model_id=layout_runtime.MODEL_ID if model else None),))
