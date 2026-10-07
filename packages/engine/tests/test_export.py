from pathlib import Path

from hanji.core import build_tree
from hanji.export import to_markdown
from hanji.formats.base import ParsedSource
from hanji_contracts import DocumentTree, PageInfo, SourceInfo

CONTRACT_FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "fixtures"


def _doc(name: str) -> DocumentTree:
    return DocumentTree.model_validate_json((CONTRACT_FIXTURES / "documents" / name).read_text(encoding="utf-8"))


def test_flow_document():
    assert to_markdown(_doc("docx_flow.json")) == (
        "# 1. 개요\n\n"
        "본 문서는 2026년 사업 추진 현황을 정리한다.\n\n"
        "자세한 내용은 별첨을 참고한다.\n\n"
        "# 2. 세부 계획\n\n"
        "- 1분기: 참여 기관 모집\n\n"
        "- 2분기: 선정 결과 통보\n\n"
        "자세한 내용은 별첨을 참고한다.\n"
    )


def test_page_document_skips_header_footer_and_renders_table():
    assert to_markdown(_doc("pdf_table_page.json")) == (
        "# 예산 현황\n\n"
        "| 분류 | 2025년 | 2025년 |\n"
        "| --- | --- | --- |\n"
        "| 분류 | 상반기 | 하반기 |\n"
        "| 인건비 | 4,250,000 | 4,310,000 |\n"
        "| 운영비 | 1,800,000 | 1,800,000 |\n\n"
        "표 1. 예산 현황(단위: 원)\n"
    )


def test_empty_document_is_empty_string():
    assert to_markdown(_doc("empty.json")) == ""


def test_numbered_items_and_levels():
    def spec(kind: str, text: str, **kw) -> dict:
        return {"kind": kind, "text": text, "confidence": 1.0, "state": "det", "text_source": "native",
                "locator": {"kind": "lines", "line_start": 1, "line_end": 1}, **kw}

    tree = build_tree(ParsedSource(mime="text/markdown", blocks=[
        spec("heading", "셋째 수준", level=3), spec("list_item", "3. 셋째"), spec("list_item", "7) 괄호"),
        spec("list_item", "2026년 계획"), spec("figure", "조직도"),
    ]), "d", 1, SourceInfo(name="t.md", mime="text/markdown", content_hash="sha256:" + "0" * 64))
    assert to_markdown(tree) == "### 셋째 수준\n\n3. 셋째\n\n7) 괄호\n\n- 2026년 계획\n\n조직도\n"


FIG = "sha256:" + "ab" * 32
A4 = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144)


def figure_tree(caption: str | None = "그림 1. [예산] 현황", text: str = "1분기\n2분기", image: bool = True) -> DocumentTree:
    def spec(kind: str, body: str, y: float, **kw) -> dict:
        return {"kind": kind, "text": body, "confidence": 0.7, "state": "det", "text_source": "text_layer",
                "locator": {"kind": "page", "page": 1, "bbox": {"x0": 0.1, "y0": y, "x1": 0.9, "y1": y + 0.1}}, **kw}

    figure = {"asset": FIG, "mime": "image/png", "width_px": 10, "height_px": 10, "dpi": 200, "category": "chart",
              "caption_ref": 1 if caption is not None else None}
    specs = [spec("figure", text, 0.1, **({"figure": figure} if image else {}))]
    if caption is not None:
        specs.append(spec("caption", caption, 0.3))
    return build_tree(ParsedSource(mime="application/pdf", pages=(A4,), blocks=specs), "d", 1,
                      SourceInfo(name="f.pdf", mime="application/pdf", content_hash="sha256:" + "0" * 64,
                                 page_count=1))


def test_figure_with_assets_dir_links_its_png_with_the_caption_as_alt_text():
    md = to_markdown(figure_tree(), "그림 폴더")
    assert md == ("![그림 1. \\[예산\\] 현황](%EA%B7%B8%EB%A6%BC%20%ED%8F%B4%EB%8D%94/abababababababab.png)\n\n"
                  "1분기\n2분기\n\n그림 1. [예산] 현황\n")


def test_without_assets_dir_figures_are_text_and_empty_ones_are_skipped():
    assert to_markdown(figure_tree()) == "1분기\n2분기\n\n그림 1. [예산] 현황\n"
    assert to_markdown(figure_tree(caption=None, text="")) == ""
    assert to_markdown(figure_tree(caption=None, text=""), "assets") == "![](assets/abababababababab.png)\n"


def test_figure_without_image_stays_text_even_with_assets_dir():
    assert to_markdown(figure_tree(image=False), "assets") == "1분기\n2분기\n\n그림 1. [예산] 현황\n"


def test_assets_dir_link_is_percent_encoded_and_a_file_uri_is_kept():
    tree = figure_tree(caption=None, text="")
    assert to_markdown(tree, "out/a #b?c%20(d)") == "![](out/a%20%23b%3Fc%2520%28d%29/abababababababab.png)\n"
    assert to_markdown(tree, "/") == "![](/abababababababab.png)\n"
    assert to_markdown(tree, "file:///C:/a%20b") == "![](file:///C:/a%20b/abababababababab.png)\n"
