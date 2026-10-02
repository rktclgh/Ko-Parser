from pathlib import Path

from ko_parser.core import build_tree
from ko_parser.export import to_markdown
from ko_parser.formats.base import ParsedSource
from ko_parser_contracts import DocumentTree, SourceInfo

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
