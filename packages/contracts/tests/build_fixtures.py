"""골든 예제 생성기(합성 데이터만).

    uv run python packages/contracts/tests/build_fixtures.py          # fixtures/ 재생성
    uv run python packages/contracts/tests/build_fixtures.py --check  # 최신인지 확인
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ko_parser_contracts import (
    Attempt, BBox, Block, Cell, ChangeBatch, CorrectionSummary, DocumentChange, DocumentTree, GateCheck, GateResult,
    ImagePayload, LineageEdge, PageInfo, PageRef, ProcessingHistory, RegionRecord, SourceInfo, Table, Usage,
    VlmBlock, VlmRequest, VlmResult, build_blocks, compute_block_id, compute_content_hash,
)
from ko_parser_contracts.testing import Recording, RecordingMeta, request_fingerprint

ROOT = Path(__file__).resolve().parents[1] / "fixtures"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
A4 = PageInfo(page=1, width_pt=595.0, height_pt=842.0, rotation=0, render_dpi=144)
# 예제용 그림 해시(이미지 바이트는 계약 밖, 저장소 자산이다)
FIGURE_ASSET = "sha256:" + hashlib.sha256(b"ko-parser figure example").hexdigest()

# lab/glm-ocr-smoke 합성 샘플 table_simple.png에 dots.mocr-8bit(oMLX)가 실제로 낸 출력
DOTS_TABLE_SIMPLE_RAW = """<table>
  <tr>
    <td>분기</td>
    <td>매출액(원)</td>
    <td>영업이익(원)</td>
    <td>증감률(%)</td>
  </tr>
  <tr>
    <td>1분기</td>
    <td>1,250,000</td>
    <td>320,500</td>
    <td>12.5</td>
  </tr>
  <tr>
    <td>2분기</td>
    <td>1,480,300</td>
    <td>410,200</td>
    <td>18.4</td>
  </tr>
  <tr>
    <td>3분기</td>
    <td>1,395,750</td>
    <td>365,900</td>
    <td>-6.2</td>
  </tr>
  <tr>
    <td>4분기</td>
    <td>1,720,000</td>
    <td>498,400</td>
    <td>23.2</td>
  </tr>
</table>"""
TABLE_SIMPLE_ROWS = [
    ["분기", "매출액(원)", "영업이익(원)", "증감률(%)"],
    ["1분기", "1,250,000", "320,500", "12.5"],
    ["2분기", "1,480,300", "410,200", "18.4"],
    ["3분기", "1,395,750", "365,900", "-6.2"],
    ["4분기", "1,720,000", "498,400", "23.2"],
]


def source(name: str, mime: str, page_count: int | None) -> SourceInfo:
    return SourceInfo(name=name, mime=mime, content_hash="sha256:" + hashlib.sha256(name.encode()).hexdigest(),
                      page_count=page_count)


def flow(i: int, path: tuple[str, ...] = ()) -> dict:
    return {"kind": "flow", "section_path": list(path), "paragraph_index": i}


def page(y0: float, y1: float, x0: float = 0.1, x1: float = 0.9) -> dict:
    return {"kind": "page", "page": 1, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}


def native(kind: str, text: str, loc: dict, **kw) -> dict:
    return {"kind": kind, "text": text, "locator": loc, "confidence": 1.0, "state": "det",
            "text_source": "native", **kw}


def budget_table(src: str) -> Table:
    def c(r, k, text, rs=1, cs=1, header="none"):
        return Cell(row=r, col=k, rowspan=rs, colspan=cs, text=text, header=header, text_source=src)

    return Table(n_rows=4, n_cols=3, cells=[
        c(0, 0, "분류", rs=2, header="column"), c(0, 1, "2025년", cs=2, header="column"),
        c(1, 1, "상반기", header="column"), c(1, 2, "하반기", header="column"),
        c(2, 0, "인건비", header="row"), c(2, 1, "4,250,000"), c(2, 2, "4,310,000"),
        c(3, 0, "운영비", header="row"), c(3, 1, "1,800,000"), c(3, 2, "1,800,000"),
    ])


def simple_table(src: str) -> Table:
    return Table(n_rows=5, n_cols=4, cells=[
        Cell(row=r, col=k, text=text, header="column" if r == 0 else "none", text_source=src)
        for r, row in enumerate(TABLE_SIMPLE_ROWS) for k, text in enumerate(row)
    ])


def documents() -> dict[str, DocumentTree]:
    docx = DocumentTree(document_id="fx-docx", version=1, layer_state="det", source=source("보고서.docx", DOCX, None),
                        blocks=build_blocks("fx-docx", [
                            native("heading", "1. 개요", flow(0), level=1),
                            native("paragraph", "본 문서는 2026년 사업 추진 현황을 정리한다.", flow(1, ("1. 개요",)),
                                   section_path=("1. 개요",)),
                            native("paragraph", "자세한 내용은 별첨을 참고한다.", flow(2, ("1. 개요",)),
                                   section_path=("1. 개요",)),
                            native("heading", "2. 세부 계획", flow(3), level=1),
                            native("list_item", "1분기: 참여 기관 모집", flow(4, ("2. 세부 계획",)),
                                   section_path=("2. 세부 계획",)),
                            native("list_item", "2분기: 선정 결과 통보", flow(5, ("2. 세부 계획",)),
                                   section_path=("2. 세부 계획",)),
                            native("paragraph", "자세한 내용은 별첨을 참고한다.", flow(6, ("2. 세부 계획",)),
                                   section_path=("2. 세부 계획",)),
                        ]))
    pdf = DocumentTree(document_id="fx-pdf", version=1, layer_state="det", source=source("예산.pdf", "application/pdf", 1),
                       pages=(A4,), blocks=build_blocks("fx-pdf", [
                           dict(native("page_header", "2026년 사업 계획", page(0.02, 0.05)), text_source="text_layer"),
                           dict(native("heading", "예산 현황", page(0.08, 0.12), level=1), text_source="text_layer"),
                           {"kind": "table", "table": budget_table("text_layer"), "locator": page(0.15, 0.40),
                            "confidence": 0.92, "state": "det", "text_source": "text_layer", "region_id": "r-table-1",
                            "section_path": ("예산 현황",)},
                           dict(native("caption", "표 1. 예산 현황(단위: 원)", page(0.41, 0.44)), text_source="text_layer",
                                section_path=("예산 현황",)),
                           dict(native("page_footer", "- 1 -", page(0.95, 0.98, 0.45, 0.55)), text_source="text_layer"),
                       ]))
    slide = DocumentTree(document_id="fx-pptx", version=1, layer_state="det", source=source("실적.pptx", PPTX, 1),
                         blocks=build_blocks("fx-pptx", [
                             native("heading", "분기 실적", {"kind": "slide", "slide": 1, "shape_index": 0}, level=1),
                             native("list_item", "매출 1,720,000원", {"kind": "slide", "slide": 1, "shape_index": 1}),
                             native("list_item", "증감률 23.2%", {"kind": "slide", "slide": 1, "shape_index": 2}),
                         ]))
    empty = DocumentTree(document_id="fx-empty", version=1, layer_state="det", source=source("빈문서.docx", DOCX, None))
    return {"docx_flow": docx, "pdf_table_page": pdf, "pptx_slide": slide, "empty": empty,
            "pdf_figure_page": figure_page()}


def figure_page() -> DocumentTree:
    """계약 0.3: 차트 그림(이미지 참조) + 바로 아래 캡션 짝."""
    blocks = build_blocks("fx-figure", [
        dict(native("heading", "추진 현황", page(0.06, 0.09), level=1), text_source="text_layer"),
        dict(native("paragraph", "분기별 처리 건수는 아래 그림과 같다.", page(0.11, 0.13)), text_source="text_layer",
             section_path=("추진 현황",)),
        {"kind": "figure", "text": "1분기\n2분기\n3분기\n32\n55\n41", "locator": page(0.15, 0.45), "confidence": 0.7,
         "state": "det", "text_source": "text_layer", "section_path": ("추진 현황",),
         "figure": {"asset": FIGURE_ASSET, "mime": "image/png", "width_px": 1322, "height_px": 702, "dpi": 200,
                    "category": "chart"}},
        dict(native("caption", "그림 1. 분기별 처리 건수(단위: 건)", page(0.46, 0.48)), text_source="text_layer",
             section_path=("추진 현황",)),
    ])
    fig = blocks[2]
    linked = Block.model_validate({**fig.model_dump(),
                                   "figure": {**fig.figure.model_dump(), "caption_block_id": blocks[3].block_id}})
    return DocumentTree(document_id="fx-figure", version=1, layer_state="det",
                        source=source("현황.pdf", "application/pdf", 1), pages=(A4,),
                        blocks=(*blocks[:2], linked, blocks[3]))


def lifecycle() -> tuple[DocumentTree, DocumentTree, DocumentTree, ChangeBatch]:
    src = source("분기보고.pdf", "application/pdf", 1)
    det_table = Table(n_rows=2, n_cols=2, cells=[
        Cell(row=0, col=0, text="분기", header="column", text_source="ocr"),
        Cell(row=0, col=1, text="증감률(%)", header="column", text_source="ocr"),
        Cell(row=1, col=0, text="3분기", text_source="ocr"),
        Cell(row=1, col=1, text="6.2", text_source="ocr"),
    ])
    specs = [
        {"kind": "heading", "text": "분기 실적", "level": 1, "locator": page(0.05, 0.09), "confidence": 1.0,
         "state": "det", "text_source": "text_layer"},
        {"kind": "paragraph", "text": "3분기 영업이익은 365,900원이다.", "locator": page(0.10, 0.14), "confidence": 1.0,
         "state": "det", "text_source": "text_layer"},
        {"kind": "table", "table": det_table, "locator": page(0.20, 0.35), "confidence": 0.6, "state": "det",
         "text_source": "ocr", "region_id": "r-table-1"},
    ]
    v1 = DocumentTree(document_id="fx-life", version=1, layer_state="det", source=src, pages=(A4,),
                      blocks=build_blocks("fx-life", specs))
    v2 = DocumentTree(document_id="fx-life", version=2, layer_state="vlm_running", source=src, pages=(A4,),
                      blocks=tuple(b.model_copy(update={"state": "unverified"}) for b in v1.blocks))
    vlm_table = Table(n_rows=2, n_cols=2, cells=[
        Cell(row=0, col=0, text="분기", header="column", text_source="text_layer"),
        Cell(row=0, col=1, text="증감률(%)", header="column", text_source="text_layer"),
        Cell(row=1, col=0, text="3분기", text_source="text_layer"),
        Cell(row=1, col=1, text="-6.2", text_source="text_layer"),
    ])
    v3_specs = [dict(s, state="vlm") for s in specs[:2]] + [
        dict(specs[2], table=vlm_table, confidence=0.95, state="vlm", text_source="text_layer")]
    v3 = DocumentTree(document_id="fx-life", version=3, layer_state="vlm_done", source=src, pages=(A4,),
                      blocks=build_blocks("fx-life", v3_specs))
    ids1 = [b.block_id for b in v1.blocks]
    new_table_id = v3.blocks[2].block_id
    batch = ChangeBatch(cursor_from=None, next_cursor=3, changes=[
        DocumentChange(document_id="fx-life", version=1, added=ids1),
        DocumentChange(document_id="fx-life", version=2, previous_version=1, updated=ids1),
        DocumentChange(document_id="fx-life", version=3, previous_version=2, added=[new_table_id],
                       updated=ids1[:2], removed=[ids1[2]],
                       lineage=[LineageEdge(old_id=ids1[2], new_id=new_table_id, kind="replaced")]),
    ])
    return v1, v2, v3, batch


def change_examples() -> dict[str, ChangeBatch]:
    def bid(text: str) -> str:
        return compute_block_id("fx-split", compute_content_hash("paragraph", text, None, None), 0)

    # 이전: A(분할 대상), B+C(병합 대상) / 이후: A→X,Y 분할, B+C→Z 병합
    old_a, old_b, old_c = (bid(t) for t in ("1. 사업 개요 및 추진 일정", "예산은 4,250,000원이다.", "집행은 분기별로 한다."))
    new_x, new_y, new_z = (bid(t) for t in ("1. 사업 개요", "추진 일정", "예산은 4,250,000원이다. 집행은 분기별로 한다."))
    split_merge = ChangeBatch(cursor_from=10, next_cursor=11, changes=[DocumentChange(
        document_id="fx-split", version=5, previous_version=4,
        added=[new_x, new_y, new_z], removed=[old_a, old_b, old_c],
        lineage=[
            LineageEdge(old_id=old_a, new_id=new_x, kind="split"),
            LineageEdge(old_id=old_a, new_id=new_y, kind="split"),
            LineageEdge(old_id=old_b, new_id=new_z, kind="merged"),
            LineageEdge(old_id=old_c, new_id=new_z, kind="merged"),
        ])])
    resync = ChangeBatch(cursor_from=5, next_cursor=120, resync_required=True)
    return {"split_merge": split_merge, "resync": resync}


def _region_history(document_id: str, vlm: Attempt, gate: GateResult, chosen: str,
                    fallback_reason: str | None) -> ProcessingHistory:
    region = RegionRecord(region_id="r-table-1", locator=page(0.20, 0.35), kind="table",
                          attempts=[Attempt(layer="det"), vlm], chosen=chosen, gate=gate,
                          fallback_reason=fallback_reason)
    return ProcessingHistory(document_id=document_id, version=3, regions=[region])


def history() -> ProcessingHistory:
    """게이트 실패로 det로 되돌린 별도 문서(lifecycle fx-life와 무관)."""
    vlm = Attempt(layer="vlm_small", driver_id="omlx", model_id="dots.mocr-8bit", usage=Usage(latency_ms=2300),
                  correction=CorrectionSummary(segments_total=20, fixed=2, unchanged=16, unmatched=1, skipped=1,
                                               chars_replaced=3, chars_dropped=1, unmatched_with_digits=1))
    gate = GateResult(passed=False, checks=[
        GateCheck(name="char_f1_after_correction", passed=True, value=0.97, threshold=">=0.90"),
        GateCheck(name="numbers_preserved", passed=False, threshold="missing=0,extra=0"),
    ])
    return _region_history("fx-fallback", vlm, gate, "det", "unmatched segment with digits")


def lifecycle_history() -> ProcessingHistory:
    """lifecycle v3(fx-life)에서 VLM 표가 채택된 처리 이력."""
    vlm = Attempt(layer="vlm_small", driver_id="omlx", model_id="dots.mocr-8bit", usage=Usage(latency_ms=2300),
                  correction=CorrectionSummary(segments_total=20, fixed=3, unchanged=16, unmatched=0, skipped=1,
                                               chars_replaced=4, chars_dropped=0, unmatched_with_digits=0))
    gate = GateResult(passed=True, checks=[
        GateCheck(name="char_f1_after_correction", passed=True, value=0.99, threshold=">=0.90"),
        GateCheck(name="numbers_preserved", passed=True, threshold="missing=0,extra=0"),
    ])
    return _region_history("fx-life", vlm, gate, "vlm", None)


def vlm_examples() -> tuple[VlmRequest, VlmResult, Recording]:
    png = (ROOT / "images" / "table_simple.png").read_bytes()
    request = VlmRequest(request_id="req-0001", task="REGION_TABLE",
                         image=ImagePayload.from_png(png, BBox(x0=0.1, y0=0.15, x1=0.6, y1=0.40), 144),
                         region_id="r-table-1", page_ref=PageRef(document_id="fx-life", page=1),
                         language_hints=("ko",))
    result = VlmResult(request_id="req-0001", blocks=[VlmBlock(kind="table", table=simple_table("vlm"))],
                       raw_text=DOTS_TABLE_SIMPLE_RAW, output_format="html", usage=Usage(latency_ms=2600),
                       driver_id="omlx", model_id="dots.mocr-8bit")
    recording = Recording(fingerprint=request_fingerprint(request), result=result, meta=RecordingMeta(
        model_id="dots.mocr-8bit", profile_id="dots_ocr", prompt="Extract the text content from this image.",
        runtime="oMLX 127.0.0.1:8000", recorded_at="2026-10-02T00:00:00+09:00"))
    return request, result, recording


def _dump(model) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


def build_all() -> dict[str, str]:
    out = {f"documents/{name}.json": _dump(doc) for name, doc in documents().items()}
    v1, v2, v3, batch = lifecycle()
    out.update({"lifecycle/v1_det.json": _dump(v1), "lifecycle/v2_unverified.json": _dump(v2),
                "lifecycle/v3_vlm.json": _dump(v3), "lifecycle/changes.json": _dump(batch)})
    out.update({f"changes/{name}.json": _dump(b) for name, b in change_examples().items()})
    out["history/gate_fail_fallback.json"] = _dump(history())
    out["history/lifecycle_vlm_success.json"] = _dump(lifecycle_history())
    request, result, recording = vlm_examples()
    out.update({"vlm/request_table.json": _dump(request), "vlm/result_table.json": _dump(result),
                "recordings/table_simple.json": _dump(recording)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ko-parser-contracts fixtures")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for rel, text in build_all().items():
        path = ROOT / rel
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(rel)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(path)
    for rel in stale:
        print(f"stale fixture: {rel}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
