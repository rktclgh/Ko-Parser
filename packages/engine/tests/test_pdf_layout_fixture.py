"""레이아웃 예제(fixtures/layout/figures.pdf, build_layout_fixture.py로 한 번 만든 합성 PDF): 그림·캡션·표가 기대대로인지
(그림 상자는 그린 자리와 IoU ≥ 0.8), 글자 누락 없음, CLI --no-layout·export --assets·뷰어. 모델 결과는 CPU·OS마다
조금 달라 골든 바이트 비교를 하지 않는다."""

import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from hanji import LocalEngine, MemoryStore, models
from hanji.cli import main
from hanji.errors import LayoutUnavailable
from hanji.formats.pdf import PdfParser, layout

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "layout" / "figures.pdf"
DATA = re.compile(r'<script type="application/json" id="ko-data">(.*?)</script>', re.DOTALL)
W, H = 595.0, 842.0
TRUTH = {(1, "chart"): (94.0, 130.0, 491.0, 336.0), (1, "image"): (150.0, 464.5, 450.0, 652.0),
         (2, "chart"): (94.0, 102.0, 506.0, 296.0)}  # 그린 자리(보이는 쪽 pt, 축·값 이름표 포함)
CAPTIONS = [(1, "그림 1. 분기별 처리 건수(단위: 건)"), (1, "그림 2. 현장 사진"), (2, "그림 3. 월별 이용자 추이")]
KINDS = [(1, "heading"), (1, "paragraph"), (1, "figure"), (1, "caption"), (1, "figure"), (1, "caption"),
         (1, "paragraph"), (1, "list_item"), (2, "heading"), (2, "caption"), (2, "figure"), (2, "paragraph"),
         (2, "table"), (2, "paragraph"), (2, "list_item")]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("HANJI_DB", raising=False)
    return tmp_path / "state.db"


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def glyphs(blocks) -> Counter:
    return Counter(ch for b in blocks for ch in b["text"] if not ch.isspace())


def no_layout_runtime(monkeypatch) -> None:
    """--no-layout은 설치를 확인하지도 모델을 돌리지도 않는다(onnxruntime 없이 돈다)."""
    monkeypatch.setattr(layout, "available", lambda: pytest.fail("--no-layout은 레이아웃 설치를 확인하지 않는다"))
    monkeypatch.setattr(layout, "get_detector", lambda: pytest.fail("--no-layout은 모델을 돌리지 않는다"))


def view_data(capsys, html: Path, *argv) -> dict:
    """뷰어 HTML 안의 데이터(Task 16의 뷰어 테스트가 쓴다)."""
    assert run(capsys, "view", FIXTURE, *argv, "--out", html)[0] == 0
    return json.loads(DATA.search(html.read_text(encoding="utf-8")).group(1))


def require_models(*names: str) -> None:
    """실제 모델 파일이 필요한 테스트: 찾을 수 없으면 건너뛰고, HANJI_CI_REQUIRE_MODELS=1이면 실패한다."""
    missing = [name for name in names if models.find(name) is None]
    if missing:
        if os.environ.get("HANJI_CI_REQUIRE_MODELS") == "1":
            pytest.fail(f"model files not found: {missing}; run `hanji models fetch`")
        pytest.skip(f"model files not found: {missing} (hanji models fetch)")


def test_fixture_figures_captions_tables_and_history():
    pytest.importorskip("onnxruntime")
    require_models("layout", "layout-config")
    engine = LocalEngine(MemoryStore())  # 계약 검증(build_tree)까지
    tree = engine.get_tree(engine.ingest(str(FIXTURE)).document_id)
    assert [(b.locator.page, b.kind) for b in tree.blocks] == KINDS
    figs = [b for b in tree.blocks if b.kind == "figure"]
    for b in figs:
        box = (b.locator.bbox.x0 * W, b.locator.bbox.y0 * H, b.locator.bbox.x1 * W, b.locator.bbox.y1 * H)
        assert iou(box, TRUTH[(b.locator.page, b.figure.category)]) >= 0.8, (b.locator.page, b.figure.category, box)
        assert engine.get_asset(b.figure.asset).startswith(b"\x89PNG")
    by_id = {b.block_id: b for b in tree.blocks}
    assert [(by_id[f.figure.caption_block_id].locator.page, by_id[f.figure.caption_block_id].text) for f in figs] == CAPTIONS
    (table,) = [b for b in tree.blocks if b.kind == "table"]  # 차트 눈금 격자는 표가 아니다(그림이 이긴다)
    assert (table.table.n_rows, table.table.n_cols) == (4, 4)
    assert "표 1. 예산 현황(단위: 천원)" in [b.text for b in tree.blocks if b.kind == "paragraph"]  # 표 제목은 문단
    # 축 이름표는 모델 상자 아랫변 가까이 있어 OS 글꼴 렌더에 따라 그림 글자나 문단 어느 쪽에 들 수 있다(글자 보존은
    # glyphs 테스트가 본다): 그 쪽 그림 글자나 문단 어느 쪽에든 있으면 된다(사전 리뷰 6)
    def words(page):
        return {w for b in tree.blocks if b.locator.page == page and b.kind in ("figure", "paragraph")
                for w in b.text.split()}

    assert {"1Q", "5Q"} <= words(figs[0].locator.page) and {"1월", "11월"} <= words(figs[2].locator.page)
    regions = {r.region_id for r in engine.history(tree.document_id).regions}
    assert {"p2-table-in-figure-1", "p2-layout-table-1"} <= regions


def test_fixture_glyphs_are_the_same_with_layout_on_and_off():
    pytest.importorskip("onnxruntime")
    require_models("layout", "layout-config")
    data = FIXTURE.read_bytes()
    on = PdfParser(ocr=False, layout=True).parse(data, FIXTURE.name)
    off = PdfParser(ocr=False, layout=False).parse(data, FIXTURE.name)
    assert glyphs(on.blocks) == glyphs(off.blocks) and sum(glyphs(on.blocks).values()) > 200
    assert [b["kind"] for b in off.blocks].count("figure") == 1  # 레이아웃 없이도 사진(이미지 객체)은 그림


def test_cli_parse_no_layout_keeps_only_the_photo_figure(capsys, db, monkeypatch):
    no_layout_runtime(monkeypatch)
    code, out, _ = run(capsys, "parse", FIXTURE, "--no-layout", "--db", db)
    blocks = json.loads(out)["blocks"]
    kinds = [b["kind"] for b in blocks]
    assert code == 0 and kinds.count("figure") == 1 and "caption" not in kinds
    (photo,) = [b for b in blocks if b["kind"] == "figure"]
    assert photo["figure"]["category"] == "image" and photo["figure"]["caption_block_id"] is None


def test_cli_export_assets_writes_the_figure_png(capsys, db, tmp_path, monkeypatch):
    no_layout_runtime(monkeypatch)
    code, out, _ = run(capsys, "parse", FIXTURE, "--no-layout", "--db", db)
    tree = json.loads(out)
    (photo,) = [b for b in tree["blocks"] if b["kind"] == "figure"]
    folder = tmp_path / "assets"
    code, md, _ = run(capsys, "export", tree["document_id"], "--db", db, "--format", "md", "--assets", folder)
    (png,) = folder.iterdir()
    assert code == 0 and png.name == photo["figure"]["asset"].removeprefix("sha256:")[:16] + ".png"
    with Image.open(png) as image:  # 닫아야 Windows에서 임시 폴더를 지울 수 있다
        assert image.size == (photo["figure"]["width_px"], photo["figure"]["height_px"])
    assert f"/{png.name})" in md or f"/{png.name}>)" in md


def test_broken_layout_install_is_a_configuration_error(capsys, db, monkeypatch):
    """레이아웃 추가 설치가 있는데 모델을 못 열면 파싱 실패(4)가 아니라 그 밖의 오류(1)와 설치·--no-layout 안내."""
    def broken():
        raise LayoutUnavailable('layout model could not be loaded: x; reinstall with pip install "hanji[layout]"')

    monkeypatch.setattr(layout, "available", lambda: True)
    monkeypatch.setattr(layout, "_detector", None)
    monkeypatch.setattr(layout, "_build", broken)
    code, _, err = run(capsys, "parse", FIXTURE, "--db", db)
    assert code == 1 and "hanji[layout]" in err and "Traceback" not in err
    assert run(capsys, "parse", FIXTURE, "--no-layout", "--db", db)[0] == 0


def test_viewer_shows_figure_categories_and_caption_pairs(capsys, db, tmp_path):
    pytest.importorskip("onnxruntime")
    require_models("layout", "layout-config")
    data = view_data(capsys, tmp_path / "v.html", "--db", db)
    figs = [b for b in data["blocks"] if b["kind"] == "figure"]
    assert [f["figure"]["category"] for f in figs] == ["chart", "image", "chart"]
    assert {b["figure_of"] for b in data["blocks"] if b["kind"] == "caption"} == {f["id"] for f in figs}
    assert data["document"]["layout_notice"] is None


def test_viewer_says_layout_needs_the_install(capsys, db, tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "available", lambda: False)
    data = view_data(capsys, tmp_path / "n.html", "--db", db)
    assert data["document"]["layout_notice"].startswith("선·도형 그림·캡션은 레이아웃 추가 설치가 필요")
    assert [b["kind"] for b in data["blocks"]].count("figure") == 1


def test_viewer_no_layout_has_no_install_notice(capsys, db, tmp_path, monkeypatch):
    no_layout_runtime(monkeypatch)
    data = view_data(capsys, tmp_path / "o.html", "--no-layout", "--db", db)
    assert data["document"]["layout_notice"] is None
    assert [b["kind"] for b in data["blocks"]].count("figure") == 1
