"""OCR 예제(fixtures/ocr/scanned.pdf, build_ocr_fixture.py로 한 번 만든 스캔 흉내 PDF): 글자는 정확히, 상자는 ±0.01.
CLI --no-ocr와 뷰어 안내도 이 파일로 본다. OCR 결과는 CPU마다 조금 달라 골든 바이트 비교를 하지 않는다."""

import json
import re
from pathlib import Path

import pytest

from ko_parser import LocalEngine, MemoryStore
from ko_parser.cli import main
from ko_parser.errors import OcrUnavailable
from ko_parser.formats.pdf import ocr

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ocr" / "scanned.pdf"
DATA = re.compile(r'<script type="application/json" id="ko-data">(.*?)</script>', re.DOTALL)
EXPECTED = [  # (출처, 글자, (x0, y0, x1, y1)) — 2026-10-05 macOS 실측 상자
    ("ocr", "스캔 문서 읽기 시험", (0.117, 0.063, 0.525, 0.099)),
    ("ocr", "이 쪽은 그림 한 장으로 된 스캔 쪽이다.\n그림 속 글자는 문단 블록이 된다.", (0.116, 0.137, 0.662, 0.195)),
    ("ocr", "두 번째 문단은 한 줄이다.", (0.118, 0.241, 0.476, 0.265)),
    ("text_layer", "1", (0.488, 0.938, 0.499, 0.952)),
]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("KO_PARSER_DB", raising=False)
    return tmp_path / "state.db"


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_ocr_fixture_text_is_exact_and_boxes_within_tolerance():
    pytest.importorskip("onnxruntime")
    engine = LocalEngine(MemoryStore())  # 계약 검증(build_tree)까지
    tree = engine.get_tree(engine.ingest(str(FIXTURE)).document_id)
    assert tree.pages[0].text_layer == "scanned"
    got = [(b.text_source, b.text, b.locator.bbox) for b in tree.blocks]
    assert [(s, t) for s, t, _ in got] == [(s, t) for s, t, _ in EXPECTED]
    for (_, _, box), (_, _, want) in zip(got, EXPECTED):
        assert all(abs(a - b) <= 0.01 for a, b in zip((box.x0, box.y0, box.x1, box.y1), want))
    for block in tree.blocks[:3]:
        assert (block.kind, block.state, block.section_path) == ("paragraph", "det", ())
        assert 0.45 <= block.confidence <= 0.5  # 평균 점수 × 0.5


def test_cli_parse_no_ocr(capsys, db, tmp_path):
    pytest.importorskip("onnxruntime")
    code, out, _ = run(capsys, "parse", FIXTURE, "--db", db)
    assert code == 0 and [b["text_source"] for b in json.loads(out)["blocks"]] == ["ocr", "ocr", "ocr", "text_layer"]
    code, out, _ = run(capsys, "parse", FIXTURE, "--no-ocr", "--db", tmp_path / "off.db")
    assert code == 0 and [b["text"] for b in json.loads(out)["blocks"]] == ["1"]


def test_viewer_notice_follows_ocr_state(capsys, db, tmp_path):
    pytest.importorskip("onnxruntime")
    out_html = tmp_path / "on.html"
    assert run(capsys, "view", FIXTURE, "--db", db, "--out", out_html)[0] == 0
    page = json.loads(DATA.search(out_html.read_text(encoding="utf-8")).group(1))["pages"][0]
    assert page["notice"] == "그림 속 글자는 OCR로 읽음(검증 전)"
    off_html = tmp_path / "off.html"
    assert run(capsys, "view", FIXTURE, "--no-ocr", "--db", tmp_path / "off.db", "--out", off_html)[0] == 0
    page = json.loads(DATA.search(off_html.read_text(encoding="utf-8")).group(1))["pages"][0]
    assert page["notice"] == "그림 속 글자는 OCR 필요(보이는 글자만 블록)"


def test_broken_ocr_install_is_a_configuration_error(capsys, db, monkeypatch):
    """OCR 추가 설치가 있는데 모델을 못 열면 파싱 실패(4)가 아니라 그 밖의 오류(1)와 설치 안내."""
    pytest.importorskip("onnxruntime")  # 설치돼 있어야 자동 모드가 _build까지 간다
    def broken():
        raise OcrUnavailable('OCR models could not be loaded: x; reinstall with pip install "ko-parser-engine[ocr]"')

    monkeypatch.setattr(ocr, "_reader", None)
    monkeypatch.setattr(ocr, "_build", broken)
    code, _, err = run(capsys, "parse", FIXTURE, "--db", db)
    assert code == 1 and "ko-parser-engine[ocr]" in err
    assert run(capsys, "parse", FIXTURE, "--no-ocr", "--db", db)[0] == 0
