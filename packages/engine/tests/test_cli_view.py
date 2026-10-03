import base64
import io
import json
import re

import pytest
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from ko_parser.cli import main

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
DATA = re.compile(r'<script type="application/json" id="ko-data">(.*?)</script>', re.DOTALL)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("KO_PARSER_DB", raising=False)
    return tmp_path / "상태" / "state.db"


def make_pdf(*lines: str) -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(595.0, 842.0), invariant=1, pageCompression=0)
    for i, text in enumerate(lines):
        c.setFont(FONT, 11)
        c.drawString(72, 770 - 40 * i, text)
    c.showPage()
    c.save()
    return buf.getvalue()


def write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def view_data(path) -> dict:
    (raw,) = DATA.findall(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def test_view_pdf_writes_html_with_page_images(capsys, db, tmp_path):
    pdf = write(tmp_path / "한글 폴더" / "보고서.pdf", make_pdf("첫째 문단이다.", "둘째 문단이다."))
    out = tmp_path / "결과" / "보기.html"
    code, stdout, _ = run(capsys, "view", pdf, "--db", db, "--out", out, "--dpi", "36")
    assert (code, stdout) == (0, f"{out}\n")
    data = view_data(out)
    assert data["document"]["name"] == "보고서.pdf" and [b["text"] for b in data["blocks"]] == [
        "첫째 문단이다.", "둘째 문단이다."]
    image = data["pages"][0]["image"]
    assert image.startswith("data:image/jpeg;base64,")
    jpeg = Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1])))
    assert jpeg.size == (298, 421)  # 36 DPI
    assert b"\r\n" not in out.read_bytes()


def test_view_default_output_is_cwd_file_stem(capsys, db, tmp_path, monkeypatch):
    pdf = write(tmp_path / "입력" / "보고서.pdf", make_pdf("본문"))
    work = tmp_path / "작업"
    work.mkdir()
    monkeypatch.chdir(work)
    code, stdout, _ = run(capsys, "view", pdf, "--db", db)
    assert code == 0 and (work / "보고서.view.html").exists()
    assert stdout.strip().endswith("보고서.view.html")


def test_view_reuses_stored_version_when_hash_matches(capsys, db, tmp_path):
    pdf = write(tmp_path / "a.pdf", make_pdf("본문"))
    run(capsys, "parse", pdf, "--db", db, "--id", "d1")
    code, _, _ = run(capsys, "view", pdf, "--db", db, "--id", "d1", "--out", tmp_path / "a.html")
    assert code == 0
    assert json.loads(run(capsys, "documents", "--db", db)[1]) == [
        {"document_id": "d1", "version": 1, "layer_state": "det"}]
    assert len(json.loads(run(capsys, "changes", "--db", db)[1])["changes"]) == 1


def test_view_marks_changes_against_previous_version(capsys, db, tmp_path):
    md = write(tmp_path / "메모.md", "가\n\n나\n".encode("utf-8"))
    run(capsys, "view", md, "--db", db, "--id", "m", "--out", tmp_path / "v1.html")
    write(md, "가\n\n다\n".encode("utf-8"))
    run(capsys, "view", md, "--db", db, "--id", "m", "--out", tmp_path / "v2.html")
    data = view_data(tmp_path / "v2.html")
    assert data["pages"] == [] and data["document"]["previous_version"] == 1
    assert [(b["text"], b["change"]) for b in data["blocks"]] == [("가", None), ("다", "added")]
    assert [r["text"] for r in data["removed"]] == ["나"]


@pytest.mark.parametrize("argv", [["view"], ["view", "a.pdf", "--dpi", "0"], ["view", "a.pdf", "--dpi", "601"],
                                  ["view", "a.pdf", "--dpi", "x"], ["view", "a.pdf", "--id", ""]])
def test_view_usage_errors_exit_2(capsys, tmp_path, monkeypatch, argv):
    state = tmp_path / "state.db"
    monkeypatch.setenv("KO_PARSER_DB", str(state))
    assert run(capsys, *argv)[0] == 2
    assert not state.exists()


def test_view_error_exit_codes(capsys, db, tmp_path):
    assert run(capsys, "view", write(tmp_path / "a.txt", b"x"), "--db", db)[0] == 3
    code, _, err = run(capsys, "view", write(tmp_path / "깨진.pdf", b"%PDF-1.4\n"), "--db", db)
    assert code == 4 and "깨진.pdf" in err
    assert run(capsys, "view", tmp_path / "없음.pdf", "--db", db)[0] == 1
    assert json.loads(run(capsys, "documents", "--db", db)[1]) == []


def test_view_out_directory_fails_before_store(capsys, db, tmp_path):
    pdf = write(tmp_path / "a.pdf", make_pdf("본문"))
    out_dir = tmp_path / "출력"
    out_dir.mkdir()
    code, out, err = run(capsys, "view", pdf, "--db", db, "--out", out_dir)
    assert (code, out) == (1, "") and "Traceback" not in err
    assert not db.exists()


def test_view_fails_when_file_changes_between_ingest_and_render(capsys, db, tmp_path, monkeypatch):
    """수집한 원본과 렌더할 원본이 다르면 다른 문서의 쪽 그림 위에 블록 상자가 깔린다: 쓰지 않고 실패한다."""
    from ko_parser.engine import LocalEngine

    pdf = write(tmp_path / "a.pdf", make_pdf("본문"))
    ingest = LocalEngine.ingest

    def ingest_then_change(self, *args, **kwargs):
        ref = ingest(self, *args, **kwargs)
        write(pdf, make_pdf("바뀐 본문"))
        return ref

    monkeypatch.setattr(LocalEngine, "ingest", ingest_then_change)
    out = tmp_path / "a.html"
    code, stdout, err = run(capsys, "view", pdf, "--db", db, "--out", out)
    assert (code, stdout) == (1, "") and "file changed during view" in err and "Traceback" not in err
    assert not out.exists()


def test_view_writes_lone_surrogate_text_as_replacement(capsys, db, tmp_path, monkeypatch):
    """짝 없는 서로게이트가 든 블록 글자도 HTML을 쓴다(인코딩 못 하는 글자는 '?')."""
    from ko_parser.engine import LocalEngine

    get_tree = LocalEngine.get_tree

    def get_tree_with_surrogate(self, *args, **kwargs):
        tree = get_tree(self, *args, **kwargs)
        blocks = [tree.blocks[0].model_copy(update={"text": "가\ud800"}), *tree.blocks[1:]]
        return tree.model_copy(update={"blocks": blocks})

    monkeypatch.setattr(LocalEngine, "get_tree", get_tree_with_surrogate)
    md = write(tmp_path / "메모.md", "가\n".encode())
    out = tmp_path / "s.html"
    code, stdout, err = run(capsys, "view", md, "--db", db, "--out", out)
    assert (code, stdout) == (0, f"{out}\n") and err == ""
    assert [b["text"] for b in view_data(out)["blocks"]] == ["가?"]
