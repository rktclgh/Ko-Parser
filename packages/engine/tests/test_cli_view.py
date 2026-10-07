import base64
import io
import json
import os
import re
import stat
import sys

import pytest
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from hanji.cli import main

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))
DATA = re.compile(r'<script type="application/json" id="ko-data">(.*?)</script>', re.DOTALL)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("HANJI_DB", raising=False)
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
                                  ["view", "a.pdf", "--dpi", "x"], ["view", "a.pdf", "--id", ""],
                                  ["view", "a.pdf", "--out", ""]])
def test_view_usage_errors_exit_2(capsys, tmp_path, monkeypatch, argv):
    state = tmp_path / "state.db"
    monkeypatch.setenv("HANJI_DB", str(state))
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


def test_view_reuses_version_for_byte_different_pdf_with_same_tree(capsys, db, tmp_path):
    """바이트만 다르고 트리가 같은 PDF(%%EOF 뒤 주석)는 버전 1을 그대로 쓰고 새 바이트로 쪽 그림을 만든다."""
    pdf = write(tmp_path / "a.pdf", make_pdf("본문"))
    assert run(capsys, "view", pdf, "--db", db, "--id", "d1", "--out", tmp_path / "v1.html")[0] == 0
    write(pdf, pdf.read_bytes() + b"% comment\n")
    out = tmp_path / "v2.html"
    assert run(capsys, "view", pdf, "--db", db, "--id", "d1", "--out", out) == (0, f"{out}\n", "")
    assert view_data(out)["pages"][0]["image"].startswith("data:image/jpeg;base64,")
    assert json.loads(run(capsys, "documents", "--db", db)[1]) == [
        {"document_id": "d1", "version": 1, "layer_state": "det"}]


def test_view_renders_exactly_the_ingested_bytes(capsys, db, tmp_path, monkeypatch):
    """수집 뒤 파일이 바뀌어도 수집한 바이트 그대로 쪽 그림을 만든다(파일은 한 번만 읽는다)."""
    import hanji.cli
    from hanji.engine import LocalEngine

    pdf = write(tmp_path / "a.pdf", make_pdf("본문"))
    seen = {}
    ingest_bytes, render = LocalEngine.ingest_bytes, hanji.cli.render_page_images

    def ingest_then_change(self, data, *args, **kwargs):
        seen["ingested"] = data
        ref = ingest_bytes(self, data, *args, **kwargs)
        write(pdf, make_pdf("바뀐 본문"))
        return ref

    def record_render(data, *args, **kwargs):
        seen["rendered"] = data
        return render(data, *args, **kwargs)

    monkeypatch.setattr(LocalEngine, "ingest_bytes", ingest_then_change)
    monkeypatch.setattr(hanji.cli, "render_page_images", record_render)
    out = tmp_path / "a.html"
    assert run(capsys, "view", pdf, "--db", db, "--out", out)[0] == 0
    assert seen["rendered"] is seen["ingested"] and [b["text"] for b in view_data(out)["blocks"]] == ["본문"]


def test_view_writes_lone_surrogate_text_as_replacement(capsys, db, tmp_path, monkeypatch):
    """짝 없는 서로게이트가 든 블록 글자도 HTML을 쓴다(인코딩 못 하는 글자는 '?')."""
    from hanji.engine import LocalEngine

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


def test_view_default_output_directory_fails_before_store(capsys, db, tmp_path, monkeypatch):
    pdf = write(tmp_path / "입력" / "보고서.pdf", make_pdf("본문"))
    work = tmp_path / "작업"
    (work / "보고서.view.html").mkdir(parents=True)
    monkeypatch.chdir(work)
    code, out, err = run(capsys, "view", pdf, "--db", db)
    assert (code, out) == (1, "") and "Traceback" not in err
    assert not db.exists()


def test_view_failure_keeps_existing_html_and_leaves_no_temp(capsys, db, tmp_path, monkeypatch):
    """실패하면 이전 HTML은 바이트 그대로, 임시 파일은 남지 않는다."""
    import hanji.cli

    out_dir = tmp_path / "결과"
    out = write(out_dir / "a.html", b"old html\n")
    pdf = write(tmp_path / "a.pdf", make_pdf("본문"))

    def fail_render(*args, **kwargs):
        raise RuntimeError("render failed")

    with monkeypatch.context() as m:
        m.setattr(hanji.cli, "render_html", fail_render)
        code, stdout, err = run(capsys, "view", pdf, "--db", db, "--out", out)
    assert (code, stdout) == (1, "") and "render failed" in err
    assert out.read_bytes() == b"old html\n" and sorted(out_dir.iterdir()) == [out]

    replace = os.replace

    def fail_for_temp(src, dst, *args, **kwargs):  # 임시 파일 → out 바꿔 끼우기만 실패시킨다
        if os.path.dirname(src) == str(out_dir) and str(src).endswith(".tmp"):
            raise OSError("replace failed")
        return replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(hanji.cli.os, "replace", fail_for_temp)
    code, stdout, err = run(capsys, "view", pdf, "--db", db, "--out", out)
    assert (code, stdout) == (1, "") and "replace failed" in err
    assert out.read_bytes() == b"old html\n" and sorted(out_dir.iterdir()) == [out]


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_view_temp_file_never_touches_existing_tmp_path(capsys, db, tmp_path, kind):
    """임시 파일은 새로 만든 고유 이름: 이미 있는 <out>.tmp 파일·심볼릭 링크(가리키는 파일 포함)는 그대로."""
    out_dir = tmp_path / "결과"
    out = out_dir / "a.html"
    tmp = out_dir / "a.html.tmp"
    unrelated = write(tmp_path / "다른.txt", b"unrelated\n")
    if kind == "file":
        write(tmp, b"keep me\n")
    else:
        out_dir.mkdir()
        try:
            tmp.symlink_to(unrelated)
        except OSError:  # Windows에서 권한이 없으면 심볼릭 링크를 만들 수 없다
            pytest.skip("symlinks are not available")
    pdf = write(tmp_path / "a.pdf", make_pdf("본문"))
    assert run(capsys, "view", pdf, "--db", db, "--out", out)[0] == 0
    assert view_data(out)["blocks"][0]["text"] == "본문"
    assert unrelated.read_bytes() == b"unrelated\n"
    if kind == "file":
        assert tmp.read_bytes() == b"keep me\n"
    else:
        assert tmp.is_symlink() and tmp.resolve() == unrelated.resolve()
    assert sorted(p.name for p in out_dir.iterdir()) == ["a.html", "a.html.tmp"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_view_html_gets_normal_permissions(capsys, db, tmp_path):
    """임시 파일(0600)을 바꿔 끼워도 새 HTML은 umask 기본 권한, 이미 있던 HTML은 원래 권한."""
    md = write(tmp_path / "메모.md", "가\n".encode())
    fresh, existing = tmp_path / "새.html", write(tmp_path / "있던.html", b"old\n")
    existing.chmod(0o640)
    old_umask = os.umask(0o022)
    try:
        assert run(capsys, "view", md, "--db", db, "--out", fresh)[0] == 0
        assert run(capsys, "view", md, "--db", db, "--out", existing)[0] == 0
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o644
    assert stat.S_IMODE(existing.stat().st_mode) == 0o640 and b"old" not in existing.read_bytes()


def test_view_long_output_name_within_file_name_limit(capsys, db, tmp_path):
    """임시 파일 이름이 출력 이름보다 길어지지 않는다: 쓸 수 있는 긴 이름(245바이트)도 그대로 쓴다."""
    pdf = write(tmp_path / "a.pdf", make_pdf("가나다"))
    out = tmp_path / ("a" * 240 + ".html")
    code, stdout, _ = run(capsys, "view", pdf, "--db", db, "--out", out)
    assert code == 0 and out.is_file() and stdout.strip() == str(out)
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
