import hashlib
import io
import json
import sqlite3
import sys

import pytest

from ko_parser import cli
from ko_parser.cli import main
from ko_parser.core import build_tree, diff_trees
from ko_parser.errors import AssetNotFound
from ko_parser.formats.base import ParsedSource
from ko_parser.store import SqliteStore
from ko_parser_contracts import ChangeBatch, DocumentTree, PageInfo, ProcessingHistory, SourceInfo


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("KO_PARSER_DB", raising=False)
    return tmp_path / "상태" / "state.db"


def write(path, text: str, encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_parse_prints_contract_json(capsys, db, tmp_path):
    path = write(tmp_path / "한글 폴더" / "보고서.md", "# 제목\n\n본문 문단\n")
    code, out, _ = run(capsys, "parse", path, "--db", db)
    assert code == 0
    tree = DocumentTree.model_validate_json(out)
    assert [b.text for b in tree.blocks] == ["제목", "본문 문단"] and tree.source.name == "보고서.md"
    assert "본문 문단" in out  # ensure_ascii=False


def test_parse_markdown_to_file(capsys, db, tmp_path):
    path = write(tmp_path / "a.md", "# 제목\n\n- 항목\n", encoding="cp949")
    out_file = tmp_path / "결과" / "a.md"
    code, out, _ = run(capsys, "parse", path, "--db", db, "--format", "md", "--out", out_file)
    assert (code, out) == (0, "")
    assert out_file.read_bytes() == "# 제목\n\n- 항목\n".encode("utf-8")


def test_reparse_with_id_and_force(capsys, db, tmp_path):
    path = write(tmp_path / "a.md", "가\n")
    assert run(capsys, "parse", path, "--db", db, "--id", "문서")[0] == 0
    write(path, "가\n\n나\n")
    code, out, _ = run(capsys, "parse", path, "--db", db, "--id", "문서")
    assert code == 0 and DocumentTree.model_validate_json(out).version == 2
    code, out, _ = run(capsys, "parse", path, "--db", db, "--id", "문서", "--force")
    assert code == 0 and DocumentTree.model_validate_json(out).version == 2


def test_export_documents_changes_history(capsys, db, tmp_path):
    path = write(tmp_path / "a.md", "가\n")
    run(capsys, "parse", path, "--db", db, "--id", "d1")
    write(path, "나\n")
    run(capsys, "parse", path, "--db", db, "--id", "d1")
    code, out, _ = run(capsys, "export", "d1", "--version", "1", "--db", db, "--format", "md")
    assert (code, out) == (0, "가\n")
    code, out, _ = run(capsys, "documents", "--db", db)
    assert code == 0 and json.loads(out) == [{"document_id": "d1", "version": 2, "layer_state": "det"}]
    code, out, _ = run(capsys, "changes", "--db", db, "--cursor", "1", "--limit", "5")
    batch = ChangeBatch.model_validate_json(out)
    assert code == 0 and (batch.cursor_from, batch.next_cursor, len(batch.changes)) == (1, 2, 1)
    code, out, _ = run(capsys, "history", "d1", "--db", db)
    assert code == 0 and ProcessingHistory.model_validate_json(out).version == 2


@pytest.mark.parametrize("argv", [[], ["parse"], ["unknown"], ["changes", "--limit", "0"],
                                  ["export", "d", "--version", "0"], ["changes", "--cursor", "-1"],
                                  ["parse", "a.md", "--id", ""]])
def test_usage_errors_exit_2(capsys, tmp_path, monkeypatch, argv):
    state = tmp_path / "state.db"
    monkeypatch.setenv("KO_PARSER_DB", str(state))
    assert run(capsys, *argv)[0] == 2  # 인자 해석 단계에서 끝나 상태 파일을 열지 않는다
    assert not state.exists()


def test_db_option_before_and_after_subcommand(capsys, tmp_path, monkeypatch):
    env_db = tmp_path / "env.db"
    monkeypatch.setenv("KO_PARSER_DB", str(env_db))
    path = write(tmp_path / "a.md", "가\n")
    before, after = tmp_path / "before.db", tmp_path / "after.db"
    assert run(capsys, "--db", before, "parse", path, "--id", "d1")[0] == 0
    assert run(capsys, "parse", path, "--db", after, "--id", "d2")[0] == 0
    assert before.exists() and after.exists() and not env_db.exists()
    assert [d["document_id"] for d in json.loads(run(capsys, "--db", before, "documents")[1])] == ["d1"]
    assert [d["document_id"] for d in json.loads(run(capsys, "documents", "--db", after)[1])] == ["d2"]


def test_db_option_given_twice_later_wins(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("KO_PARSER_DB", raising=False)
    first, second = tmp_path / "first.db", tmp_path / "second.db"
    assert run(capsys, "--db", first, "documents", "--db", second)[0] == 0
    assert second.exists() and not first.exists()


def test_parse_out_directory_fails_before_store(capsys, db, tmp_path):
    path = write(tmp_path / "a.md", "가\n")
    out_dir = tmp_path / "출력"
    out_dir.mkdir()
    code, out, err = run(capsys, "parse", path, "--db", db, "--out", out_dir)
    assert (code, out) == (1, "") and err.count("\n") == 1 and "Traceback" not in err
    assert not db.exists()
    assert json.loads(run(capsys, "documents", "--db", db)[1]) == []


def test_stderr_backslashreplaces_unencodable(monkeypatch):
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(raw, encoding="ascii", newline=""))
    cli._configure_streams()  # main()이 쓰는 것과 같은 재설정
    print("ko-parser: \udc80 문서", file=sys.stderr)  # 짝 없는 서로게이트가 든 메시지
    sys.stderr.flush()
    assert raw.getvalue() == "ko-parser: \\udc80 문서\n".encode("utf-8")


def test_help_exits_0(capsys):
    code, out, _ = run(capsys, "--help")
    assert code == 0 and "parse" in out


def test_unsupported_format_exit_3(capsys, db, tmp_path):
    code, _, err = run(capsys, "parse", write(tmp_path / "a.txt", "가"), "--db", db)
    assert code == 3 and "unsupported format" in err


def test_parse_failure_exit_4_and_store_unchanged(capsys, db, tmp_path):
    path = tmp_path / "깨진.md"
    path.write_bytes(b"\xff\xfe\x00")
    code, _, err = run(capsys, "parse", path, "--db", db)
    assert code == 4 and "깨진.md" in err
    assert json.loads(run(capsys, "documents", "--db", db)[1]) == []


def test_missing_document_or_version_exit_5(capsys, db, tmp_path):
    assert run(capsys, "export", "없음", "--db", db)[0] == 5
    run(capsys, "parse", write(tmp_path / "a.md", "가\n"), "--db", db, "--id", "d1")
    assert run(capsys, "export", "d1", "--version", "9", "--db", db)[0] == 5
    assert run(capsys, "history", "d1", "--version", "9", "--db", db)[0] == 5


def test_missing_file_exit_1(capsys, db, tmp_path):
    code, _, err = run(capsys, "parse", tmp_path / "없는 파일.md", "--db", db)
    assert code == 1 and "FileNotFoundError" in err
    assert err.count("\n") == 1 and "Traceback" not in err


def test_db_not_sqlite_exit_1(capsys, tmp_path):
    bad = write(tmp_path / "state.db", "이것은 SQLite 파일이 아니다. " * 20)
    code, out, err = run(capsys, "documents", "--db", bad)
    assert (code, out) == (1, "") and err.startswith("ko-parser: ")
    assert err.count("\n") == 1 and "Traceback" not in err


@pytest.mark.parametrize("old", ["1", "2"])  # 계약 0.1·0.2 시절 상태 파일
def test_db_from_older_contracts_exit_1(capsys, db, old):
    assert run(capsys, "documents", "--db", db)[0] == 0
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'format'", (old,))
    conn.close()
    code, out, err = run(capsys, "documents", "--db", db)
    assert (code, out) == (1, "") and err.startswith("ko-parser: ") and "ingest again" in err
    assert err.count("\n") == 1 and "Traceback" not in err


def test_deeply_nested_markdown_exit_4_and_store_unchanged(capsys, db, tmp_path):
    nested = "".join("  " * i + "- 항목\n" for i in range(12))
    code, _, err = run(capsys, "parse", write(tmp_path / "깊은.md", nested), "--db", db)
    assert code == 4 and "block nesting too deep" in err and "깊은.md" in err
    assert err.count("\n") == 1 and "Traceback" not in err
    assert json.loads(run(capsys, "documents", "--db", db)[1]) == []
    assert json.loads(run(capsys, "changes", "--db", db)[1])["changes"] == []


def test_db_priority(capsys, tmp_path, monkeypatch):
    option, env, default_dir = tmp_path / "opt.db", tmp_path / "env.db", tmp_path / "기본"
    monkeypatch.setattr(cli, "user_data_dir", lambda appname, appauthor: str(default_dir))
    monkeypatch.delenv("KO_PARSER_DB", raising=False)
    assert cli.resolve_db(None) == default_dir / "state.db"
    monkeypatch.setenv("KO_PARSER_DB", str(env))
    assert cli.resolve_db(None) == env
    assert cli.resolve_db(str(option)) == option
    run(capsys, "documents", "--db", option)
    run(capsys, "documents")
    assert option.exists() and env.exists() and not (default_dir / "state.db").exists()


def test_stdout_forced_to_utf8(monkeypatch, db, tmp_path):
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii", newline=""))
    path = write(tmp_path / "a.md", "# 한글 제목\n")
    assert main(["parse", str(path), "--db", str(db), "--format", "md"]) == 0
    assert raw.getvalue().decode("utf-8") == "# 한글 제목\n"


def seed_figure(db) -> tuple[str, bytes]:
    """그림 블록·캡션과 그 이미지를 상태 파일에 바로 넣는다(이 PR의 파서는 아직 그림을 내지 않는다)."""
    png = b"\x89PNG\r\n\x1a\n-figure-"
    asset = "sha256:" + hashlib.sha256(png).hexdigest()
    loc = {"kind": "page", "page": 1, "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.4}}
    specs = [{"kind": "figure", "text": "", "confidence": 0.7, "state": "det", "text_source": "text_layer",
              "locator": loc, "figure": {"asset": asset, "mime": "image/png", "width_px": 4, "height_px": 3,
                                         "dpi": 200, "category": "image", "caption_ref": 1}},
             {"kind": "caption", "text": "그림 1. 현황", "confidence": 0.7, "state": "det", "text_source": "text_layer",
              "locator": loc}]
    page = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144)
    source = SourceInfo(name="f.pdf", mime="application/pdf", content_hash="sha256:" + "0" * 64, page_count=1)
    tree = build_tree(ParsedSource(mime="application/pdf", pages=(page,), blocks=specs), "fig", 1, source)
    with SqliteStore(db) as store:
        store.commit(tree, diff_trees(None, tree), ProcessingHistory(document_id="fig", version=1), {asset: png})
    return asset, png


def test_export_assets_writes_pngs_and_markdown_links(capsys, db, tmp_path):
    asset, png = seed_figure(db)
    folder = tmp_path / "그림"
    code, out, _ = run(capsys, "export", "fig", "--db", db, "--format", "md", "--assets", folder)
    name = asset.removeprefix("sha256:")[:16] + ".png"
    assert code == 0 and (folder / name).read_bytes() == png
    first, rest = out.split("\n", 1)
    assert first.startswith("![그림 1. 현황](") and first.rstrip(">)").endswith(f"/{name}")
    assert rest == "\n그림 1. 현황\n"


def test_export_json_with_assets_writes_files_and_the_same_json(capsys, db, tmp_path):
    asset, png = seed_figure(db)
    plain = run(capsys, "export", "fig", "--db", db)[1]
    code, out, _ = run(capsys, "export", "fig", "--db", db, "--assets", tmp_path / "a")
    assert code == 0 and out == plain
    assert (tmp_path / "a" / (asset.removeprefix("sha256:")[:16] + ".png")).read_bytes() == png


def test_export_missing_asset_exit_5(capsys, db, tmp_path, monkeypatch):
    seed_figure(db)

    def missing(self, asset):
        raise AssetNotFound(asset)

    monkeypatch.setattr(SqliteStore, "get_asset", missing)
    code, out, err = run(capsys, "export", "fig", "--db", db, "--assets", tmp_path / "a")
    assert (code, out) == (5, "") and "sha256:" in err


def test_export_assets_path_that_is_a_file_exit_1(capsys, db, tmp_path):
    seed_figure(db)
    blocker = write(tmp_path / "file", "x")
    code, _, err = run(capsys, "export", "fig", "--db", db, "--assets", blocker)
    assert code == 1 and "FileExistsError" in err
