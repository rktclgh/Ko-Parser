"""ko-parser 명령줄. main(argv)는 종료 코드를 돌려준다."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from ko_parser_contracts import DocumentTree

from .engine import LocalEngine
from .errors import DocumentNotFound, KoParserError, ParseError, UnsupportedFormat, VersionNotFound
from .export import to_markdown
from .store.sqlite import SqliteStore

APP_NAME = "ko-parser"
DB_ENV = "KO_PARSER_DB"
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_UNSUPPORTED, EXIT_PARSE, EXIT_NOT_FOUND = 0, 1, 2, 3, 4, 5
_EXIT_CODES: tuple[tuple[type[KoParserError], int], ...] = (
    (UnsupportedFormat, EXIT_UNSUPPORTED), (ParseError, EXIT_PARSE),
    (DocumentNotFound, EXIT_NOT_FOUND), (VersionNotFound, EXIT_NOT_FOUND),
)


def resolve_db(option: str | None) -> Path:
    """--db > 환경변수 KO_PARSER_DB > 사용자 데이터 폴더/state.db."""
    if option:
        return Path(option)
    env = os.environ.get(DB_ENV)
    if env:
        return Path(env)
    return Path(user_data_dir(APP_NAME, appauthor=False)) / "state.db"


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help=f"상태 파일 경로 (기본: ${DB_ENV} 또는 사용자 데이터 폴더/state.db)")
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("--format", choices=("json", "md"), default="json")
    output.add_argument("--out", help="출력 파일 경로 (기본: 표준 출력)")

    parser = argparse.ArgumentParser(prog=APP_NAME, description="한국어 특화 문서 파싱 엔진")
    sub = parser.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("parse", parents=[common, output], help="파일을 파싱·저장하고 문서 트리를 출력")
    parse.add_argument("file")
    parse.add_argument("--id", dest="document_id", help="문서 ID (기본: doc_ + 원본 sha256 앞 24자리)")
    parse.add_argument("--force", action="store_true", help="원본이 같아도 다시 파싱")
    export = sub.add_parser("export", parents=[common, output], help="저장된 문서 트리를 출력")
    export.add_argument("document_id")
    export.add_argument("--version", type=_positive)
    sub.add_parser("documents", parents=[common], help="문서마다 최신 버전")
    changes = sub.add_parser("changes", parents=[common], help="커서 뒤의 변경 내역")
    changes.add_argument("--cursor", type=_non_negative)
    changes.add_argument("--limit", type=_positive, default=100)
    history = sub.add_parser("history", parents=[common], help="처리 이력")
    history.add_argument("document_id")
    history.add_argument("--version", type=_positive)
    return parser


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _write(text: str, out: str | None) -> None:
    if out is None:
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _emit_tree(tree: DocumentTree, args: argparse.Namespace) -> None:
    _write(to_markdown(tree) if args.format == "md" else _json(tree.model_dump(mode="json")), args.out)


def _run(args: argparse.Namespace, engine: LocalEngine) -> None:
    match args.command:
        case "parse":
            ref = engine.ingest(args.file, document_id=args.document_id, force=args.force)
            _emit_tree(engine.get_tree(ref.document_id, ref.version), args)
        case "export":
            _emit_tree(engine.get_tree(args.document_id, args.version), args)
        case "documents":
            _write(_json([ref.model_dump(mode="json") for ref in engine.documents()]), None)
        case "changes":
            _write(_json(engine.changes(args.cursor, args.limit).model_dump(mode="json")), None)
        case "history":
            _write(_json(engine.history(args.document_id, args.version).model_dump(mode="json")), None)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # 콘솔 기본 인코딩(Windows cp949 등)에 기대지 않는다
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse: 사용법 오류 2, --help 0
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        with SqliteStore(resolve_db(args.db)) as store:
            _run(args, LocalEngine(store))
    except KoParserError as exc:
        print(f"{APP_NAME}: {exc}", file=sys.stderr)
        return next((code for kind, code in _EXIT_CODES if isinstance(exc, kind)), EXIT_ERROR)
    except Exception as exc:  # 그 밖(파일 없음·권한 등)
        print(f"{APP_NAME}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK
