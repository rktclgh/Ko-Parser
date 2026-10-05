"""ko-parser 명령줄. main(argv)는 종료 코드를 돌려준다."""

import argparse
import json
import os
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from ko_parser_contracts import DocumentTree

from .engine import LocalEngine
from .errors import AssetNotFound, DocumentNotFound, KoParserError, ParseError, UnsupportedFormat, VersionNotFound
from .export import asset_name, to_markdown
from .formats.detect import default_parsers
from .formats.pdf.parser import MIME as PDF_MIME
from .store.sqlite import SqliteStore
from .viewer import DEFAULT_DPI, render_html, render_page_images

APP_NAME = "ko-parser"
DB_ENV = "KO_PARSER_DB"
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_UNSUPPORTED, EXIT_PARSE, EXIT_NOT_FOUND = 0, 1, 2, 3, 4, 5
MAX_DPI = 600
OCR_HELP = "스캔 쪽 OCR을 끈다 (기본: OCR 추가 설치가 있으면 켠다. 원본이 같으면 저장된 버전을 쓰니 바꾸려면 parse --force)"
ASSETS_HELP = ("그림 이미지를 이 폴더에 <sha256 앞 16자>.png로 쓴다(--format md면 마크다운이 그 파일을 가리킨다. 링크는 "
               "--out 파일 폴더 기준 상대 경로, 표준 출력이면 현재 폴더 기준)")
_EXIT_CODES: tuple[tuple[type[KoParserError], int], ...] = (
    (UnsupportedFormat, EXIT_UNSUPPORTED), (ParseError, EXIT_PARSE),
    (DocumentNotFound, EXIT_NOT_FOUND), (VersionNotFound, EXIT_NOT_FOUND), (AssetNotFound, EXIT_NOT_FOUND),
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


def _dpi(text: str) -> int:
    value = int(text)
    if not 1 <= value <= MAX_DPI:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_DPI}")
    return value


def _non_empty(text: str) -> str:
    if not text:
        raise argparse.ArgumentTypeError("must not be empty")
    return text


def _configure_streams() -> None:
    """stdout은 UTF-8 고정, stderr는 인코딩 못 하는 글자(짝 없는 서로게이트 등)를 이스케이프해 예외를 막는다."""
    for stream, errors in ((sys.stdout, "strict"), (sys.stderr, "backslashreplace")):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # 콘솔 기본 인코딩(Windows cp949 등)에 기대지 않는다
            reconfigure(encoding="utf-8", errors=errors)


def _build_parser() -> argparse.ArgumentParser:
    db_help = f"상태 파일 경로 (기본: ${DB_ENV} 또는 사용자 데이터 폴더/state.db)"
    common = argparse.ArgumentParser(add_help=False)
    # 하위 명령 기본값이 최상위 값을 덮어쓰지 않도록 SUPPRESS. 둘 다 주면 뒤(하위 명령 쪽)가 이긴다.
    common.add_argument("--db", default=argparse.SUPPRESS, help=db_help)
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("--format", choices=("json", "md"), default="json")
    output.add_argument("--out", help="출력 파일 경로 (기본: 표준 출력)")

    parser = argparse.ArgumentParser(prog=APP_NAME, description="한국어 특화 문서 파싱 엔진")
    parser.add_argument("--db", help=db_help)
    sub = parser.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("parse", parents=[common, output], help="파일을 파싱·저장하고 문서 트리를 출력")
    parse.add_argument("file")
    parse.add_argument("--id", dest="document_id", type=_non_empty, help="문서 ID (기본: doc_ + 원본 sha256 앞 24자리)")
    parse.add_argument("--force", action="store_true", help="원본이 같아도 다시 파싱")
    parse.add_argument("--no-ocr", action="store_true", help=OCR_HELP)
    export = sub.add_parser("export", parents=[common, output], help="저장된 문서 트리를 출력")
    export.add_argument("document_id")
    export.add_argument("--version", type=_positive)
    export.add_argument("--assets", type=_non_empty, help=ASSETS_HELP)
    sub.add_parser("documents", parents=[common], help="문서마다 최신 버전")
    changes = sub.add_parser("changes", parents=[common], help="커서 뒤의 변경 내역")
    changes.add_argument("--cursor", type=_non_negative)
    changes.add_argument("--limit", type=_positive, default=100)
    history = sub.add_parser("history", parents=[common], help="처리 이력")
    history.add_argument("document_id")
    history.add_argument("--version", type=_positive)
    view = sub.add_parser("view", parents=[common], help="파싱 결과를 HTML 한 장으로 만든다(원본이 같으면 저장된 버전)")
    view.add_argument("file")
    view.add_argument("--id", dest="document_id", type=_non_empty, help="문서 ID (기본: doc_ + 원본 sha256 앞 24자리)")
    view.add_argument("--out", type=_non_empty, help="HTML 경로 (기본: 현재 폴더/<파일 이름(확장자 제외)>.view.html)")
    view.add_argument("--dpi", type=_dpi, default=DEFAULT_DPI, help=f"쪽 이미지 해상도 (기본 {DEFAULT_DPI})")
    view.add_argument("--no-ocr", action="store_true", help=OCR_HELP)
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


def _emit_tree(tree: DocumentTree, args: argparse.Namespace, assets_dir: str | None = None) -> None:
    _write(to_markdown(tree, assets_dir) if args.format == "md" else _json(tree.model_dump(mode="json")), args.out)


def _assets_link(assets: str, out: str | None) -> str:
    """마크다운 링크의 폴더 부분('/' 구분). --out이 있으면 그 파일 폴더 기준 상대 경로, 표준 출력이면 현재 폴더 기준
    입력 그대로. 상대 경로가 없거나(Windows 다른 드라이브) 드라이브가 붙은 경로를 표준 출력에 쓰면 file:// URI."""
    folder = Path(assets)
    try:
        if out is not None:
            return Path(os.path.relpath(folder.resolve(), Path(out).resolve().parent)).as_posix()
        if not folder.drive:  # C:·UNC 경로가 C:/… 같은 스킴 모양 링크가 되지 않게
            return folder.as_posix()
    except ValueError:  # Windows: 드라이브가 다르면 상대 경로가 없다
        pass
    return folder.resolve().as_uri()


def _write_assets(engine: LocalEngine, tree: DocumentTree, folder: Path) -> None:
    """트리의 그림 이미지를 폴더에 쓴다(같은 이미지는 한 번). 없는 이미지는 AssetNotFound."""
    folder.mkdir(parents=True, exist_ok=True)
    for asset in dict.fromkeys(b.figure.asset for b in tree.blocks if b.figure is not None):
        (folder / asset_name(asset)).write_bytes(engine.get_asset(asset))


def view_path(file: str, out: str | None) -> Path:
    """--out이 없으면 현재 폴더의 <파일 이름(확장자 제외, NFC)>.view.html."""
    if out:
        return Path(out)
    return Path.cwd() / (unicodedata.normalize("NFC", Path(file).stem) + ".view.html")


def _view(args: argparse.Namespace, engine: LocalEngine) -> None:
    data = Path(args.file).read_bytes()  # 한 번만 읽어 수집과 쪽 그림에 같은 바이트를 쓴다
    ref = engine.ingest_bytes(data, Path(args.file).name, document_id=args.document_id)
    tree = engine.get_tree(ref.document_id, ref.version)
    previous = engine.get_tree(ref.document_id, ref.version - 1) if ref.version > 1 else None
    images = None
    if tree.source.mime == PDF_MIME:
        images = render_page_images(data, tree.source.name, args.dpi)
    out = view_path(args.file, args.out)
    html = render_html(tree, images, previous)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 같은 폴더의 새 임시 파일(배타적으로 만든 고유 이름)에 다 쓴 뒤 바꿔 끼운다: 실패해도 이전 HTML이 반쯤 덮이지 않는다
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".ko-parser-view.", suffix=".tmp")  # 짧은 이름: 긴 출력 이름도 이름 길이 한도를 넘지 않게
    try:
        # 문서 글자에 짝 없는 서로게이트가 있어도 쓴다(인코딩 못 하는 글자는 "?")
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace", newline="\n") as f:
            f.write(html)
        if out.exists():
            mode = stat.S_IMODE(out.stat().st_mode)
        else:
            umask = os.umask(0)  # umask는 읽으려면 바꿔야 한다(CLI는 한 스레드라 바로 되돌린다)
            os.umask(umask)
            mode = 0o666 & ~umask
        try:  # mkstemp는 0600으로 만든다: 이미 있던 HTML의 권한 또는 umask 기본 권한으로
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # 지우지 못해도 원래 오류를 가리지 않는다
            pass
        raise
    _write(f"{out}\n", None)


def _run(args: argparse.Namespace, engine: LocalEngine) -> None:
    match args.command:
        case "parse":
            ref = engine.ingest(args.file, document_id=args.document_id, force=args.force)
            _emit_tree(engine.get_tree(ref.document_id, ref.version), args)
        case "export":
            tree = engine.get_tree(args.document_id, args.version)
            if args.assets:
                _write_assets(engine, tree, Path(args.assets))
            _emit_tree(tree, args, _assets_link(args.assets, args.out) if args.assets else None)
        case "documents":
            _write(_json([ref.model_dump(mode="json") for ref in engine.documents()]), None)
        case "changes":
            _write(_json(engine.changes(args.cursor, args.limit).model_dump(mode="json")), None)
        case "history":
            _write(_json(engine.history(args.document_id, args.version).model_dump(mode="json")), None)
        case "view":
            _view(args, engine)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_streams()
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse: 사용법 오류 2, --help 0
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        out = (view_path(args.file, args.out) if args.command == "view"
               else args.out if args.command in ("parse", "export") else None)
        if out and Path(out).is_dir():  # 저장소·그림 파일을 건드리기 전에 막는다(view는 기본 출력 경로도)
            raise IsADirectoryError(f"output path is a directory: {out}")
        with SqliteStore(resolve_db(args.db)) as store:
            parsers = default_parsers(ocr=False) if getattr(args, "no_ocr", False) else None
            _run(args, LocalEngine(store, parsers))
    except KoParserError as exc:
        print(f"{APP_NAME}: {exc}", file=sys.stderr)
        return next((code for kind, code in _EXIT_CODES if isinstance(exc, kind)), EXIT_ERROR)
    except Exception as exc:  # 그 밖(파일 없음·권한 등)
        print(f"{APP_NAME}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK
