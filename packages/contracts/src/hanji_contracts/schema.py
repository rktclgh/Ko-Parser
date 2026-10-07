"""루트 모델의 JSON Schema를 생성·확인한다.

    uv run python -m hanji_contracts.schema           # schemas/ 재생성
    uv run python -m hanji_contracts.schema --check   # 저장된 스키마가 최신인지 확인(다르면 종료 코드 1)
"""

import argparse
import json
import sys
from pathlib import Path

from .changes import ChangeBatch
from .document import DocumentTree
from .provenance import ProcessingHistory
from .testing.replay import Recording
from .vlm import VlmRequest, VlmResult

ROOT_MODELS = {
    "document_tree": DocumentTree,
    "processing_history": ProcessingHistory,
    "change_batch": ChangeBatch,
    "vlm_request": VlmRequest,
    "vlm_result": VlmResult,
    "recording": Recording,
}

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def render_schemas() -> dict[str, str]:
    return {
        f"{name}.schema.json": json.dumps(model.model_json_schema(mode="validation"), ensure_ascii=False,
                                          indent=2, sort_keys=True) + "\n"
        for name, model in ROOT_MODELS.items()
    }


def export_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, text in render_schemas().items():
        path = out_dir / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def check_schemas(out_dir: Path) -> list[str]:
    """저장된 파일과 다르거나 없는 스키마 파일 이름, 뒤이어 루트 모델에 없는 고아 *.schema.json 이름(정렬)."""
    rendered = render_schemas()
    stale = []
    for filename, text in rendered.items():
        path = out_dir / filename
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(filename)
    orphans = sorted(p.name for p in out_dir.glob("*.schema.json") if p.name not in rendered) if out_dir.is_dir() else []
    return stale + orphans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hanji-contracts JSON Schema")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=SCHEMA_DIR)
    args = parser.parse_args(argv)
    if args.check:
        problems = check_schemas(args.out)
        known = render_schemas()
        for name in problems:
            print(f"{'stale' if name in known else 'orphan'} schema: {name}", file=sys.stderr)
        return 1 if problems else 0
    for path in export_schemas(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
