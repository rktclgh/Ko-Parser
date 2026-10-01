"""블록 식별자와 내용 해시. 원문은 보존하고 해시·비교에만 정규화를 쓴다."""

import hashlib
import json
import unicodedata

from .table import Table

NORMALIZATION_VERSION = "norm-v1"


def normalize_text(s: str) -> str:
    """NFKC 후 모든 공백 덩어리를 공백 하나로 바꾸고 양끝을 자른다."""
    return " ".join(unicodedata.normalize("NFKC", s).split())


def _canonical_table(table: Table | None) -> dict | None:
    if table is None:
        return None
    cells = sorted(table.cells, key=lambda cell: (cell.row, cell.col))
    return {
        "n_rows": table.n_rows,
        "n_cols": table.n_cols,
        "cells": [[c.row, c.col, c.rowspan, c.colspan, c.header, normalize_text(c.text)] for c in cells],
    }


def compute_content_hash(kind: str, text: str, level: int | None, table: Table | None) -> str:
    payload = {
        "v": NORMALIZATION_VERSION,
        "kind": kind,
        "text": normalize_text(text),
        "level": level,
        "table": _canonical_table(table),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "c_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def compute_block_id(document_id: str, content_hash: str, occurrence: int) -> str:
    """occurrence: 문서 안에서 order가 더 작은 블록 중 같은 content_hash를 가진 블록의 수(0부터)."""
    if occurrence < 0:
        raise ValueError("occurrence must be >= 0")
    raw = f"{document_id}\x1f{content_hash}\x1f{occurrence}"
    return "b_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
