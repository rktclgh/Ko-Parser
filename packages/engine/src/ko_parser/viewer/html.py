"""DocumentTree → HTML 한 장(순수 함수). 외부 리소스를 요청하지 않는다.

문서 글자는 모두 데이터 JSON(<script type="application/json">)에만 들어가고 화면에는 textContent로 넣는다.
JSON 안의 '<'는 모두 \\u003c로 바꿔 '</script>'·'<!--'가 스크립트 블록을 끝내거나 바꾸지 못하게 한다.
"""

import base64
import html
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ko_parser_contracts import Block, DocumentTree, LinesLocator, PageLocator

from ..core import diff_trees


def _image_uri(data: bytes) -> str:
    mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _where(block: Block) -> str:
    loc = block.locator
    if isinstance(loc, PageLocator):
        return f"{loc.page}쪽"
    if isinstance(loc, LinesLocator):
        return f"{loc.line_start}~{loc.line_end}줄"
    if loc.kind == "slide":
        return f"슬라이드 {loc.slide}"
    return f"문단 {loc.paragraph_index}"


def _text(block: Block) -> str:
    """보이는 글자. 표 블록은 마크다운 표(살아 있는 블록과 사라진 블록 모두. 화면에는 격자로 그린다)."""
    return block.table.to_markdown() if block.table is not None else block.text


def _table(block: Block) -> dict[str, Any] | None:
    """표 격자(행·열 순서의 칸). 화면은 이것으로 <table>을 DOM으로 만든다(글자는 textContent)."""
    if block.table is None:
        return None
    cells = sorted(block.table.cells, key=lambda c: (c.row, c.col))
    return {"n_rows": block.table.n_rows, "n_cols": block.table.n_cols,
            "cells": [{"row": c.row, "col": c.col, "rowspan": c.rowspan, "colspan": c.colspan, "text": c.text,
                       "header": c.header} for c in cells]}


# 쪽 상태별 안내. scanned는 보이는 글자만 블록이 되고, unreliable은 블록이 없다(스펙 2026-10-03 보정)
_NOTICE = {"scanned": "그림 속 글자는 OCR 필요(보이는 글자만 블록)", "unreliable": "글자가 깨져 블록을 만들지 않았다"}


def _block(block: Block, change: str | None) -> dict[str, Any]:
    loc = block.locator
    page = isinstance(loc, PageLocator)
    return {
        "id": block.block_id, "order": block.order, "kind": block.kind, "level": block.level,
        "text": _text(block), "table": _table(block),
        "section_path": list(block.section_path), "text_source": block.text_source, "state": block.state,
        "confidence": block.confidence, "where": _where(block), "change": change,
        "page": loc.page if page else None,
        "bbox": [loc.bbox.x0, loc.bbox.y0, loc.bbox.x1, loc.bbox.y1] if page else None,
    }


def view_data(tree: DocumentTree, page_images: Mapping[int, bytes] | None = None,
              previous: DocumentTree | None = None) -> dict[str, Any]:
    """뷰어가 쓰는 데이터. previous는 같은 문서의 이전 버전(added·updated 표시, removed 목록)."""
    change = diff_trees(previous, tree) if previous is not None else None
    added = set(change.added) if change else set()
    updated = set(change.updated) if change else set()
    removed = set(change.removed) if change else set()
    images = page_images or {}
    states = Counter(p.text_layer for p in tree.pages)
    return {
        "document": {"name": tree.source.name, "document_id": tree.document_id, "version": tree.version,
                     "layer_state": tree.layer_state, "mime": tree.source.mime,
                     "previous_version": previous.version if previous is not None else None,
                     "page_states": dict(sorted(states.items()))},
        "pages": [{"page": p.page, "width": p.width_pt, "height": p.height_pt, "state": p.text_layer,
                   "notice": _NOTICE.get(p.text_layer),
                   "stats": p.text_stats.model_dump() if p.text_stats is not None else None,
                   "image": _image_uri(images[p.page]) if p.page in images else None} for p in tree.pages],
        "blocks": [_block(b, "added" if b.block_id in added else "updated" if b.block_id in updated else None)
                   for b in tree.blocks],
        "removed": [{"id": b.block_id, "kind": b.kind, "text": _text(b), "table": _table(b), "where": _where(b)}
                    for b in (previous.blocks if previous is not None else ()) if b.block_id in removed],
    }


def render_html(tree: DocumentTree, page_images: Mapping[int, bytes] | None = None,
                previous: DocumentTree | None = None) -> str:
    """page_images: 쪽 번호 → JPEG(또는 PNG) 바이트. 쪽이 없는 문서(MD)는 블록 목록만 보인다."""
    data = json.dumps(view_data(tree, page_images, previous), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)
    title = html.escape(f"{tree.source.name} — ko-parser 뷰어")
    return _BEFORE_TITLE + title + _BEFORE_DATA + data.replace("<", "\\u003c") + _AFTER_DATA


_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>__TITLE__</title>
<style>
:root { --line: #d0d4d9; --muted: #5f6b76; --bg: #f6f7f9; --sel: #ffd43b; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 system-ui, -apple-system, "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
       color: #1b1f24; background: var(--bg); }
header { position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid var(--line); padding: 8px 16px; }
header h1 { font-size: 16px; margin: 0 0 4px; }
.meta, .filters { color: var(--muted); font-size: 12px; display: flex; flex-wrap: wrap; gap: 4px 14px; }
.filters label { cursor: pointer; }
main { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 40%); gap: 16px; padding: 16px; }
main.no-pages { grid-template-columns: minmax(0, 1fr); }
.pages { display: flex; flex-direction: column; gap: 16px; }
.page { position: relative; background: #fff; border: 1px solid var(--line); }
.page img { display: block; width: 100%; height: 100%; }
.page-label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
.notice { background: #fff4e6; border: 1px solid #f08c00; padding: 8px 12px; margin-bottom: 4px; }
.box { position: absolute; border: 1.5px solid var(--c); background: color-mix(in srgb, var(--c) 8%, transparent); cursor: pointer; }
.box.changed { outline: 2px dashed #e03131; outline-offset: 1px; }
.box.selected, .item.selected { box-shadow: 0 0 0 3px var(--sel); }
.list { display: flex; flex-direction: column; gap: 6px; }
.item { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--c); padding: 6px 8px; cursor: pointer; }
.item .text { white-space: pre-wrap; word-break: break-word; margin-top: 2px; }
.badge { display: inline-block; font-size: 11px; padding: 0 6px; border-radius: 8px; background: #eef0f3; margin-right: 4px; }
.badge.kind { background: var(--c); color: #fff; }
.badge.added { background: #2f9e44; color: #fff; }
.badge.updated { background: #e8590c; color: #fff; }
.grid-wrap { overflow-x: auto; margin-top: 4px; }
.grid { border-collapse: collapse; font-size: 12px; }
.grid td, .grid th { border: 1px solid var(--line); padding: 2px 6px; white-space: pre-wrap; word-break: keep-all;
                     overflow-wrap: normal; vertical-align: top; text-align: left; }
.grid th { background: #eef0f3; font-weight: 600; }
.removed { margin-top: 16px; }
.removed .item { --c: #adb5bd; text-decoration: line-through; color: var(--muted); cursor: default; }
.hidden { display: none !important; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="meta" id="meta"></div>
  <div class="filters" id="filters"></div>
</header>
<main id="main">
  <section class="pages" id="pages"></section>
  <section>
    <div class="list" id="list"></div>
    <div class="removed" id="removed"></div>
  </section>
</main>
<script type="application/json" id="ko-data">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("ko-data").textContent);
const COLORS = { heading: "#d9480f", paragraph: "#1971c2", list_item: "#2f9e44", table: "#ae3ec9", figure: "#f08c00",
                 caption: "#0c8599", page_header: "#868e96", page_footer: "#868e96" };
const STATE_LABEL = { digital: "디지털", scanned: "스캔", unreliable: "글자 깨짐" };
const pct = (v) => (v * 100).toFixed(1) + "%";
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
const doc = DATA.document;
document.getElementById("title").textContent = doc.name;
const meta = document.getElementById("meta");
const states = Object.entries(doc.page_states).map(([k, n]) => (STATE_LABEL[k] || k) + " " + n).join(" · ");
[["문서 ID", doc.document_id], ["버전", doc.version + (doc.previous_version ? " (이전 " + doc.previous_version + " 대비)" : "")],
 ["레이어", doc.layer_state], ["형식", doc.mime], ["쪽", DATA.pages.length ? DATA.pages.length + "쪽 (" + states + ")" : "없음"],
 ["블록", DATA.blocks.length]].forEach(([k, v]) => meta.appendChild(el("span", "", k + ": " + v)));

// 표 블록: 칸 목록으로 <table>을 DOM으로 만든다. 글자는 textContent(줄바꿈은 CSS pre-wrap), 머리 칸은 <th>.
// 낱말·숫자는 칸 안에서 끊지 않고(keep-all) 넓은 표는 가로로 스크롤한다
function grid(t) {
  const wrap = el("div", "grid-wrap");
  const table = el("table", "grid");
  const body = el("tbody");
  const rows = [];
  for (let r = 0; r < t.n_rows; r++) rows.push(body.appendChild(el("tr")));
  for (const c of t.cells) {
    const cell = el(c.header === "none" ? "td" : "th", "", c.text);
    if (c.header === "column") cell.scope = "col";
    else if (c.header === "row") cell.scope = "row";
    if (c.rowspan > 1) cell.rowSpan = c.rowspan;
    if (c.colspan > 1) cell.colSpan = c.colspan;
    rows[c.row].appendChild(cell);
  }
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}
const boxes = new Map();
const items = new Map();
function select(id, from) {
  document.querySelectorAll(".selected").forEach((n) => n.classList.remove("selected"));
  const box = boxes.get(id), item = items.get(id);
  if (box) box.classList.add("selected");
  if (item) item.classList.add("selected");
  const target = from === "box" ? item : box;
  if (target) target.scrollIntoView({ block: "center", behavior: "smooth" });
}

const pagesNode = document.getElementById("pages");
const pageNodes = new Map();
if (!DATA.pages.length) document.getElementById("main").classList.add("no-pages");
for (const p of DATA.pages) {
  // 쪽 이름표와 안내는 그림 바깥(위)에 둔다: 그림 위에 겹치면 위쪽 블록 상자를 가린다
  const wrap = el("div", "page-wrap");
  wrap.appendChild(el("span", "page-label", p.page + "쪽 · " + (STATE_LABEL[p.state] || p.state)));
  const node = el("div", "page");
  node.style.aspectRatio = p.width + " / " + p.height;
  node.dataset.page = p.page;
  if (p.image) {
    const img = el("img");
    img.src = p.image;
    img.alt = p.page + "쪽";
    node.appendChild(img);
  }
  if (p.notice) {
    const s = p.stats;
    const reasons = s ? ". 보이는 글자 " + s.chars + " · 숨은 글자 " + pct(s.invisible_ratio) + " · 매핑 실패 " +
      pct(s.unmapped_ratio) + " · PUA " + pct(s.pua_ratio) + " · 가장 큰 그림 " + pct(s.max_image_coverage) : "";
    wrap.appendChild(el("div", "notice", (STATE_LABEL[p.state] || p.state) + " 쪽: " + p.notice + reasons));
  }
  wrap.appendChild(node);
  pageNodes.set(p.page, node);
  pagesNode.appendChild(wrap);
}

const list = document.getElementById("list");
const pending = [];
for (const b of DATA.blocks) {
  const color = COLORS[b.kind] || "#495057";
  const item = el("div", "item");
  item.style.setProperty("--c", color);
  item.dataset.kind = b.kind;
  item.dataset.state = b.state;
  const head = el("div");
  head.appendChild(el("span", "badge kind", b.kind + (b.level ? " " + b.level : "")));
  if (b.change) head.appendChild(el("span", "badge " + b.change, b.change));
  head.appendChild(el("span", "badge", b.text_source));
  head.appendChild(el("span", "badge", b.state));
  head.appendChild(el("span", "badge", "신뢰도 " + b.confidence.toFixed(2)));
  head.appendChild(el("span", "badge", "#" + b.order + " · " + b.where));
  item.appendChild(head);
  if (b.section_path.length) item.appendChild(el("div", "meta", b.section_path.join(" › ")));
  item.appendChild(b.table ? grid(b.table) : el("div", "text", b.text));
  item.addEventListener("click", () => select(b.id, "item"));
  items.set(b.id, item);
  list.appendChild(item);
  const page = b.page !== null ? pageNodes.get(b.page) : undefined;
  if (page && b.bbox) {
    const box = el("div", "box" + (b.change ? " changed" : ""));
    box.style.setProperty("--c", color);
    box.style.left = b.bbox[0] * 100 + "%";
    box.style.top = b.bbox[1] * 100 + "%";
    box.style.width = (b.bbox[2] - b.bbox[0]) * 100 + "%";
    box.style.height = (b.bbox[3] - b.bbox[1]) * 100 + "%";
    box.dataset.kind = b.kind;
    box.dataset.state = b.state;
    box.title = b.kind + " #" + b.order;
    box.addEventListener("click", () => select(b.id, "box"));
    boxes.set(b.id, box);
    pending.push({ page, box, area: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) });
  }
}
// 큰 상자부터 붙여 작은 상자가 위에 오게 한다(표·그림 안의 블록도 누를 수 있게). 목록 순서는 그대로
pending.sort((x, y) => y.area - x.area).forEach(({ page, box }) => page.appendChild(box));

if (DATA.removed.length) {
  const removed = document.getElementById("removed");
  removed.appendChild(el("h2", "", "이전 버전에서 사라진 블록 " + DATA.removed.length));
  for (const r of DATA.removed) {
    const item = el("div", "item");
    item.appendChild(el("span", "badge", r.kind + " · " + r.where));
    item.appendChild(r.table ? grid(r.table) : el("div", "text", r.text));
    removed.appendChild(item);
  }
}

const filters = document.getElementById("filters");
const hidden = { kind: new Set(), state: new Set() };
function applyFilters() {
  for (const node of [...items.values(), ...boxes.values()]) {
    node.classList.toggle("hidden", hidden.kind.has(node.dataset.kind) || hidden.state.has(node.dataset.state));
  }
}
for (const field of ["kind", "state"]) {
  const values = [...new Set(DATA.blocks.map((b) => b[field]))].sort();
  if (!values.length) continue;
  filters.appendChild(el("span", "", field === "kind" ? "종류:" : "상태:"));
  for (const value of values) {
    const label = el("label");
    const box = el("input");
    box.type = "checkbox";
    box.checked = true;
    box.addEventListener("change", () => { box.checked ? hidden[field].delete(value) : hidden[field].add(value); applyFilters(); });
    label.appendChild(box);
    label.appendChild(document.createTextNode(" " + value));
    filters.appendChild(label);
  }
}
</script>
</body>
</html>
"""

# 문서 글자에 자리표시 문자열이 있어도 다시 치환되지 않도록 미리 잘라 둔다
_BEFORE_TITLE, _REST = _TEMPLATE.split("__TITLE__")
_BEFORE_DATA, _AFTER_DATA = _REST.split("__DATA__")
