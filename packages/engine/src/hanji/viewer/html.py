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

from hanji_contracts import Block, DocumentTree, LinesLocator, PageLocator

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


# 쪽 상태별 안내. scanned는 보이는 글자만 블록이 되고(OCR 블록이 있으면 OCR로 읽음), unreliable은 깨진 글자층으로
# 만든 블록이라 믿기 어렵다(신뢰도 0.2 이하)
_NOTICE = {"scanned": "그림 속 글자는 OCR 필요(보이는 글자만 블록)",
           "unreliable": "글자층이 깨져 블록 글자를 믿기 어렵다(신뢰도 0.2 이하)"}
_NOTICE_OCR = "그림 속 글자는 OCR로 읽음(검증 전)"
LAYOUT_NOTICE = ('선·도형 그림·캡션은 레이아웃 추가 설치가 필요(pip install "hanji[layout]", '
                 "hanji models fetch layout)")


def _notice(state: str, page: int, ocr_pages: set[int]) -> str | None:
    return _NOTICE_OCR if state == "scanned" and page in ocr_pages else _NOTICE.get(state)


def _block(block: Block, change: str | None, figure_of: str | None = None) -> dict[str, Any]:
    """figure_of는 이 블록이 캡션인 그림 블록 id(없으면 None)."""
    loc = block.locator
    page = isinstance(loc, PageLocator)
    return {
        "id": block.block_id, "order": block.order, "kind": block.kind, "level": block.level,
        "text": _text(block), "table": _table(block),
        "figure": ({"category": block.figure.category, "caption": block.figure.caption_block_id}
                   if block.figure is not None else None),
        "figure_of": figure_of,
        "section_path": list(block.section_path), "text_source": block.text_source, "state": block.state,
        "confidence": block.confidence, "where": _where(block), "change": change,
        "page": loc.page if page else None,
        "bbox": [loc.bbox.x0, loc.bbox.y0, loc.bbox.x1, loc.bbox.y1] if page else None,
    }


def view_data(tree: DocumentTree, page_images: Mapping[int, bytes] | None = None,
              previous: DocumentTree | None = None, layout_notice: bool = False) -> dict[str, Any]:
    """뷰어가 쓰는 데이터. previous는 같은 문서의 이전 버전(added·updated 표시, removed 목록). layout_notice가 참이면
    레이아웃 추가 설치 안내를 문서 머리에 보인다."""
    change = diff_trees(previous, tree) if previous is not None else None
    added = set(change.added) if change else set()
    updated = set(change.updated) if change else set()
    removed = set(change.removed) if change else set()
    images = page_images or {}
    states = Counter(p.text_layer for p in tree.pages)
    ocr_pages = {b.locator.page for b in tree.blocks if b.text_source == "ocr" and isinstance(b.locator, PageLocator)}
    owners = {b.figure.caption_block_id: b.block_id for b in tree.blocks
              if b.figure is not None and b.figure.caption_block_id is not None}
    return {
        "document": {"name": tree.source.name, "document_id": tree.document_id, "version": tree.version,
                     "layer_state": tree.layer_state, "mime": tree.source.mime,
                     "previous_version": previous.version if previous is not None else None,
                     "page_states": dict(sorted(states.items())),
                     "layout_notice": LAYOUT_NOTICE if layout_notice else None},
        "pages": [{"page": p.page, "width": p.width_pt, "height": p.height_pt, "state": p.text_layer,
                   "notice": _notice(p.text_layer, p.page, ocr_pages),
                   "stats": p.text_stats.model_dump() if p.text_stats is not None else None,
                   "image": _image_uri(images[p.page]) if p.page in images else None} for p in tree.pages],
        "blocks": [_block(b, "added" if b.block_id in added else "updated" if b.block_id in updated else None,
                          owners.get(b.block_id)) for b in tree.blocks],
        "removed": [{"id": b.block_id, "kind": b.kind, "text": _text(b), "table": _table(b), "where": _where(b)}
                    for b in (previous.blocks if previous is not None else ()) if b.block_id in removed],
    }


def render_html(tree: DocumentTree, page_images: Mapping[int, bytes] | None = None,
                previous: DocumentTree | None = None, layout_notice: bool = False) -> str:
    """page_images: 쪽 번호 → JPEG(또는 PNG) 바이트. 쪽이 없는 문서(MD)는 블록 목록만 보인다. layout_notice: 레이아웃
    추가 설치 안내를 보인다(PDF인데 설치가 없을 때)."""
    data = json.dumps(view_data(tree, page_images, previous, layout_notice), ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
    title = html.escape(f"{tree.source.name} — hanji 뷰어")
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
.grid-note { font-size: 11px; color: var(--muted); margin-top: 2px; }
.removed { margin-top: 16px; }
.removed .item { --c: #adb5bd; text-decoration: line-through; color: var(--muted); cursor: default; }
.thumb { width: min(100%, 240px); margin-top: 4px; border: 1px solid var(--line); background-repeat: no-repeat; }
.badge.link { cursor: pointer; text-decoration: underline; }
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
const FIGURE_COLORS = { image: "#f08c00", chart: "#e64980" };  // 그림 블록은 분류별 색
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
if (doc.layout_notice) document.querySelector("header").appendChild(el("div", "notice", doc.layout_notice));

// 표 블록: 칸 목록으로 <table>을 DOM으로 만든다. 글자는 textContent(줄바꿈은 CSS pre-wrap), 머리 칸은 <th>.
// 낱말·숫자는 칸 안에서 끊지 않고(keep-all) 넓은 표는 가로로 스크롤한다
// 브라우저는 colSpan을 1000, rowSpan을 65534로 자른다. 넘는 칸이 있으면 격자 대신 마크다운 글자로
const MAX_COLSPAN = 1000, MAX_ROWSPAN = 65534;
function grid(t, text) {
  if (t.cells.some((c) => c.colspan > MAX_COLSPAN || c.rowspan > MAX_ROWSPAN)) {
    const box = el("div");
    box.appendChild(el("div", "grid-note", "표가 커서 글자로 표시"));
    box.appendChild(el("div", "text", text));
    return box;
  }
  const wrap = el("div", "grid-wrap");
  const table = el("table", "grid");
  const body = el("tbody");
  const rows = [];
  for (let r = 0; r < t.n_rows; r++) rows.push(body.appendChild(el("tr")));
  for (const c of t.cells) {
    const cell = el(c.header === "column" || c.header === "row" ? "th" : "td", "", c.text);
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

// 그림 블록 썸네일: 이미 넣은 쪽 그림(data URI)을 배경으로 다시 쓰고 상자만큼 잘라 보인다(HTML에 이미지를 더 넣지 않는다)
function thumb(p, bbox) {
  const w = bbox[2] - bbox[0], h = bbox[3] - bbox[1];
  const node = el("div", "thumb");
  node.style.backgroundImage = "url(" + p.image + ")";
  node.style.backgroundSize = (100 / w) + "% " + (100 / h) + "%";
  node.style.backgroundPosition = (w < 1 ? bbox[0] / (1 - w) * 100 : 0) + "% " + (h < 1 ? bbox[1] / (1 - h) * 100 : 0) + "%";
  node.style.aspectRatio = (w * p.width) + " / " + (h * p.height);
  return node;
}
const pageData = new Map(DATA.pages.map((p) => [p.page, p]));
const orderOf = new Map(DATA.blocks.map((b) => [b.id, b.order]));
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
  const color = (b.figure && FIGURE_COLORS[b.figure.category]) || COLORS[b.kind] || "#495057";
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
  if (b.figure) head.appendChild(el("span", "badge", b.figure.category === "chart" ? "차트" : "사진·그림"));
  const partner = b.figure ? b.figure.caption : b.figure_of;
  if (partner) {  // 캡션 짝: 누르면 짝 블록으로 간다
    const link = el("span", "badge link", b.figure ? "캡션 #" + orderOf.get(partner) : "그림 #" + orderOf.get(partner) + "의 캡션");
    link.addEventListener("click", (event) => { event.stopPropagation(); select(partner, "box"); });
    head.appendChild(link);
  }
  item.appendChild(head);
  if (b.section_path.length) item.appendChild(el("div", "meta", b.section_path.join(" › ")));
  item.appendChild(b.table ? grid(b.table, b.text) : el("div", "text", b.text));
  const shot = b.kind === "figure" && b.page !== null && b.bbox ? pageData.get(b.page) : undefined;
  if (shot && shot.image) item.appendChild(thumb(shot, b.bbox));
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
    item.appendChild(r.table ? grid(r.table, r.text) : el("div", "text", r.text));
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
