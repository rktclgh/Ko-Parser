from pathlib import Path

import pytest

from hanji.core import build_tree, diff_trees
from hanji.formats.base import ParsedSource
from hanji_contracts import ChangeBatch, DocumentChange, DocumentTree, PageInfo, SourceInfo, TextLayerStats

CONTRACT_FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "fixtures"
SOURCE = SourceInfo(name="메모.md", mime="text/markdown", content_hash="sha256:" + "0" * 64)


def spec(text: str, line: int, kind: str = "paragraph", **kw) -> dict:
    return {"kind": kind, "text": text, "confidence": 1.0, "state": "det", "text_source": "native",
            "locator": {"kind": "lines", "line_start": line, "line_end": line}, **kw}


def tree(version: int, *specs: dict, doc: str = "d1") -> DocumentTree:
    return build_tree(ParsedSource(mime="text/markdown", blocks=specs), doc, version, SOURCE)


def ids(t: DocumentTree) -> list[str]:
    return [b.block_id for b in t.blocks]


def test_build_tree_is_det_layer():
    t = tree(1, spec("제목", 1, "heading", level=1), spec("본문", 3))
    assert (t.document_id, t.version, t.layer_state, t.source) == ("d1", 1, "det", SOURCE)
    assert [b.order for b in t.blocks] == [0, 1] and {b.state for b in t.blocks} == {"det"}
    assert tree(1, spec("제목", 1, "heading", level=1), spec("본문", 3)) == t  # 결정론


def test_first_version_adds_everything():
    t = tree(1, spec("가", 1), spec("나", 2))
    assert diff_trees(None, t) == DocumentChange(document_id="d1", version=1, added=ids(t))


def test_first_version_of_empty_document_is_still_a_change():
    change = diff_trees(None, tree(1))
    assert change == DocumentChange(document_id="d1", version=1)


def test_identical_blocks_give_none():
    assert diff_trees(tree(1, spec("가", 1)), tree(2, spec("가", 1))) is None


def test_pages_only_change_is_an_empty_change():
    """블록이 같아도 쪽 판정이 바뀌면 새 버전(빈 변경). 쪽 정보까지 같으면 None."""
    digital = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144)
    scanned = digital.model_copy(update={"text_layer": "scanned", "text_stats": TextLayerStats(
        chars=3, invisible_ratio=0.0, unmapped_ratio=0.0, pua_ratio=0.0, max_image_coverage=0.6)})

    def pdf(version: int, page: PageInfo) -> DocumentTree:
        return build_tree(ParsedSource(mime="application/pdf", pages=(page,)), "d1", version, SOURCE)

    assert diff_trees(pdf(1, digital), pdf(2, scanned)) == DocumentChange(document_id="d1", version=2,
                                                                          previous_version=1)
    assert diff_trees(pdf(1, digital), pdf(2, digital)) is None


def test_add_remove_and_unchanged():
    v1 = tree(1, spec("가", 1), spec("나", 2), spec("다", 3))
    v2 = tree(2, spec("새", 1), spec("가", 2), spec("다", 3))
    change = diff_trees(v1, v2)
    a, b, _ = ids(v1)
    assert (change.version, change.previous_version) == (2, 1)
    assert change.added == (ids(v2)[0],)
    assert change.removed == (b,)
    assert change.updated == (a,)  # 다는 order·줄이 그대로라 빠진다


def test_reorder_updates_in_new_tree_order():
    v1 = tree(1, spec("가", 1), spec("나", 2))
    v2 = tree(2, spec("나", 1), spec("가", 2))
    assert diff_trees(v1, v2).updated == tuple(ids(v2))


def test_removed_follows_previous_tree_order():
    v1 = tree(1, spec("가", 1), spec("나", 2), spec("다", 3))
    v2 = tree(2, spec("라", 1))
    assert diff_trees(v1, v2).removed == tuple(ids(v1))


def test_moved_block_only_locator_changed_is_updated():
    v1 = tree(1, spec("가", 1), spec("나", 3))
    v2 = tree(2, spec("가", 1), spec("나", 5))
    assert diff_trees(v1, v2).updated == (ids(v1)[1],)


def test_whitespace_only_edit_keeps_id_but_is_updated():
    v1 = tree(1, spec("가 나", 1))
    v2 = tree(2, spec("가   나", 1))  # 정규화 해시는 같고 원문은 다르다
    assert ids(v1) == ids(v2)
    assert diff_trees(v1, v2).updated == tuple(ids(v1))


def test_duplicate_text_inserted_before_shifts_occurrence():
    v1 = tree(1, spec("같은 문단", 1), spec("끝", 2))
    v2 = tree(2, spec("같은 문단", 1), spec("같은 문단", 2), spec("끝", 3))
    change = diff_trees(v1, v2)
    assert ids(v2)[0] == ids(v1)[0]
    assert change.added == (ids(v2)[1],)
    assert change.updated == (ids(v1)[1],) and not change.removed


def test_change_batch_accepts_consecutive_diffs():
    v1 = tree(1, spec("가", 1))
    v2 = tree(2, spec("가", 1), spec("나", 2))
    v3 = tree(3, spec("나", 1))
    ChangeBatch(next_cursor=3, changes=[diff_trees(None, v1), diff_trees(v1, v2), diff_trees(v2, v3)])


def test_mismatched_documents_rejected():
    with pytest.raises(ValueError):
        diff_trees(tree(1, spec("가", 1)), tree(2, spec("가", 1), doc="d2"))


def _load(rel: str, model):
    return model.model_validate_json((CONTRACT_FIXTURES / rel).read_text(encoding="utf-8"))


def test_matches_contract_lifecycle_golden():
    v1, v2, v3 = (_load(f"lifecycle/{n}.json", DocumentTree) for n in ("v1_det", "v2_unverified", "v3_vlm"))
    c1, c2, c3 = _load("lifecycle/changes.json", ChangeBatch).changes
    assert diff_trees(None, v1) == c1
    assert diff_trees(v1, v2) == c2  # 상태 변경 = 전부 updated
    ours = diff_trees(v2, v3)  # lineage는 만들지 않는다(D5)
    assert (ours.added, ours.updated, ours.removed) == (c3.added, c3.updated, c3.removed) and ours.lineage == ()


def test_figure_image_change_replaces_the_block_but_pairing_metadata_updates_it():
    """그림 이미지(asset)는 해시에 들어가 바뀌면 블록이 removed+added, 캡션 짝·dpi·종류만 바뀌면 같은 id로 updated."""
    def figure(asset_hex: str, **kw) -> dict:
        return spec("", 1, "figure", figure={"asset": "sha256:" + asset_hex * 64, "mime": "image/png", "width_px": 2,
                                             "height_px": 2, "dpi": 72, "category": "image", **kw})
    caption = spec("그림 1. 합성", 2, "caption")
    v1 = tree(1, figure("a", caption_ref=1), caption)
    v2 = tree(2, figure("b", caption_ref=1), caption)  # 이미지가 바뀜
    change = diff_trees(v1, v2)
    assert change.removed == (ids(v1)[0],) and change.added == (ids(v2)[0],) and change.updated == ()
    v3 = tree(3, figure("b"), caption)  # 캡션 짝만 풀림
    change = diff_trees(v2, v3)
    assert ids(v3) == ids(v2) and v3.blocks[0].figure.caption_block_id is None
    assert change.updated == (ids(v2)[0],) and change.added == change.removed == ()
    v4 = tree(4, figure("b", dpi=144, category="chart"), caption)  # dpi·종류만 바뀜
    assert ids(v4) == ids(v3) and diff_trees(v3, v4).updated == (ids(v3)[0],)
