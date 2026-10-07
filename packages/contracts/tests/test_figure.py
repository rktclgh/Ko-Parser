"""계약 0.3 그림 참조: FigureImage 필드·상한, 내용 해시, 캡션 짝 검증."""

import hashlib

import pytest
from pydantic import ValidationError

from hanji_contracts import (
    MAX_FIGURE_SIDE, Block, DocumentTree, FigureImage, PageInfo, SourceInfo, build_blocks, compute_content_hash,
)

ASSET = "sha256:" + hashlib.sha256(b"png").hexdigest()
OTHER = "sha256:" + hashlib.sha256(b"other png").hexdigest()
SRC = SourceInfo(name="a.pdf", mime="application/pdf", content_hash="sha256:" + "0" * 64, page_count=1)
A4 = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144)


def image(**kw) -> dict:
    return {"asset": ASSET, "mime": "image/png", "width_px": 400, "height_px": 300, "dpi": 200, "category": "chart",
            **kw}


def spec(kind: str, text: str, y: float, **kw) -> dict:
    return {"kind": kind, "text": text, "confidence": 0.7, "state": "det", "text_source": "text_layer",
            "locator": {"kind": "page", "page": 1, "bbox": {"x0": 0.1, "y0": y, "x1": 0.9, "y1": y + 0.05}}, **kw}


def with_caption(blocks: tuple[Block, ...], figure_at: int, caption_id: str) -> tuple[Block, ...]:
    """figure_at 블록의 figure에 caption_block_id를 넣어 다시 검증한다(해시에 들어가지 않아 id는 그대로)."""
    fig = blocks[figure_at]
    linked = Block.model_validate({**fig.model_dump(), "figure": {**fig.figure.model_dump(), "caption_block_id": caption_id}})
    return blocks[:figure_at] + (linked,) + blocks[figure_at + 1:]


def tree(blocks) -> DocumentTree:
    return DocumentTree(document_id="d", version=1, layer_state="det", source=SRC, pages=(A4,), blocks=blocks)


def test_figure_image_fields_and_limits():
    assert FigureImage(**image()).caption_block_id is None
    for bad in ({"asset": "sha256:abc"}, {"asset": "md5:" + "0" * 64}, {"mime": "image/jpeg"}, {"width_px": 0},
                {"height_px": MAX_FIGURE_SIDE + 1}, {"dpi": 0}, {"category": "table"}, {"caption_block_id": ""}):
        with pytest.raises(ValidationError):
            FigureImage(**image(**bad))
    assert FigureImage(**image(width_px=MAX_FIGURE_SIDE, height_px=MAX_FIGURE_SIDE)).width_px == 4000


def test_figure_only_on_figure_blocks():
    (fig,) = build_blocks("d", [spec("figure", "", 0.1, figure=image())])
    assert fig.figure.asset == ASSET
    with pytest.raises(ValidationError, match="kind == 'figure'"):
        build_blocks("d", [spec("paragraph", "본문", 0.1, figure=image())])
    (plain,) = build_blocks("d", [spec("figure", "조직도", 0.1)])  # 그림 블록도 이미지 없이 된다(MD, 바이트 상한)
    assert plain.figure is None


def test_figure_asset_enters_content_hash_but_pairing_dpi_and_category_do_not():
    base = compute_content_hash("figure", "가", None, None, ASSET)
    assert base != compute_content_hash("figure", "가", None, None, OTHER)
    assert base != compute_content_hash("figure", "가", None, None)
    a, b = build_blocks("d", [spec("figure", "가", 0.1, figure=image()),
                              spec("figure", "가", 0.3, figure=image(dpi=144, category="image", width_px=10))])
    assert a.content_hash == b.content_hash == base and a.block_id != b.block_id  # 같은 해시는 순번으로 id가 갈린다


def test_blocks_without_figure_keep_their_0_2_hashes():
    """그림이 없으면 해시 입력이 0.2와 같다: 기존 블록 id가 바뀌지 않는다(값은 develop b60616b에서 잰 것)."""
    assert compute_content_hash("paragraph", "본문", None, None) == "c_dce18f49317badffbe1a2d7a1b1173bf"
    assert compute_content_hash("figure", "", None, None) == "c_4b4a3456b4b3bcff05395b4db5c801ac"


def test_caption_block_id_must_point_to_a_caption_in_the_tree():
    blocks = build_blocks("d", [spec("figure", "", 0.1, figure=image()), spec("caption", "그림 1. 현황", 0.2),
                                spec("paragraph", "본문", 0.3)])
    linked = tree(with_caption(blocks, 0, blocks[1].block_id))
    assert linked.blocks[0].figure.caption_block_id == blocks[1].block_id
    with pytest.raises(ValidationError, match="caption block"):
        tree(with_caption(blocks, 0, blocks[2].block_id))  # 문단은 캡션이 아니다
    with pytest.raises(ValidationError, match="caption block"):
        tree(with_caption(blocks, 0, "b_" + "0" * 24))  # 트리에 없는 id


def test_one_caption_belongs_to_one_figure():
    blocks = build_blocks("d", [spec("figure", "", 0.1, figure=image()), spec("figure", "", 0.3, figure=image(asset=OTHER)),
                                spec("caption", "그림 1. 현황", 0.5)])
    one = with_caption(blocks, 0, blocks[2].block_id)
    tree(one)
    with pytest.raises(ValidationError, match="one figure"):
        tree(with_caption(one, 1, blocks[2].block_id))


def test_figure_round_trips_through_json():
    blocks = build_blocks("d", [spec("figure", "1분기\n2분기", 0.1, figure=image()), spec("caption", "그림 1.", 0.2)])
    doc = tree(with_caption(blocks, 0, blocks[1].block_id))
    again = DocumentTree.model_validate_json(doc.model_dump_json())
    assert again == doc and again.blocks[0].figure.caption_block_id == blocks[1].block_id
