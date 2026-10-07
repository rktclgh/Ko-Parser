import math

import pytest
from pydantic import ValidationError

from hanji_contracts.geometry import BBox, PageInfo, TextLayerStats


def test_bbox_accepts_normalized_box():
    b = BBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8)
    assert (b.x0, b.y1) == (0.1, 0.8)


@pytest.mark.parametrize(
    "x0,y0,x1,y1",
    [(-0.1, 0, 0.5, 0.5), (0, 0, 1.2, 0.5), (0.5, 0, 0.5, 0.5), (0, 0.6, 0.5, 0.5), (math.nan, 0, 0.5, 0.5)],
)
def test_bbox_rejects_bad_values(x0, y0, x1, y1):
    with pytest.raises(ValidationError):
        BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def test_bbox_is_frozen_and_forbids_extra():
    b = BBox(x0=0, y0=0, x1=1, y1=1)
    with pytest.raises(ValidationError):
        b.x0 = 0.5
    with pytest.raises(ValidationError):
        BBox(x0=0, y0=0, x1=1, y1=1, z=1)


def test_page_info_defaults_and_rotation():
    p = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144)
    assert p.rotation == 0
    with pytest.raises(ValidationError):
        PageInfo(page=1, width_pt=595.0, height_pt=842.0, rotation=45, render_dpi=144)


def test_page_info_rejects_inf():
    with pytest.raises(ValidationError):
        PageInfo(page=1, width_pt=math.inf, height_pt=842.0, render_dpi=144)
    with pytest.raises(ValidationError):
        PageInfo(page=0, width_pt=595.0, height_pt=842.0, render_dpi=144)


def stats(**kw) -> TextLayerStats:
    base = {"chars": 4, "invisible_ratio": 0.0, "unmapped_ratio": 0.0, "pua_ratio": 0.0, "max_image_coverage": 0.6}
    return TextLayerStats(**{**base, **kw})


def test_page_info_text_layer_defaults_to_digital_without_stats():
    p = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144)
    assert (p.text_layer, p.text_stats) == ("digital", None)
    digital = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144, text_stats=stats(chars=0))
    assert digital.text_stats.chars == 0


@pytest.mark.parametrize("state", ["scanned", "unreliable"])
def test_non_digital_page_requires_stats(state):
    with pytest.raises(ValidationError, match="require text_stats"):
        PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144, text_layer=state)
    page = PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144, text_layer=state, text_stats=stats())
    assert PageInfo.model_validate_json(page.model_dump_json()) == page


@pytest.mark.parametrize("field,value", [("chars", -1), ("invisible_ratio", 1.01), ("unmapped_ratio", -0.1),
                                         ("pua_ratio", math.nan), ("max_image_coverage", math.inf)])
def test_text_stats_rejects_out_of_range(field, value):
    with pytest.raises(ValidationError):
        stats(**{field: value})


def test_text_layer_rejects_unknown_state():
    with pytest.raises(ValidationError):
        PageInfo(page=1, width_pt=595.0, height_pt=842.0, render_dpi=144, text_layer="ocr", text_stats=stats())
    with pytest.raises(ValidationError):
        stats(extra=1)
