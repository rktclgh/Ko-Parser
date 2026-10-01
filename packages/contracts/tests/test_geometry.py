import math

import pytest
from pydantic import ValidationError

from ko_parser_contracts.geometry import BBox, PageInfo


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
