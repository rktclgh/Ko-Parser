import pytest
from pydantic import TypeAdapter, ValidationError

from ko_parser_contracts.locator import FlowLocator, LinesLocator, Locator, PageLocator, SlideLocator

adapter = TypeAdapter(Locator)


@pytest.mark.parametrize(
    "data,cls",
    [
        ({"kind": "page", "page": 1, "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1}}, PageLocator),
        ({"kind": "flow", "section_path": ["1. 개요"], "paragraph_index": 0}, FlowLocator),
        ({"kind": "slide", "slide": 2, "shape_index": 3}, SlideLocator),
        ({"kind": "lines", "section_path": [], "line_start": 3, "line_end": 5}, LinesLocator),
    ],
)
def test_discriminated_roundtrip(data, cls):
    loc = adapter.validate_python(data)
    assert isinstance(loc, cls)
    assert adapter.validate_json(adapter.dump_json(loc)) == loc


def test_section_path_becomes_tuple():
    loc = FlowLocator(section_path=["가", "나"], paragraph_index=1)
    assert loc.section_path == ("가", "나")


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "cell", "row": 1})


def test_lines_range_order():
    with pytest.raises(ValidationError):
        LinesLocator(line_start=5, line_end=4)


def test_page_and_slide_are_one_based():
    with pytest.raises(ValidationError):
        PageLocator(page=0, bbox={"x0": 0, "y0": 0, "x1": 1, "y1": 1})
    with pytest.raises(ValidationError):
        SlideLocator(slide=0, shape_index=0)
