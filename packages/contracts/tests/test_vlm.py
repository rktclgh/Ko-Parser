import asyncio
import base64
import struct

import pytest
from pydantic import ValidationError

from ko_parser_contracts.geometry import BBox
from ko_parser_contracts.provenance import ErrorInfo, Usage
from ko_parser_contracts.testing import tiny_png
from ko_parser_contracts.vlm import (
    Capabilities, ImagePayload, PageRef, VlmBlock, VlmDriver, VlmError, VlmRequest, VlmResult,
)

PNG = tiny_png(2, 3)
BOX = BBox(x0=0.1, y0=0.2, x1=0.9, y1=0.6)


def request(**kw) -> VlmRequest:
    base = dict(request_id="req-1", task="REGION_TABLE", image=ImagePayload.from_png(PNG, BOX, 144),
                region_id="r1", page_ref=PageRef(document_id="doc-1", page=1), language_hints=["ko"])
    base.update(kw)
    return VlmRequest(**base)


def test_image_payload_hash_and_base64_roundtrip():
    img = ImagePayload.from_png(PNG, BOX, 144)
    dumped = img.model_dump(mode="json")
    assert dumped["png"] == base64.b64encode(PNG).decode("ascii")
    assert ImagePayload.model_validate_json(img.model_dump_json()) == img
    with pytest.raises(ValidationError, match="sha256"):
        ImagePayload(png=PNG, page_bbox=BOX, dpi=144, sha256="0" * 64)


def _with_ihdr(width: int, height: int) -> bytes:
    return PNG[:16] + struct.pack(">II", width, height) + PNG[24:]


@pytest.mark.parametrize("png, message", [
    (b"", "png signature missing"),
    (b"not a PNG", "png signature missing"),
    (PNG[:8], "png must start with an IHDR chunk"),
    (PNG[:-12], "png is truncated"),
    (_with_ihdr(0, 3), "png dimensions out of range"),
    (_with_ihdr(10_000, 10_000), "png dimensions out of range"),
])
def test_image_payload_rejects_non_png(png, message):
    with pytest.raises(ValidationError, match=message):
        ImagePayload.from_png(png, BOX, 144)


def test_image_payload_rejects_oversized_bytes(monkeypatch):
    monkeypatch.setattr("ko_parser_contracts.vlm.MAX_IMAGE_BYTES", len(PNG) - 1)
    with pytest.raises(ValidationError, match="MAX_IMAGE_BYTES"):
        ImagePayload.from_png(PNG, BOX, 144)


def test_image_payload_width_height():
    img = ImagePayload.from_png(PNG, BOX, 144)
    assert (img.width, img.height) == (2, 3)
    assert "width" not in img.model_dump()


def test_tiny_png_is_deterministic_and_valid():
    assert tiny_png(4, 5, 7) == tiny_png(4, 5, 7)
    assert tiny_png(gray=0) != tiny_png(gray=255)
    assert ImagePayload.from_png(tiny_png(4, 5, 7), BOX, 72).height == 5


def test_request_region_rule():
    assert request().region_id == "r1"
    with pytest.raises(ValidationError, match="region_id"):
        request(region_id=None)
    with pytest.raises(ValidationError, match="region_id"):
        request(task="PAGE_FULL")
    with pytest.raises(ValidationError):
        request(task="REGION_TEXT", region_id="")
    assert request(task="PAGE_FULL", region_id=None).task == "PAGE_FULL"


def test_vlm_block_is_a_proposal_with_kind_rules():
    VlmBlock(kind="paragraph", text="본문", bbox=BOX)
    with pytest.raises(ValidationError, match="level"):
        VlmBlock(kind="heading", text="제목")


def test_vlm_result_defaults():
    r = VlmResult(request_id="req-1", blocks=[], raw_text="", output_format="text", driver_id="d", model_id="m")
    assert r.usage == Usage() and r.warnings == ()


def test_capabilities_tasks_frozenset():
    caps = Capabilities(tasks=["REGION_TABLE", "REGION_TABLE"], languages=["ko"], max_image_pixels=1_000_000,
                        max_concurrency=1, structured_output=False, location="local", tier="small")
    assert caps.tasks == frozenset({"REGION_TABLE"})


def test_vlm_error_carries_info():
    info = ErrorInfo(code="TIMEOUT", retryable=True, message="slow")
    err = VlmError(info)
    assert err.info is info and "TIMEOUT" in str(err)


def test_driver_protocol_is_runtime_checkable():
    class Dummy:
        id = "dummy"

        def capabilities(self):
            return None

        async def health(self):
            return True

        async def run(self, req):
            return None

    assert isinstance(Dummy(), VlmDriver)
    assert asyncio.run(Dummy().health()) is True
