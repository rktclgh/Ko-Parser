import asyncio
import base64

import pytest
from pydantic import ValidationError

from ko_parser_contracts.geometry import BBox
from ko_parser_contracts.provenance import ErrorInfo, Usage
from ko_parser_contracts.vlm import (
    Capabilities, ImagePayload, PageRef, VlmBlock, VlmDriver, VlmError, VlmRequest, VlmResult,
)

PNG = b"\x89PNG\r\n\x1a\nfake-bytes"
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


def test_request_region_rule():
    assert request().region_id == "r1"
    with pytest.raises(ValidationError, match="region_id"):
        request(region_id=None)
    with pytest.raises(ValidationError, match="region_id"):
        request(task="PAGE_FULL")
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
