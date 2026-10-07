"""녹화 키. request_id·page_ref·region_id는 넣지 않는다."""

import hashlib
import json

from ..vlm import VlmRequest


def request_fingerprint(req: VlmRequest) -> str:
    payload = {
        "task": req.task,
        "image_sha256": req.image.sha256,
        "anchor_sha256": None if req.anchor_text is None else hashlib.sha256(req.anchor_text.encode("utf-8")).hexdigest(),
        "language_hints": sorted(req.language_hints),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
