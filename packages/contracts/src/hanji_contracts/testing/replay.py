"""녹화 재생 드라이버: 실제 모델 응답을 녹화해 두고 그대로 돌려준다(테스트·CI 전용)."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import Field

from ..base import ContractModel, VersionedModel
from ..vlm import Capabilities, VlmRequest, VlmResult
from .defaults import DEFAULT_CAPABILITIES
from .fingerprint import request_fingerprint


class RecordingMeta(ContractModel):
    model_id: str
    profile_id: str
    prompt: str
    runtime: str
    recorded_at: str


class Recording(VersionedModel):
    fingerprint: str = Field(pattern=r"^[0-9a-f]{32}$")
    result: VlmResult
    meta: RecordingMeta


class ReplayMiss(AssertionError):
    """녹화본이 없는 요청. 런타임 장애(VlmError)와 구별되는 테스트 오류다."""


class ReplayDriver:
    def __init__(self, recordings: Iterable[Recording], driver_id: str = "replay",
                 capabilities: Capabilities | None = None) -> None:
        self.id = driver_id
        self._caps = capabilities or DEFAULT_CAPABILITIES
        self._by_fingerprint: dict[str, Recording] = {}
        for rec in recordings:
            if rec.fingerprint in self._by_fingerprint:
                raise ValueError(f"duplicate recording fingerprint {rec.fingerprint}")
            self._by_fingerprint[rec.fingerprint] = rec

    @classmethod
    def from_dir(cls, path: str | Path, **kwargs) -> "ReplayDriver":
        files = sorted(Path(path).glob("*.json"))
        return cls((Recording.model_validate_json(f.read_text(encoding="utf-8")) for f in files), **kwargs)

    def capabilities(self) -> Capabilities:
        return self._caps

    async def health(self) -> bool:
        return True

    async def run(self, req: VlmRequest) -> VlmResult:
        fingerprint = request_fingerprint(req)
        rec = self._by_fingerprint.get(fingerprint)
        if rec is None:
            raise ReplayMiss(f"no recording for fingerprint {fingerprint} (task={req.task}, image={req.image.sha256[:12]})")
        return rec.result.model_copy(update={"request_id": req.request_id})
