"""순서 드라이버: 호출 순서대로 정해 둔 결과나 오류를 낸다. 실제로 기다리지 않는다."""

from collections.abc import Sequence

from ..provenance import ErrorInfo
from ..vlm import Capabilities, VlmError, VlmRequest, VlmResult
from .defaults import DEFAULT_CAPABILITIES


class ScriptedDriver:
    def __init__(self, steps: Sequence[VlmResult | ErrorInfo], driver_id: str = "scripted",
                 capabilities: Capabilities | None = None) -> None:
        self.id = driver_id
        self._steps = list(steps)
        self._caps = capabilities or DEFAULT_CAPABILITIES
        self.calls: list[VlmRequest] = []

    def capabilities(self) -> Capabilities:
        return self._caps

    async def health(self) -> bool:
        return True

    async def run(self, req: VlmRequest) -> VlmResult:
        self.calls.append(req)
        if len(self.calls) > len(self._steps):
            raise AssertionError(f"ScriptedDriver exhausted after {len(self._steps)} steps")
        step = self._steps[len(self.calls) - 1]
        if isinstance(step, ErrorInfo):
            raise VlmError(step)
        return step.model_copy(update={"request_id": req.request_id})
