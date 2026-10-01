"""엔진 공개 인터페이스. 구현이 아니라 계약이다."""

from typing import Literal, Protocol

from pydantic import Field

from .base import ContractModel
from .changes import ChangeBatch
from .document import DocumentTree, LayerState
from .provenance import ErrorInfo, ProcessingHistory


class DocRef(ContractModel):
    document_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    layer_state: LayerState


class DocFilter(ContractModel):
    """둘 다 None이면 전체 문서."""

    document_ids: tuple[str, ...] | None = None
    layer_states: tuple[LayerState, ...] | None = None


class JobRef(ContractModel):
    job_id: str = Field(min_length=1)


class JobStatus(ContractModel):
    job_id: str = Field(min_length=1)
    state: Literal["queued", "running", "done", "failed"]
    document_ids: tuple[str, ...]
    progress: float = Field(ge=0.0, le=1.0)
    error: ErrorInfo | None = None


class Engine(Protocol):
    def ingest(self, path: str) -> DocRef: ...

    def documents(self) -> tuple[DocRef, ...]: ...

    def get_tree(self, document_id: str, version: int | None = None) -> DocumentTree: ...

    def run_vlm(self, target: DocFilter) -> JobRef: ...

    def job(self, job_id: str) -> JobStatus: ...

    def changes(self, cursor: int | None, limit: int = 100) -> ChangeBatch: ...

    def history(self, document_id: str, version: int | None = None) -> ProcessingHistory: ...
