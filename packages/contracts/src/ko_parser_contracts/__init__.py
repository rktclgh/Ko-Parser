"""ko-parser 데이터 계약."""

from .base import SCHEMA_VERSION, ContractModel, VersionedModel
from .changes import ChangeBatch, DocumentChange, LineageEdge
from .document import (
    ALLOWED_BLOCK_STATES, Block, BlockKind, BlockState, DocumentTree, LayerState, SourceInfo, TextSource, build_blocks,
    check_kind_fields,
)
from .engine import DocFilter, DocRef, Engine, JobRef, JobStatus
from .geometry import BBox, PageInfo
from .ids import NORMALIZATION_VERSION, compute_block_id, compute_content_hash, normalize_text
from .locator import FlowLocator, LinesLocator, Locator, PageLocator, SlideLocator
from .provenance import (
    ATTEMPT_LAYER, Attempt, CorrectionSummary, ErrorCode, ErrorInfo, GateCheck, GateResult, ProcessingHistory,
    RegionRecord, Usage,
)
from .table import MAX_TABLE_CELLS, MAX_TABLE_EXPANDED_CHARS, Cell, CellTextSource, HeaderRole, Table
from .vlm import (
    MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, Capabilities, ImagePayload, PageRef, Task, VlmBlock, VlmDriver, VlmError,
    VlmRequest, VlmResult,
)

__all__ = [
    "SCHEMA_VERSION", "NORMALIZATION_VERSION", "ContractModel", "VersionedModel",
    "BBox", "PageInfo", "Locator", "PageLocator", "FlowLocator", "SlideLocator", "LinesLocator",
    "Cell", "CellTextSource", "HeaderRole", "Table", "MAX_TABLE_CELLS", "MAX_TABLE_EXPANDED_CHARS",
    "normalize_text", "compute_content_hash", "compute_block_id",
    "Block", "BlockKind", "BlockState", "TextSource", "LayerState", "SourceInfo", "DocumentTree",
    "build_blocks", "check_kind_fields", "ALLOWED_BLOCK_STATES",
    "Usage", "ErrorCode", "ErrorInfo", "CorrectionSummary", "GateCheck", "GateResult", "Attempt", "RegionRecord",
    "ProcessingHistory", "ATTEMPT_LAYER",
    "LineageEdge", "DocumentChange", "ChangeBatch",
    "Task", "ImagePayload", "PageRef", "VlmRequest", "VlmBlock", "VlmResult", "Capabilities", "VlmError", "VlmDriver",
    "MAX_IMAGE_BYTES", "MAX_IMAGE_PIXELS",
    "DocRef", "DocFilter", "JobRef", "JobStatus", "Engine",
]
