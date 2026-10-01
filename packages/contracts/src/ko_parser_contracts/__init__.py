"""ko-parser 데이터 계약."""

from .base import SCHEMA_VERSION, ContractModel, VersionedModel
from .changes import ChangeBatch, DocumentChange, LineageEdge
from .document import (
    Block, BlockKind, BlockState, DocumentTree, LayerState, SourceInfo, TextSource, build_blocks, check_kind_fields,
)
from .engine import DocFilter, DocRef, Engine, JobRef, JobStatus
from .geometry import BBox, PageInfo
from .ids import NORMALIZATION_VERSION, compute_block_id, compute_content_hash, normalize_text
from .locator import FlowLocator, LinesLocator, Locator, PageLocator, SlideLocator
from .provenance import (
    Attempt, CorrectionSummary, ErrorCode, ErrorInfo, GateCheck, GateResult, ProcessingHistory, RegionRecord, Usage,
)
from .table import Cell, CellTextSource, HeaderRole, Table
from .vlm import Capabilities, ImagePayload, PageRef, Task, VlmBlock, VlmDriver, VlmError, VlmRequest, VlmResult

__all__ = [
    "SCHEMA_VERSION", "NORMALIZATION_VERSION", "ContractModel", "VersionedModel",
    "BBox", "PageInfo", "Locator", "PageLocator", "FlowLocator", "SlideLocator", "LinesLocator",
    "Cell", "CellTextSource", "HeaderRole", "Table",
    "normalize_text", "compute_content_hash", "compute_block_id",
    "Block", "BlockKind", "BlockState", "TextSource", "LayerState", "SourceInfo", "DocumentTree",
    "build_blocks", "check_kind_fields",
    "Usage", "ErrorCode", "ErrorInfo", "CorrectionSummary", "GateCheck", "GateResult", "Attempt", "RegionRecord",
    "ProcessingHistory",
    "LineageEdge", "DocumentChange", "ChangeBatch",
    "Task", "ImagePayload", "PageRef", "VlmRequest", "VlmBlock", "VlmResult", "Capabilities", "VlmError", "VlmDriver",
    "DocRef", "DocFilter", "JobRef", "JobStatus", "Engine",
]
