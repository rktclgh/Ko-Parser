"""변경 내역: 소비자가 커서로 증분 갱신한다. 바뀐 블록의 내용은 get_tree(document_id, version)으로 얻는다."""

from typing import Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel, VersionedModel


class LineageEdge(ContractModel):
    """옛 블록 → 새 블록. 분할은 같은 old_id의 여러 간선, 병합은 같은 new_id의 여러 간선."""

    old_id: str
    new_id: str | None = None
    kind: Literal["replaced", "split", "merged", "removed"]

    @model_validator(mode="after")
    def _check_new_id(self) -> Self:
        if (self.kind == "removed") != (self.new_id is None):
            raise ValueError("new_id must be None if and only if kind == 'removed'")
        return self


class DocumentChange(ContractModel):
    document_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    previous_version: int | None = Field(default=None, ge=1)
    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    lineage: tuple[LineageEdge, ...] = ()

    @model_validator(mode="after")
    def _check_change(self) -> Self:
        added, updated, removed = set(self.added), set(self.updated), set(self.removed)
        if added & updated or added & removed or updated & removed:
            raise ValueError("added, updated and removed must be disjoint")
        for edge in self.lineage:
            if edge.old_id not in removed:
                raise ValueError(f"lineage old_id {edge.old_id!r} must be in removed")
            if edge.new_id is not None and edge.new_id not in added:
                raise ValueError(f"lineage new_id {edge.new_id!r} must be in added")
        if self.previous_version is None:
            if self.updated or self.removed or self.lineage:
                raise ValueError("first version may only contain added blocks")
        elif self.previous_version >= self.version:
            raise ValueError("previous_version must be smaller than version")
        return self


class ChangeBatch(VersionedModel):
    """커서는 엔진의 단조 증가 시퀀스 번호. resync_required면 소비자는 전체를 다시 색인한다."""

    cursor_from: int | None = Field(default=None, ge=0)
    next_cursor: int = Field(ge=0)
    changes: tuple[DocumentChange, ...] = ()
    resync_required: bool = False

    @model_validator(mode="after")
    def _check_batch(self) -> Self:
        if self.resync_required and self.changes:
            raise ValueError("resync_required batches must not contain changes")
        if self.cursor_from is not None and self.next_cursor < self.cursor_from:
            raise ValueError("next_cursor must be >= cursor_from")
        return self
