import pytest
from pydantic import ValidationError

from ko_parser_contracts.changes import ChangeBatch, DocumentChange, LineageEdge


def test_lineage_kinds():
    LineageEdge(old_id="a", new_id="x", kind="replaced")
    LineageEdge(old_id="a", kind="removed")
    with pytest.raises(ValidationError, match="new_id"):
        LineageEdge(old_id="a", kind="split")
    with pytest.raises(ValidationError, match="new_id"):
        LineageEdge(old_id="a", new_id="x", kind="removed")


def test_split_and_merge_change():
    change = DocumentChange(
        document_id="doc-1", version=3, previous_version=2,
        added=["x", "y", "z"], removed=["a", "b", "c"],
        lineage=[
            LineageEdge(old_id="a", new_id="x", kind="split"),
            LineageEdge(old_id="a", new_id="y", kind="split"),
            LineageEdge(old_id="b", new_id="z", kind="merged"),
            LineageEdge(old_id="c", new_id="z", kind="merged"),
        ],
    )
    assert len(change.lineage) == 4


def test_sets_must_be_disjoint():
    with pytest.raises(ValidationError, match="disjoint"):
        DocumentChange(document_id="d", version=2, previous_version=1, added=["a"], updated=["a"])


def test_lineage_must_reference_removed_and_added():
    with pytest.raises(ValidationError, match="removed"):
        DocumentChange(document_id="d", version=2, previous_version=1, added=["x"],
                       lineage=[LineageEdge(old_id="a", new_id="x", kind="replaced")])
    with pytest.raises(ValidationError, match="added"):
        DocumentChange(document_id="d", version=2, previous_version=1, removed=["a"],
                       lineage=[LineageEdge(old_id="a", new_id="x", kind="replaced")])


def test_first_version_rules():
    DocumentChange(document_id="d", version=1, added=["a"])
    with pytest.raises(ValidationError, match="first version"):
        DocumentChange(document_id="d", version=1, updated=["a"])
    with pytest.raises(ValidationError, match="previous_version"):
        DocumentChange(document_id="d", version=2, previous_version=2)


def test_change_batch_rules():
    ChangeBatch(cursor_from=None, next_cursor=1, changes=[DocumentChange(document_id="d", version=1, added=["a"])])
    resync = ChangeBatch(cursor_from=5, next_cursor=120, resync_required=True)
    assert resync.changes == ()
    with pytest.raises(ValidationError, match="resync"):
        ChangeBatch(cursor_from=5, next_cursor=6, resync_required=True,
                    changes=[DocumentChange(document_id="d", version=1, added=["a"])])
    with pytest.raises(ValidationError, match="next_cursor"):
        ChangeBatch(cursor_from=10, next_cursor=9)
