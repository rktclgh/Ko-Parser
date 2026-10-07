import pytest
from pydantic import ValidationError

from hanji_contracts.changes import ChangeBatch, DocumentChange, LineageEdge


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
    with pytest.raises(ValidationError, match="removed or updated"):
        DocumentChange(document_id="d", version=2, previous_version=1, added=["x"],
                       lineage=[LineageEdge(old_id="a", new_id="x", kind="replaced")])
    with pytest.raises(ValidationError, match="added or updated"):
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


def chg(version, previous_version, doc="d"):
    return DocumentChange(document_id=doc, version=version, previous_version=previous_version, added=[f"n{version}"])


def test_lineage_may_reference_surviving_ids():
    change = DocumentChange(
        document_id="d", version=2, previous_version=1, updated=["a"], removed=["b"],
        lineage=[LineageEdge(old_id="a", new_id="a", kind="merged"),
                 LineageEdge(old_id="b", new_id="a", kind="merged")],
    )
    assert len(change.lineage) == 2
    with pytest.raises(ValidationError, match="must be in removed"):
        DocumentChange(document_id="d", version=2, previous_version=1, updated=["a"], added=["x"],
                       lineage=[LineageEdge(old_id="a", kind="removed")])


def test_cursor_must_advance_with_changes():
    with pytest.raises(ValidationError, match="must advance"):
        ChangeBatch(cursor_from=10, next_cursor=10, changes=[chg(1, None)])
    assert ChangeBatch(cursor_from=10, next_cursor=10, changes=()).changes == ()
    assert ChangeBatch(cursor_from=10, next_cursor=11, changes=[chg(1, None)]).next_cursor == 11


def test_versions_increase_and_chain_within_batch():
    with pytest.raises(ValidationError, match="must increase"):
        ChangeBatch(next_cursor=5, changes=[chg(3, 2), chg(2, 1)])
    with pytest.raises(ValidationError, match="must chain"):
        ChangeBatch(next_cursor=5, changes=[chg(2, 1), chg(3, 1)])
    ChangeBatch(next_cursor=5, changes=[chg(2, 1), chg(3, 2)])
    ChangeBatch(next_cursor=5, changes=[chg(2, 1, "d1"), chg(5, 4, "d2"), chg(3, 2, "d1"), chg(6, 5, "d2")])


@pytest.mark.parametrize("field", ["added", "updated", "removed"])
def test_duplicate_ids_rejected(field):
    with pytest.raises(ValidationError, match="duplicate"):
        DocumentChange(document_id="doc-1", version=2, previous_version=1, **{field: ["x", "x"]})


def test_duplicate_lineage_edges_rejected():
    edge = LineageEdge(old_id="a", new_id="x", kind="replaced")
    with pytest.raises(ValidationError, match="duplicate"):
        DocumentChange(document_id="doc-1", version=2, previous_version=1, added=["x"], removed=["a"],
                       lineage=[edge, edge])


def test_later_versions_require_previous_version():
    with pytest.raises(ValidationError, match="previous_version"):
        DocumentChange(document_id="doc-1", version=3, added=["x"])
    DocumentChange(document_id="doc-1", version=1, added=["x"])
