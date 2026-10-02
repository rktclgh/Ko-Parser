import inspect
import math

import pytest
from pydantic import ValidationError

from ko_parser_contracts.engine import DocFilter, DocRef, Engine, JobRef, JobStatus


def test_refs_and_filter():
    assert DocRef(document_id="d", version=1, layer_state="det").version == 1
    f = DocFilter()
    assert f.document_ids is None and f.layer_states is None
    assert DocFilter(document_ids=["a"], layer_states=["det"]).document_ids == ("a",)
    assert JobRef(job_id="j1").job_id == "j1"


def test_job_status_progress_bounds():
    JobStatus(job_id="j1", state="running", document_ids=["d"], progress=0.5)
    for bad in (1.5, -0.1, math.nan):
        with pytest.raises(ValidationError):
            JobStatus(job_id="j1", state="running", document_ids=["d"], progress=bad)


def test_engine_protocol_shape():
    names = {"ingest", "documents", "get_tree", "run_vlm", "job", "changes", "history"}
    assert names <= set(dir(Engine))


def test_engine_ingest_accepts_document_id_and_force():
    params = inspect.signature(Engine.ingest).parameters
    assert list(params) == ["self", "path", "document_id", "force"]
    assert params["document_id"].default is None and params["force"].default is False
    assert params["document_id"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
