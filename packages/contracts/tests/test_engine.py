import inspect
import math
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from hanji_contracts.engine import DocFilter, DocRef, Engine, JobRef, JobStatus


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
    sig = inspect.signature(Engine.ingest)
    params = sig.parameters

    # 인자 순서·이름
    assert list(params) == ["self", "path", "document_id", "force"]

    # 모두 위치·키워드 겸용(기존 위치 호출 호환)
    assert params["path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["document_id"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["force"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    # 기본값
    assert params["document_id"].default is None
    assert params["force"].default is False

    # 타입 주석
    hints = get_type_hints(Engine.ingest)
    assert hints["path"] is str
    assert hints["document_id"] == str | None
    assert hints["force"] is bool
    assert hints["return"] is DocRef


def test_engine_protocol_declares_get_asset():
    """계약 0.3: 그림 이미지 바이트는 트리 밖 자산이라 엔진이 내용 해시로 돌려준다."""
    params = inspect.signature(Engine.get_asset).parameters
    assert list(params) == ["self", "asset"]
    hints = get_type_hints(Engine.get_asset)
    assert hints["asset"] is str and hints["return"] is bytes
