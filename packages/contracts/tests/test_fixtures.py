import asyncio
import importlib.util
from pathlib import Path

import pytest

from ko_parser_contracts import ChangeBatch, DocumentTree, ProcessingHistory, VlmRequest, VlmResult
from ko_parser_contracts.testing import Recording, ReplayDriver

ROOT = Path(__file__).resolve().parents[1] / "fixtures"
MODEL_BY_GLOB = {
    "documents/*.json": DocumentTree,
    "lifecycle/v*.json": DocumentTree,
    "lifecycle/changes.json": ChangeBatch,
    "changes/*.json": ChangeBatch,
    "history/*.json": ProcessingHistory,
    "vlm/request_*.json": VlmRequest,
    "vlm/result_*.json": VlmResult,
    "recordings/*.json": Recording,
}
CASES = [(p, model) for pattern, model in MODEL_BY_GLOB.items() for p in sorted(ROOT.glob(pattern))]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_fixtures", Path(__file__).with_name("build_fixtures.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_fixture_files_exist():
    names = {p.relative_to(ROOT).as_posix() for p, _ in CASES}
    assert {
        "documents/docx_flow.json", "documents/pdf_table_page.json", "documents/pptx_slide.json",
        "documents/empty.json", "lifecycle/v1_det.json", "lifecycle/v2_unverified.json", "lifecycle/v3_vlm.json",
        "lifecycle/changes.json", "changes/split_merge.json", "changes/resync.json",
        "history/gate_fail_fallback.json", "vlm/request_table.json", "vlm/result_table.json",
        "recordings/table_simple.json",
    } <= names


@pytest.mark.parametrize("path,model", CASES, ids=lambda x: getattr(x, "name", str(x)))
def test_fixture_roundtrip(path, model):
    obj = model.model_validate_json(path.read_text(encoding="utf-8"))
    assert model.model_validate_json(obj.model_dump_json()) == obj


def test_fixtures_are_up_to_date():
    builder = _load_builder()
    for rel, text in builder.build_all().items():
        assert (ROOT / rel).read_text(encoding="utf-8") == text, rel


def test_lifecycle_versions_share_ids_until_replacement():
    v1, v2, v3 = (DocumentTree.model_validate_json((ROOT / f"lifecycle/{n}.json").read_text(encoding="utf-8"))
                  for n in ("v1_det", "v2_unverified", "v3_vlm"))
    assert [b.block_id for b in v1.blocks] == [b.block_id for b in v2.blocks]
    assert {b.state for b in v2.blocks} == {"unverified"} and v2.layer_state == "vlm_running"
    assert v3.layer_state == "vlm_done" and v1.blocks[2].block_id not in {b.block_id for b in v3.blocks}


def test_replay_serves_fixture_request():
    request = VlmRequest.model_validate_json((ROOT / "vlm/request_table.json").read_text(encoding="utf-8"))
    driver = ReplayDriver.from_dir(ROOT / "recordings")
    out = asyncio.run(driver.run(request.model_copy(update={"request_id": "req-test"})))
    assert out.request_id == "req-test"
    assert out.blocks[0].table.to_grid()[3][3] == "-6.2"
