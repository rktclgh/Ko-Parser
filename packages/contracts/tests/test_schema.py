import json

from ko_parser_contracts import schema
import ko_parser_contracts as kpc


def test_render_has_all_roots_with_version_const():
    rendered = schema.render_schemas()
    assert set(rendered) == {f"{name}.schema.json" for name in schema.ROOT_MODELS}
    doc = json.loads(rendered["document_tree.schema.json"])
    assert doc["properties"]["schema_version"]["const"] == "0.1"


def test_export_and_check_roundtrip(tmp_path):
    schema.export_schemas(tmp_path)
    assert schema.check_schemas(tmp_path) == []
    (tmp_path / "vlm_result.schema.json").write_text("{}", encoding="utf-8")
    assert schema.check_schemas(tmp_path) == ["vlm_result.schema.json"]
    (tmp_path / "old_root.schema.json").write_text("{}", encoding="utf-8")
    assert schema.check_schemas(tmp_path) == ["vlm_result.schema.json", "old_root.schema.json"]


def test_main_check_exit_codes(tmp_path):
    assert schema.main(["--check", "--out", str(tmp_path)]) == 1
    schema.export_schemas(tmp_path)
    assert schema.main(["--check", "--out", str(tmp_path)]) == 0


def test_committed_schemas_are_current():
    assert schema.check_schemas(schema.SCHEMA_DIR) == []


def test_public_api_exports():
    for name in ("DocumentTree", "Block", "Table", "Cell", "BBox", "Locator", "ChangeBatch", "ProcessingHistory",
                 "VlmRequest", "VlmResult", "VlmDriver", "VlmError", "Engine", "build_blocks",
                 "compute_content_hash", "compute_block_id", "SCHEMA_VERSION", "NORMALIZATION_VERSION",
                 "MAX_TABLE_CELLS", "MAX_TABLE_EXPANDED_CHARS", "MAX_IMAGE_BYTES", "MAX_IMAGE_PIXELS",
                 "ALLOWED_BLOCK_STATES", "ATTEMPT_LAYER"):
        assert name in kpc.__all__ and hasattr(kpc, name), name
    assert all(hasattr(kpc, name) for name in kpc.__all__)


def test_public_rule_mappings_are_read_only():
    import pytest

    for mapping in (kpc.ALLOWED_BLOCK_STATES, kpc.ATTEMPT_LAYER):
        with pytest.raises(TypeError):
            mapping["det"] = "x"
        with pytest.raises(TypeError):
            del mapping["det"]
    assert isinstance(kpc.ALLOWED_BLOCK_STATES["det"], frozenset)
