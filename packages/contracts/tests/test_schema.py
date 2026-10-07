import json

from hanji_contracts import schema
import hanji_contracts as kpc


def test_render_has_all_roots_with_version_const():
    rendered = schema.render_schemas()
    assert set(rendered) == {f"{name}.schema.json" for name in schema.ROOT_MODELS}
    doc = json.loads(rendered["document_tree.schema.json"])
    assert doc["properties"]["schema_version"]["const"] == "0.3"


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
                 "TextLayerState", "TextLayerStats", "FigureImage", "FigureCategory", "MAX_FIGURE_SIDE",
                 "MAX_DOCUMENT_ASSET_BYTES",
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


def _page_cases() -> tuple[list[dict], list[dict]]:
    """pdf_table_page 골든 예제의 첫 쪽을 바꾼 문서들: (둘 다 거부할 것, 둘 다 받을 것)."""
    path = schema.SCHEMA_DIR.parent / "fixtures" / "documents" / "pdf_table_page.json"
    base = json.loads(path.read_text(encoding="utf-8"))
    stats = {"chars": 0, "invisible_ratio": 0.0, "unmapped_ratio": 0.0, "pua_ratio": 0.0, "max_image_coverage": 1.0}

    def with_page(**changes) -> dict:
        doc = json.loads(json.dumps(base))
        page = doc["pages"][0]
        page.pop("text_stats")
        page.update(changes)
        return doc

    rejected = [with_page(text_layer=state, **extra) for state in ("scanned", "unreliable")
                for extra in ({}, {"text_stats": None})]
    accepted = [with_page(), with_page(text_layer="digital", text_stats=None),
                with_page(text_layer="scanned", text_stats=stats)]
    return rejected, accepted


def test_json_schema_matches_model_on_text_stats_rule():
    import pytest
    from jsonschema import Draft202012Validator
    from pydantic import ValidationError

    committed = (schema.SCHEMA_DIR / "document_tree.schema.json").read_text(encoding="utf-8")
    validator = Draft202012Validator(json.loads(committed))
    rejected, accepted = _page_cases()
    for doc in rejected:
        assert not validator.is_valid(doc), doc["pages"][0]
        with pytest.raises(ValidationError, match="require text_stats"):
            kpc.DocumentTree.model_validate(doc)
    for doc in accepted:
        validator.validate(doc)
        kpc.DocumentTree.model_validate(doc)
