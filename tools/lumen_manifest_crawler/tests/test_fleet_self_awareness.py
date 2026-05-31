from pathlib import Path

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts


def test_fleet_artifacts_include_source_code_map_and_whole_system_records():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)

    assert artifacts.system_prompts
    assert artifacts.cross_model_training
    assert "## System Identity" in artifacts.markdown
    assert "## Source Integrity" in artifacts.markdown

    first_prompt = next(iter(artifacts.system_prompts.values()))
    payload = first_prompt["contextPayload"]
    assert "sourceCodeMap" in payload
    assert payload["sourceCodeMap"]["fileCount"] == len(manifest.sourceIntegrity.files)
    assert payload["sourceCodeMap"]["boundary"]
    assert "source_code_map" in first_prompt

    task_types = {record.get("taskType") for record in artifacts.cross_model_training}
    assert "fleet_whole_system_identity" in task_types
    assert "source_code_self_knowledge" in task_types
    assert "source_tool_registry_knowledge" in task_types
    assert "source_routing_knowledge" in task_types


def test_fleet_records_teach_peer_source_awareness_and_private_boundaries():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    task_types = {record.get("taskType") for record in artifacts.cross_model_training}

    if len(manifest.fleet.slots) > 1:
        assert "fleet_peer_source_knowledge" in task_types
        assert "fleet_private_state_boundary" in task_types

    serialized = "\n".join(str(record) for record in artifacts.cross_model_training)
    assert "single logical agent" in serialized or "one logical agent" in serialized
    assert "must not claim direct access" in serialized or "cannot inspect" in serialized
