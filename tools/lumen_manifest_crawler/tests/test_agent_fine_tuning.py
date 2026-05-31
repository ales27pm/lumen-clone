from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.fine_tuning import AGENTS, compile_agent_fine_tuning_datasets
from lumen_manifest_crawler.output.writer import write_outputs
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets, validate_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def compiled_fine_tuning() -> tuple:
    repo_root = _repo_root()
    manifest = generate_manifest(repo_root)
    datasets = generate_all_datasets(manifest)
    fine_tuning = compile_agent_fine_tuning_datasets(manifest, datasets)
    return manifest, datasets, fine_tuning


def _write_fine_tuning_fixture(tmp_path: Path, compiled_fine_tuning: tuple) -> Path:
    manifest, datasets, fine_tuning = compiled_fine_tuning
    report = validate_manifest(manifest, datasets)
    output = tmp_path / "agent_manifest"
    fine_tuning_output = tmp_path / "fine_tuning"

    write_outputs(
        output,
        manifest,
        report,
        datasets,
        pretty=True,
        fine_tuning_datasets=fine_tuning,
        fine_tuning_output_dir=fine_tuning_output,
    )
    return fine_tuning_output


def test_per_agent_directories_are_produced(tmp_path: Path, compiled_fine_tuning: tuple) -> None:
    fine_tuning_output = _write_fine_tuning_fixture(tmp_path, compiled_fine_tuning)

    assert (fine_tuning_output / "adapter_runtime_manifest.json").exists()
    for agent in AGENTS:
        agent_dir = fine_tuning_output / agent
        assert agent_dir.exists()
        for filename in (
            "train_sft.jsonl",
            "val_sft.jsonl",
            "eval.jsonl",
            "dataset_card.json",
            "unsloth_config.json",
            "adapter_export_plan.json",
        ):
            assert (agent_dir / filename).exists(), f"missing {agent}/{filename}"


def test_written_fine_tuning_outputs_are_adapter_first(tmp_path: Path, compiled_fine_tuning: tuple) -> None:
    fine_tuning_output = _write_fine_tuning_fixture(tmp_path, compiled_fine_tuning)
    runtime_manifest = json.loads((fine_tuning_output / "adapter_runtime_manifest.json").read_text(encoding="utf-8"))

    assert runtime_manifest["mode"] == "adapter_first"
    assert runtime_manifest["runtimeStrategy"]["loadBaseModelOnce"] is True
    assert runtime_manifest["runtimeStrategy"]["selectAdapterByAgentSlot"] is True
    assert runtime_manifest["runtimeStrategy"]["mergeAdaptersByDefault"] is False
    assert runtime_manifest["runtimeStrategy"]["mergedExportPhase"] == "optional_release_bake"
    assert runtime_manifest["releaseBakePolicy"]["enabledByDefault"] is False

    adapters_by_agent = {entry["agent"]: entry for entry in runtime_manifest["adapters"]}
    for agent in AGENTS:
        expected_adapter_dir = f"models/lora/{agent}"
        agent_dir = fine_tuning_output / agent
        config = json.loads((agent_dir / "unsloth_config.json").read_text(encoding="utf-8"))
        plan = json.loads((agent_dir / "adapter_export_plan.json").read_text(encoding="utf-8"))

        assert config["artifactMode"] == "adapter_first"
        assert config["defaultExportArtifact"] == "lora_adapter"
        assert config["adapter_output_dir"] == expected_adapter_dir
        assert config["output_dir"] == expected_adapter_dir
        assert config["adapterExport"]["agent"] == agent
        assert config["adapterExport"]["adapterArtifact"] == expected_adapter_dir
        assert config["adapterExport"]["adapterDirectory"] == expected_adapter_dir
        assert config["adapterExport"]["trainBaseModelWeights"] is False
        assert config["adapterExport"]["saveAdapterByDefault"] is True
        assert config["adapterExport"]["mergeAdaptersByDefault"] is False
        assert config["mergeExport"]["enabledByDefault"] is False
        assert config["mergeExport"]["phase"] == "optional_release_bake"

        assert adapters_by_agent[agent]["adapterArtifact"] == expected_adapter_dir
        assert adapters_by_agent[agent]["adapterDirectory"] == expected_adapter_dir
        assert plan["mode"] == "adapter_first"
        assert plan["agent"] == agent
        assert plan["adapterArtifact"] == expected_adapter_dir
        assert plan["adapterDirectory"] == expected_adapter_dir
        assert plan["expectedArtifacts"]["adapterDirectory"] == expected_adapter_dir
        assert plan["runtimeBinding"]["loadBaseModelOnce"] is True
        assert plan["runtimeBinding"]["selectAdapterByAgentSlot"] is True
        assert plan["exportPolicy"]["defaultArtifact"] == "adapter"
        assert plan["exportPolicy"]["mergeAdaptersByDefault"] is False
        assert plan["exportPolicy"]["mergedExportPhase"] == "optional_release_bake"


def test_sft_records_use_chat_format(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        for record in (fine_tuning[agent].train_sft + fine_tuning[agent].val_sft)[:20]:
            messages = record["messages"]
            assert len(messages) == 3
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            assert isinstance(messages[2]["content"], str)
            assert messages[2]["content"].strip()


def test_dpo_records_have_prompt_chosen_rejected(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        for record in (fine_tuning[agent].train_dpo + fine_tuning[agent].val_dpo)[:20]:
            assert isinstance(record["prompt"], list)
            assert record["prompt"][0]["role"] == "system"
            assert record["prompt"][1]["role"] == "user"
            assert record["chosen"]["role"] == "assistant"
            assert record["rejected"]["role"] == "assistant"
            assert record["chosen"]["content"] != record["rejected"]["content"]


def test_no_unknown_agent_roles_unknown_tools_or_sentinel_leaks(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    failures = validate_agent_fine_tuning_datasets(manifest, fine_tuning)
    blocked = {
        "unknown_agent_role",
        "unknown_tool_id",
        "sentinel_leak",
        "dpo_chosen_equals_rejected",
        "eval_missing_expected",
        "missing_required_args_executor_examples",
    }
    failing_codes = {failure.code for failure in failures}
    assert blocked.isdisjoint(failing_codes), failures


def test_executor_has_tool_coverage(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    expected_tools = {tool.id for tool in manifest.tools}
    covered_tools: set[str] = set()
    for record in fine_tuning["executor"].train_sft + fine_tuning["executor"].val_sft:
        covered_tools.update(record["metadata"]["toolIDs"])
    assert expected_tools.issubset(covered_tools)


def test_fleet_has_model_slot_coverage(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    blob = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in (fine_tuning["fleet"].train_sft + fine_tuning["fleet"].val_sft))
    for slot in manifest.fleet.slots:
        assert slot.id in blob


def test_unsloth_configs_include_required_keys(compiled_fine_tuning: tuple) -> None:
    required = {
        "agent",
        "base_model_name",
        "max_seq_length",
        "load_in_4bit",
        "lora_r",
        "lora_alpha",
        "learning_rate",
        "dataset_dir",
        "output_dir",
    }
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        config = fine_tuning[agent].unsloth_config
        assert required.issubset(config.keys()), f"{agent} missing keys"

    config_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "configs"
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert required.issubset(cfg.keys()), f"{path} missing required keys"


def test_unsloth_output_dirs_include_agent_and_finetune_marker(compiled_fine_tuning: tuple) -> None:
    markers = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned"}
    _, _, fine_tuning = compiled_fine_tuning

    for agent in AGENTS:
        output_dir = str(fine_tuning[agent].unsloth_config["output_dir"])
        tokens = set("".join(ch.lower() if ch.isalnum() else " " for ch in output_dir).split())
        assert agent in tokens, f"{agent} output_dir missing slot token: {output_dir}"
        assert markers.intersection(tokens), f"{agent} output_dir missing finetune marker: {output_dir}"

    config_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "configs"
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        agent = str(cfg["agent"]).lower()
        output_dir = str(cfg["output_dir"])
        tokens = set("".join(ch.lower() if ch.isalnum() else " " for ch in output_dir).split())
        assert agent in tokens, f"{path} output_dir missing slot token: {output_dir}"
        assert markers.intersection(tokens), f"{path} output_dir missing finetune marker: {output_dir}"


def test_static_unsloth_configs_are_adapter_first_with_optional_release_bake() -> None:
    config_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "configs"
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg.get("artifact_mode") == "adapter_first", f"{path} must default to adapter-first artifacts"
        assert cfg.get("default_export_artifact") == "lora_adapter", f"{path} must save LoRA adapter by default"
        assert cfg.get("merge_adapters_by_default") is False, f"{path} must not merge adapters by default"
        assert cfg.get("release_bake_enabled_by_default") is False, f"{path} release bake must be opt-in"

        agent = str(cfg["agent"]).lower()
        gguf_output_dir = str(cfg.get("gguf_output_dir", ""))
        assert gguf_output_dir, f"{path} missing optional gguf_output_dir"
        tokens = set("".join(ch.lower() if ch.isalnum() else " " for ch in gguf_output_dir).split())
        assert agent in tokens, f"{path} optional gguf_output_dir missing slot token: {gguf_output_dir}"
        assert "gguf" in tokens, f"{path} optional gguf_output_dir missing gguf marker: {gguf_output_dir}"
        assert {"release", "bake"}.issubset(tokens), f"{path} optional gguf_output_dir missing release-bake marker: {gguf_output_dir}"
