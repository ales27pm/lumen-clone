from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")


def _load_export_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py"
    spec = importlib.util.spec_from_file_location("lumen_unsloth_export_gguf", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path, *, agent: str = "cortex", merge_by_default: bool = False) -> Path:
    config = {
        "agent": agent,
        "base_model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "baseModelRevision": "a" * 40,
        "baseModelIndexDigest": "b" * 64,
        "baseModelIndexReferencedShardNames": ["model.safetensors"],
        "baseModelIndexShardBindingSHA256": "f" * 64,
        "baseModelArtifactDigest": "c" * 64,
        "baseModelWeightShards": [
            {"filename": "model.safetensors", "size": 1, "sha256": "d" * 64}
        ],
        "baseModelTokenizerDigest": "e" * 64,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "output_dir": f"{tmp_path}/models/lora/{agent}",
        "artifact_mode": "adapter_first",
        "default_export_artifact": "lora_adapter",
        "merge_adapters_by_default": merge_by_default,
        "release_bake_enabled_by_default": False,
        "gguf_output_dir": f"{tmp_path}/models/gguf_release_bake/{agent}_merged_gguf",
    }
    path = tmp_path / f"{agent}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_load_config_requires_adapter_first_merge_disabled(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, merge_by_default=False)

    cfg = export_gguf.load_config(config_path)

    assert cfg["artifact_mode"] == "adapter_first"
    assert cfg["merge_adapters_by_default"] is False
    assert cfg["release_bake_enabled_by_default"] is False


def test_load_config_rejects_merge_by_default(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, merge_by_default=True)

    try:
        export_gguf.load_config(config_path)
    except ValueError as error:
        assert "merge_adapters_by_default=false" in str(error)
    else:
        raise AssertionError("load_config should reject configs that merge adapters by default")


def test_load_config_rejects_legacy_config_without_base_lineage(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("baseModelIndexDigest")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="baseModelIndexDigest"):
        export_gguf.load_config(config_path)


def test_gather_configs_prefers_generated_nested_configs(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path)
    generated = tmp_path / "generated" / "cortex"
    generated.mkdir(parents=True)
    (generated / "unsloth_config.json").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    configs = export_gguf.gather_configs([], str(tmp_path / "generated"), ["cortex"])

    assert [config["agent"] for config in configs] == ["cortex"]


def test_release_bake_skipped_manifest_is_adapter_first(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, merge_by_default=False)
    cfg = export_gguf.load_config(config_path)

    class Args:
        manifest_output = str(tmp_path / "release_bake_gguf_manifest.json")

    manifest = export_gguf._release_bake_skipped_manifest([cfg], Args())

    assert manifest["mode"] == "adapter_first"
    assert manifest["release_bake_requested"] is False
    assert manifest["skipped"] is True
    assert "--release-bake" in manifest["reason"]
    assert manifest["agents"]["cortex"]["merge_adapters_by_default"] is False
    assert manifest["agents"]["cortex"]["release_bake_enabled_by_default"] is False


def test_static_agent_configs_disable_default_release_bake() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = repo_root / "tools" / "fine_tuning" / "unsloth" / "configs"

    for agent in AGENTS:
        cfg = json.loads((config_dir / f"{agent}.json").read_text(encoding="utf-8"))
        assert cfg["agent"] == agent
        assert cfg["artifact_mode"] == "adapter_first"
        assert cfg["default_export_artifact"] == "lora_adapter"
        assert cfg["merge_adapters_by_default"] is False
        assert cfg["release_bake_enabled_by_default"] is False
        assert "lora" in cfg["output_dir"].lower()
        assert "release_bake" in cfg["gguf_output_dir"].lower()


def test_skip_existing_requires_matching_current_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path)
    cfg = export_gguf.load_config(config_path)
    output_dir = tmp_path / "models" / "gguf_release_bake" / "cortex_merged_gguf"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "lumen-cortex-release-bake-q4_k_m.gguf"
    target.write_bytes(b"GGUF-current")
    lineage = {
        "adapterSHA256": "1" * 64,
        "adapterTrainingPhase": "sft",
        "finalizedVariantManifestSHA256": "2" * 64,
        "sourceVariantManifestSHA256": "3" * 64,
        "baseModelRevision": cfg["baseModelRevision"],
        "baseModelIndexDigest": cfg["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": cfg["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": cfg["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": cfg["baseModelArtifactDigest"],
        "baseModelTokenizerDigest": cfg["baseModelTokenizerDigest"],
        "trainingEnvironmentSHA256": "4" * 64,
    }
    calls: list[str] = []

    def verify_lineage(_cfg: dict[str, object]) -> dict[str, object]:
        calls.append("verified")
        return lineage

    monkeypatch.setattr(export_gguf, "_verified_release_bake_lineage", verify_lineage)
    report = {
        "agent": "cortex",
        "mode": "optional_release_bake",
        "quantization": "q4_k_m",
        "adapter_dir": str(Path(cfg["output_dir"]).resolve()),
        "gguf_output_dir": str(output_dir.resolve()),
        "gguf_file": target.name,
        "gguf_path": str(target.resolve()),
        "size_bytes": target.stat().st_size,
        "sha256": export_gguf.sha256sum(target),
        "base_model_name": cfg["base_model_name"],
        **lineage,
    }
    (output_dir / "gguf_release_bake_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    summary = export_gguf.existing_summary_for_agent(
        cfg,
        output_root=tmp_path / "unused",
        quantization_override=None,
    )

    assert calls == ["verified"]
    assert summary is not None and summary["reused_existing"] is True

    target.write_bytes(b"GGUF-stale")
    with pytest.raises(ValueError, match="does not match current (size_bytes|sha256)"):
        export_gguf.existing_summary_for_agent(
            cfg,
            output_root=tmp_path / "unused",
            quantization_override=None,
        )


def test_ubuntu_post_training_gate_requires_finalized_source_and_artifact_lineage() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = (repo_root / "tools/fine_tuning/unsloth/ubuntu_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "Container image digest must be sha256" in script
    assert '"sourceVariantManifestSHA256": config.get("variantManifestSHA256")' in script
    assert "_shared_finalized_variant_validator" in script
    assert "canonical_sha256(unsigned) != digest" in script
    assert 'artifact.get("trainingPhase") != "sft"' in script
    assert "expected_adapter_sha256=str(artifact.get" in script


def test_release_bake_loads_pinned_base_before_verified_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (
        repo_root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py"
    ).read_text(encoding="utf-8")

    assert 'model_name=cfg["base_model_name"]' in source
    assert 'revision=cfg["baseModelRevision"]' in source
    assert "PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)" in source


def test_dpo_release_bake_requires_external_parent_sft_digest() -> None:
    export_gguf = _load_export_module()
    parent = "a" * 64
    artifact = {"parentSFTAdapterSHA256": parent}

    assert export_gguf._expected_adapter_training_lineage(
        {
            "adapter_training_phase": "sft_dpo",
            "parent_sft_adapter_sha256": parent,
        },
        artifact,
    ) == ("sft_dpo", parent)
    with pytest.raises(ValueError, match="configured parent SFT digest"):
        export_gguf._expected_adapter_training_lineage(
            {"adapter_training_phase": "sft_dpo"}, artifact
        )
    with pytest.raises(ValueError, match="configured parent SFT digest"):
        export_gguf._expected_adapter_training_lineage(
            {
                "adapter_training_phase": "sft_dpo",
                "parent_sft_adapter_sha256": "b" * 64,
            },
            artifact,
        )
