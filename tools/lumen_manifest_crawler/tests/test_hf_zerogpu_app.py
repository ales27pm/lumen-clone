from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


class _DummyComponent:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> _DummyComponent:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def click(self, **_kwargs: Any) -> None:
        return None

    def queue(self) -> _DummyComponent:
        return self

    def launch(self) -> None:
        return None


def _load_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    source = Path(__file__).resolve().parents[2] / "hf_zerogpu" / "space_template" / "app.py"
    app_root = tmp_path / "space"
    app_root.mkdir()
    shutil.copy2(source, app_root / "app.py")
    (app_root / "lumen_zero_gpu_defaults.json").write_text(
        json.dumps(
            {
                "dataset_repo": "user/datasets",
                "dataset_path_in_repo": "runs/test/fine_tuning",
                "adapter_repo": "user/adapters",
                "run_id": "test",
                "agents": ["executor"],
                "container_image_digest": "sha256:" + "c" * 64,
            }
        ),
        encoding="utf-8",
    )

    gradio = ModuleType("gradio")
    for name in ("Blocks", "Row", "Textbox", "Number", "Dropdown", "Checkbox", "JSON", "Button"):
        setattr(gradio, name, _DummyComponent)
    gradio.Markdown = lambda *_args, **_kwargs: None
    spaces = ModuleType("spaces")
    spaces.GPU = lambda **_kwargs: (lambda function: function)
    hub = ModuleType("huggingface_hub")
    hub.HfApi = object
    hub.snapshot_download = lambda **_kwargs: ""
    monkeypatch.setitem(sys.modules, "gradio", gradio)
    monkeypatch.setitem(sys.modules, "spaces", spaces)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    spec = importlib.util.spec_from_file_location("lumen_zerogpu_test_app", app_root / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_variant_fixture(module: Any, root: Path) -> tuple[Path, dict[str, Any]]:
    agent_root = root / "executor"
    variant_root = agent_root / "experiments" / "internal_plus_public_optimized"
    variant_root.mkdir(parents=True, exist_ok=True)
    lanes = {
        "train_sft": [{"messages": [{"role": "assistant", "content": "ok"}]}],
        "val_sft": [{"messages": [{"role": "assistant", "content": "valid"}]}],
        "train_dpo": [{"prompt": "x", "chosen": "y", "rejected": "z"}],
        "val_dpo": [],
    }
    for lane, records in lanes.items():
        _write_jsonl(variant_root / f"{lane}.jsonl", records)
    config = {
        "agent": "executor",
        "baseModelID": "Qwen/Qwen3-1.7B",
        "base_model_name": "Qwen/Qwen3-1.7B",
        "baseModelRevision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "baseModelArtifactDigest": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
        "baseModelTokenizerDigest": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "max_seq_length": 128,
        "load_in_4bit": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "learning_rate": 0.0002,
        "seed": 42,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 1,
        "warmup_steps": 0,
        "merge_adapters_by_default": False,
        "release_bake_enabled_by_default": False,
    }
    environment_lock = {
        "schemaVersion": "lumen.adapter-training-environment-lock/1.0.0",
        "pythonVersion": "3.10",
        "cudaVersion": "12.8",
        "packageVersions": {"torch": "2.8.0"},
        "unslothRevision": "935474c20aabc2aadb1da17338959c7c6f9bdafe",
        "llamaCppRevision": "34558825a27f4d74dcfd7a91bfde4464baa2a30a",
        "baseTokenizerSHA256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "containerImageDigestPolicy": "required_immutable_sha256_at_training",
    }
    config["trainingEnvironmentLock"] = environment_lock
    (agent_root / "unsloth_config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = {
        "agent": "executor",
        "variant": "internal_plus_public_optimized",
        "baseModelID": "Qwen/Qwen3-1.7B",
        "baseModelRevision": config["baseModelRevision"],
        "baseModelArtifactDigest": config["baseModelArtifactDigest"],
        "baseModelTokenizerDigest": config["baseModelTokenizerDigest"],
        "trainingEnvironmentLock": environment_lock,
        "trainingEnvironmentLockSHA256": module._canonical_sha256(environment_lock),
        "trainingEnvironmentSHA256": None,
        "seed": 42,
        "controlledTrainingConfig": config,
        "trainingConfigSHA256": module._canonical_sha256(config),
        "trainingCorpusSHA256": module._canonical_sha256(
            [*lanes["train_sft"], *lanes["val_sft"], *lanes["train_dpo"], *lanes["val_dpo"]]
        ),
        "datasets": {
            "trainSFT": {"count": 1, "sha256": module._canonical_sha256(lanes["train_sft"])},
            "validationSFT": {"count": 1, "sha256": module._canonical_sha256(lanes["val_sft"])},
            "trainDPO": {"count": 1, "sha256": module._canonical_sha256(lanes["train_dpo"])},
            "validationDPO": {"count": 0, "sha256": module._canonical_sha256(lanes["val_dpo"])},
        },
    }
    manifest["variantManifestSHA256"] = module._canonical_sha256(manifest)
    (variant_root / "variant_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "adapter_runtime_manifest.json").write_text(
        json.dumps(
            {
                "sharedBaseModelID": "Qwen/Qwen3-1.7B",
                "adapters": [{"agent": "executor", "baseModelID": "Qwen/Qwen3-1.7B"}],
            }
        ),
        encoding="utf-8",
    )
    return variant_root, manifest


def test_prepare_configs_selects_and_attests_optimized_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    variant_root, manifest = _write_variant_fixture(module, source_root)

    prepared = module._prepare_configs(
        source_root=source_root,
        run_root=tmp_path / "run",
        agents=["executor"],
        base_model_override="",
        seed=42,
        variant="internal_plus_public_optimized",
    )

    config = json.loads(Path(prepared[0]["config"]).read_text(encoding="utf-8"))
    assert config["dataset_dir"] == str(variant_root)
    assert config["variantManifestSHA256"] == manifest["variantManifestSHA256"]
    assert config["variantAttestation"]["trainingCorpusSHA256"] == manifest["trainingCorpusSHA256"]
    assert config["trainingContainerImageDigest"] == "sha256:" + "c" * 64
    assert config["trainingEnvironmentSHA256"] == module._canonical_sha256(
        {
            "schemaVersion": "lumen.adapter-training-environment/1.0.0",
            "containerImageDigest": "sha256:" + "c" * 64,
            "environmentLock": manifest["trainingEnvironmentLock"],
        }
    )
    assert config["variantAttestation"]["baseModelRevision"] == manifest["baseModelRevision"]
    assert config["variantAttestation"]["trainingEnvironmentSHA256"] == config["trainingEnvironmentSHA256"]


def test_variant_dataset_rejects_tampered_lane_and_control_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    variant_root, _ = _write_variant_fixture(module, source_root)
    (variant_root / "train_sft.jsonl").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "tampered"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        module._variant_dataset(
            source_root / "executor",
            agent="executor",
            variant="internal_plus_public_optimized",
        )

    _write_variant_fixture(module, source_root)
    config_path = source_root / "executor" / "unsloth_config.json"
    drifted_config = json.loads(config_path.read_text(encoding="utf-8"))
    drifted_config["max_train_records"] = 1
    config_path.write_text(json.dumps(drifted_config), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        module._prepare_configs(
            source_root=source_root,
            run_root=tmp_path / "run",
            agents=["executor"],
            base_model_override="",
            seed=42,
            variant="internal_plus_public_optimized",
        )

    _write_variant_fixture(module, source_root)
    with pytest.raises(ValueError, match="Seed override"):
        module._prepare_configs(
            source_root=source_root,
            run_root=tmp_path / "run",
            agents=["executor"],
            base_model_override="",
            seed=7,
            variant="internal_plus_public_optimized",
        )
