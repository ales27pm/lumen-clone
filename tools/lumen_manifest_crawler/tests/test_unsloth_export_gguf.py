from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


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
