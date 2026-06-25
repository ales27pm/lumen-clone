from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.hf_zerogpu.build_lumen_zerogpu_space import parse_agents, write_space_bundle


def _write_agent_fixture(root: Path, agent: str) -> None:
    agent_dir = root / agent
    agent_dir.mkdir(parents=True)
    (agent_dir / "train_sft.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}) + "\n",
        encoding="utf-8",
    )
    (agent_dir / "val_sft.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}) + "\n",
        encoding="utf-8",
    )
    (agent_dir / "unsloth_config.json").write_text(
        json.dumps(
            {
                "agent": agent,
                "base_model_name": "Qwen/Qwen3-1.7B",
                "max_seq_length": 128,
                "load_in_4bit": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0,
                "learning_rate": 0.0002,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_train_epochs": 1,
                "warmup_steps": 0,
                "output_dir": f"models/lora_qwen3_bootstrap/{agent}",
                "dataset_dir": f"generated/fine_tuning/{agent}",
            }
        ),
        encoding="utf-8",
    )


def test_parse_agents_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError):
        parse_agents("executor,unknown")


def test_write_space_bundle_copies_dataset_and_writes_defaults(tmp_path: Path) -> None:
    dataset = tmp_path / "fine_tuning"
    dataset.mkdir()
    (dataset / "adapter_runtime_manifest.json").write_text(
        json.dumps(
            {
                "sharedBaseModelID": "Qwen/Qwen3-1.7B",
                "adapterRepoID": "user/adapters",
                "adapters": [{"agent": "executor", "baseModelID": "Qwen/Qwen3-1.7B"}],
            }
        ),
        encoding="utf-8",
    )
    _write_agent_fixture(dataset, "executor")

    build = write_space_bundle(
        root=Path(__file__).resolve().parents[3],
        run_id="test-run",
        run_root=tmp_path / "run",
        dataset_source=dataset,
        space_repo="user/space",
        dataset_repo="user/datasets",
        adapter_repo="user/adapters",
        agents=["executor"],
        base_model="",
        gpu_size="large",
        gpu_duration_seconds=3600,
    )

    defaults = json.loads(build.defaults_path.read_text(encoding="utf-8"))
    assert defaults["fresh_run"] is True
    assert defaults["resume_default"] is False
    assert defaults["adapter_first"] is True
    assert defaults["dataset_path_in_repo"] == "runs/test-run/fine_tuning"
    assert (build.space_dir / "app.py").exists()
    assert (build.space_dir / "lumen_train_sft.py").exists()
    assert (build.dataset_dir / "executor" / "train_sft.jsonl").exists()
