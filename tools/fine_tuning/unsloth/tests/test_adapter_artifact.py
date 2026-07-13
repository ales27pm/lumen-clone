from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth.adapter_artifact import (
    verify_adapter_artifact,
    write_adapter_artifact_manifest,
)


def _write_adapter(path: Path) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_adapter_artifact_manifest_rejects_missing_extra_and_modified_files(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    manifest = write_adapter_artifact_manifest(adapter, training_phase="sft")

    assert verify_adapter_artifact(
        adapter,
        expected_adapter_sha256=manifest["adapterSHA256"],
        expected_training_phase="sft",
    ) == manifest

    (adapter / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unrecognized files"):
        verify_adapter_artifact(adapter)
    (adapter / "unexpected.bin").unlink()

    (adapter / "adapter_model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="do not match"):
        verify_adapter_artifact(adapter)

    (adapter / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="canonical PEFT weight file"):
        verify_adapter_artifact(adapter)


def test_dpo_artifact_binds_parent_sft_digest(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    parent = "a" * 64
    manifest = write_adapter_artifact_manifest(
        adapter,
        training_phase="sft_dpo",
        parent_sft_adapter_sha256=parent,
    )

    assert manifest["trainingPhase"] == "sft_dpo"
    assert manifest["parentSFTAdapterSHA256"] == parent
    with pytest.raises(ValueError, match="expected finalized lineage"):
        verify_adapter_artifact(adapter, expected_adapter_sha256="b" * 64)
