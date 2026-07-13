from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth.adapter_artifact import (
    verify_adapter_artifact,
    write_adapter_artifact_manifest,
)


def _safetensors_bytes(data: bytes = b"\x00\x00\x00\x00") -> bytes:
    header = json.dumps(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(data)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return len(header).to_bytes(8, "little") + header + data


def _write_adapter(path: Path, *, weights: bytes | None = None) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(
        weights if weights is not None else _safetensors_bytes()
    )
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

    (adapter / "adapter_model.safetensors").write_bytes(
        _safetensors_bytes(b"\x01\x00\x00\x00")
    )
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
    assert verify_adapter_artifact(
        adapter,
        expected_training_phase="sft_dpo",
        expected_parent_sft_adapter_sha256=parent,
    ) == manifest
    with pytest.raises(ValueError, match="expected finalized lineage"):
        verify_adapter_artifact(adapter, expected_adapter_sha256="b" * 64)
    with pytest.raises(ValueError, match="parent SFT digest"):
        verify_adapter_artifact(
            adapter,
            expected_parent_sft_adapter_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "weights,error",
    [
        (b"", "truncated or empty"),
        (b"not-a-safetensors-file", "invalid header length"),
    ],
)
def test_adapter_artifact_rejects_malformed_safetensors(
    tmp_path: Path,
    weights: bytes,
    error: str,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter, weights=weights)

    with pytest.raises(ValueError, match=error):
        write_adapter_artifact_manifest(adapter, training_phase="sft")


def test_adapter_artifact_rejects_weights_outside_declared_lora_targets(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                "target_modules": ["k_proj"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match.*target_modules"):
        write_adapter_artifact_manifest(adapter, training_phase="sft")


def test_adapter_artifact_rejects_safetensor_shape_byte_mismatch(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    malformed = json.loads(
        _safetensors_bytes()[8 : 8 + int.from_bytes(_safetensors_bytes()[:8], "little")]
    )
    malformed[
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ]["shape"] = [2]
    header = json.dumps(malformed, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    (adapter / "adapter_model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"
    )

    with pytest.raises(ValueError, match="byte length does not match"):
        write_adapter_artifact_manifest(adapter, training_phase="sft")


def test_adapter_artifact_rejects_forged_pytorch_zip_magic(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    (adapter / "adapter_model.safetensors").unlink()
    (adapter / "adapter_model.bin").write_bytes(b"PK\x03\x04" + b"x" * 32)

    with pytest.raises(ValueError, match="valid PyTorch ZIP"):
        write_adapter_artifact_manifest(adapter, training_phase="sft")


def test_adapter_artifact_accepts_structural_pytorch_zip(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    (adapter / "adapter_model.safetensors").unlink()
    with zipfile.ZipFile(adapter / "adapter_model.bin", "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"weight": "storage"}))
        archive.writestr("archive/data/0", b"tensor-bytes")

    manifest = write_adapter_artifact_manifest(adapter, training_phase="sft")

    assert verify_adapter_artifact(adapter) == manifest


def test_adapter_artifact_rejects_invalid_pytorch_pickle(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    (adapter / "adapter_model.safetensors").unlink()
    with zipfile.ZipFile(adapter / "adapter_model.bin", "w") as archive:
        archive.writestr("archive/data.pkl", b"not-a-pickle")
        archive.writestr("archive/data/0", b"tensor-bytes")

    with pytest.raises(ValueError, match="invalid PyTorch pickle"):
        write_adapter_artifact_manifest(adapter, training_phase="sft")
