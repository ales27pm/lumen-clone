from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth.adapter_artifact import (
    portable_adapter_model_card,
    verify_adapter_artifact,
    write_adapter_artifact_manifest,
    write_portable_adapter_model_card,
)


BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
BASE_MODEL_REVISION = "a" * 40


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
                "base_model_name_or_path": BASE_MODEL_ID,
                "revision": BASE_MODEL_REVISION,
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
    write_portable_adapter_model_card(
        path,
        base_model_id=BASE_MODEL_ID,
        base_model_revision=BASE_MODEL_REVISION,
    )


def test_portable_adapter_model_card_is_deterministic_and_path_free(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    private_path = "/outputs/run/training/base_model_runtime_snapshot"
    (adapter / "README.md").write_text(private_path, encoding="utf-8")

    first = write_portable_adapter_model_card(
        adapter,
        base_model_id=BASE_MODEL_ID,
        base_model_revision=BASE_MODEL_REVISION,
    )
    second = write_portable_adapter_model_card(
        adapter,
        base_model_id=BASE_MODEL_ID,
        base_model_revision=BASE_MODEL_REVISION,
    )

    assert first == second == portable_adapter_model_card(
        BASE_MODEL_ID,
        BASE_MODEL_REVISION,
    )
    assert (adapter / "README.md").read_text(encoding="utf-8") == first
    assert private_path not in first
    assert BASE_MODEL_ID in first
    assert BASE_MODEL_REVISION in first


@pytest.mark.parametrize(
    "drift",
    (
        "/outputs/run/training/base_model_runtime_snapshot",
        "f" * 40,
    ),
)
def test_adapter_artifact_rejects_private_path_or_wrong_revision_in_readme(
    tmp_path: Path,
    drift: str,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    manifest = write_adapter_artifact_manifest(adapter, training_phase="sft")
    readme = (adapter / "README.md").read_text(encoding="utf-8")
    source = BASE_MODEL_REVISION if drift == "f" * 40 else BASE_MODEL_ID
    (adapter / "README.md").write_text(
        readme.replace(source, drift),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="base model and exact revision"):
        verify_adapter_artifact(
            adapter,
            expected_adapter_sha256=manifest["adapterSHA256"],
        )


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
    with pytest.raises(
        ValueError,
        match=re.escape("require adapter_model.safetensors"),
    ):
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


@pytest.mark.parametrize("revision", (None, "main", "f" * 40))
def test_adapter_artifact_requires_exact_base_revision(
    tmp_path: Path,
    revision: str | None,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["revision"] = BASE_MODEL_REVISION
    config_path.write_text(json.dumps(config), encoding="utf-8")
    write_adapter_artifact_manifest(
        adapter,
        training_phase="sft",
        expected_base_model=BASE_MODEL_ID,
        expected_base_revision=BASE_MODEL_REVISION,
    )
    config["revision"] = revision
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="revision"):
        verify_adapter_artifact(
            adapter,
            expected_base_model=BASE_MODEL_ID,
            expected_base_revision=BASE_MODEL_REVISION,
        )


@pytest.mark.parametrize(
    "payload,error",
    [
        ('{"schemaVersion":"first","schemaVersion":"second"}', "strict JSON"),
        ('{"adapterSHA256":NaN}', "strict JSON"),
        ('{"adapterSHA256":Infinity}', "strict JSON"),
        ('{"adapterSHA256":-Infinity}', "strict JSON"),
        ('{"adapterSHA256":1e400}', "strict JSON"),
    ],
)
def test_adapter_artifact_manifest_requires_strict_json(
    tmp_path: Path,
    payload: str,
    error: str,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    write_adapter_artifact_manifest(adapter, training_phase="sft")
    (adapter / "adapter_artifact_manifest.json").write_text(
        payload,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        verify_adapter_artifact(adapter)


def test_adapter_artifact_manifest_must_not_be_a_symlink(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    write_adapter_artifact_manifest(adapter, training_phase="sft")
    manifest_path = adapter / "adapter_artifact_manifest.json"
    manifest_copy = tmp_path / "manifest.json"
    manifest_copy.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(manifest_copy)

    with pytest.raises(ValueError, match="regular file"):
        verify_adapter_artifact(adapter)


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
                "base_model_name_or_path": BASE_MODEL_ID,
                "revision": BASE_MODEL_REVISION,
                "target_modules": ["k_proj"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=re.escape("do not match adapter_config.json target_modules"),
    ):
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


def test_adapter_artifact_rejects_pytorch_bin_for_finalized_artifacts(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)
    (adapter / "adapter_model.safetensors").unlink()
    (adapter / "adapter_model.bin").write_bytes(b"PK\x03\x04" + b"x" * 32)

    with pytest.raises(ValueError, match=re.escape("adapter_model.bin")):
        write_adapter_artifact_manifest(adapter, training_phase="sft")
