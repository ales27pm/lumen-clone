from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import train_dpo, train_sft
from tools.fine_tuning.unsloth.adapter_artifact import write_adapter_artifact_manifest


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


def _write_sft_adapter(path: Path) -> dict:
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
    (path / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return write_adapter_artifact_manifest(path, training_phase="sft")


def _write_finalized_sft_manifest(
    path: Path,
    artifact: dict,
    *,
    agent: str = "executor",
    variant: str = "internal_plus_public_optimized",
    source_variant_sha256: str = "c" * 64,
) -> None:
    payload = {
        "agent": agent,
        "variant": variant,
        "sourceVariantManifestSHA256": source_variant_sha256,
        "artifact": {
            "status": "trained",
            "trainingPhase": "sft",
            "adapterSHA256": artifact["adapterSHA256"],
            "adapterManifestSHA256": artifact["adapterSHA256"],
        },
    }
    payload["variantManifestSHA256"] = train_dpo._canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verified_sft_parent_rejects_identity_digest_and_file_drift(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "sft" / "executor"
    adapter.parent.mkdir()
    artifact = _write_sft_adapter(adapter)
    finalized = tmp_path / "sft-finalized.json"
    cfg = {
        "agent": "executor",
        "variant": "internal_plus_public_optimized",
        "variantManifestSHA256": "c" * 64,
    }

    _write_finalized_sft_manifest(finalized, artifact, agent="cortex")
    with pytest.raises(RuntimeError, match="finalized SFT artifact"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact, variant="internal_only")
    with pytest.raises(RuntimeError, match="finalized SFT artifact"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact, source_variant_sha256="d" * 64)
    with pytest.raises(RuntimeError, match="finalized SFT artifact"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact)
    payload = json.loads(finalized.read_text(encoding="utf-8"))
    payload["artifact"]["adapterSHA256"] = "f" * 64
    payload["variantManifestSHA256"] = train_dpo._canonical_sha256(
        {key: value for key, value in payload.items() if key != "variantManifestSHA256"}
    )
    finalized.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected finalized lineage"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )

    _write_finalized_sft_manifest(finalized, artifact)
    (adapter / "adapter_model.safetensors").write_bytes(
        _safetensors_bytes(b"\x01\x00\x00\x00")
    )
    with pytest.raises(ValueError, match="do not match"):
        train_dpo._verified_sft_parent(
            cfg, adapter_dir=adapter, finalized_manifest_path=finalized
        )


def test_dpo_output_path_must_be_role_scoped_and_separate(tmp_path: Path) -> None:
    sft = tmp_path / "lora" / "executor"
    cfg = {
        "agent": "executor",
        "adapter_output_dir": str(sft),
        "output_dir": str(tmp_path / "training" / "executor"),
        "dpo_output_dir": str(tmp_path / "lora_dpo" / "executor"),
    }
    work, output = train_dpo.validate_dpo_artifact_paths(cfg, sft_adapter_dir=sft)
    assert work.name == "dpo"
    assert output == Path(cfg["dpo_output_dir"])

    cfg["dpo_output_dir"] = str(tmp_path / "result")
    with pytest.raises(ValueError, match="agent role"):
        train_dpo.validate_dpo_artifact_paths(cfg, sft_adapter_dir=sft)


def test_verify_base_model_lineage_checks_pinned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    shard_name = "weights.safetensors"
    artifacts = {
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"layer.weight": shard_name}}, sort_keys=True
        ).encode(),
        "tokenizer.json": b"tokenizer",
        shard_name: b"weights",
    }
    downloaded: list[tuple[str, str, str]] = []
    for filename, content in artifacts.items():
        (tmp_path / filename).write_bytes(content)

    def hf_hub_download(*, repo_id: str, filename: str, revision: str) -> str:
        downloaded.append((repo_id, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    train_dpo._verify_base_model_lineage(
        {
            "base_model_name": "example/model",
            "baseModelRevision": revision,
            "baseModelIndexDigest": hashlib.sha256(artifacts["model.safetensors.index.json"]).hexdigest(),
            "baseModelWeightShards": [
                {
                    "filename": shard_name,
                    "size": len(artifacts[shard_name]),
                    "sha256": hashlib.sha256(artifacts[shard_name]).hexdigest(),
                }
            ],
            "baseModelTokenizerDigest": hashlib.sha256(artifacts["tokenizer.json"]).hexdigest(),
            "baseModelArtifactDigest": train_dpo._canonical_sha256(
                {
                    "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
                    "shards": [
                        {
                            "filename": shard_name,
                            "size": len(artifacts[shard_name]),
                            "sha256": hashlib.sha256(artifacts[shard_name]).hexdigest(),
                        }
                    ],
                }
            ),
        }
    )

    assert downloaded == [
        ("example/model", "model.safetensors.index.json", revision),
        ("example/model", "tokenizer.json", revision),
        ("example/model", shard_name, revision),
    ]


def test_verify_base_model_lineage_rejects_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "model.safetensors.index.json"
    artifact.write_bytes(b"unexpected")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_: str(artifact)),
    )

    with pytest.raises(RuntimeError, match="Pinned base-model artifact digest mismatch"):
        train_dpo._verify_base_model_lineage(
            {
                "base_model_name": "example/model",
                "baseModelRevision": "a" * 40,
                "baseModelIndexDigest": "0" * 64,
                "baseModelArtifactDigest": train_dpo._canonical_sha256(
                    {
                        "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
                        "shards": [
                            {
                                "filename": "weights.safetensors",
                                "size": 1,
                                "sha256": "2" * 64,
                            }
                        ],
                    }
                ),
                "baseModelWeightShards": [
                    {
                        "filename": "weights.safetensors",
                        "size": 1,
                        "sha256": "2" * 64,
                    }
                ],
                "baseModelTokenizerDigest": "1" * 64,
            }
        )


def test_sft_lineage_rejects_modified_weight_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "model.safetensors.index.json"
    tokenizer = tmp_path / "tokenizer.json"
    shard = tmp_path / "weights.safetensors"
    index.write_text(
        json.dumps({"weight_map": {"layer.weight": shard.name}}),
        encoding="utf-8",
    )
    tokenizer.write_bytes(b"tokenizer")
    shard.write_bytes(b"modified")
    files = {path.name: path for path in (index, tokenizer, shard)}
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda **kwargs: str(files[kwargs["filename"]])
        ),
    )
    declared_shards = [
        {"filename": shard.name, "size": 8, "sha256": hashlib.sha256(b"expected").hexdigest()}
    ]
    with pytest.raises(RuntimeError, match="weight shard digest mismatch"):
        train_sft._verify_base_model_lineage(
            {
                "base_model_name": "example/model",
                "baseModelRevision": "a" * 40,
                "baseModelIndexDigest": hashlib.sha256(index.read_bytes()).hexdigest(),
                "baseModelArtifactDigest": train_sft._canonical_sha256(
                    {
                        "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
                        "shards": declared_shards,
                    }
                ),
                "baseModelWeightShards": declared_shards,
                "baseModelTokenizerDigest": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            }
        )
