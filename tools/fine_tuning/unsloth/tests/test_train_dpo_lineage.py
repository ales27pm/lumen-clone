from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import train_dpo


def test_verify_base_model_lineage_checks_pinned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    artifacts = {
        "model.safetensors.index.json": b"model-index",
        "tokenizer.json": b"tokenizer",
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
            "baseModelArtifactDigest": hashlib.sha256(artifacts["model.safetensors.index.json"]).hexdigest(),
            "baseModelTokenizerDigest": hashlib.sha256(artifacts["tokenizer.json"]).hexdigest(),
        }
    )

    assert downloaded == [
        ("example/model", "model.safetensors.index.json", revision),
        ("example/model", "tokenizer.json", revision),
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
                "baseModelArtifactDigest": "0" * 64,
                "baseModelTokenizerDigest": "1" * 64,
            }
        )
