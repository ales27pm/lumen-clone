from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


ADAPTER_ARTIFACT_SCHEMA_VERSION = "lumen.peft-lora-adapter-artifact/1.0.0"
ADAPTER_ARTIFACT_MANIFEST_FILENAME = "adapter_artifact_manifest.json"
_REQUIRED_FILES = {"adapter_config.json", "tokenizer.json", "tokenizer_config.json"}
_WEIGHT_FILES = {"adapter_model.safetensors", "adapter_model.bin"}
_OPTIONAL_FILES = {
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
}
_ALLOWED_FILES = _REQUIRED_FILES | _WEIGHT_FILES | _OPTIONAL_FILES


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(adapter_dir: Path) -> list[Path]:
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter artifact directory does not exist: {adapter_dir}")
    entries = list(adapter_dir.iterdir())
    directories = sorted(entry.name for entry in entries if entry.is_dir())
    symlinks = sorted(entry.name for entry in entries if entry.is_symlink())
    if directories or symlinks:
        raise ValueError(
            "Adapter artifact must be a flat directory without subdirectories or symlinks: "
            + ", ".join([*directories, *symlinks])
        )
    files = sorted(
        (
            entry
            for entry in entries
            if entry.is_file() and entry.name != ADAPTER_ARTIFACT_MANIFEST_FILENAME
        ),
        key=lambda entry: entry.name,
    )
    names = {entry.name for entry in files}
    missing = sorted(_REQUIRED_FILES - names)
    weights = sorted(names & _WEIGHT_FILES)
    extra = sorted(names - _ALLOWED_FILES)
    if missing:
        raise ValueError(f"Adapter artifact is missing required files: {', '.join(missing)}")
    if len(weights) != 1:
        raise ValueError("Adapter artifact must contain exactly one canonical PEFT weight file")
    if extra:
        raise ValueError(f"Adapter artifact contains unrecognized files: {', '.join(extra)}")

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or str(config.get("peft_type") or "").upper() != "LORA":
        raise ValueError("adapter_config.json must declare peft_type=LORA")
    return files


def build_adapter_artifact_manifest(
    adapter_dir: Path,
    *,
    training_phase: str,
    parent_sft_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    if training_phase not in {"sft", "sft_dpo"}:
        raise ValueError("training_phase must be sft or sft_dpo")
    if training_phase == "sft_dpo":
        if re.fullmatch(r"[0-9a-f]{64}", str(parent_sft_adapter_sha256 or "")) is None:
            raise ValueError("sft_dpo artifacts require parent_sft_adapter_sha256")
    elif parent_sft_adapter_sha256 is not None:
        raise ValueError("sft artifacts cannot declare a parent SFT adapter")

    files = _artifact_files(adapter_dir)
    payload: dict[str, Any] = {
        "schemaVersion": ADAPTER_ARTIFACT_SCHEMA_VERSION,
        "artifactType": "peft_lora_directory",
        "trainingPhase": training_phase,
        "parentSFTAdapterSHA256": parent_sft_adapter_sha256,
        "files": [
            {
                "path": path.name,
                "sizeBytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in files
        ],
    }
    payload["adapterSHA256"] = canonical_sha256(payload)
    return payload


def write_adapter_artifact_manifest(
    adapter_dir: Path,
    *,
    training_phase: str,
    parent_sft_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = build_adapter_artifact_manifest(
        adapter_dir,
        training_phase=training_phase,
        parent_sft_adapter_sha256=parent_sft_adapter_sha256,
    )
    (adapter_dir / ADAPTER_ARTIFACT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_adapter_artifact(
    adapter_dir: Path,
    *,
    expected_adapter_sha256: str | None = None,
    expected_training_phase: str | None = None,
) -> dict[str, Any]:
    manifest_path = adapter_dir / ADAPTER_ARTIFACT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"Adapter artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Adapter artifact manifest must be a JSON object")
    phase = str(manifest.get("trainingPhase") or "")
    rebuilt = build_adapter_artifact_manifest(
        adapter_dir,
        training_phase=phase,
        parent_sft_adapter_sha256=(
            str(manifest.get("parentSFTAdapterSHA256"))
            if manifest.get("parentSFTAdapterSHA256") is not None
            else None
        ),
    )
    if dict(manifest) != rebuilt:
        raise ValueError("Adapter artifact files do not match the canonical manifest")
    digest = rebuilt["adapterSHA256"]
    if expected_adapter_sha256 is not None and digest != expected_adapter_sha256:
        raise ValueError("Adapter artifact digest does not match the expected finalized lineage")
    if expected_training_phase is not None and phase != expected_training_phase:
        raise ValueError(f"Expected a {expected_training_phase} adapter artifact, got {phase or '<missing>'}")
    return rebuilt
