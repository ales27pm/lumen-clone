from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


ADAPTER_ARTIFACT_SCHEMA_VERSION = "lumen.peft-lora-adapter-artifact/1.0.0"
ADAPTER_ARTIFACT_MANIFEST_FILENAME = "adapter_artifact_manifest.json"
_REQUIRED_FILES = {"adapter_config.json", "tokenizer.json", "tokenizer_config.json"}
_WEIGHT_FILES = {"adapter_model.safetensors"}
_OPTIONAL_FILES = {
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
}
_ALLOWED_FILES = _REQUIRED_FILES | _WEIGHT_FILES | _OPTIONAL_FILES
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_nonfinite_json_constant(value)
    return parsed


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


def _load_json_object(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid strict JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _validate_safetensors_file(path: Path) -> set[str]:
    size = path.stat().st_size
    if size < 9:
        raise ValueError("adapter_model.safetensors is truncated or empty")
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        header_size = int.from_bytes(header_size_bytes, byteorder="little", signed=False)
        if header_size == 0 or header_size > size - 8:
            raise ValueError("adapter_model.safetensors has an invalid header length")
        header = _load_json_object(
            handle.read(header_size),
            label="adapter_model.safetensors header",
        )

    data_size = size - 8 - header_size
    tensor_entries = [
        (name, value) for name, value in header.items() if name != "__metadata__"
    ]
    if not tensor_entries:
        raise ValueError("adapter_model.safetensors must contain at least one tensor")
    tensor_names = {name for name, _ in tensor_entries if isinstance(name, str)}
    if not any(".lora_" in name or "lora_embedding_" in name for name in tensor_names):
        raise ValueError("adapter_model.safetensors does not contain LoRA tensors")

    ranges: list[tuple[int, int]] = []
    for name, value in tensor_entries:
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ValueError("adapter_model.safetensors contains an invalid tensor entry")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"adapter_model.safetensors tensor {name!r} has an unsupported dtype")
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ValueError(f"adapter_model.safetensors tensor {name!r} has an invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise ValueError(f"adapter_model.safetensors tensor {name!r} has invalid offsets")
        start, end = offsets
        if start < 0 or end <= start or end > data_size:
            raise ValueError(f"adapter_model.safetensors tensor {name!r} is out of bounds")
        expected_bytes = math.prod(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ValueError(
                f"adapter_model.safetensors tensor {name!r} byte length does not match its dtype and shape"
            )
        ranges.append((start, end))

    expected_start = 0
    for start, end in sorted(ranges):
        if start != expected_start:
            raise ValueError("adapter_model.safetensors tensor data has gaps or overlaps")
        expected_start = end
    if expected_start != data_size:
        raise ValueError("adapter_model.safetensors contains unreferenced tensor data")
    return tensor_names


def _validate_weight_file(path: Path) -> set[str]:
    return _validate_safetensors_file(path)


def _validate_lora_config(
    config: Mapping[str, Any],
    *,
    tensor_names: set[str] | None,
) -> None:
    base_model = config.get("base_model_name_or_path")
    if base_model is not None and (
        not isinstance(base_model, str) or not base_model.strip()
    ):
        raise ValueError("adapter_config.json base_model_name_or_path must be non-empty")

    raw_targets = config.get("target_modules")
    target_modules: list[str] | None
    if raw_targets is None or (
        isinstance(raw_targets, str) and raw_targets.strip()
    ):
        target_modules = None
    elif isinstance(raw_targets, list) and raw_targets and all(
        isinstance(target, str) and target.strip() for target in raw_targets
    ):
        target_modules = raw_targets
    else:
        raise ValueError("adapter_config.json must declare non-empty LoRA target_modules")

    if tensor_names is not None and target_modules is not None and any(
        not any(
            f".{target}.lora_" in tensor_name
            or f".{target}.lora_embedding_" in tensor_name
            for tensor_name in tensor_names
        )
        for target in target_modules
    ):
        raise ValueError(
            "adapter_model.safetensors LoRA tensors do not match adapter_config.json target_modules"
        )


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
    if "adapter_model.bin" in names:
        raise ValueError(
            "Finalized adapter artifacts prohibit adapter_model.bin; "
            "save adapter_model.safetensors instead"
        )
    if extra:
        raise ValueError(f"Adapter artifact contains unrecognized files: {', '.join(extra)}")
    if weights != ["adapter_model.safetensors"]:
        raise ValueError(
            "Finalized adapter artifacts require adapter_model.safetensors"
        )

    config = _load_json_object(
        (adapter_dir / "adapter_config.json").read_bytes(),
        label="adapter_config.json",
    )
    if str(config.get("peft_type") or "").upper() != "LORA":
        raise ValueError("adapter_config.json must declare peft_type=LORA")
    tensor_names = _validate_weight_file(adapter_dir / weights[0])
    _validate_lora_config(config, tensor_names=tensor_names)
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
    expected_parent_sft_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = adapter_dir / ADAPTER_ARTIFACT_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            f"Adapter artifact manifest must be a regular file: {manifest_path}"
        )
    manifest = _load_json_object(
        manifest_path.read_bytes(),
        label="Adapter artifact manifest",
    )
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
    if (
        expected_parent_sft_adapter_sha256 is not None
        and rebuilt["parentSFTAdapterSHA256"]
        != expected_parent_sft_adapter_sha256
    ):
        raise ValueError(
            "Adapter artifact parent SFT digest does not match the expected finalized lineage"
        )
    return rebuilt
