from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .train_sft import (
        RUNTIME_MODEL_BINDING_SCHEMA,
        RUNTIME_TOKENIZER_BINDING_SCHEMA,
        load_config,
    )
except ImportError:
    from train_sft import (
        RUNTIME_MODEL_BINDING_SCHEMA,
        RUNTIME_TOKENIZER_BINDING_SCHEMA,
        load_config,
    )


REPORT_SCHEMA = "lumen.runtime-binding-smoke-gate/1.0.0"
CONTRACT_SCHEMA = "lumen.runtime-load-contract/1.0.0"
SMOKE_SCHEMA = "lumen.runtime-binding-smoke/1.0.0"
FORWARD_PROBE_SCHEMA = "lumen.runtime-forward-kernel-probe/1.0.0"
PARAMETER_PLACEMENT_SCHEMA = "lumen.runtime-parameter-placement/1.0.0"
REPORT_FILENAME = "runtime_binding_smoke.json"
REPORT_HASH_FIELD = "runtimeBindingSmokeGateSHA256"
SMOKE_HASH_FIELD = "runtimeBindingSmokeSHA256"
FORWARD_PROBE_HASH_FIELD = "runtimeForwardKernelProbeSHA256"
PARAMETER_PLACEMENT_HASH_FIELD = "runtimeParameterPlacementSHA256"
_AGENT = re.compile(r"[a-z]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BNB_ACTIVE_SKIP_MODULES = [
    "lm_head",
    "multi_modal_projector",
    "merger",
    "modality_projection",
    "router",
    "mlp.gate",
    "block_sparse_moe.gate",
    "mamba",
    "audio_tower",
    "vision_tower",
    "vision_embedder",
    "embed_vision",
    "embed_audio",
    "score",
    "classifier",
    "qa_outputs",
]
_FIXED_FORWARD_INPUT = {
    "inputIDs": [[1, 2, 3, 4]],
    "attentionMask": [[1, 1, 1, 1]],
    "useCache": False,
}
_PINNED_QWEN_LAYER_COUNT = 28
_PINNED_QWEN_VOCAB_SIZE = 151_936

# These are every prepared-config input consumed before or by the real model
# loader in train_sft's --runtime-binding-smoke path. Agent-specific datasets,
# output paths, adapter hyperparameters, and experiment labels are deliberately
# absent: they cannot change the base model load. Any loader-bearing addition
# must be added here before it can share an existing smoke result.
_RUNTIME_LOAD_CONFIG_FIELDS = (
    "base_model_name",
    "baseModelID",
    "baseModelRevision",
    "baseModelIndexDigest",
    "baseModelIndexReferencedShardNames",
    "baseModelIndexShardBindingSHA256",
    "baseModelArtifactDigest",
    "baseModelWeightShards",
    "baseModelGenerationConfigFile",
    "baseModelTokenizerDigest",
    "baseModelTokenizerFiles",
    "baseModelTokenizerClosureSHA256",
    "baseModelTokenizerSnapshotPath",
    "baseModelTokenizerSnapshotVerification",
    "baseModelRuntimeSnapshotPath",
    "baseModelRuntimeSnapshotVerification",
    "chatTemplateContract",
    "max_seq_length",
    "load_in_4bit",
    "bf16",
    "fp16",
    "seed",
    "trainingEnvironmentLock",
    "trainingContainerImageDigest",
    "trainingContainerImageDigestSource",
    "trainingRuntimeImageBindingStatus",
    "trainingRuntimeImageBindingVerified",
    "trainingEnvironmentSHA256",
    "trainingCodeManifest",
    "trainingCodeSHA256",
    "trainingDependencyLock",
    "trainingDependencyLockSHA256",
    "requirementsSHA256",
    "resolvedTrainingEnvironment",
    "resolvedTrainingEnvironmentSHA256",
    "resolvedTrainingEnvironmentCacheAttestation",
    "resolvedTrainingEnvironmentScanAudit",
    "spaceConfigurationSHA256",
    "zeroGPUSize",
    "zeroGPUDurationSeconds",
    "observedAccelerator",
    "runtimeSourceKind",
    "runtimeSourceRevision",
    "expectedRuntimeSourceRevision",
    "observedRepositoryRevision",
    "observedRuntimeRevision",
    "runtimeSourceBindingStatus",
    "runtimeSourceBindingMethod",
    "workingTreeDigest",
    "ubuntuOrchestrationCodeSHA256",
    "ubuntuSourceIntegritySHA256",
    "ubuntuSourceIntegrity",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file(
    path: Path,
    *,
    label: str,
    require_private_owner: bool = False,
    maximum_bytes: int = 32 << 20,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeError(f"{label} verification requires O_NOFOLLOW")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to open {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        if require_private_owner and (
            before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise RuntimeError(f"{label} must be process-owned mode 0600")
        if before.st_size > maximum_bytes:
            raise RuntimeError(f"{label} exceeds its bounded size")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        rebound = path.stat(follow_symlinks=False)
        if (
            offset != before.st_size
            or _file_signature(before) != _file_signature(after)
            or _file_signature(before) != _file_signature(rebound)
        ):
            raise RuntimeError(f"{label} changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _private_directory(path: Path, *, label: str) -> Path:
    try:
        observed = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Unable to inspect {label}: {path}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
        or resolved != path.absolute()
    ):
        raise RuntimeError(f"{label} must be a process-owned mode-0700 directory")
    return resolved


def _parse_agents(value: str) -> tuple[str, ...]:
    agents = tuple(value.split(","))
    if (
        not agents
        or any(_AGENT.fullmatch(agent) is None for agent in agents)
        or len(set(agents)) != len(agents)
    ):
        raise ValueError("agents must be a unique lowercase comma-separated list")
    return agents


def _runtime_load_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _RUNTIME_LOAD_CONFIG_FIELDS if field not in config]
    # Resolved/source-attestation fields are not required by non-Ubuntu configs,
    # but every field used by the loader itself is mandatory.
    required_missing = [
        field
        for field in missing
        if field
        not in {
            "resolvedTrainingEnvironment",
            "resolvedTrainingEnvironmentSHA256",
            "resolvedTrainingEnvironmentScanAudit",
            "workingTreeDigest",
            "ubuntuOrchestrationCodeSHA256",
            "ubuntuSourceIntegritySHA256",
            "ubuntuSourceIntegrity",
        }
    ]
    if required_missing:
        raise RuntimeError(
            "Prepared config lacks runtime-load contract fields: "
            + ", ".join(required_missing)
        )
    inputs = {
        field: config.get(field)
        for field in _RUNTIME_LOAD_CONFIG_FIELDS
    }
    return {
        "schemaVersion": CONTRACT_SCHEMA,
        "entryPoint": "tools.fine_tuning.unsloth.train_sft",
        "mode": "--runtime-binding-smoke",
        "fixedLoaderArguments": {
            "localFilesOnly": True,
            "trustRemoteCode": False,
            "useExactModelName": True,
            "modelNameSource": "baseModelRuntimeSnapshotPath",
            "tokenizerNameSource": "baseModelRuntimeSnapshotPath",
        },
        "preparedConfigInputs": inputs,
    }


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(
        _read_regular_file(path, label="prepared agent config")
    ).hexdigest()


def _prepared_contracts(
    run_root: Path,
    agents: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for agent in agents:
        config_path = run_root / "configs" / f"{agent}.json"
        config_sha256 = _config_sha256(config_path)
        config = load_config(config_path)
        if _config_sha256(config_path) != config_sha256:
            raise RuntimeError(f"Prepared config changed while loading: {agent}")
        if config.get("agent") != agent:
            raise RuntimeError(f"Prepared config agent mismatch: {agent}")
        contract = _runtime_load_contract(config)
        digest = canonical_sha256(contract)
        record = grouped.setdefault(
            digest,
            {
                "runtimeLoadContract": contract,
                "runtimeLoadContractSHA256": digest,
                "agents": [],
                "representativeAgent": agent,
                "configSHA256ByAgent": {},
            },
        )
        record["agents"].append(agent)
        record["configSHA256ByAgent"][agent] = config_sha256
        configs[agent] = config
    records = sorted(grouped.values(), key=lambda item: item["runtimeLoadContractSHA256"])
    return records, configs


def _self_hashed_object(
    value: Mapping[str, Any],
    *,
    schema: str,
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    result = dict(value)
    declared = result.pop(hash_field, None)
    if (
        result.get("schemaVersion") != schema
        or not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or canonical_sha256(result) != declared
    ):
        raise RuntimeError(f"{label} failed its self-hash contract")
    return dict(value)


def _require_exact_json_object(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Require exact JSON-schema closure, including primitive JSON types."""

    observed = dict(value)
    expected_object = dict(expected)
    try:
        matches = canonical_sha256(observed) == canonical_sha256(expected_object)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not canonical JSON evidence") from exc
    if not matches:
        raise RuntimeError(f"{label} drifted from its reconstructed contract")
    return observed


def _bound_runtime_model_config(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Read the exact pinned config.json bytes used by the runtime smoke."""

    records = inputs.get("baseModelTokenizerFiles")
    runtime_snapshot = inputs.get("baseModelRuntimeSnapshotPath")
    config_records = [
        item
        for item in records
        if isinstance(item, Mapping) and item.get("path") == "config.json"
    ] if isinstance(records, list) else []
    if len(config_records) != 1 or not isinstance(runtime_snapshot, str):
        raise RuntimeError("Runtime-load contract lacks one bound model config")
    record = config_records[0]
    size_bytes = record.get("sizeBytes")
    expected_sha256 = record.get("sha256")
    if (
        type(size_bytes) is not int
        or size_bytes <= 0
        or not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise RuntimeError("Runtime-load contract model config binding is invalid")
    payload = _read_regular_file(
        Path(runtime_snapshot) / "config.json",
        label="bound runtime model config",
        maximum_bytes=4 << 20,
    )
    if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError("Bound runtime model config drifted from prepared bytes")
    return _parse_object(payload, label="bound runtime model config")


def _expected_qwen_projection_names(layer_count: int) -> list[str]:
    return sorted(
        {
            *(
                f"model.layers.{layer}.self_attn.{projection}"
                for layer in range(layer_count)
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            ),
            *(
                f"model.layers.{layer}.mlp.{projection}"
                for layer in range(layer_count)
                for projection in ("gate_proj", "up_proj", "down_proj")
            ),
        }
    )


def _expected_qwen_parameter_names(
    layer_count: int,
    *,
    tied_embeddings: bool,
) -> list[str]:
    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        *(
            f"model.layers.{layer}.{name}"
            for layer in range(layer_count)
            for name in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            )
        ),
    }
    if not tied_embeddings:
        names.add("lm_head.weight")
    return sorted(names)


def _validate_forward_probe(
    value: Any,
    *,
    compute_dtype: str,
    vocab_size: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Runtime model binding lacks a CUDA forward probe")
    verified = _self_hashed_object(
        value,
        schema=FORWARD_PROBE_SCHEMA,
        hash_field=FORWARD_PROBE_HASH_FIELD,
        label="runtime CUDA forward probe",
    )
    expected_unsigned = {
        "schemaVersion": FORWARD_PROBE_SCHEMA,
        "status": "passed",
        "fixedInputSHA256": canonical_sha256(_FIXED_FORWARD_INPUT),
        "batchSize": 1,
        "tokenCount": 4,
        "logitsShape": [1, 4, vocab_size],
        "logitsDType": compute_dtype,
        "logitsDeviceType": "cuda",
        "allFinite": True,
        "requiresGrad": False,
        "useCache": False,
    }
    expected = {
        **expected_unsigned,
        FORWARD_PROBE_HASH_FIELD: canonical_sha256(expected_unsigned),
    }
    return _require_exact_json_object(
        verified,
        expected=expected,
        label="Runtime CUDA forward probe",
    )


def _validate_parameter_placement(
    value: Any,
    *,
    expected_parameter_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Runtime model binding lacks parameter placement evidence")
    verified = _self_hashed_object(
        value,
        schema=PARAMETER_PLACEMENT_SCHEMA,
        hash_field=PARAMETER_PLACEMENT_HASH_FIELD,
        label="runtime parameter placement",
    )
    parameter_count = len(expected_parameter_names)
    expected_unsigned = {
        "schemaVersion": PARAMETER_PLACEMENT_SCHEMA,
        "status": "passed",
        "totalParameterCount": parameter_count,
        "cudaParameterCount": parameter_count,
        "deviceTypeCounts": {"cuda": parameter_count},
        "parameterNamesSHA256": canonical_sha256(list(expected_parameter_names)),
        "allParametersOnCUDA": True,
    }
    expected = {
        **expected_unsigned,
        PARAMETER_PLACEMENT_HASH_FIELD: canonical_sha256(expected_unsigned),
    }
    return _require_exact_json_object(
        verified,
        expected=expected,
        label="Runtime parameter placement",
    )


def verify_runtime_load_materialization_evidence(
    value: Mapping[str, Any],
    cfg: Mapping[str, Any],
    bound_config_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Purely reconstruct the live NF4 and forward-kernel evidence contract."""

    if not isinstance(value, Mapping):
        raise RuntimeError("Runtime model binding lacks 4-bit materialization evidence")
    materialization = dict(value)
    bf16 = cfg.get("bf16")
    fp16 = cfg.get("fp16")
    max_seq_length = cfg.get("max_seq_length")
    if (
        cfg.get("load_in_4bit") is not True
        or type(bf16) is not bool
        or type(fp16) is not bool
        or bf16 == fp16
        or type(max_seq_length) is not int
        or max_seq_length <= 0
    ):
        raise RuntimeError("Prepared runtime precision/context contract is invalid")
    compute_dtype = "bfloat16" if bf16 else "float16"
    if bound_config_payload is not None and not isinstance(
        bound_config_payload,
        Mapping,
    ):
        raise RuntimeError("Bound runtime model config evidence is invalid")
    model_config = (
        dict(bound_config_payload)
        if isinstance(bound_config_payload, Mapping)
        else _bound_runtime_model_config(cfg)
    )
    layer_count = model_config.get("num_hidden_layers")
    vocab_size = model_config.get("vocab_size")
    tied_embeddings = model_config.get("tie_word_embeddings")
    if (
        model_config.get("model_type") != "qwen3"
        or type(layer_count) is not int
        or layer_count != _PINNED_QWEN_LAYER_COUNT
        or model_config.get("attention_bias") is not False
        or tied_embeddings is not True
        or type(vocab_size) is not int
        or vocab_size != _PINNED_QWEN_VOCAB_SIZE
    ):
        raise RuntimeError("Bound runtime model is not the pinned Qwen3-1.7B topology")
    expected_names = _expected_qwen_projection_names(layer_count)
    expected_outer_config = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "quant_method": "bitsandbytes",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute_dtype,
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_has_fp16_weight": False,
        "llm_int8_threshold": 6.0,
        "llm_int8_skip_modules": None,
    }
    expected_active_config = {
        **expected_outer_config,
        "llm_int8_skip_modules": _BNB_ACTIVE_SKIP_MODULES,
    }
    expected_representative = {
        "moduleName": expected_names[0],
        "moduleClass": "bitsandbytes.nn.modules.Linear4bit",
        "parameterClass": "bitsandbytes.nn.modules.Params4bit",
        "deviceType": "cuda",
        "storageDType": "uint8",
        "computeDType": compute_dtype,
        "quantType": "nf4",
        "doubleQuantized": True,
        "bnbQuantized": True,
        "requiresGrad": False,
        "quantStatePresent": True,
        "quantStateClass": "bitsandbytes.functional.QuantState",
        "quantStateQuantType": "nf4",
        "quantStateNested": True,
        "quantStateBlocksize": 64,
        "quantStateDType": compute_dtype,
        "quantStateNestedBlocksize": 256,
    }
    expected_fields = {
        "requestedMaxSequenceLength": max_seq_length,
        "runtimeMaxSequenceLength": max_seq_length,
        "requestedComputeDType": compute_dtype,
        "runtimeIsLoadedIn4Bit": True,
        "runtimeIsQuantized": True,
        "runtimeQuantizationMethod": "bitsandbytes",
        "quantizerClass": (
            "transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer"
        ),
        "activeQuantizationConfigClass": (
            "transformers.utils.quantization_config.BitsAndBytesConfig"
        ),
        "outerQuantizationConfig": expected_outer_config,
        "activeQuantizationConfig": expected_active_config,
        "expectedTargetModuleCount": len(expected_names),
        "materializedTargetModuleCount": len(expected_names),
        "targetModuleNamesSHA256": canonical_sha256(expected_names),
        "representativeMaterializedTarget": expected_representative,
    }
    expected_module_records = [
        {**expected_representative, "moduleName": module_name}
        for module_name in expected_names
    ]
    expected_materialized_digest = canonical_sha256(expected_module_records)
    forward_probe = _validate_forward_probe(
        materialization.get("forwardKernelProbe"),
        compute_dtype=compute_dtype,
        vocab_size=vocab_size,
    )
    parameter_placement = _validate_parameter_placement(
        materialization.get("parameterPlacement"),
        expected_parameter_names=_expected_qwen_parameter_names(
            layer_count,
            tied_embeddings=tied_embeddings,
        ),
    )
    expected_materialization = {
        **expected_fields,
        "materializedTargetModulesSHA256": expected_materialized_digest,
        "parameterPlacement": parameter_placement,
        "forwardKernelProbe": forward_probe,
    }
    return _require_exact_json_object(
        materialization,
        expected=expected_materialization,
        label="Runtime NF4 materialization",
    )


def _validate_smoke(
    smoke: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _self_hashed_object(
        smoke,
        schema=SMOKE_SCHEMA,
        hash_field=SMOKE_HASH_FIELD,
        label="train_sft runtime-binding smoke",
    )
    inputs = contract.get("preparedConfigInputs")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("Runtime-load contract inputs are invalid")
    expected_fields = {
        "baseModelTokenizerDigest": inputs.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": inputs.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": inputs.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": inputs.get("baseModelGenerationConfigFile"),
        "baseModelTokenizerSnapshotPath": inputs.get(
            "baseModelTokenizerSnapshotPath"
        ),
        "baseModelTokenizerSnapshotVerification": inputs.get(
            "baseModelTokenizerSnapshotVerification"
        ),
        "baseModelRuntimeSnapshotPath": inputs.get("baseModelRuntimeSnapshotPath"),
        "baseModelRuntimeSnapshotVerification": inputs.get(
            "baseModelRuntimeSnapshotVerification"
        ),
    }
    if any(verified.get(field) != value for field, value in expected_fields.items()):
        raise RuntimeError("Runtime-binding smoke drifted from its prepared load contract")
    model_binding = verified.get("runtimeModelBinding")
    tokenizer_binding = verified.get("runtimeTokenizerBinding")
    if not isinstance(model_binding, Mapping) or not isinstance(tokenizer_binding, Mapping):
        raise RuntimeError("Runtime-binding smoke lacks model/tokenizer evidence")
    _self_hashed_object(
        model_binding,
        schema=RUNTIME_MODEL_BINDING_SCHEMA,
        hash_field="runtimeModelBindingSHA256",
        label="runtime model binding",
    )
    _self_hashed_object(
        tokenizer_binding,
        schema=RUNTIME_TOKENIZER_BINDING_SCHEMA,
        hash_field="runtimeTokenizerBindingSHA256",
        label="runtime tokenizer binding",
    )
    expected_snapshot_digest = inputs.get(
        "baseModelRuntimeSnapshotVerification", {}
    )
    expected_snapshot_digest = (
        expected_snapshot_digest.get("snapshotVerificationSHA256")
        if isinstance(expected_snapshot_digest, Mapping)
        else None
    )
    for binding in (model_binding, tokenizer_binding):
        if (
            binding.get("baseModelID") != inputs.get("baseModelID")
            or binding.get("baseModelRevision") != inputs.get("baseModelRevision")
            or binding.get("runtimeSnapshotPath")
            != inputs.get("baseModelRuntimeSnapshotPath")
            or binding.get("runtimeSnapshotVerificationSHA256")
            != expected_snapshot_digest
        ):
            raise RuntimeError("Runtime-binding smoke contains cross-contract evidence")
    expected_model_fields = {
        "baseModelIndexDigest": inputs.get("baseModelIndexDigest"),
        "baseModelIndexShardBindingSHA256": inputs.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": inputs.get("baseModelArtifactDigest"),
        "baseModelTokenizerClosureSHA256": inputs.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": inputs.get(
            "baseModelGenerationConfigFile"
        ),
    }
    if any(
        model_binding.get(field) != expected
        for field, expected in expected_model_fields.items()
    ):
        raise RuntimeError("Runtime model binding drifted from prepared base artifacts")
    if model_binding.get("localFilesOnly") is not True:
        raise RuntimeError("Runtime-binding smoke did not prove local-only model loading")
    verify_runtime_load_materialization_evidence(
        model_binding.get("runtimeLoadMaterialization", {}),
        inputs,
    )
    if any("peft" in str(key).lower() or "trainer" in str(key).lower() for key in verified):
        raise RuntimeError("Runtime-binding smoke unexpectedly contains trainer state")
    return verified


def _invoke_train_sft_smoke(config_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "tools.fine_tuning.unsloth.train_sft",
        "--config",
        str(config_path),
        "--runtime-binding-smoke",
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0 or not stdout_lines:
        if result.stdout:
            sys.stderr.write(result.stdout)
        raise RuntimeError(
            f"train_sft runtime-binding smoke failed with exit code {result.returncode}"
        )
    if len(stdout_lines) > 1:
        sys.stderr.write("\n".join(stdout_lines[:-1]) + "\n")
    return _parse_object(
        stdout_lines[-1].encode("utf-8"),
        label="train_sft runtime-binding smoke output",
    )


def _expected_report_prefix(
    run_root: Path,
    agents: Sequence[str],
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schemaVersion": REPORT_SCHEMA,
        "status": "passed",
        "runRoot": str(run_root),
        "agents": list(agents),
        "distinctRuntimeLoadContractCount": len(contracts),
    }


def _validate_report(
    report: Mapping[str, Any],
    *,
    run_root: Path,
    agents: Sequence[str],
    expected_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    verified = _self_hashed_object(
        report,
        schema=REPORT_SCHEMA,
        hash_field=REPORT_HASH_FIELD,
        label="runtime-binding smoke gate report",
    )
    prefix = _expected_report_prefix(run_root, agents, expected_contracts)
    if any(verified.get(field) != value for field, value in prefix.items()):
        raise RuntimeError("Runtime-binding smoke report does not match the prepared run")
    observed_contracts = verified.get("contracts")
    if not isinstance(observed_contracts, list) or len(observed_contracts) != len(
        expected_contracts
    ):
        raise RuntimeError("Runtime-binding smoke report contract inventory drifted")
    for observed, expected in zip(observed_contracts, expected_contracts, strict=True):
        if not isinstance(observed, Mapping):
            raise RuntimeError("Runtime-binding smoke report contract is invalid")
        expected_without_smoke = dict(expected)
        for field, value in expected_without_smoke.items():
            if observed.get(field) != value:
                raise RuntimeError(
                    "Runtime-binding smoke report no longer matches prepared configs"
                )
        _validate_smoke(
            observed.get("smoke", {}),
            contract=expected["runtimeLoadContract"],
        )
    return verified


def _remove_interrupted_staging(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("Unable to inspect interrupted smoke-report staging") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise RuntimeError("Interrupted smoke-report staging is unsafe")
    path.unlink()


def _write_report_atomic(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError("Refusing to replace an existing runtime-binding smoke report")
    staging = path.with_name(f".{path.name}.staging")
    _remove_interrupted_staging(staging)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeError("Smoke-report persistence requires O_NOFOLLOW")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
            0o600,
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(staging, path)
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | nofollow,
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if staging.exists() and not staging.is_symlink():
            staging.unlink()


def verify_existing_report(
    run_root: Path,
    agents: Sequence[str],
) -> dict[str, Any]:
    """Purely re-verify existing gate evidence; never load a model or write state."""

    requested_agents = tuple(agents)
    if isinstance(agents, (str, bytes)) or requested_agents != _parse_agents(
        ",".join(requested_agents)
    ):
        raise ValueError("agents must be a unique lowercase sequence")
    root = _private_directory(run_root, label="run root")
    _private_directory(root / "configs", label="prepared config directory")
    training = _private_directory(root / "training", label="training evidence directory")
    expected_contracts, _configs = _prepared_contracts(root, requested_agents)
    report_path = training / REPORT_FILENAME
    if not report_path.exists() and not report_path.is_symlink():
        raise RuntimeError(f"Missing runtime-binding smoke gate report: {report_path}")
    report = _parse_object(
        _read_regular_file(
            report_path,
            label="runtime-binding smoke gate report",
            require_private_owner=True,
        ),
        label="runtime-binding smoke gate report",
    )
    return _validate_report(
        report,
        run_root=root,
        agents=requested_agents,
        expected_contracts=expected_contracts,
    )


def run_gate(run_root: Path, agents: Sequence[str]) -> tuple[dict[str, Any], bool]:
    root = _private_directory(run_root, label="run root")
    _private_directory(root / "configs", label="prepared config directory")
    training = _private_directory(root / "training", label="training evidence directory")
    expected_contracts, _configs = _prepared_contracts(root, agents)
    report_path = training / REPORT_FILENAME
    if report_path.exists() or report_path.is_symlink():
        return verify_existing_report(root, agents), True

    completed_contracts: list[dict[str, Any]] = []
    for contract_record in expected_contracts:
        representative = str(contract_record["representativeAgent"])
        smoke = _invoke_train_sft_smoke(
            root / "configs" / f"{representative}.json"
        )
        verified_smoke = _validate_smoke(
            smoke,
            contract=contract_record["runtimeLoadContract"],
        )
        completed_contracts.append({**contract_record, "smoke": verified_smoke})
    unsigned = {
        **_expected_report_prefix(root, agents, expected_contracts),
        "contracts": completed_contracts,
    }
    report = {**unsigned, REPORT_HASH_FIELD: canonical_sha256(unsigned)}
    _validate_report(
        report,
        run_root=root,
        agents=agents,
        expected_contracts=expected_contracts,
    )
    _write_report_atomic(report_path, report)
    persisted = _parse_object(
        _read_regular_file(
            report_path,
            label="runtime-binding smoke gate report",
            require_private_owner=True,
        ),
        label="runtime-binding smoke gate report",
    )
    return (
        _validate_report(
            persisted,
            run_root=root,
            agents=agents,
            expected_contracts=expected_contracts,
        ),
        False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or verify the pre-training Unsloth runtime-binding smoke gate."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--agents", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents = _parse_agents(args.agents)
    report, reused = run_gate(Path(args.run_root), agents)
    action = "reverified" if reused else "created"
    print(
        f"Runtime-binding smoke gate {action}: "
        f"{report['distinctRuntimeLoadContractCount']} distinct load contract(s), "
        f"{len(report['agents'])} agent(s)"
    )


if __name__ == "__main__":
    main()
