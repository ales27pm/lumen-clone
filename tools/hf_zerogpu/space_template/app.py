from __future__ import annotations

import hashlib
import hmac
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fcntl

import gradio as gr
import spaces
from huggingface_hub import HfApi, snapshot_download
try:
    from lumen_manifest_crawler.dataset.optimization_policy import (
        EXPERIMENT_VARIANT_SCHEMA_VERSION,
        NON_TRAINING_CONFIG_FIELDS as _BASE_CONFIG_NON_TRAINING_FIELDS,
        effective_variant_training_config as _effective_variant_training_config,
        invariant_training_config as _normalized_invariant_training_config,
    )
except ImportError:
    from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.optimization_policy import (
        EXPERIMENT_VARIANT_SCHEMA_VERSION,
        NON_TRAINING_CONFIG_FIELDS as _BASE_CONFIG_NON_TRAINING_FIELDS,
        effective_variant_training_config as _effective_variant_training_config,
        invariant_training_config as _normalized_invariant_training_config,
    )
try:
    from lumen_training.adapter_artifact import verify_adapter_artifact
    from lumen_training.training_lineage import (
        build_resolved_training_environment_snapshot,
        create_private_base_model_runtime_snapshot,
        DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE,
        installed_controlled_package_versions,
        private_base_model_runtime_snapshot_required_bytes,
        RUN_RESUME_LINEAGE_SCHEMA,
        TRAINING_VARIANT_ATTESTATION_SCHEMA,
        sign_resolved_training_environment_cache,
        validate_runtime_source,
        verify_resolved_training_environment,
        verify_resolved_training_environment_cache,
        verify_base_model_tokenizer_snapshot,
        verify_private_base_model_conversion_snapshot,
        verify_private_base_model_tokenizer_snapshot,
        verify_space_configuration,
        verify_training_code_manifest,
        verify_training_dependency_lock,
        ZERO_GPU_ALLOWED_SIZES,
    )
except ImportError:
    # Allows the template to be loaded directly by repository tests. Built
    # Spaces always use the package import above.
    from adapter_artifact import verify_adapter_artifact
    from training_lineage import (
        build_resolved_training_environment_snapshot,
        create_private_base_model_runtime_snapshot,
        DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE,
        installed_controlled_package_versions,
        private_base_model_runtime_snapshot_required_bytes,
        RUN_RESUME_LINEAGE_SCHEMA,
        TRAINING_VARIANT_ATTESTATION_SCHEMA,
        sign_resolved_training_environment_cache,
        validate_runtime_source,
        verify_resolved_training_environment,
        verify_resolved_training_environment_cache,
        verify_base_model_tokenizer_snapshot,
        verify_private_base_model_conversion_snapshot,
        verify_private_base_model_tokenizer_snapshot,
        verify_space_configuration,
        verify_training_code_manifest,
        verify_training_dependency_lock,
        ZERO_GPU_ALLOWED_SIZES,
    )


APP_ROOT = Path(__file__).resolve().parent
DEFAULTS = json.loads((APP_ROOT / "lumen_zero_gpu_defaults.json").read_text(encoding="utf-8"))
DEFAULT_GPU_SIZE = os.environ.get("LUMEN_ZERO_GPU_SIZE", str(DEFAULTS.get("gpu_size", "large")))
MAX_ZERO_GPU_DURATION = int(os.environ.get("LUMEN_ZERO_GPU_MAX_DURATION_SECONDS", "1200"))
REQUESTED_GPU_DURATION = int(os.environ.get("LUMEN_ZERO_GPU_DURATION_SECONDS", str(DEFAULTS.get("gpu_duration_seconds", 1200))))
DEFAULT_GPU_DURATION = min(REQUESTED_GPU_DURATION, MAX_ZERO_GPU_DURATION)
AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
EXPERIMENT_VARIANTS = (
    "internal_only",
    "internal_plus_public_baseline",
    "internal_plus_public_optimized",
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
IMMUTABLE_HUB_REVISION = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MIN_ADMIN_TOKEN_LENGTH = 32
RUN_MANIFEST_NAME = "lumen_zerogpu_run_manifest.json"
PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME = "base_model_tokenizer_snapshot"
PRIVATE_BASE_MODEL_RUNTIME_SNAPSHOT_DIRNAME = "base_model_runtime_snapshot"
CHECKPOINT_LINEAGE_SCHEMA = "lumen.zerogpu.checkpoint_lineage/1.0.0"
TRAINING_RUN_SCHEMA = "lumen.zerogpu.training_run/2.1.0"
TRAINING_SUMMARY_SCHEMA = "lumen.zerogpu.training_summary/2.1.0"
CONTAINER_IMAGE_DIGEST_SOURCE = "operator_declared"
RUNTIME_IMAGE_BINDING_STATUS = "manual_validation_required"
RUNTIME_SOURCE_BINDING_UNVERIFIED = "operator_declared_unverified"
RUNTIME_SOURCE_BINDING_METHOD_REPOSITORY_HEAD = (
    "huggingface_repository_head_supplemental"
)
RUNTIME_SOURCE_BINDING_METHOD_DECLARATION = "operator_declared_only"
RUNTIME_SOURCE_LINEAGE_FIELDS = (
    "runtimeSourceKind",
    "runtimeSourceRevision",
    "expectedRuntimeSourceRevision",
    "observedRepositoryRevision",
    "observedRuntimeRevision",
    "runtimeSourceBindingStatus",
    "runtimeSourceBindingMethod",
)
ZERO_GPU_LINEAGE_FIELDS = (
    "zeroGPUSize",
    "zeroGPUDurationSeconds",
    "observedAccelerator",
)
REQUIRED_VARIANT_DATASET_FILES = (
    "train_sft.jsonl",
    "val_sft.jsonl",
    "train_dpo.jsonl",
    "val_dpo.jsonl",
)
UNCONTROLLED_CONFIG_FIELDS = {
    "adapterExport",
    "adapter_gguf_output_path",
    "adapter_output_dir",
    "dataset_dir",
    "dpo_output_dir",
    "gguf_output_dir",
    "gguf_repo_id",
    "mergeExport",
    "output_dir",
}
RUNTIME_LINEAGE_CONFIG_FIELDS = {
    "trainingContainerImageDigestSource",
    "trainingRuntimeImageBindingStatus",
    "trainingRuntimeImageBindingVerified",
    "trainingContainerImageDigest",
    "trainingEnvironmentSHA256",
    "checkpointLineagePath",
    "datasetPath",
    "datasetRepository",
    "datasetRevision",
    "localDatasetSnapshot",
    "requirementsSHA256",
    "resolvedTrainingEnvironment",
    "resolvedTrainingEnvironmentCacheAttestation",
    "resolvedTrainingEnvironmentScanAudit",
    "resolvedTrainingEnvironmentSHA256",
    "runResumeLineage",
    "runResumeLineageSHA256",
    "runtimeSourceKind",
    "runtimeSourceRevision",
    "expectedRuntimeSourceRevision",
    "observedRepositoryRevision",
    "observedRuntimeRevision",
    "runtimeSourceBindingStatus",
    "runtimeSourceBindingMethod",
    "spaceConfigurationSHA256",
    *ZERO_GPU_LINEAGE_FIELDS,
    "trainingCodeSHA256",
    "trainingCodeManifest",
    "trainingDependencyLock",
    "trainingDependencyLockSHA256",
    "variant",
    "variantAttestation",
    "variantManifestSHA256",
    "baseModelTokenizerSnapshotPath",
    "baseModelTokenizerSnapshotVerification",
    "baseModelGenerationConfigFile",
    "baseModelRuntimeSnapshotPath",
    "baseModelRuntimeSnapshotVerification",
}


class RequestAuthorizationError(Exception):
    pass


class AuthorizationConfigurationError(Exception):
    pass


class RepositoryCredentialConfigurationError(Exception):
    pass


class TrainingConflictError(Exception):
    pass


LOGGER = logging.getLogger("lumen.zerogpu")
_STARTUP_ENVIRONMENT_BYTES: bytes | None = None
_STARTUP_ENVIRONMENT_SCAN: dict[str, Any] | None = None
_STARTUP_ENVIRONMENT_ATTESTATION: dict[str, Any] | None = None
_STARTUP_ENVIRONMENT_HMAC_KEY: bytes | None = None
_STARTUP_ENVIRONMENT_ERROR: str | None = None


def _csv_agents(value: str) -> list[str]:
    agents = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = [agent for agent in agents if agent not in AGENTS]
    if unsupported:
        raise ValueError(f"Unsupported agents: {', '.join(unsupported)}")
    if not agents:
        raise ValueError("Select at least one agent")
    if len(agents) != len(set(agents)):
        raise ValueError("Selected agents must be unique")
    return agents


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _variant_effective_training_config(
    *,
    agent: str,
    base_config: dict[str, Any],
    controlled_config: dict[str, Any],
    train_sft_record_count: int,
    train_dpo_record_count: int,
    declared_invariant_sha256: Any,
) -> dict[str, Any]:
    try:
        effective = _effective_variant_training_config(
            agent=agent,
            base_config=base_config,
            controlled_config=controlled_config,
            noncontrolled_fields=_BASE_CONFIG_NON_TRAINING_FIELDS,
            sft_train_record_count=train_sft_record_count,
            dpo_train_record_count=train_dpo_record_count,
        )
        controlled_invariant = _normalized_invariant_training_config(
            controlled_config,
            agent=agent,
            sft_train_record_count=train_sft_record_count,
            dpo_train_record_count=train_dpo_record_count,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Variant optimization-step policy is invalid") from exc
    invariant_sha256 = _canonical_sha256(controlled_invariant)
    if declared_invariant_sha256 != invariant_sha256:
        raise ValueError(
            "Variant invariant training config differs from the base config"
        )
    return effective


def _initialize_startup_environment_cache() -> None:
    """Hash the installed environment once, before any ZeroGPU lease exists."""

    global _STARTUP_ENVIRONMENT_ATTESTATION
    global _STARTUP_ENVIRONMENT_BYTES
    global _STARTUP_ENVIRONMENT_ERROR
    global _STARTUP_ENVIRONMENT_HMAC_KEY
    global _STARTUP_ENVIRONMENT_SCAN
    try:
        environment, scan = build_resolved_training_environment_snapshot()
        verify_resolved_training_environment(environment)
        key = secrets.token_bytes(32)
        attestation = sign_resolved_training_environment_cache(
            environment,
            scan,
            key=key,
            startup_id=uuid.uuid4().hex,
        )
        _STARTUP_ENVIRONMENT_BYTES = json.dumps(
            environment,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _STARTUP_ENVIRONMENT_SCAN = dict(scan)
        _STARTUP_ENVIRONMENT_ATTESTATION = attestation
        _STARTUP_ENVIRONMENT_HMAC_KEY = key
        _STARTUP_ENVIRONMENT_ERROR = None
    except Exception:
        _STARTUP_ENVIRONMENT_BYTES = None
        _STARTUP_ENVIRONMENT_SCAN = None
        _STARTUP_ENVIRONMENT_ATTESTATION = None
        _STARTUP_ENVIRONMENT_HMAC_KEY = None
        _STARTUP_ENVIRONMENT_ERROR = "startup_environment_attestation_failed"
        LOGGER.error(
            "ZeroGPU startup environment attestation failed\n%s",
            traceback.format_exc(),
        )


def _verified_startup_environment_cache() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        _STARTUP_ENVIRONMENT_ERROR is not None
        or _STARTUP_ENVIRONMENT_BYTES is None
        or _STARTUP_ENVIRONMENT_SCAN is None
        or _STARTUP_ENVIRONMENT_ATTESTATION is None
        or _STARTUP_ENVIRONMENT_HMAC_KEY is None
    ):
        raise RuntimeError("ZeroGPU startup environment attestation is unavailable")
    environment = json.loads(_STARTUP_ENVIRONMENT_BYTES)
    if not isinstance(environment, dict):
        raise RuntimeError("ZeroGPU startup environment cache is malformed")
    scan = verify_resolved_training_environment_cache(
        environment,
        _STARTUP_ENVIRONMENT_ATTESTATION,
        key=_STARTUP_ENVIRONMENT_HMAC_KEY,
    )
    if scan != _STARTUP_ENVIRONMENT_SCAN:
        raise RuntimeError("ZeroGPU startup environment scan audit drifted")
    return environment, dict(scan), dict(_STARTUP_ENVIRONMENT_ATTESTATION)


def _startup_environment_child_variable() -> dict[str, str]:
    if (
        _STARTUP_ENVIRONMENT_HMAC_KEY is None
        or _STARTUP_ENVIRONMENT_ATTESTATION is None
    ):
        raise RuntimeError("ZeroGPU startup environment attestation is unavailable")
    return {
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_HMAC_KEY": (
            _STARTUP_ENVIRONMENT_HMAC_KEY.hex()
        ),
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_ATTESTATION": json.dumps(
            _STARTUP_ENVIRONMENT_ATTESTATION,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _startup_environment_status() -> str:
    if _STARTUP_ENVIRONMENT_ERROR is not None or _STARTUP_ENVIRONMENT_SCAN is None:
        return "Runtime environment attestation: unavailable. Training is disabled."
    return (
        "Runtime environment attestation: ready; digest `"
        f"{_STARTUP_ENVIRONMENT_SCAN['resolvedTrainingEnvironmentSHA256']}`; "
        f"distributions {_STARTUP_ENVIRONMENT_SCAN['distributionCount']}; "
        f"hashed bytes {_STARTUP_ENVIRONMENT_SCAN['totalHashedBytes']}; "
        f"startup scan {_STARTUP_ENVIRONMENT_SCAN['durationMilliseconds']} ms."
    )


def _deployed_zero_gpu_contract(*, requested_size: str | None = None) -> dict[str, Any]:
    configured_size = str(DEFAULTS.get("gpu_size", "large"))
    configured_duration = int(DEFAULTS.get("gpu_duration_seconds", 1200))
    if configured_size not in ZERO_GPU_ALLOWED_SIZES:
        raise ValueError("Deployed ZeroGPU size is unsupported")
    if configured_duration <= 0:
        raise ValueError("Deployed ZeroGPU duration must be positive")
    if DEFAULT_GPU_SIZE != configured_size:
        raise ValueError("ZeroGPU size drifted from the deployed Space configuration")
    if (
        REQUESTED_GPU_DURATION != configured_duration
        or DEFAULT_GPU_DURATION != configured_duration
        or MAX_ZERO_GPU_DURATION < configured_duration
    ):
        raise ValueError(
            "ZeroGPU duration drifted or would be clamped from the deployed Space configuration"
        )
    if requested_size is not None and requested_size != configured_size:
        raise ValueError("Requested GPU size differs from deployed Space configuration")
    return {
        "zeroGPUSize": configured_size,
        "zeroGPUDurationSeconds": configured_duration,
    }


def _observed_accelerator() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ZeroGPU lease did not expose a CUDA accelerator")
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        raise RuntimeError("ZeroGPU lease exposed no CUDA devices")
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "totalMemoryBytes": int(properties.total_memory),
                "computeCapability": [int(capability[0]), int(capability[1])],
            }
        )
    return {
        "bindingStatus": "runtime_observed_unverified",
        "backend": "cuda",
        "deviceCount": device_count,
        "devices": devices,
    }


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _immutable_hub_revision(value: Any, *, label: str) -> str:
    revision = str(value or "").strip().lower()
    if IMMUTABLE_HUB_REVISION.fullmatch(revision) is None:
        raise ValueError(f"{label} must be a full immutable Hub commit SHA")
    return revision


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass


def _self_hashed(payload: dict[str, Any], *, field: str) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return {**unsigned, field: _canonical_sha256(unsigned)}


def _optional_observed_revision(value: Any, *, label: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return _immutable_hub_revision(value, label=label)


def _resolve_runtime_source_binding(
    *,
    kind: Any,
    expected_revision: Any,
) -> dict[str, Any]:
    runtime_source_kind, revision = validate_runtime_source(
        kind=kind,
        revision=expected_revision,
    )
    if runtime_source_kind != "huggingface_space":
        raise ValueError("ZeroGPU runtimeSourceKind must be huggingface_space")

    observed_repository_revision: str | None = None
    observed_runtime_revision: str | None = None
    method = RUNTIME_SOURCE_BINDING_METHOD_DECLARATION
    space_repo = str(
        os.environ.get("SPACE_ID") or DEFAULTS.get("space_repo") or ""
    ).strip()
    token = os.environ.get("LUMEN_ZERO_GPU_HUB_TOKEN")
    if space_repo:
        try:
            api = HfApi(token=token)
            repository_info = api.space_info(
                repo_id=space_repo,
                files_metadata=False,
                token=token,
            )
            observed_repository_revision = _optional_observed_revision(
                getattr(repository_info, "sha", None),
                label="Observed Space repository revision",
            )
        except ValueError:
            raise
        except Exception:
            LOGGER.exception(
                "Unable to resolve supplemental Space source evidence"
            )

    # Hugging Face exposes repository identity and head revision, but no
    # independently attested executing-container revision. Repository-head
    # equality is useful audit context only and must never upgrade this binding.
    if (
        observed_repository_revision is not None
        and observed_repository_revision != revision
    ):
        raise ValueError(
            "Observed Space repository revision does not match the expected revision"
        )
    status = RUNTIME_SOURCE_BINDING_UNVERIFIED
    if observed_repository_revision is not None:
        method = RUNTIME_SOURCE_BINDING_METHOD_REPOSITORY_HEAD

    return {
        "runtimeSourceKind": runtime_source_kind,
        # Retained as a compatibility alias. It is always the expected revision,
        # never an assertion about the running container.
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": observed_repository_revision,
        "observedRuntimeRevision": observed_runtime_revision,
        "runtimeSourceBindingStatus": status,
        "runtimeSourceBindingMethod": method,
    }


def _read_self_hashed_json(
    path: Path,
    *,
    schema: str,
    hash_field: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required lineage manifest: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"Invalid lineage manifest contract: {path.name}")
    expected = payload.get(hash_field)
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    if (
        not isinstance(expected, str)
        or SHA256_PATTERN.fullmatch(expected) is None
        or _canonical_sha256(unsigned) != expected
    ):
        raise ValueError(f"Lineage manifest integrity check failed: {path.name}")
    return payload


def _checkpoint_directory_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError("Checkpoint directory is missing")
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    if not files:
        raise ValueError("Checkpoint directory is empty")
    entries = [
        {
            "path": candidate.relative_to(path).as_posix(),
            "sizeBytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        for candidate in sorted(files, key=lambda value: value.relative_to(path).as_posix())
    ]
    payload = {
        "schema": "lumen.zerogpu.checkpoint_directory/1.0.0",
        "files": entries,
    }
    return {**payload, "checkpointSHA256": _canonical_sha256(payload)}


def _validate_admin_token(value: str | None) -> str:
    token = value or ""
    if (
        len(token) < MIN_ADMIN_TOKEN_LENGTH
        or any(character.isspace() for character in token)
        or len(set(token)) < 12
    ):
        raise AuthorizationConfigurationError(
            "ZeroGPU administrative authorization is not configured"
        )
    return token


def _request_header(request: Any, name: str) -> str:
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    getter = getattr(headers, "get", None)
    if getter is None:
        return ""
    return str(getter(name) or getter(name.lower()) or "")


def _authorize_request(request: Any) -> None:
    expected = _validate_admin_token(os.environ.get("LUMEN_ZERO_GPU_ADMIN_TOKEN"))
    supplied = _request_header(request, "X-Lumen-Admin-Token")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise RequestAuthorizationError("Administrative authorization required")


@contextmanager
def _exclusive_training_operation() -> Any:
    work_root = Path(
        os.environ.get("LUMEN_ZERO_GPU_WORKDIR", "/tmp/lumen_zerogpu_runs")
    ).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    lock_path = work_root / ".lumen-training.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TrainingConflictError("Another training operation is active") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _external_failure(*, code: str, correlation_id: str, message: str) -> dict[str, Any]:
    return {
        "schema": "lumen.zerogpu.training_response/1.0.0",
        "ok": False,
        "error_code": code,
        "correlation_id": correlation_id,
        "message": message,
    }


def _installed_unsloth_revision() -> str:
    distribution = importlib_metadata.distribution("unsloth")
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        raise ValueError("Installed Unsloth package lacks VCS provenance")
    payload = json.loads(direct_url)
    revision = ((payload.get("vcs_info") or {}).get("commit_id"))
    if not isinstance(revision, str) or IMMUTABLE_HUB_REVISION.fullmatch(revision) is None:
        raise ValueError("Installed Unsloth package lacks an immutable VCS revision")
    return revision


def _verify_runtime_lineage() -> dict[str, Any]:
    code_manifest = DEFAULTS.get("trainingCodeManifest")
    code_digest = _require_sha256(
        DEFAULTS.get("trainingCodeSHA256"),
        label="trainingCodeSHA256",
    )
    if not isinstance(code_manifest, dict):
        raise ValueError("Missing training-code manifest")
    if verify_training_code_manifest(code_manifest, root=APP_ROOT) != code_digest:
        raise ValueError("Deployed training-code digest mismatch")

    space_configuration = DEFAULTS.get("spaceConfiguration")
    space_configuration_digest = _require_sha256(
        DEFAULTS.get("spaceConfigurationSHA256"),
        label="spaceConfigurationSHA256",
    )
    if (
        not isinstance(space_configuration, dict)
        or verify_space_configuration(
            space_configuration,
            readme_path=APP_ROOT / "README.md",
        )
        != space_configuration_digest
    ):
        raise ValueError("Deployed Space runtime configuration mismatch")

    dependency_lock = DEFAULTS.get("trainingDependencyLock")
    dependency_digest = _require_sha256(
        DEFAULTS.get("trainingDependencyLockSHA256"),
        label="trainingDependencyLockSHA256",
    )
    requirements_digest = _require_sha256(
        DEFAULTS.get("requirementsSHA256"),
        label="requirementsSHA256",
    )
    if not isinstance(dependency_lock, dict):
        raise ValueError("Missing training dependency lock")
    installed_versions = installed_controlled_package_versions(dependency_lock)
    import torch

    runtime_python_version = ".".join(platform.python_version_tuple()[:2])
    runtime_cuda_version = str(torch.version.cuda or "")
    if (
        verify_training_dependency_lock(
            dependency_lock,
            requirements_path=APP_ROOT / "requirements.txt",
            installed_versions=installed_versions,
            installed_unsloth_revision=_installed_unsloth_revision(),
            runtime_python_version=runtime_python_version,
            runtime_cuda_version=runtime_cuda_version,
        )
        != dependency_digest
        or dependency_lock.get("requirementsSHA256") != requirements_digest
    ):
        raise ValueError("Training dependency lineage mismatch")
    (
        resolved_environment,
        resolved_environment_scan,
        resolved_environment_cache_attestation,
    ) = _verified_startup_environment_cache()
    resolved_environment_digest = verify_resolved_training_environment(
        resolved_environment,
    )

    configured_revision = os.environ.get(
        "LUMEN_ZERO_GPU_EXPECTED_RUNTIME_SOURCE_REVISION"
    )
    legacy_revision = os.environ.get("LUMEN_ZERO_GPU_RUNTIME_SOURCE_REVISION")
    if (
        configured_revision is not None
        and legacy_revision is not None
        and configured_revision != legacy_revision
    ):
        raise ValueError("Configured runtime-source revisions disagree")
    runtime_source = _resolve_runtime_source_binding(
        kind=os.environ.get(
            "LUMEN_ZERO_GPU_RUNTIME_SOURCE_KIND",
            DEFAULTS.get("runtimeSourceKind"),
        ),
        expected_revision=configured_revision or legacy_revision,
    )
    return {
        "trainingCodeManifest": code_manifest,
        "trainingCodeSHA256": code_digest,
        "trainingDependencyLock": dependency_lock,
        "trainingDependencyLockSHA256": dependency_digest,
        "requirementsSHA256": requirements_digest,
        "resolvedTrainingEnvironment": resolved_environment,
        "resolvedTrainingEnvironmentCacheAttestation": (
            resolved_environment_cache_attestation
        ),
        "resolvedTrainingEnvironmentScanAudit": resolved_environment_scan,
        "resolvedTrainingEnvironmentSHA256": resolved_environment_digest,
        "spaceConfigurationSHA256": space_configuration_digest,
        **_deployed_zero_gpu_contract(),
        **runtime_source,
    }


def _experiment_variant(value: str) -> str:
    variant = value.strip()
    if variant not in EXPERIMENT_VARIANTS:
        raise ValueError(
            f"Unsupported experiment variant: {variant or '<empty>'}. "
            f"Expected one of: {', '.join(EXPERIMENT_VARIANTS)}"
        )
    return variant


def _resolve_run_workspace(run_id: str, variant: str) -> tuple[str, Path]:
    marker = f"-{variant}"
    qualified_run_id = run_id if run_id.endswith(marker) else f"{run_id}{marker}"
    if RUN_ID_PATTERN.fullmatch(qualified_run_id) is None:
        raise ValueError("run_id contains unsupported characters or exceeds 128 characters")

    work_root = Path(
        os.environ.get("LUMEN_ZERO_GPU_WORKDIR", "/tmp/lumen_zerogpu_runs")
    ).resolve()
    run_root = (work_root / qualified_run_id).resolve()
    if run_root.parent != work_root:
        raise ValueError("run_id escapes the ZeroGPU work directory")
    return qualified_run_id, run_root


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{lineno}")
        records.append(record)
    return records


def _require_dataset_contract(
    manifest: dict[str, Any],
    *,
    key: str,
    records: list[dict[str, Any]],
    manifest_path: Path,
) -> None:
    datasets = manifest.get("datasets")
    contract = datasets.get(key) if isinstance(datasets, dict) else None
    if not isinstance(contract, dict):
        raise ValueError(f"Experiment variant manifest is missing datasets.{key}: {manifest_path}")
    if type(contract.get("count")) is not int or contract["count"] != len(records):
        raise ValueError(f"Experiment variant dataset count mismatch for datasets.{key}: {manifest_path}")
    if contract.get("sha256") != _canonical_sha256(records):
        raise ValueError(f"Experiment variant dataset hash mismatch for datasets.{key}: {manifest_path}")


def _variant_dataset(agent_root: Path, *, agent: str, variant: str) -> tuple[Path, dict[str, Any]]:
    variant = _experiment_variant(variant)
    variant_root = agent_root / "experiments" / variant
    for filename in REQUIRED_VARIANT_DATASET_FILES:
        path = variant_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated experiment dataset for {agent}/{variant}: {path}")
    manifest_path = variant_root / "variant_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing generated experiment variant manifest for {agent}/{variant}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Experiment variant manifest is not an object: {manifest_path}")
    if manifest.get("schemaVersion") != EXPERIMENT_VARIANT_SCHEMA_VERSION:
        raise ValueError(f"Experiment variant manifest schema is unsupported: {manifest_path}")
    if manifest.get("agent") != agent or manifest.get("variant") != variant:
        raise ValueError(f"Experiment variant manifest identity mismatch: {manifest_path}")
    expected_sha = manifest.get("variantManifestSHA256")
    unsigned = dict(manifest)
    unsigned.pop("variantManifestSHA256", None)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or _canonical_sha256(unsigned) != expected_sha:
        raise ValueError(f"Experiment variant manifest integrity check failed: {manifest_path}")
    lanes = {
        filename.removesuffix(".jsonl"): _read_jsonl(variant_root / filename)
        for filename in REQUIRED_VARIANT_DATASET_FILES
    }
    _require_dataset_contract(manifest, key="trainSFT", records=lanes["train_sft"], manifest_path=manifest_path)
    _require_dataset_contract(manifest, key="validationSFT", records=lanes["val_sft"], manifest_path=manifest_path)
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    if "trainDPO" in datasets or "validationDPO" in datasets:
        _require_dataset_contract(manifest, key="trainDPO", records=lanes["train_dpo"], manifest_path=manifest_path)
        _require_dataset_contract(manifest, key="validationDPO", records=lanes["val_dpo"], manifest_path=manifest_path)
    else:
        _require_dataset_contract(
            manifest,
            key="dpo",
            records=[*lanes["train_dpo"], *lanes["val_dpo"]],
            manifest_path=manifest_path,
        )
    training_corpus = [
        *lanes["train_sft"],
        *lanes["val_sft"],
        *lanes["train_dpo"],
        *lanes["val_dpo"],
    ]
    if manifest.get("trainingCorpusSHA256") != _canonical_sha256(training_corpus):
        raise ValueError(f"Experiment variant training-corpus hash mismatch: {manifest_path}")
    controlled = manifest.get("controlledTrainingConfig")
    if (
        not isinstance(controlled, dict)
        or manifest.get("trainingConfigSHA256") != _canonical_sha256(controlled)
    ):
        raise ValueError(
            f"Experiment variant training-config hash mismatch: {manifest_path}"
        )
    if (
        type(manifest.get("seed")) is not int
        or type(controlled.get("seed")) is not int
        or manifest.get("seed") != controlled.get("seed")
    ):
        raise ValueError(f"Experiment variant seed contract is invalid: {manifest_path}")
    try:
        invariant = _normalized_invariant_training_config(
            controlled,
            agent=agent,
            sft_train_record_count=len(lanes["train_sft"]),
            dpo_train_record_count=len(lanes["train_dpo"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Experiment variant optimization policy is invalid: {manifest_path}"
        ) from exc
    if manifest.get("trainingConfigInvariantSHA256") != _canonical_sha256(
        invariant
    ):
        raise ValueError(
            f"Experiment variant invariant training-config hash mismatch: {manifest_path}"
        )
    return variant_root, manifest


def _training_attestation(cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    datasets = manifest["datasets"]
    controlled = manifest["controlledTrainingConfig"]
    missing_controlled = set(controlled) - set(cfg)
    effective_controlled = {key: cfg[key] for key in controlled if key in cfg}
    unexpected_fields = set(cfg) - set(controlled) - UNCONTROLLED_CONFIG_FIELDS - RUNTIME_LINEAGE_CONFIG_FIELDS
    effective_digest = _canonical_sha256(effective_controlled)
    controlled_digest = _canonical_sha256(controlled)
    if (
        missing_controlled
        or effective_digest != controlled_digest
        or effective_digest != manifest.get("trainingConfigSHA256")
        or unexpected_fields
    ):
        raise ValueError("Effective training configuration drifted from the controlled variant")
    return {
        "schema": TRAINING_VARIANT_ATTESTATION_SCHEMA,
        "variant": manifest["variant"],
        "variantManifestSHA256": manifest["variantManifestSHA256"],
        "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
        "laneHashes": {
            name: contract["sha256"]
            for name, contract in sorted(datasets.items())
            if isinstance(contract, dict) and isinstance(contract.get("sha256"), str)
        },
        "effectiveTrainingConfigSHA256": effective_digest,
        "trainingConfigInvariantSHA256": manifest[
            "trainingConfigInvariantSHA256"
        ],
        "baseModelRevision": manifest["baseModelRevision"],
        "baseModelIndexDigest": manifest["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": manifest["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": manifest["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": manifest["baseModelArtifactDigest"],
        "baseModelWeightShards": manifest["baseModelWeightShards"],
        "baseModelTokenizerDigest": manifest["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": manifest["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": manifest[
            "baseModelTokenizerClosureSHA256"
        ],
        "trainingEnvironmentLockSHA256": manifest["trainingEnvironmentLockSHA256"],
        "trainingEnvironmentSHA256": cfg["trainingEnvironmentSHA256"],
        "trainingCodeSHA256": cfg.get("trainingCodeSHA256"),
        "trainingDependencyLockSHA256": cfg.get(
            "trainingDependencyLockSHA256"
        ),
        "requirementsSHA256": cfg.get("requirementsSHA256"),
        "resolvedTrainingEnvironmentSHA256": cfg.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "zeroGPUSize": cfg.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": cfg.get("zeroGPUDurationSeconds"),
        "observedAccelerator": cfg.get("observedAccelerator"),
        "spaceConfigurationSHA256": cfg.get("spaceConfigurationSHA256"),
        "runtimeSourceKind": cfg.get("runtimeSourceKind"),
        "runtimeSourceRevision": cfg.get("runtimeSourceRevision"),
        "expectedRuntimeSourceRevision": cfg.get(
            "expectedRuntimeSourceRevision"
        ),
        "observedRepositoryRevision": cfg.get("observedRepositoryRevision"),
        "observedRuntimeRevision": cfg.get("observedRuntimeRevision"),
        "runtimeSourceBindingStatus": cfg.get("runtimeSourceBindingStatus"),
        "runtimeSourceBindingMethod": cfg.get("runtimeSourceBindingMethod"),
        "runtimeImageBindingStatus": cfg["trainingRuntimeImageBindingStatus"],
        "runtimeImageBindingVerified": cfg["trainingRuntimeImageBindingVerified"],
    }


def _training_environment(
    manifest: dict[str, Any],
    *,
    runtime_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    container_digest = str(DEFAULTS.get("container_image_digest") or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", container_digest) is None:
        raise ValueError("An explicit operator-declared container image digest is required before training")
    lock = manifest.get("trainingEnvironmentLock")
    if not isinstance(lock, dict) or _canonical_sha256(lock) != manifest.get("trainingEnvironmentLockSHA256"):
        raise ValueError("Experiment variant training-environment lock is invalid")
    digest_source = DEFAULTS.get("container_image_digest_source")
    binding_status = DEFAULTS.get("runtime_image_binding_status")
    binding_verified = DEFAULTS.get("runtime_image_binding_verified")
    if (
        digest_source != CONTAINER_IMAGE_DIGEST_SOURCE
        or binding_status != RUNTIME_IMAGE_BINDING_STATUS
        or binding_verified is not False
    ):
        raise ValueError("ZeroGPU runtime-image provenance must remain explicitly unverified")
    payload = {
        "schemaVersion": "lumen.adapter-training-environment/1.0.0",
        "containerImageDigest": container_digest,
        "containerImageDigestSource": digest_source,
        "runtimeImageBindingStatus": binding_status,
        "runtimeImageBindingVerified": binding_verified,
        "effectiveSeed": int(manifest["seed"]),
        "environmentLock": lock,
        "zeroGPUSize": (
            runtime_lineage.get("zeroGPUSize")
            if runtime_lineage is not None
            else None
        ),
        "zeroGPUDurationSeconds": (
            runtime_lineage.get("zeroGPUDurationSeconds")
            if runtime_lineage is not None
            else None
        ),
        "observedAccelerator": (
            runtime_lineage.get("observedAccelerator")
            if runtime_lineage is not None
            else None
        ),
    }
    if runtime_lineage is not None:
        payload.update(
            {
                "trainingCodeSHA256": runtime_lineage["trainingCodeSHA256"],
                "trainingDependencyLockSHA256": runtime_lineage[
                    "trainingDependencyLockSHA256"
                ],
                "requirementsSHA256": runtime_lineage["requirementsSHA256"],
                "resolvedTrainingEnvironment": runtime_lineage[
                    "resolvedTrainingEnvironment"
                ],
                "resolvedTrainingEnvironmentSHA256": runtime_lineage[
                    "resolvedTrainingEnvironmentSHA256"
                ],
            }
        )
    return {**payload, "trainingEnvironmentSHA256": _canonical_sha256(payload)}


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _copy_dataset_snapshot(run_root: Path, dataset_repo: str, revision: str, path_in_repo: str, token: str) -> Path:
    revision = _immutable_hub_revision(revision, label="Dataset revision")
    allow_pattern = f"{path_in_repo}/**"
    snapshot = Path(
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[allow_pattern],
            token=token,
        )
    )
    source = snapshot / path_in_repo
    if not source.exists():
        raise FileNotFoundError(f"Downloaded dataset snapshot did not contain {path_in_repo}")
    target = run_root / "generated" / "fine_tuning"
    if target.exists():
        raise FileExistsError("Fresh dataset snapshot destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def _shared_base_tokenizer_lineage(
    run_lineage: dict[str, Any],
) -> dict[str, Any]:
    agents = run_lineage.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("ZeroGPU run lineage lacks agent tokenizer bindings")
    fields = (
        "baseModelID",
        "baseModelRevision",
        "baseModelTokenizerDigest",
        "baseModelTokenizerFiles",
        "baseModelTokenizerClosureSHA256",
        "baseModelTokenizerSnapshotPath",
        "baseModelGenerationConfigFile",
        "baseModelRuntimeSnapshotPath",
        "baseModelIndexDigest",
        "baseModelIndexReferencedShardNames",
        "baseModelIndexShardBindingSHA256",
        "baseModelArtifactDigest",
        "baseModelWeightShards",
    )
    first = agents[0]
    if not isinstance(first, dict):
        raise ValueError("ZeroGPU run tokenizer lineage is invalid")
    contract = {field: first.get(field) for field in fields}
    if any(
        not isinstance(item, dict)
        or any(item.get(field) != contract[field] for field in fields)
        for item in agents
    ):
        raise ValueError("ZeroGPU agents do not share one tokenizer closure")
    return contract


def _verify_private_tokenizer_for_run(
    run_root: Path,
    run_lineage: dict[str, Any],
) -> dict[str, Any]:
    contract = _shared_base_tokenizer_lineage(run_lineage)
    expected_path = (run_root / PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME).resolve()
    if contract["baseModelTokenizerSnapshotPath"] != str(expected_path):
        raise ValueError("ZeroGPU private tokenizer snapshot path drifted")
    return verify_private_base_model_tokenizer_snapshot(
        expected_path,
        base_model_id=contract["baseModelID"],
        base_model_name=contract["baseModelID"],
        base_model_revision=contract["baseModelRevision"],
        tokenizer_files=contract["baseModelTokenizerFiles"],
        tokenizer_digest=contract["baseModelTokenizerDigest"],
        tokenizer_closure_sha256=contract[
            "baseModelTokenizerClosureSHA256"
        ],
    )


def _materialize_private_tokenizer_for_run(
    run_root: Path,
    run_lineage: dict[str, Any],
    token: str,
) -> dict[str, Any]:
    contract = _shared_base_tokenizer_lineage(run_lineage)
    destination = run_root / PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "Fresh ZeroGPU private tokenizer snapshot already exists"
        )
    source = Path(
        snapshot_download(
            repo_id=contract["baseModelID"],
            revision=contract["baseModelRevision"],
            allow_patterns=[
                item["path"]
                for item in contract["baseModelTokenizerFiles"]
            ],
            token=token,
        )
    )
    source_verification = verify_base_model_tokenizer_snapshot(
        source,
        base_model_id=contract["baseModelID"],
        base_model_name=contract["baseModelID"],
        base_model_revision=contract["baseModelRevision"],
        tokenizer_files=contract["baseModelTokenizerFiles"],
        tokenizer_digest=contract["baseModelTokenizerDigest"],
        tokenizer_closure_sha256=contract[
            "baseModelTokenizerClosureSHA256"
        ],
    )
    source = Path(source_verification["snapshotPath"])
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME}.",
            dir=run_root,
        )
    )
    try:
        staging.chmod(0o700)
        for expected in contract["baseModelTokenizerFiles"]:
            filename = expected["path"]
            source_file = (source / filename).resolve(strict=True)
            target = staging / filename
            shutil.copyfile(source_file, target, follow_symlinks=False)
            target.chmod(0o400)
        verification = verify_private_base_model_tokenizer_snapshot(
            staging,
            base_model_id=contract["baseModelID"],
            base_model_name=contract["baseModelID"],
            base_model_revision=contract["baseModelRevision"],
            tokenizer_files=contract["baseModelTokenizerFiles"],
            tokenizer_digest=contract["baseModelTokenizerDigest"],
            tokenizer_closure_sha256=contract[
                "baseModelTokenizerClosureSHA256"
            ],
        )
        os.replace(staging, destination)
        staging = None
        final = _verify_private_tokenizer_for_run(run_root, run_lineage)
        if (
            final["baseModelTokenizerFiles"]
            != verification["baseModelTokenizerFiles"]
            or final["baseModelTokenizerClosureSHA256"]
            != verification["baseModelTokenizerClosureSHA256"]
        ):
            raise RuntimeError(
                "ZeroGPU private tokenizer snapshot changed during promotion"
            )
        return final
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _verify_private_base_model_for_run(
    run_root: Path,
    run_lineage: dict[str, Any],
) -> dict[str, Any]:
    contract = _shared_base_tokenizer_lineage(run_lineage)
    expected_path = (
        run_root / PRIVATE_BASE_MODEL_RUNTIME_SNAPSHOT_DIRNAME
    ).resolve()
    if contract["baseModelRuntimeSnapshotPath"] != str(expected_path):
        raise ValueError("ZeroGPU private base-model runtime path drifted")
    return verify_private_base_model_conversion_snapshot(
        expected_path,
        base_model_id=contract["baseModelID"],
        base_model_name=contract["baseModelID"],
        base_model_revision=contract["baseModelRevision"],
        tokenizer_files=contract["baseModelTokenizerFiles"],
        tokenizer_digest=contract["baseModelTokenizerDigest"],
        tokenizer_closure_sha256=contract[
            "baseModelTokenizerClosureSHA256"
        ],
        generation_config_file=contract["baseModelGenerationConfigFile"],
        model_index_digest=contract["baseModelIndexDigest"],
        index_referenced_shard_names=contract[
            "baseModelIndexReferencedShardNames"
        ],
        index_shard_binding_sha256=contract[
            "baseModelIndexShardBindingSHA256"
        ],
        model_artifact_digest=contract["baseModelArtifactDigest"],
        weight_shards=contract["baseModelWeightShards"],
    )


def _materialize_private_base_model_for_run(
    run_root: Path,
    run_lineage: dict[str, Any],
    token: str,
) -> dict[str, Any]:
    contract = _shared_base_tokenizer_lineage(run_lineage)
    required_runtime_bytes = private_base_model_runtime_snapshot_required_bytes(
        weight_shards=contract["baseModelWeightShards"],
        tokenizer_files=contract["baseModelTokenizerFiles"],
        generation_config_file=contract["baseModelGenerationConfigFile"],
    )
    if shutil.disk_usage(run_root).free < required_runtime_bytes:
        raise RuntimeError(
            "Insufficient free space for private base-model runtime snapshot"
        )
    source = Path(
        snapshot_download(
            repo_id=contract["baseModelID"],
            revision=contract["baseModelRevision"],
            token=token,
        )
    )
    return create_private_base_model_runtime_snapshot(
        source_snapshot_dir=source,
        private_tokenizer_snapshot_dir=(
            run_root / PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME
        ),
        destination=(
            run_root / PRIVATE_BASE_MODEL_RUNTIME_SNAPSHOT_DIRNAME
        ),
        base_model_id=contract["baseModelID"],
        base_model_name=contract["baseModelID"],
        base_model_revision=contract["baseModelRevision"],
        tokenizer_files=contract["baseModelTokenizerFiles"],
        tokenizer_digest=contract["baseModelTokenizerDigest"],
        tokenizer_closure_sha256=contract[
            "baseModelTokenizerClosureSHA256"
        ],
        generation_config_file=contract["baseModelGenerationConfigFile"],
        model_index_digest=contract["baseModelIndexDigest"],
        index_referenced_shard_names=contract[
            "baseModelIndexReferencedShardNames"
        ],
        index_shard_binding_sha256=contract[
            "baseModelIndexShardBindingSHA256"
        ],
        model_artifact_digest=contract["baseModelArtifactDigest"],
        weight_shards=contract["baseModelWeightShards"],
    )


def _agent_run_lineage(
    *,
    source_root: Path,
    run_root: Path,
    agent: str,
    variant: str,
) -> dict[str, Any]:
    variant_root, manifest = _variant_dataset(
        source_root / agent,
        agent=agent,
        variant=variant,
    )
    dataset_files = {
        filename: _sha256(variant_root / filename)
        for filename in REQUIRED_VARIANT_DATASET_FILES
    }
    lane_hashes = {
        name: contract["sha256"]
        for name, contract in sorted(manifest["datasets"].items())
        if isinstance(contract, dict) and isinstance(contract.get("sha256"), str)
    }
    return {
        "agent": agent,
        "sourceVariantManifestSHA256": _require_sha256(
            manifest.get("variantManifestSHA256"),
            label=f"{agent} variant manifest",
        ),
        "laneHashes": lane_hashes,
        "datasetFileSHA256": dataset_files,
        "trainingCorpusSHA256": _require_sha256(
            manifest.get("trainingCorpusSHA256"),
            label=f"{agent} training corpus",
        ),
        "controlledTrainingConfigSHA256": _require_sha256(
            manifest.get("trainingConfigSHA256"),
            label=f"{agent} training config",
        ),
        "trainingConfigInvariantSHA256": _require_sha256(
            manifest.get("trainingConfigInvariantSHA256"),
            label=f"{agent} invariant training config",
        ),
        "baseModelID": manifest.get("baseModelID"),
        "baseModelRevision": manifest.get("baseModelRevision"),
        "baseModelIndexDigest": manifest.get("baseModelIndexDigest"),
        "baseModelIndexReferencedShardNames": manifest.get(
            "baseModelIndexReferencedShardNames"
        ),
        "baseModelIndexShardBindingSHA256": manifest.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": manifest.get("baseModelArtifactDigest"),
        "baseModelWeightShards": manifest.get("baseModelWeightShards"),
        "baseModelTokenizerDigest": manifest.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": manifest.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": manifest.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelTokenizerSnapshotPath": str(
            run_root / PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME
        ),
        "baseModelGenerationConfigFile": (
            DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE
        ),
        "baseModelRuntimeSnapshotPath": str(
            run_root / PRIVATE_BASE_MODEL_RUNTIME_SNAPSHOT_DIRNAME
        ),
        "seed": manifest.get("seed"),
        "trainingEnvironmentLockSHA256": manifest.get(
            "trainingEnvironmentLockSHA256"
        ),
        "configPath": str(run_root / "configs" / f"{agent}.json"),
        "checkpointLineagePath": str(
            run_root / "checkpoint_lineage" / f"{agent}.json"
        ),
        "checkpointRoot": str(run_root / "training" / agent),
        "outputDirectory": str(run_root / "training" / agent),
        "adapterOutputDirectory": str(
            run_root / "models" / "lora_qwen3_bootstrap" / agent
        ),
    }


def _build_run_resume_lineage(
    *,
    run_id: str,
    run_root: Path,
    source_root: Path,
    dataset_repo: str,
    dataset_revision: str,
    dataset_path: str,
    agents: list[str],
    variant: str,
    seed: int,
    assistant_only_loss: bool,
    runtime_lineage: dict[str, Any],
) -> dict[str, Any]:
    if type(seed) is not int:
        raise ValueError("Run-resume lineage seed must be an exact integer")
    dataset_revision = _immutable_hub_revision(
        dataset_revision,
        label="Dataset revision",
    )
    agent_lineage = [
        _agent_run_lineage(
            source_root=source_root,
            run_root=run_root,
            agent=agent,
            variant=variant,
        )
        for agent in agents
    ]
    if any(
        type(item.get("seed")) is not int or item.get("seed") != seed
        for item in agent_lineage
    ):
        raise ValueError("Requested seed drifted from the controlled agent lineage")
    payload = {
        "schema": RUN_RESUME_LINEAGE_SCHEMA,
        "runID": run_id,
        "datasetRepository": dataset_repo,
        "datasetRevision": dataset_revision,
        "datasetPath": dataset_path,
        "localDatasetSnapshot": str(run_root / "generated" / "fine_tuning"),
        "selectedAgents": agents,
        "experimentVariant": variant,
        "seed": seed,
        "assistantOnlyLoss": bool(assistant_only_loss),
        "trainingCodeSHA256": runtime_lineage["trainingCodeSHA256"],
        "trainingDependencyLockSHA256": runtime_lineage[
            "trainingDependencyLockSHA256"
        ],
        "requirementsSHA256": runtime_lineage["requirementsSHA256"],
        "resolvedTrainingEnvironment": runtime_lineage[
            "resolvedTrainingEnvironment"
        ],
        "resolvedTrainingEnvironmentSHA256": runtime_lineage[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "zeroGPUSize": runtime_lineage["zeroGPUSize"],
        "zeroGPUDurationSeconds": runtime_lineage[
            "zeroGPUDurationSeconds"
        ],
        "observedAccelerator": runtime_lineage["observedAccelerator"],
        "spaceConfigurationSHA256": runtime_lineage[
            "spaceConfigurationSHA256"
        ],
        "runtimeSourceKind": runtime_lineage["runtimeSourceKind"],
        "runtimeSourceRevision": runtime_lineage["runtimeSourceRevision"],
        "expectedRuntimeSourceRevision": runtime_lineage[
            "expectedRuntimeSourceRevision"
        ],
        "observedRepositoryRevision": runtime_lineage[
            "observedRepositoryRevision"
        ],
        "observedRuntimeRevision": runtime_lineage[
            "observedRuntimeRevision"
        ],
        "runtimeSourceBindingStatus": runtime_lineage[
            "runtimeSourceBindingStatus"
        ],
        "runtimeSourceBindingMethod": runtime_lineage[
            "runtimeSourceBindingMethod"
        ],
        "agents": agent_lineage,
    }
    return {**payload, "runResumeLineageSHA256": _canonical_sha256(payload)}


def _initial_checkpoint_lineage(
    *,
    run_lineage: dict[str, Any],
    agent_lineage: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_LINEAGE_SCHEMA,
        "agent": agent_lineage["agent"],
        "runResumeLineageSHA256": run_lineage["runResumeLineageSHA256"],
        "configSHA256": _sha256(config_path),
        "datasetFileSHA256": agent_lineage["datasetFileSHA256"],
        "laneHashes": agent_lineage["laneHashes"],
        "resolvedTrainingEnvironmentSHA256": run_lineage[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "zeroGPUSize": run_lineage["zeroGPUSize"],
        "zeroGPUDurationSeconds": run_lineage["zeroGPUDurationSeconds"],
        "observedAccelerator": run_lineage["observedAccelerator"],
        "spaceConfigurationSHA256": run_lineage[
            "spaceConfigurationSHA256"
        ],
        "runtimeSourceBinding": {
            field: run_lineage[field]
            for field in RUNTIME_SOURCE_LINEAGE_FIELDS
        },
        "checkpointRoot": agent_lineage["checkpointRoot"],
        "outputDirectory": agent_lineage["outputDirectory"],
        "checkpoints": [],
    }
    return _self_hashed(payload, field="checkpointLineageSHA256")


def _validate_checkpoint_lineage(
    *,
    run_lineage: dict[str, Any],
    agent_lineage: dict[str, Any],
) -> dict[str, Any]:
    record_path = Path(agent_lineage["checkpointLineagePath"])
    record = _read_self_hashed_json(
        record_path,
        schema=CHECKPOINT_LINEAGE_SCHEMA,
        hash_field="checkpointLineageSHA256",
    )
    expected_static = {
        "agent": agent_lineage["agent"],
        "runResumeLineageSHA256": run_lineage["runResumeLineageSHA256"],
        "configSHA256": _sha256(Path(agent_lineage["configPath"])),
        "datasetFileSHA256": agent_lineage["datasetFileSHA256"],
        "laneHashes": agent_lineage["laneHashes"],
        "resolvedTrainingEnvironmentSHA256": run_lineage[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "zeroGPUSize": run_lineage["zeroGPUSize"],
        "zeroGPUDurationSeconds": run_lineage["zeroGPUDurationSeconds"],
        "observedAccelerator": run_lineage["observedAccelerator"],
        "spaceConfigurationSHA256": run_lineage[
            "spaceConfigurationSHA256"
        ],
        "runtimeSourceBinding": {
            field: run_lineage[field]
            for field in RUNTIME_SOURCE_LINEAGE_FIELDS
        },
        "checkpointRoot": agent_lineage["checkpointRoot"],
        "outputDirectory": agent_lineage["outputDirectory"],
    }
    if any(record.get(key) != value for key, value in expected_static.items()):
        raise ValueError("Checkpoint lineage drifted from the requested run")
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Resume requires at least one checkpoint bound to the run")
    root = Path(agent_lineage["checkpointRoot"]).resolve()
    valid: list[dict[str, str]] = []
    steps: list[int] = []
    for entry in checkpoints:
        if not isinstance(entry, dict):
            raise ValueError("Checkpoint lineage entries must be objects")
        relative = str(entry.get("path") or "")
        if re.fullmatch(r"checkpoint-[1-9][0-9]*", relative) is None:
            raise ValueError("Checkpoint lineage contains an invalid checkpoint path")
        steps.append(int(relative.removeprefix("checkpoint-")))
        checkpoint = (root / relative).resolve()
        if checkpoint.parent != root:
            raise ValueError("Checkpoint lineage escapes the checkpoint root")
        manifest = _checkpoint_directory_manifest(checkpoint)
        if manifest["checkpointSHA256"] != entry.get("checkpointSHA256"):
            raise ValueError("Checkpoint contents do not match checkpoint lineage")
        valid.append(
            {"path": relative, "checkpointSHA256": manifest["checkpointSHA256"]}
        )
    if valid != checkpoints:
        raise ValueError("Checkpoint lineage entries are not canonical")
    if steps != sorted(set(steps)):
        raise ValueError("Checkpoint lineage entries must be unique and step-sorted")
    return record


def _write_fresh_run_contract(
    *,
    run_root: Path,
    run_lineage: dict[str, Any],
    prepared: list[dict[str, Any]],
    resolved_environment_scan_audit: dict[str, Any],
) -> Path:
    for agent_lineage in run_lineage["agents"]:
        config_path = Path(agent_lineage["configPath"])
        checkpoint_record = _initial_checkpoint_lineage(
            run_lineage=run_lineage,
            agent_lineage=agent_lineage,
            config_path=config_path,
        )
        _atomic_write_json(
            Path(agent_lineage["checkpointLineagePath"]),
            checkpoint_record,
        )
    payload = {
        "schema": TRAINING_RUN_SCHEMA,
        "runResumeLineage": run_lineage,
        "runResumeLineageSHA256": run_lineage["runResumeLineageSHA256"],
        "resolvedTrainingEnvironmentScanAudit": resolved_environment_scan_audit,
        "preparedAgents": prepared,
    }
    manifest = _self_hashed(payload, field="runManifestSHA256")
    path = run_root / RUN_MANIFEST_NAME
    _atomic_write_json(path, manifest)
    return path


def _load_resume_contract(
    *,
    run_root: Path,
    expected_lineage: dict[str, Any],
    existing_manifest: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    path = run_root / RUN_MANIFEST_NAME
    manifest = existing_manifest or _read_self_hashed_json(
        path,
        schema=TRAINING_RUN_SCHEMA,
        hash_field="runManifestSHA256",
    )
    if (
        manifest.get("runResumeLineage") != expected_lineage
        or manifest.get("runResumeLineageSHA256")
        != expected_lineage["runResumeLineageSHA256"]
    ):
        raise ValueError("Resume lineage does not match the original run")
    prepared = manifest.get("preparedAgents")
    if not isinstance(prepared, list) or len(prepared) != len(expected_lineage["agents"]):
        raise ValueError("Run manifest prepared-agent contract is invalid")
    prepared_by_agent = {
        item.get("agent"): item for item in prepared if isinstance(item, dict)
    }
    if set(prepared_by_agent) != set(expected_lineage["selectedAgents"]):
        raise ValueError("Run manifest selected-agent contract is invalid")
    for agent_lineage in expected_lineage["agents"]:
        agent = agent_lineage["agent"]
        item = prepared_by_agent[agent]
        if (
            item.get("config") != agent_lineage["configPath"]
            or item.get("checkpointLineagePath")
            != agent_lineage["checkpointLineagePath"]
            or item.get("baseModelTokenizerSnapshotPath")
            != agent_lineage["baseModelTokenizerSnapshotPath"]
            or item.get("baseModelRuntimeSnapshotPath")
            != agent_lineage["baseModelRuntimeSnapshotPath"]
            or item.get("datasetRepository")
            != expected_lineage["datasetRepository"]
            or item.get("datasetRevision") != expected_lineage["datasetRevision"]
            or item.get("datasetPath") != expected_lineage["datasetPath"]
            or item.get("runResumeLineageSHA256")
            != expected_lineage["runResumeLineageSHA256"]
            or item.get("trainingCodeSHA256")
            != expected_lineage["trainingCodeSHA256"]
            or item.get("trainingDependencyLockSHA256")
            != expected_lineage["trainingDependencyLockSHA256"]
            or item.get("requirementsSHA256")
            != expected_lineage["requirementsSHA256"]
            or item.get("resolvedTrainingEnvironmentSHA256")
            != expected_lineage["resolvedTrainingEnvironmentSHA256"]
            or any(
                item.get(field) != expected_lineage.get(field)
                for field in ZERO_GPU_LINEAGE_FIELDS
            )
            or item.get("spaceConfigurationSHA256")
            != expected_lineage["spaceConfigurationSHA256"]
            or item.get("runtimeSourceKind")
            != expected_lineage["runtimeSourceKind"]
            or item.get("runtimeSourceRevision")
            != expected_lineage["runtimeSourceRevision"]
            or any(
                item.get(field) != expected_lineage.get(field)
                for field in RUNTIME_SOURCE_LINEAGE_FIELDS
            )
        ):
            raise ValueError("Run manifest path lineage drifted")
        config_path = Path(agent_lineage["configPath"])
        if not config_path.is_file():
            raise FileNotFoundError("Prepared resume config is missing")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            not isinstance(config, dict)
            or config.get("runResumeLineage") != expected_lineage
            or config.get("runResumeLineageSHA256")
            != expected_lineage["runResumeLineageSHA256"]
            or config.get("checkpointLineagePath")
            != agent_lineage["checkpointLineagePath"]
            or config.get("baseModelTokenizerSnapshotPath")
            != agent_lineage["baseModelTokenizerSnapshotPath"]
            or config.get("baseModelTokenizerSnapshotVerification")
            != item.get("baseModelTokenizerSnapshotVerification")
            or config.get("baseModelRuntimeSnapshotPath")
            != agent_lineage["baseModelRuntimeSnapshotPath"]
            or config.get("baseModelRuntimeSnapshotVerification")
            != item.get("baseModelRuntimeSnapshotVerification")
        ):
            raise ValueError("Prepared resume config lineage drifted")
        _validate_checkpoint_lineage(
            run_lineage=expected_lineage,
            agent_lineage=agent_lineage,
        )
    return path, [prepared_by_agent[agent] for agent in expected_lineage["selectedAgents"]]


def _prepare_configs(
    *,
    source_root: Path,
    run_root: Path,
    agents: list[str],
    base_model_override: str,
    seed: int,
    variant: str,
    run_lineage: dict[str, Any] | None = None,
    runtime_lineage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    runtime_manifest = json.loads((source_root / "adapter_runtime_manifest.json").read_text(encoding="utf-8"))
    base_by_agent = {
        item["agent"]: item.get("baseModelID") or runtime_manifest.get("sharedBaseModelID") or "Qwen/Qwen3-1.7B"
        for item in runtime_manifest.get("adapters", [])
        if isinstance(item, dict) and item.get("agent")
    }
    prepared: list[dict[str, Any]] = []
    config_root = run_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        agent_dir = source_root / agent
        variant_dir, variant_manifest = _variant_dataset(agent_dir, agent=agent, variant=variant)
        cfg_path = agent_dir / "unsloth_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"Generated training config is not an object: {cfg_path}")
        controlled = variant_manifest.get("controlledTrainingConfig")
        datasets = variant_manifest.get("datasets")
        if not isinstance(controlled, dict) or not isinstance(datasets, dict):
            raise ValueError(
                f"Generated training config is not bound to the variant manifest: {cfg_path}"
            )
        cfg = _variant_effective_training_config(
            agent=agent,
            base_config=cfg,
            controlled_config=controlled,
            train_sft_record_count=datasets["trainSFT"]["count"],
            train_dpo_record_count=(
                datasets["trainDPO"]["count"]
                if "trainDPO" in datasets
                else datasets["dpo"]["count"]
            ),
            declared_invariant_sha256=variant_manifest.get(
                "trainingConfigInvariantSHA256"
            ),
        )
        base = base_model_override.strip() or base_by_agent.get(agent) or cfg.get("base_model_name") or "Qwen/Qwen3-1.7B"
        if variant_manifest.get("baseModelID") != base:
            raise ValueError(f"Base-model override would break the controlled variant for {agent}: {base}")
        if (
            type(seed) is not int
            or type(variant_manifest.get("seed")) is not int
            or variant_manifest.get("seed") != seed
            or type(cfg.get("seed")) is not int
            or cfg.get("seed") != seed
        ):
            raise ValueError(f"Seed override would break the controlled variant for {agent}: {seed}")
        training_dir = run_root / "training" / agent
        adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
        dpo_adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
        adapter_gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"

        cfg["base_model_name"] = base
        cfg["baseModelID"] = base
        for field in (
            "baseModelRevision",
            "baseModelIndexDigest",
            "baseModelIndexReferencedShardNames",
            "baseModelIndexShardBindingSHA256",
            "baseModelArtifactDigest",
            "baseModelWeightShards",
            "baseModelTokenizerDigest",
            "baseModelTokenizerFiles",
            "baseModelTokenizerClosureSHA256",
            "trainingEnvironmentLock",
        ):
            if cfg.get(field) != variant_manifest.get(field):
                raise ValueError(f"{field} drifted from the controlled variant for {agent}")
        environment = _training_environment(
            variant_manifest,
            runtime_lineage=runtime_lineage,
        )
        cfg["trainingContainerImageDigest"] = environment["containerImageDigest"]
        cfg["trainingContainerImageDigestSource"] = environment["containerImageDigestSource"]
        cfg["trainingRuntimeImageBindingStatus"] = environment["runtimeImageBindingStatus"]
        cfg["trainingRuntimeImageBindingVerified"] = environment["runtimeImageBindingVerified"]
        cfg["trainingEnvironmentSHA256"] = environment["trainingEnvironmentSHA256"]
        cfg["dataset_dir"] = str(variant_dir)
        cfg["variant"] = variant
        cfg["variantManifestSHA256"] = variant_manifest["variantManifestSHA256"]
        cfg["output_dir"] = str(training_dir)
        cfg["adapter_output_dir"] = str(adapter_dir)
        cfg["dpo_output_dir"] = str(dpo_adapter_dir)
        cfg["adapter_gguf_output_path"] = str(adapter_gguf)
        cfg["seed"] = seed
        cfg["merge_adapters_by_default"] = False
        cfg["release_bake_enabled_by_default"] = False
        tokenizer_snapshot_path = (
            run_root / PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME
        ).resolve()
        tokenizer_snapshot_verification = (
            verify_private_base_model_tokenizer_snapshot(
                tokenizer_snapshot_path,
                base_model_id=cfg["baseModelID"],
                base_model_name=cfg["base_model_name"],
                base_model_revision=cfg["baseModelRevision"],
                tokenizer_files=cfg["baseModelTokenizerFiles"],
                tokenizer_digest=cfg["baseModelTokenizerDigest"],
                tokenizer_closure_sha256=cfg[
                    "baseModelTokenizerClosureSHA256"
                ],
            )
        )
        cfg["baseModelTokenizerSnapshotPath"] = str(
            tokenizer_snapshot_path
        )
        cfg["baseModelTokenizerSnapshotVerification"] = (
            tokenizer_snapshot_verification
        )
        runtime_snapshot_path = (
            run_root / PRIVATE_BASE_MODEL_RUNTIME_SNAPSHOT_DIRNAME
        ).resolve()
        runtime_snapshot_verification = (
            verify_private_base_model_conversion_snapshot(
                runtime_snapshot_path,
                base_model_id=cfg["baseModelID"],
                base_model_name=cfg["base_model_name"],
                base_model_revision=cfg["baseModelRevision"],
                tokenizer_files=cfg["baseModelTokenizerFiles"],
                tokenizer_digest=cfg["baseModelTokenizerDigest"],
                tokenizer_closure_sha256=cfg[
                    "baseModelTokenizerClosureSHA256"
                ],
                generation_config_file=(
                    DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE
                ),
                model_index_digest=cfg["baseModelIndexDigest"],
                index_referenced_shard_names=cfg[
                    "baseModelIndexReferencedShardNames"
                ],
                index_shard_binding_sha256=cfg[
                    "baseModelIndexShardBindingSHA256"
                ],
                model_artifact_digest=cfg["baseModelArtifactDigest"],
                weight_shards=cfg["baseModelWeightShards"],
            )
        )
        cfg["baseModelGenerationConfigFile"] = (
            DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE
        )
        cfg["baseModelRuntimeSnapshotPath"] = str(runtime_snapshot_path)
        cfg["baseModelRuntimeSnapshotVerification"] = (
            runtime_snapshot_verification
        )
        if run_lineage is not None:
            agent_lineage = next(
                item
                for item in run_lineage["agents"]
                if item["agent"] == agent
            )
            cfg["runResumeLineage"] = run_lineage
            cfg["runResumeLineageSHA256"] = run_lineage[
                "runResumeLineageSHA256"
            ]
            cfg["checkpointLineagePath"] = agent_lineage[
                "checkpointLineagePath"
            ]
            cfg["datasetRepository"] = run_lineage["datasetRepository"]
            cfg["datasetRevision"] = run_lineage["datasetRevision"]
            cfg["datasetPath"] = run_lineage["datasetPath"]
            cfg["localDatasetSnapshot"] = run_lineage[
                "localDatasetSnapshot"
            ]
            for field in RUNTIME_SOURCE_LINEAGE_FIELDS:
                cfg[field] = run_lineage[field]
        if runtime_lineage is not None:
            for field in (
                "trainingCodeManifest",
                "trainingCodeSHA256",
                "trainingDependencyLock",
                "trainingDependencyLockSHA256",
                "requirementsSHA256",
                "resolvedTrainingEnvironment",
                "resolvedTrainingEnvironmentCacheAttestation",
                "resolvedTrainingEnvironmentScanAudit",
                "resolvedTrainingEnvironmentSHA256",
                *ZERO_GPU_LINEAGE_FIELDS,
                "spaceConfigurationSHA256",
            ):
                cfg[field] = runtime_lineage[field]
        for env_name, key in (
            ("LUMEN_ZERO_GPU_MAX_TRAIN_RECORDS", "max_train_records"),
            ("LUMEN_ZERO_GPU_MAX_VAL_RECORDS", "max_val_records"),
            ("LUMEN_ZERO_GPU_MAX_SEQ_LENGTH", "max_seq_length"),
            ("LUMEN_ZERO_GPU_NUM_TRAIN_EPOCHS", "num_train_epochs"),
        ):
            override = _optional_int_env(env_name)
            if override is not None and override != cfg.get(key):
                raise ValueError(f"{env_name} would break the controlled variant for {agent}: {override}")
        cfg.setdefault("adapterExport", {})
        cfg["adapterExport"]["trainBaseModelWeights"] = False
        cfg["adapterExport"]["mergeAdaptersByDefault"] = False
        cfg["adapterExport"]["adapterArtifact"] = str(adapter_dir)
        cfg["adapterExport"]["adapterDirectory"] = str(adapter_dir)
        cfg["adapterExport"]["adapterGGUFArtifact"] = str(adapter_gguf)
        attestation = _training_attestation(cfg, variant_manifest)
        cfg["variantAttestation"] = attestation

        out = config_root / f"{agent}.json"
        _atomic_write_json(out, cfg)
        prepared.append(
            {
                "agent": agent,
                "config": str(out),
                "dataset_dir": str(variant_dir),
                "variant": variant,
                "variantManifestSHA256": variant_manifest["variantManifestSHA256"],
                "variantAttestation": attestation,
                "base_model_name": base,
                "baseModelRevision": cfg["baseModelRevision"],
                "baseModelIndexDigest": cfg["baseModelIndexDigest"],
                "baseModelIndexReferencedShardNames": cfg["baseModelIndexReferencedShardNames"],
                "baseModelIndexShardBindingSHA256": cfg["baseModelIndexShardBindingSHA256"],
                "baseModelArtifactDigest": cfg["baseModelArtifactDigest"],
                "baseModelWeightShards": cfg["baseModelWeightShards"],
                "baseModelTokenizerDigest": cfg["baseModelTokenizerDigest"],
                "baseModelTokenizerFiles": cfg["baseModelTokenizerFiles"],
                "baseModelTokenizerClosureSHA256": cfg[
                    "baseModelTokenizerClosureSHA256"
                ],
                "baseModelTokenizerSnapshotPath": cfg[
                    "baseModelTokenizerSnapshotPath"
                ],
                "baseModelTokenizerSnapshotVerification": cfg[
                    "baseModelTokenizerSnapshotVerification"
                ],
                "baseModelGenerationConfigFile": cfg[
                    "baseModelGenerationConfigFile"
                ],
                "baseModelRuntimeSnapshotPath": cfg[
                    "baseModelRuntimeSnapshotPath"
                ],
                "baseModelRuntimeSnapshotVerification": cfg[
                    "baseModelRuntimeSnapshotVerification"
                ],
                "trainingEnvironmentSHA256": cfg["trainingEnvironmentSHA256"],
                "zeroGPUSize": cfg.get("zeroGPUSize"),
                "zeroGPUDurationSeconds": cfg.get("zeroGPUDurationSeconds"),
                "observedAccelerator": cfg.get("observedAccelerator"),
                "runtimeImageBindingStatus": cfg["trainingRuntimeImageBindingStatus"],
                "runtimeImageBindingVerified": cfg["trainingRuntimeImageBindingVerified"],
                "adapter_dir": str(adapter_dir),
                "training_dir": str(training_dir),
                "finalized_variant_manifest": str(
                    training_dir / "finalized_variant_manifest.json"
                ),
                "adapter_gguf": str(adapter_gguf),
                "checkpointLineagePath": (
                    agent_lineage["checkpointLineagePath"]
                    if run_lineage is not None
                    else str(run_root / "checkpoint_lineage" / f"{agent}.json")
                ),
                **(
                    {
                        "datasetRepository": run_lineage["datasetRepository"],
                        "datasetRevision": run_lineage["datasetRevision"],
                        "datasetPath": run_lineage["datasetPath"],
                        "runResumeLineageSHA256": run_lineage[
                            "runResumeLineageSHA256"
                        ],
                        "trainingCodeSHA256": run_lineage[
                            "trainingCodeSHA256"
                        ],
                        "trainingDependencyLockSHA256": run_lineage[
                            "trainingDependencyLockSHA256"
                        ],
                        "requirementsSHA256": run_lineage[
                            "requirementsSHA256"
                        ],
                        "resolvedTrainingEnvironmentSHA256": run_lineage[
                            "resolvedTrainingEnvironmentSHA256"
                        ],
                        "zeroGPUSize": run_lineage["zeroGPUSize"],
                        "zeroGPUDurationSeconds": run_lineage[
                            "zeroGPUDurationSeconds"
                        ],
                        "observedAccelerator": run_lineage[
                            "observedAccelerator"
                        ],
                        "spaceConfigurationSHA256": run_lineage[
                            "spaceConfigurationSHA256"
                        ],
                        "runtimeSourceKind": run_lineage["runtimeSourceKind"],
                        "runtimeSourceRevision": run_lineage[
                            "runtimeSourceRevision"
                        ],
                        "expectedRuntimeSourceRevision": run_lineage[
                            "expectedRuntimeSourceRevision"
                        ],
                        "observedRepositoryRevision": run_lineage[
                            "observedRepositoryRevision"
                        ],
                        "observedRuntimeRevision": run_lineage[
                            "observedRuntimeRevision"
                        ],
                        "runtimeSourceBindingStatus": run_lineage[
                            "runtimeSourceBindingStatus"
                        ],
                        "runtimeSourceBindingMethod": run_lineage[
                            "runtimeSourceBindingMethod"
                        ],
                    }
                    if run_lineage is not None
                    else {}
                ),
            }
        )
    return prepared


def _validate_nonempty_assistant_outputs(source_root: Path, agents: list[str], variant: str) -> None:
    bad: list[str] = []
    for agent in agents:
        variant_root, _ = _variant_dataset(source_root / agent, agent=agent, variant=variant)
        for split in ("train_sft.jsonl", "val_sft.jsonl"):
            path = variant_root / split
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = record.get("messages") or []
                assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
                if not str(assistant).strip() or str(assistant).strip().lower() in {"null", "none"}:
                    bad.append(f"{path}:{lineno}")
    if bad:
        raise RuntimeError("Refusing to train on empty/null assistant outputs:\n" + "\n".join(bad[:20]))


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        child_environment = dict(os.environ)
        if environment:
            child_environment.update(environment)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        rc = process.wait()
    if rc != 0:
        tail = ""
        if log_path.exists():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"Command failed with exit {rc}: {' '.join(command)}. See {log_path}\n{tail}")


@contextmanager
def _private_conversion_base_snapshot(
    run_root: Path,
    item: dict[str, Any],
    base_snapshot: Path,
) -> Any:
    del run_root, base_snapshot
    runtime_snapshot = Path(item["baseModelRuntimeSnapshotPath"])
    verification = verify_private_base_model_conversion_snapshot(
        runtime_snapshot,
        base_model_id=item["base_model_name"],
        base_model_name=item["base_model_name"],
        base_model_revision=item["baseModelRevision"],
        tokenizer_files=item["baseModelTokenizerFiles"],
        tokenizer_digest=item["baseModelTokenizerDigest"],
        tokenizer_closure_sha256=item[
            "baseModelTokenizerClosureSHA256"
        ],
        generation_config_file=item["baseModelGenerationConfigFile"],
        model_index_digest=item["baseModelIndexDigest"],
        index_referenced_shard_names=item[
            "baseModelIndexReferencedShardNames"
        ],
        index_shard_binding_sha256=item[
            "baseModelIndexShardBindingSHA256"
        ],
        model_artifact_digest=item["baseModelArtifactDigest"],
        weight_shards=item["baseModelWeightShards"],
    )
    if verification != item.get("baseModelRuntimeSnapshotVerification"):
        raise RuntimeError(
            "Prepared private base-model verification drifted before conversion"
        )
    yield runtime_snapshot, verification


def _convert_lora_to_gguf(run_root: Path, prepared: list[dict[str, Any]], token: str) -> None:
    del token
    # The revision itself is carried in each generated config's immutable environment lock.
    first_config = json.loads(Path(prepared[0]["config"]).read_text(encoding="utf-8"))
    llama_cpp_revision = str(first_config["trainingEnvironmentLock"]["llamaCppRevision"])
    converter = Path(os.environ.get("LUMEN_LORA_CONVERTER", str(Path.home() / ".unsloth/llama.cpp/convert_lora_to_gguf.py")))
    if not converter.exists():
        clone_dir = run_root / "llama.cpp"
        _run(["git", "init", str(clone_dir)], cwd=run_root, log_path=run_root / "logs" / "clone_llama_cpp.log")
        _run(["git", "-C", str(clone_dir), "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp"], cwd=run_root, log_path=run_root / "logs" / "clone_llama_cpp_remote.log")
        _run(["git", "-C", str(clone_dir), "fetch", "--depth", "1", "origin", llama_cpp_revision], cwd=run_root, log_path=run_root / "logs" / "clone_llama_cpp_fetch.log")
        _run(["git", "-C", str(clone_dir), "checkout", "--detach", "FETCH_HEAD"], cwd=run_root, log_path=run_root / "logs" / "clone_llama_cpp_checkout.log")
        converter = clone_dir / "convert_lora_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Missing convert_lora_to_gguf.py: {converter}")
    converter_revision = subprocess.check_output(
        ["git", "-C", str(converter.parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if converter_revision != llama_cpp_revision:
        raise RuntimeError(
            f"llama.cpp converter revision drifted: expected {llama_cpp_revision}, got {converter_revision}"
        )

    for item in prepared:
        agent = item["agent"]
        outfile = Path(item["adapter_gguf"])
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with _private_conversion_base_snapshot(
            run_root,
            item,
            Path(item["baseModelRuntimeSnapshotPath"]),
        ) as (conversion_snapshot, conversion_verification):
            _run(
                [
                    sys.executable,
                    str(converter),
                    item["adapter_dir"],
                    "--outfile",
                    str(outfile),
                    "--base",
                    str(conversion_snapshot),
                ],
                cwd=run_root,
                log_path=run_root / "logs" / f"convert_{agent}.log",
            )
            after = verify_private_base_model_conversion_snapshot(
                conversion_snapshot,
                base_model_id=item["base_model_name"],
                base_model_name=item["base_model_name"],
                base_model_revision=item["baseModelRevision"],
                tokenizer_files=item["baseModelTokenizerFiles"],
                tokenizer_digest=item["baseModelTokenizerDigest"],
                tokenizer_closure_sha256=item[
                    "baseModelTokenizerClosureSHA256"
                ],
                generation_config_file=item[
                    "baseModelGenerationConfigFile"
                ],
                model_index_digest=item["baseModelIndexDigest"],
                index_referenced_shard_names=item[
                    "baseModelIndexReferencedShardNames"
                ],
                index_shard_binding_sha256=item[
                    "baseModelIndexShardBindingSHA256"
                ],
                model_artifact_digest=item["baseModelArtifactDigest"],
                weight_shards=item["baseModelWeightShards"],
            )
            if after != conversion_verification:
                raise RuntimeError(
                    "Private base-model snapshot changed during GGUF conversion"
                )
            item["baseModelConversionSnapshotVerification"] = after


def _upload_outputs(run_root: Path, prepared: list[dict[str, Any]], adapter_repo: str, run_id: str, token: str, include_gguf: bool) -> dict[str, Any]:
    api = HfApi(token=token)
    private = os.environ.get("LUMEN_ZERO_GPU_PRIVATE_ADAPTERS", "1") == "1"
    info = api.model_info(
        repo_id=adapter_repo,
        files_metadata=False,
        token=token,
    )
    actual_private = getattr(info, "private", None)
    if not isinstance(actual_private, bool) or actual_private is not private:
        expected = "private" if private else "public"
        actual = (
            "private"
            if actual_private is True
            else "public"
            if actual_private is False
            else "unknown"
        )
        raise RuntimeError(
            "Adapter repository visibility postcondition failed before upload: "
            f"expected {expected}, observed {actual}"
        )

    uploaded: dict[str, Any] = {}
    for item in prepared:
        agent = item["agent"]
        adapter_dir, finalized = _verify_trained_adapter(item)
        adapter_path = f"runs/{run_id}/adapters/{agent}"
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=adapter_repo,
            repo_type="model",
            path_in_repo=adapter_path,
            commit_message=f"Upload Lumen {agent} adapter {run_id}",
            token=token,
        )
        finalized_manifest = Path(item["finalized_variant_manifest"])
        if not finalized_manifest.is_file():
            raise FileNotFoundError(
                f"Missing finalized experiment variant manifest: {finalized_manifest}"
            )
        artifact = finalized.get("artifact") if isinstance(finalized, dict) else None
        if not isinstance(artifact, dict) or artifact.get("adapterSHA256") is None:
            raise ValueError(f"Finalized manifest lacks adapter lineage: {finalized_manifest}")
        manifest_path = f"runs/{run_id}/manifests/{agent}/variant_manifest.json"
        api.upload_file(
            path_or_fileobj=str(finalized_manifest),
            repo_id=adapter_repo,
            repo_type="model",
            path_in_repo=manifest_path,
            commit_message=f"Upload Lumen {agent} finalized variant manifest {run_id}",
            token=token,
        )
        entry: dict[str, Any] = {
            "adapter_dir": str(adapter_dir),
            "adapter_repo": adapter_repo,
            "adapter_path_in_repo": adapter_path,
            "variant": item["variant"],
            "sourceVariantManifestSHA256": item["variantManifestSHA256"],
            "variantManifestSHA256": finalized["variantManifestSHA256"],
            "adapterSHA256": artifact["adapterSHA256"],
            "finalized_variant_manifest": str(finalized_manifest),
            "finalized_variant_manifest_path_in_repo": manifest_path,
        }
        gguf = Path(item["adapter_gguf"])
        if include_gguf and gguf.exists():
            gguf_path = f"runs/{run_id}/lora_gguf/{gguf.name}"
            api.upload_file(
                path_or_fileobj=str(gguf),
                repo_id=adapter_repo,
                repo_type="model",
                path_in_repo=gguf_path,
                commit_message=f"Upload Lumen {agent} LoRA GGUF {run_id}",
                token=token,
            )
            entry["adapter_gguf_path_in_repo"] = gguf_path
            entry["adapter_gguf_sha256"] = _sha256(gguf)
            entry["adapter_gguf_size_bytes"] = gguf.stat().st_size
            entry["baseModelTokenizerSnapshotVerification"] = item[
                "baseModelTokenizerSnapshotVerification"
            ]
            entry["baseModelConversionSnapshotVerification"] = item[
                "baseModelConversionSnapshotVerification"
            ]
        uploaded[agent] = entry
    return uploaded


def _verify_trained_adapter(item: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    adapter_dir = Path(item["adapter_dir"])
    finalized_manifest = Path(item["finalized_variant_manifest"])
    if not finalized_manifest.is_file():
        raise FileNotFoundError(
            f"Missing finalized experiment variant manifest: {finalized_manifest}"
        )
    finalized = json.loads(finalized_manifest.read_text(encoding="utf-8"))
    _verify_finalized_variant_lineage(item, finalized, finalized_manifest)
    artifact = finalized.get("artifact") if isinstance(finalized, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError(f"Finalized manifest lacks adapter lineage: {finalized_manifest}")
    verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=artifact["adapterSHA256"],
        expected_training_phase="sft",
        expected_base_model=str(item.get("base_model_name") or ""),
        expected_base_revision=str(item.get("baseModelRevision") or ""),
    )
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(adapter_config, dict)
        or adapter_config.get("base_model_name_or_path")
        != item.get("base_model_name")
        or adapter_config.get("revision") != item.get("baseModelRevision")
        or IMMUTABLE_HUB_REVISION.fullmatch(
            str(adapter_config.get("revision") or "")
        )
        is None
    ):
        raise ValueError("Trained adapter is not bound to the prepared base model")
    return adapter_dir, finalized


def _verify_finalized_variant_lineage(
    item: dict[str, Any],
    finalized: Any,
    finalized_manifest: Path,
) -> None:
    if not isinstance(finalized, dict):
        raise ValueError(f"Finalized manifest must be a JSON object: {finalized_manifest}")
    expected_sha = finalized.get("variantManifestSHA256")
    unsigned = dict(finalized)
    unsigned.pop("variantManifestSHA256", None)
    if (
        not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or _canonical_sha256(unsigned) != expected_sha
    ):
        raise ValueError(f"Finalized manifest integrity check failed: {finalized_manifest}")
    if (
        finalized.get("agent") != item.get("agent")
        or finalized.get("variant") != item.get("variant")
        or finalized.get("sourceVariantManifestSHA256")
        != item.get("variantManifestSHA256")
    ):
        raise ValueError(f"Finalized manifest identity or source lineage mismatch: {finalized_manifest}")

    artifact = finalized.get("artifact")
    if (
        not isinstance(artifact, dict)
        or artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft"
        or artifact.get("parentSFTAdapterSHA256") is not None
        or re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("adapterSHA256") or ""))
        is None
        or artifact.get("adapterManifestSHA256") != artifact.get("adapterSHA256")
    ):
        raise ValueError(f"Finalized manifest lacks valid SFT adapter lineage: {finalized_manifest}")

    for field in (
        "baseModelRevision",
        "baseModelIndexDigest",
        "baseModelIndexReferencedShardNames",
        "baseModelIndexShardBindingSHA256",
        "baseModelArtifactDigest",
        "baseModelWeightShards",
        "baseModelTokenizerDigest",
        "baseModelTokenizerFiles",
        "baseModelTokenizerClosureSHA256",
        "trainingEnvironmentSHA256",
        "trainingCodeSHA256",
        "trainingDependencyLockSHA256",
        "requirementsSHA256",
        "resolvedTrainingEnvironmentSHA256",
        *ZERO_GPU_LINEAGE_FIELDS,
        "spaceConfigurationSHA256",
        *RUNTIME_SOURCE_LINEAGE_FIELDS,
    ):
        if finalized.get(field) != item.get(field):
            raise ValueError(f"Finalized manifest {field} mismatch: {finalized_manifest}")
    training_environment = finalized.get("trainingEnvironment")
    if (
        finalized.get("baseModelID") != item.get("base_model_name")
        or not isinstance(training_environment, dict)
        or _canonical_sha256(training_environment)
        != item.get("trainingEnvironmentSHA256")
        or training_environment.get("runtimeImageBindingStatus")
        != item.get("runtimeImageBindingStatus")
        or training_environment.get("runtimeImageBindingVerified")
        is not item.get("runtimeImageBindingVerified")
        or any(
            training_environment.get(field) != item.get(field)
            for field in ZERO_GPU_LINEAGE_FIELDS
        )
    ):
        raise ValueError(f"Finalized manifest base or runtime lineage mismatch: {finalized_manifest}")
    attestation = item.get("variantAttestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("variant") != item.get("variant")
        or attestation.get("variantManifestSHA256") != item.get("variantManifestSHA256")
        or attestation.get("trainingEnvironmentSHA256")
        != item.get("trainingEnvironmentSHA256")
        or finalized.get("trainingCorpusSHA256") != attestation.get("trainingCorpusSHA256")
        or finalized.get("trainingConfigSHA256")
        != attestation.get("effectiveTrainingConfigSHA256")
        or finalized.get("trainingConfigInvariantSHA256")
        != attestation.get("trainingConfigInvariantSHA256")
        or {
            name: contract.get("sha256")
            for name, contract in sorted((finalized.get("datasets") or {}).items())
            if isinstance(contract, dict)
            and isinstance(contract.get("sha256"), str)
        }
        != attestation.get("laneHashes")
    ):
        raise ValueError(f"Finalized manifest does not match the prepared attestation: {finalized_manifest}")


@spaces.GPU(size=DEFAULT_GPU_SIZE, duration=DEFAULT_GPU_DURATION)
def _train_lumen_adapters_gpu(
    run_id: str,
    agents_csv: str,
    base_model_override: str,
    seed: int,
    assistant_only_loss: bool,
    resume: bool,
    convert_gguf: bool,
    upload_outputs: bool,
    gpu_size: str,
    experiment_variant: str = "",
    confirm_experiment_variant: bool = False,
    destructive_reset: bool = False,
) -> dict[str, Any]:
    _deployed_zero_gpu_contract(requested_size=gpu_size)
    agents = _csv_agents(agents_csv)
    experiment_variant = _experiment_variant(experiment_variant)
    if not confirm_experiment_variant:
        raise RuntimeError("Confirm the explicitly selected experiment variant before training")
    run_id = run_id.strip() or os.environ.get(
        "LUMEN_ZERO_GPU_RUN_ID", str(DEFAULTS["run_id"])
    )
    run_id, run_root = _resolve_run_workspace(run_id, experiment_variant)
    existing_run_manifest: dict[str, Any] | None = None
    if resume:
        if destructive_reset:
            raise ValueError("Resume and destructive reset are mutually exclusive")
        if not run_root.is_dir():
            raise FileNotFoundError("Resume requires an existing run workspace")
        existing_run_manifest = _read_self_hashed_json(
            run_root / RUN_MANIFEST_NAME,
            schema=TRAINING_RUN_SCHEMA,
            hash_field="runManifestSHA256",
        )
    dataset_repo = os.environ.get(
        "LUMEN_ZERO_GPU_DATASET_REPO", str(DEFAULTS["dataset_repo"])
    )
    if dataset_repo != str(DEFAULTS["dataset_repo"]):
        raise ValueError("Dataset repository drifted from the deployed defaults")
    default_dataset_revision = _immutable_hub_revision(
        DEFAULTS.get("dataset_revision"),
        label="Deployed dataset revision",
    )
    dataset_revision = _immutable_hub_revision(
        os.environ.get(
            "LUMEN_ZERO_GPU_DATASET_REVISION",
            str(DEFAULTS.get("dataset_revision", "")),
        ),
        label="Dataset revision",
    )
    if dataset_revision != default_dataset_revision:
        raise ValueError("Dataset revision drifted from the deployed immutable snapshot")
    dataset_path = os.environ.get(
        "LUMEN_ZERO_GPU_DATASET_PATH", str(DEFAULTS["dataset_path_in_repo"])
    )
    if dataset_path != str(DEFAULTS["dataset_path_in_repo"]):
        raise ValueError("Dataset path drifted from the deployed defaults")
    adapter_repo = os.environ.get(
        "LUMEN_ZERO_GPU_ADAPTER_REPO", str(DEFAULTS["adapter_repo"])
    )
    runtime_lineage = {
        **_verify_runtime_lineage(),
        "observedAccelerator": _observed_accelerator(),
    }

    if resume:
        assert existing_run_manifest is not None
        source_root = run_root / "generated" / "fine_tuning"
        if not source_root.is_dir():
            raise FileNotFoundError("Resume requires the original local dataset snapshot")
        expected_lineage = _build_run_resume_lineage(
            run_id=run_id,
            run_root=run_root,
            source_root=source_root,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            dataset_path=dataset_path,
            agents=agents,
            variant=experiment_variant,
            seed=int(seed),
            assistant_only_loss=bool(assistant_only_loss),
            runtime_lineage=runtime_lineage,
        )
        if base_model_override.strip() and any(
            item.get("baseModelID") != base_model_override.strip()
            for item in expected_lineage["agents"]
        ):
            raise ValueError("Base-model override drifted from the original run")
        _verify_private_tokenizer_for_run(run_root, expected_lineage)
        _verify_private_base_model_for_run(run_root, expected_lineage)
        run_manifest_path, prepared = _load_resume_contract(
            run_root=run_root,
            expected_lineage=expected_lineage,
            existing_manifest=existing_run_manifest,
        )
    else:
        if run_root.exists():
            if not destructive_reset:
                raise FileExistsError(
                    "Fresh run destination already exists; explicitly request destructive reset"
                )
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("LUMEN_ZERO_GPU_HUB_TOKEN")
        if not token:
            raise RuntimeError("A fine-grained LUMEN_ZERO_GPU_HUB_TOKEN Space secret is required")
        source_root = _copy_dataset_snapshot(run_root, dataset_repo, dataset_revision, dataset_path, token)
        _validate_nonempty_assistant_outputs(source_root, agents, experiment_variant)
        expected_lineage = _build_run_resume_lineage(
            run_id=run_id,
            run_root=run_root,
            source_root=source_root,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            dataset_path=dataset_path,
            agents=agents,
            variant=experiment_variant,
            seed=int(seed),
            assistant_only_loss=bool(assistant_only_loss),
            runtime_lineage=runtime_lineage,
        )
        _materialize_private_tokenizer_for_run(
            run_root,
            expected_lineage,
            token,
        )
        _materialize_private_base_model_for_run(
            run_root,
            expected_lineage,
            token,
        )
        prepared = _prepare_configs(
            source_root=source_root,
            run_root=run_root,
            agents=agents,
            base_model_override=base_model_override,
            seed=int(seed),
            variant=experiment_variant,
            run_lineage=expected_lineage,
            runtime_lineage=runtime_lineage,
        )
        run_manifest_path = _write_fresh_run_contract(
            run_root=run_root,
            run_lineage=expected_lineage,
            prepared=prepared,
            resolved_environment_scan_audit=runtime_lineage[
                "resolvedTrainingEnvironmentScanAudit"
            ],
        )

    # Access the repository credential only after authorization and the complete
    # fresh/resume lineage contract have passed.
    token = os.environ.get("LUMEN_ZERO_GPU_HUB_TOKEN")
    if not token:
        raise RuntimeError("A fine-grained LUMEN_ZERO_GPU_HUB_TOKEN Space secret is required")

    for item in prepared:
        agent = item["agent"]
        command = [
            sys.executable,
            "-m",
            "lumen_training.train_sft",
            "--config",
            item["config"],
            "--seed",
            str(seed),
        ]
        if assistant_only_loss:
            command.append("--assistant-only-loss")
        if resume:
            command.append("--resume-from-checkpoint")
        _run(
            command,
            cwd=APP_ROOT,
            log_path=run_root / "logs" / f"train_{agent}.log",
            environment=_startup_environment_child_variable(),
        )

    for item in prepared:
        _, finalized = _verify_trained_adapter(item)
        item["adapterSHA256"] = finalized["artifact"]["adapterSHA256"]
        item["finalizedVariantManifestSHA256"] = finalized[
            "variantManifestSHA256"
        ]

    if convert_gguf:
        _convert_lora_to_gguf(run_root, prepared, token)

    for item in prepared:
        gguf = Path(item["adapter_gguf"])
        if gguf.exists():
            item["adapter_gguf_sha256"] = _sha256(gguf)
            item["adapter_gguf_size_bytes"] = gguf.stat().st_size

    uploads = (
        _upload_outputs(run_root, prepared, adapter_repo, run_id, token, convert_gguf)
        if upload_outputs
        else {}
    )
    summary = {
        "schema": TRAINING_SUMMARY_SCHEMA,
        "ok": True,
        "run_id": run_id,
        "run_root": str(run_root),
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "dataset_path": dataset_path,
        "adapter_repo": adapter_repo,
        "variant": experiment_variant,
        "run_manifest": str(run_manifest_path),
        "runResumeLineageSHA256": expected_lineage[
            "runResumeLineageSHA256"
        ],
        "trainingCodeSHA256": expected_lineage["trainingCodeSHA256"],
        "trainingDependencyLockSHA256": expected_lineage[
            "trainingDependencyLockSHA256"
        ],
        "requirementsSHA256": expected_lineage["requirementsSHA256"],
        "resolvedTrainingEnvironmentSHA256": expected_lineage[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "resolvedTrainingEnvironmentScanAudit": runtime_lineage[
            "resolvedTrainingEnvironmentScanAudit"
        ],
        "zeroGPUSize": expected_lineage["zeroGPUSize"],
        "zeroGPUDurationSeconds": expected_lineage[
            "zeroGPUDurationSeconds"
        ],
        "observedAccelerator": expected_lineage["observedAccelerator"],
        "spaceConfigurationSHA256": expected_lineage[
            "spaceConfigurationSHA256"
        ],
        "runtimeSourceKind": expected_lineage["runtimeSourceKind"],
        "runtimeSourceRevision": expected_lineage["runtimeSourceRevision"],
        "expectedRuntimeSourceRevision": expected_lineage[
            "expectedRuntimeSourceRevision"
        ],
        "observedRepositoryRevision": expected_lineage[
            "observedRepositoryRevision"
        ],
        "observedRuntimeRevision": expected_lineage[
            "observedRuntimeRevision"
        ],
        "runtimeSourceBindingStatus": expected_lineage[
            "runtimeSourceBindingStatus"
        ],
        "runtimeSourceBindingMethod": expected_lineage[
            "runtimeSourceBindingMethod"
        ],
        "agents": prepared,
        "uploads": uploads,
        "fresh_run": not resume,
        "resume": bool(resume),
        "assistant_only_loss": bool(assistant_only_loss),
        "convert_gguf": bool(convert_gguf),
    }
    _atomic_write_json(run_root / "lumen_zerogpu_training_summary.json", summary)
    return summary


def train_lumen_adapters(
    run_id: str,
    agents_csv: str,
    base_model_override: str,
    seed: int,
    assistant_only_loss: bool,
    resume: bool,
    convert_gguf: bool,
    upload_outputs: bool,
    gpu_size: str,
    experiment_variant: str = "",
    confirm_experiment_variant: bool = False,
    destructive_reset: bool = False,
    request: gr.Request = None,
) -> dict[str, Any]:
    correlation_id = str(uuid.uuid4())
    try:
        _authorize_request(request)
        _deployed_zero_gpu_contract(requested_size=gpu_size)
        if not os.environ.get("LUMEN_ZERO_GPU_HUB_TOKEN"):
            raise RepositoryCredentialConfigurationError(
                "ZeroGPU repository authorization is not configured"
            )
        _verified_startup_environment_cache()
        with _exclusive_training_operation():
            return _train_lumen_adapters_gpu(
                run_id,
                agents_csv,
                base_model_override,
                seed,
                assistant_only_loss,
                resume,
                convert_gguf,
                upload_outputs,
                gpu_size,
                experiment_variant,
                confirm_experiment_variant,
                destructive_reset,
            )
    except RequestAuthorizationError:
        return _external_failure(
            code="unauthorized",
            correlation_id=correlation_id,
            message="Administrative authorization is required.",
        )
    except AuthorizationConfigurationError:
        return _external_failure(
            code="authorization_not_configured",
            correlation_id=correlation_id,
            message="Administrative authorization is not configured.",
        )
    except RepositoryCredentialConfigurationError:
        return _external_failure(
            code="repository_authorization_not_configured",
            correlation_id=correlation_id,
            message="Repository authorization is not configured.",
        )
    except TrainingConflictError:
        return _external_failure(
            code="training_already_active",
            correlation_id=correlation_id,
            message="Another training operation is already active.",
        )
    except Exception:
        LOGGER.error(
            "ZeroGPU training request failed correlation_id=%s\n%s",
            correlation_id,
            traceback.format_exc(),
        )
        return _external_failure(
            code="training_failed",
            correlation_id=correlation_id,
            message="Training failed. Consult server logs with the correlation ID.",
        )

_initialize_startup_environment_cache()


with gr.Blocks() as demo:
    gr.Markdown("# Lumen ZeroGPU Adapter Trainer")
    gr.Markdown(
        "Training is API-only. Use the authenticated Lumen ZeroGPU launcher; "
        "browser requests cannot provide the required administrative header."
    )
    gr.Markdown(_startup_environment_status())
    with gr.Row():
        run_id = gr.Textbox(
            value=str(DEFAULTS.get("run_id", "")),
            label="Run ID",
            visible=False,
        )
        agents = gr.Textbox(
            value=",".join(DEFAULTS.get("agents", AGENTS)),
            label="Agents",
            visible=False,
        )
    with gr.Row():
        base_model = gr.Textbox(
            value=str(DEFAULTS.get("base_model_override", "")),
            label="Base model override",
            visible=False,
        )
        seed = gr.Number(value=42, precision=0, label="Seed", visible=False)
        gpu_size = gr.Dropdown(
            choices=["large", "xlarge"],
            value=DEFAULT_GPU_SIZE,
            label="ZeroGPU size",
            visible=False,
        )
        experiment_variant = gr.Dropdown(
            choices=[("Select a variant", ""), *[(value, value) for value in EXPERIMENT_VARIANTS]],
            value="",
            label="Experiment variant",
            visible=False,
        )
    with gr.Row():
        assistant_loss = gr.Checkbox(
            value=True,
            label="Assistant-only loss",
            visible=False,
        )
        resume = gr.Checkbox(value=False, label="Resume", visible=False)
        convert = gr.Checkbox(
            value=True,
            label="Convert LoRA to GGUF",
            visible=False,
        )
        upload = gr.Checkbox(value=True, label="Upload outputs", visible=False)
        confirm_variant = gr.Checkbox(
            value=False,
            label="I confirm this experiment variant",
            visible=False,
        )
        destructive_reset = gr.Checkbox(
            value=False,
            label="Explicitly replace an existing fresh-run workspace",
            visible=False,
        )
    output = gr.JSON(label="Training summary", visible=False)
    run = gr.Button("Authenticated API endpoint", visible=False)
    run.click(
        fn=train_lumen_adapters,
        inputs=[run_id, agents, base_model, seed, assistant_loss, resume, convert, upload, gpu_size, experiment_variant, confirm_variant, destructive_reset],
        outputs=output,
        api_name="train_lumen_adapters",
        api_visibility="undocumented",
    )


if __name__ == "__main__":
    demo.queue().launch()
