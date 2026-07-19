from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import importlib.metadata as importlib_metadata
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit


TRAINING_CODE_MANIFEST_SCHEMA_VERSION = "lumen.training-code-manifest/2.0.0"
TRAINING_CODE_BUNDLE_SCHEMA_VERSION = "lumen.training-code-bundle/2.0.0"
TRAINING_DEPENDENCY_LOCK_SCHEMA_VERSION = (
    "lumen.adapter-training-dependency-lock/1.0.0"
)
RESOLVED_TRAINING_ENVIRONMENT_SCHEMA_VERSION = (
    "lumen.resolved-training-environment/1.0.0"
)
RESOLVED_TRAINING_ENVIRONMENT_CACHE_SCHEMA_VERSION = (
    "lumen.resolved-training-environment-cache/1.0.0"
)
ZERO_GPU_ALLOWED_SIZES = frozenset({"large", "xlarge"})
RESOLVED_TRAINING_ENVIRONMENT_RECORD_POLICY = {
    "hashAlgorithm": "sha256",
    "verifyDeclaredFileHashes": True,
    "excludeUnhashedSelfRecord": True,
    "hashUnattestedGeneratedBytecode": True,
    "hashRegeneratedBytecodePairs": True,
    "requireAttestedSourceForGeneratedBytecode": True,
    "rejectOtherUnhashedFiles": True,
}
SPACE_CONFIGURATION_SCHEMA_VERSION = "lumen.zerogpu.space-configuration/1.0.0"
BASE_MODEL_TOKENIZER_CLOSURE_SCHEMA_VERSION = (
    "lumen.base-model-tokenizer-closure/1.0.0"
)
BASE_MODEL_TOKENIZER_SNAPSHOT_VERIFICATION_SCHEMA_VERSION = (
    "lumen.base-model-tokenizer-snapshot-verification/1.0.0"
)
PRIVATE_BASE_MODEL_TOKENIZER_SNAPSHOT_VERIFICATION_SCHEMA_VERSION = (
    "lumen.private-base-model-tokenizer-snapshot-verification/1.0.0"
)
PRIVATE_BASE_MODEL_CONVERSION_SNAPSHOT_VERIFICATION_SCHEMA_VERSION = (
    "lumen.private-base-model-conversion-snapshot-verification/1.1.0"
)
BASE_MODEL_TOKENIZER_REQUIRED_PATHS = (
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE = {
    "path": "generation_config.json",
    "sizeBytes": 239,
    "sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "huggingFaceBlobID": "20a8a9156fc8c3f25295ca067f61fdf120d517c5",
}

DEFAULT_PYTHON_VERSION = "3.10"
DEFAULT_CUDA_VERSION = "12.8"
DEFAULT_UNSLOTH_REVISION = "935474c20aabc2aadb1da17338959c7c6f9bdafe"
DEFAULT_LLAMA_CPP_REVISION = "34558825a27f4d74dcfd7a91bfde4464baa2a30a"
DEFAULT_PACKAGE_VERSIONS: dict[str, str] = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "datasets": "4.3.0",
    "gradio": "6.17.3",
    "hf_transfer": "0.1.9",
    "huggingface_hub": "0.36.2",
    "peft": "0.19.1",
    "protobuf": "7.35.1",
    "sentencepiece": "0.2.2",
    "spaces": "0.51.0",
    "torch": "2.9.1",
    "torchaudio": "2.9.1",
    "torchvision": "0.24.1",
    "trackio": "0.20.2",
    "transformers": "4.57.6",
    "trl": "0.24.0",
    "unsloth_zoo": "2026.7.2",
}
_CUDA_WHEEL_LOCAL_VERSION_PACKAGES = frozenset(
    {"torch", "torchaudio", "torchvision"}
)

_REQUIREMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_SOURCE_AUDIT_FIELDS = (
    "runtimeSourceKind",
    "runtimeSourceRevision",
    "expectedRuntimeSourceRevision",
    "observedRepositoryRevision",
    "observedRuntimeRevision",
    "runtimeSourceBindingStatus",
    "runtimeSourceBindingMethod",
)
RUNTIME_SOURCE_BINDING_SPACE_UNVERIFIED = "operator_declared_unverified"
RUNTIME_SOURCE_BINDING_SPACE_REPOSITORY_HEAD = (
    "huggingface_repository_head_supplemental"
)
RUNTIME_SOURCE_BINDING_SPACE_DECLARATION = "operator_declared_only"
RUNTIME_SOURCE_BINDING_LOCAL = "local_checkout_observed"
RUNTIME_SOURCE_BINDING_LOCAL_METHOD = "git_head_plus_training_code_manifest"
RUNTIME_SOURCE_BINDING_ATTESTED = "verified_clean_snapshot"
RUNTIME_SOURCE_BINDING_ATTESTED_METHOD = (
    "git_clean_worktree_plus_ubuntu_orchestration_manifest"
)
_PHASES = ("sft", "dpo", "orpo")
_TRAINING_CODE_EXTENSIONS = (
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)
_TRAINING_CODE_VOLATILE_DIRECTORIES = (
    ".git",
    "__pycache__",
    "checkpoints",
    "logs",
    "outputs",
    "uploads",
)
_DEPLOYED_TRAINING_CODE_PATHS = (
    "app.py",
    "lumen_manifest_crawler",
    "lumen_training",
    "requirements.txt",
)
_DEPLOYED_TRAINING_CODE_EXCLUSIONS = (
    "lumen_zero_gpu_defaults.json",
    "lumen_zero_gpu_run_manifest.json",
)
_SPACE_FRONT_MATTER_KEYS = {
    "app_file",
    "python_version",
    "sdk",
    "title",
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requirements_sha256(path: Path) -> str:
    return file_sha256(path)


def canonical_base_model_tokenizer_closure(
    *,
    base_model_id: str,
    base_model_revision: str,
    files: Any,
) -> dict[str, Any]:
    """Validate and canonicalize the exact tokenizer/config closure."""

    if (
        not isinstance(base_model_id, str)
        or not base_model_id
        or re.fullmatch(_REVISION_PATTERN, base_model_revision) is None
        or not isinstance(files, list)
    ):
        raise ValueError(
            "Base-model tokenizer closure requires an ID, full revision, and files"
        )
    normalized: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sizeBytes",
            "sha256",
            "huggingFaceBlobID",
        }:
            raise ValueError("Base-model tokenizer files have an invalid schema")
        logical_path = item.get("path")
        size = item.get("sizeBytes")
        digest = item.get("sha256")
        blob_id = item.get("huggingFaceBlobID")
        if (
            logical_path not in BASE_MODEL_TOKENIZER_REQUIRED_PATHS
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or re.fullmatch(_SHA256_PATTERN, digest) is None
            or not isinstance(blob_id, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", blob_id) is None
        ):
            raise ValueError("Base-model tokenizer file binding is invalid")
        normalized.append(
            {
                "path": logical_path,
                "sizeBytes": size,
                "sha256": digest,
                "huggingFaceBlobID": blob_id,
            }
        )
    normalized.sort(key=lambda item: item["path"])
    if [item["path"] for item in normalized] != list(
        BASE_MODEL_TOKENIZER_REQUIRED_PATHS
    ):
        raise ValueError(
            "Base-model tokenizer closure must bind the exact required files"
        )
    return {
        "schemaVersion": BASE_MODEL_TOKENIZER_CLOSURE_SCHEMA_VERSION,
        "baseModelID": base_model_id,
        "baseModelRevision": base_model_revision,
        "files": normalized,
    }


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def verify_base_model_tokenizer_snapshot(
    snapshot_dir: Path,
    *,
    base_model_id: str,
    base_model_name: str,
    base_model_revision: str,
    tokenizer_files: Any,
    tokenizer_digest: str,
    tokenizer_closure_sha256: str,
) -> dict[str, Any]:
    """Verify the exact Hugging Face snapshot that a converter will consume.

    Hugging Face cache snapshots are symlink farms whose targets are immutable
    blob names. Requiring that layout binds both the selected revision and each
    configured blob identity, while descriptor-stability checks reject a file
    replacement during verification.
    """

    if base_model_id != base_model_name:
        raise ValueError("baseModelID must exactly match base_model_name")
    closure = canonical_base_model_tokenizer_closure(
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        files=tokenizer_files,
    )
    if (
        re.fullmatch(_SHA256_PATTERN, str(tokenizer_digest or "")) is None
        or re.fullmatch(
            _SHA256_PATTERN,
            str(tokenizer_closure_sha256 or ""),
        )
        is None
        or canonical_sha256(closure) != tokenizer_closure_sha256
    ):
        raise ValueError("Base-model tokenizer closure digest drifted")
    tokenizer_json = next(
        item for item in closure["files"] if item["path"] == "tokenizer.json"
    )
    if tokenizer_json["sha256"] != tokenizer_digest:
        raise ValueError("tokenizer.json digest drifted from the tokenizer closure")

    snapshot = Path(snapshot_dir)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("Base-model snapshot must be a regular directory")
    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != base_model_revision or snapshot.parent.name != "snapshots":
        raise ValueError(
            "Base-model snapshot is not bound to the requested immutable revision"
        )
    blob_root = snapshot.parent.parent / "blobs"
    if blob_root.is_symlink() or not blob_root.is_dir():
        raise ValueError("Base-model snapshot blob store is unavailable")
    blob_root = blob_root.resolve(strict=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise ValueError("Tokenizer snapshot verification requires O_NOFOLLOW")

    verified_files: list[dict[str, Any]] = []
    for expected in closure["files"]:
        logical_path = str(expected["path"])
        candidate = snapshot / logical_path
        if not candidate.is_symlink():
            raise ValueError(
                f"Base-model tokenizer snapshot entry is not a cache link: {logical_path}"
            )
        link_before = candidate.lstat()
        try:
            target = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"Base-model tokenizer snapshot entry is unavailable: {logical_path}"
            ) from exc
        if target.parent != blob_root or target.name != expected["huggingFaceBlobID"]:
            raise ValueError(
                f"Base-model tokenizer snapshot blob identity drifted: {logical_path}"
            )
        try:
            descriptor = os.open(
                target,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
            )
        except OSError as exc:
            raise ValueError(
                f"Base-model tokenizer snapshot blob is unavailable: {logical_path}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(
                    f"Base-model tokenizer snapshot blob is not regular: {logical_path}"
                )
            sha256 = hashlib.sha256()
            git_blob_sha1 = hashlib.sha1(
                f"blob {before.st_size}\0".encode("ascii")
            )
            offset = 0
            while True:
                chunk = os.pread(descriptor, 1 << 20, offset)
                if not chunk:
                    break
                sha256.update(chunk)
                git_blob_sha1.update(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
            rebound = target.stat(follow_symlinks=False)
            link_after = candidate.lstat()
            rebound_target = candidate.resolve(strict=True)
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(before) != _stat_identity(rebound)
                or _stat_identity(link_before) != _stat_identity(link_after)
                or rebound_target != target
            ):
                raise ValueError(
                    f"Base-model tokenizer snapshot entry changed: {logical_path}"
                )
        finally:
            os.close(descriptor)
        observed_sha256 = sha256.hexdigest()
        blob_id = str(expected["huggingFaceBlobID"])
        if (
            before.st_size != expected["sizeBytes"]
            or observed_sha256 != expected["sha256"]
            or (len(blob_id) == 64 and observed_sha256 != blob_id)
            or (len(blob_id) == 40 and git_blob_sha1.hexdigest() != blob_id)
        ):
            raise ValueError(
                f"Base-model tokenizer snapshot content drifted: {logical_path}"
            )
        verified_files.append(dict(expected))

    payload = {
        "schemaVersion": (
            BASE_MODEL_TOKENIZER_SNAPSHOT_VERIFICATION_SCHEMA_VERSION
        ),
        "baseModelID": base_model_id,
        "baseModelRevision": base_model_revision,
        "baseModelTokenizerDigest": tokenizer_digest,
        "baseModelTokenizerFiles": verified_files,
        "baseModelTokenizerClosureSHA256": tokenizer_closure_sha256,
        "snapshotPath": str(snapshot),
        "verificationMethod": (
            "huggingface_snapshot_revision_and_blob_identity"
        ),
    }
    return {
        **payload,
        "snapshotVerificationSHA256": canonical_sha256(payload),
    }


def verify_private_base_model_tokenizer_snapshot(
    snapshot_dir: Path,
    *,
    base_model_id: str,
    base_model_name: str,
    base_model_revision: str,
    tokenizer_files: Any,
    tokenizer_digest: str,
    tokenizer_closure_sha256: str,
    allowed_extra_paths: Any = (),
) -> dict[str, Any]:
    """Verify a private copied tokenizer closure used by a trainer process."""

    if base_model_id != base_model_name:
        raise ValueError("baseModelID must exactly match base_model_name")
    closure = canonical_base_model_tokenizer_closure(
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        files=tokenizer_files,
    )
    if (
        re.fullmatch(_SHA256_PATTERN, str(tokenizer_digest or "")) is None
        or re.fullmatch(
            _SHA256_PATTERN,
            str(tokenizer_closure_sha256 or ""),
        )
        is None
        or canonical_sha256(closure) != tokenizer_closure_sha256
    ):
        raise ValueError("Private tokenizer closure digest drifted")
    tokenizer_json = next(
        item for item in closure["files"] if item["path"] == "tokenizer.json"
    )
    if tokenizer_json["sha256"] != tokenizer_digest:
        raise ValueError("Private tokenizer.json digest drifted from the closure")

    snapshot = Path(snapshot_dir)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("Private tokenizer snapshot must be a regular directory")
    snapshot = snapshot.resolve(strict=True)
    root_stat = snapshot.stat(follow_symlinks=False)
    if root_stat.st_uid != os.geteuid() or stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise ValueError("Private tokenizer snapshot ownership or mode drifted")
    if (
        not isinstance(allowed_extra_paths, (list, tuple))
        or any(
            not isinstance(value, str)
            or not value
            or value != Path(value).name
            or value in BASE_MODEL_TOKENIZER_REQUIRED_PATHS
            for value in allowed_extra_paths
        )
        or len(set(allowed_extra_paths)) != len(allowed_extra_paths)
    ):
        raise ValueError("Private tokenizer snapshot allowlist is invalid")
    entries = list(snapshot.iterdir())
    expected_names = {
        *BASE_MODEL_TOKENIZER_REQUIRED_PATHS,
        *allowed_extra_paths,
    }
    if {entry.name for entry in entries} != expected_names:
        raise ValueError("Private tokenizer snapshot has an unexpected file set")

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise ValueError("Private tokenizer verification requires O_NOFOLLOW")
    verified_files: list[dict[str, Any]] = []
    file_signatures: list[dict[str, Any]] = []
    for expected in closure["files"]:
        logical_path = str(expected["path"])
        candidate = snapshot / logical_path
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(
                f"Private tokenizer snapshot entry is not regular: {logical_path}"
            )
        try:
            descriptor = os.open(
                candidate,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
            )
        except OSError as exc:
            raise ValueError(
                f"Private tokenizer snapshot entry is unavailable: {logical_path}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
            ):
                raise ValueError(
                    f"Private tokenizer snapshot entry ownership or mode drifted: {logical_path}"
                )
            sha256 = hashlib.sha256()
            git_blob_sha1 = hashlib.sha1(
                f"blob {before.st_size}\0".encode("ascii")
            )
            offset = 0
            while True:
                chunk = os.pread(descriptor, 1 << 20, offset)
                if not chunk:
                    break
                sha256.update(chunk)
                git_blob_sha1.update(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
            rebound = candidate.stat(follow_symlinks=False)
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(before) != _stat_identity(rebound)
            ):
                raise ValueError(
                    f"Private tokenizer snapshot entry changed: {logical_path}"
                )
        finally:
            os.close(descriptor)
        observed_sha256 = sha256.hexdigest()
        blob_id = str(expected["huggingFaceBlobID"])
        if (
            before.st_size != expected["sizeBytes"]
            or observed_sha256 != expected["sha256"]
            or (len(blob_id) == 64 and observed_sha256 != blob_id)
            or (len(blob_id) == 40 and git_blob_sha1.hexdigest() != blob_id)
        ):
            raise ValueError(
                f"Private tokenizer snapshot content drifted: {logical_path}"
            )
        verified_files.append(dict(expected))
        file_signatures.append(
            {
                "path": logical_path,
                "device": before.st_dev,
                "inode": before.st_ino,
                "mode": stat.S_IMODE(before.st_mode),
                "sizeBytes": before.st_size,
                "mtimeNS": before.st_mtime_ns,
                "ctimeNS": before.st_ctime_ns,
            }
        )

    payload = {
        "schemaVersion": (
            PRIVATE_BASE_MODEL_TOKENIZER_SNAPSHOT_VERIFICATION_SCHEMA_VERSION
        ),
        "baseModelID": base_model_id,
        "baseModelRevision": base_model_revision,
        "baseModelTokenizerDigest": tokenizer_digest,
        "baseModelTokenizerFiles": verified_files,
        "baseModelTokenizerClosureSHA256": tokenizer_closure_sha256,
        "snapshotPath": str(snapshot),
        "verificationMethod": "private_regular_file_closure",
        "snapshotDirectorySignature": {
            "device": root_stat.st_dev,
            "inode": root_stat.st_ino,
            "mode": stat.S_IMODE(root_stat.st_mode),
            "mtimeNS": root_stat.st_mtime_ns,
            "ctimeNS": root_stat.st_ctime_ns,
        },
        "fileStabilitySignatures": file_signatures,
    }
    return {
        **payload,
        "snapshotVerificationSHA256": canonical_sha256(payload),
    }


def verify_private_base_model_conversion_snapshot(
    snapshot_dir: Path,
    *,
    base_model_id: str,
    base_model_name: str,
    base_model_revision: str,
    tokenizer_files: Any,
    tokenizer_digest: str,
    tokenizer_closure_sha256: str,
    generation_config_file: Any,
    model_index_digest: str,
    index_referenced_shard_names: Any,
    index_shard_binding_sha256: str,
    model_artifact_digest: str,
    weight_shards: Any,
) -> dict[str, Any]:
    """Verify the exact run-private base directory passed to GGUF conversion."""

    if not isinstance(weight_shards, list) or not weight_shards:
        raise ValueError("Private conversion snapshot weight shards are missing")
    normalized_shards: list[dict[str, Any]] = []
    for item in weight_shards:
        if not isinstance(item, Mapping) or set(item) != {
            "filename",
            "size",
            "sha256",
        }:
            raise ValueError("Private conversion snapshot shard schema is invalid")
        filename = item.get("filename")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or filename != Path(filename).name
            or filename in BASE_MODEL_TOKENIZER_REQUIRED_PATHS
            or filename == "model.safetensors.index.json"
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or re.fullmatch(_SHA256_PATTERN, digest) is None
        ):
            raise ValueError("Private conversion snapshot shard binding is invalid")
        normalized_shards.append(
            {"filename": filename, "size": size, "sha256": digest}
        )
    normalized_shards.sort(key=lambda item: item["filename"])
    shard_names = [item["filename"] for item in normalized_shards]
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("Private conversion snapshot shards are not unique")
    shard_contract = {
        "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
        "shards": normalized_shards,
    }
    if canonical_sha256(shard_contract) != model_artifact_digest:
        raise ValueError("Private conversion snapshot shard contract drifted")
    if (
        not isinstance(index_referenced_shard_names, list)
        or index_referenced_shard_names != shard_names
        or re.fullmatch(_SHA256_PATTERN, str(model_index_digest or "")) is None
        or re.fullmatch(
            _SHA256_PATTERN,
            str(index_shard_binding_sha256 or ""),
        )
        is None
    ):
        raise ValueError("Private conversion snapshot index binding is invalid")
    index_binding = {
        "schemaVersion": "lumen.base-model-index-shard-binding/1.0.0",
        "indexDigest": model_index_digest,
        "referencedShardNames": shard_names,
        "shardContractDigest": model_artifact_digest,
    }
    if canonical_sha256(index_binding) != index_shard_binding_sha256:
        raise ValueError("Private conversion snapshot index binding drifted")

    if (
        not isinstance(generation_config_file, Mapping)
        or set(generation_config_file) != {
            "path",
            "sizeBytes",
            "sha256",
            "huggingFaceBlobID",
        }
        or generation_config_file.get("path") != "generation_config.json"
        or type(generation_config_file.get("sizeBytes")) is not int
        or generation_config_file["sizeBytes"] <= 0
        or re.fullmatch(
            _SHA256_PATTERN,
            str(generation_config_file.get("sha256") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}",
            str(generation_config_file.get("huggingFaceBlobID") or ""),
        )
        is None
    ):
        raise ValueError("Private conversion generation-config binding is invalid")
    generation_config = dict(generation_config_file)
    extra_paths = [
        "generation_config.json",
        "model.safetensors.index.json",
        *shard_names,
    ]
    tokenizer_verification = verify_private_base_model_tokenizer_snapshot(
        snapshot_dir,
        base_model_id=base_model_id,
        base_model_name=base_model_name,
        base_model_revision=base_model_revision,
        tokenizer_files=tokenizer_files,
        tokenizer_digest=tokenizer_digest,
        tokenizer_closure_sha256=tokenizer_closure_sha256,
        allowed_extra_paths=extra_paths,
    )
    snapshot = Path(tokenizer_verification["snapshotPath"])
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    model_signatures: list[dict[str, Any]] = []

    generation_path = snapshot / "generation_config.json"
    if generation_path.is_symlink() or not generation_path.is_file():
        raise ValueError("Private generation config must be a regular file")
    generation_descriptor = os.open(
        generation_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
    )
    try:
        generation_before = os.fstat(generation_descriptor)
        generation_payload = b""
        offset = 0
        while True:
            chunk = os.pread(generation_descriptor, 1 << 20, offset)
            if not chunk:
                break
            generation_payload += chunk
            offset += len(chunk)
        generation_after = os.fstat(generation_descriptor)
        generation_rebound = generation_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(generation_before.st_mode)
            or generation_before.st_uid != os.geteuid()
            or stat.S_IMODE(generation_before.st_mode) not in {0o400, 0o600}
            or _stat_identity(generation_before) != _stat_identity(generation_after)
            or _stat_identity(generation_before) != _stat_identity(generation_rebound)
        ):
            raise ValueError("Private generation config ownership or stability drifted")
    finally:
        os.close(generation_descriptor)
    generation_sha256 = hashlib.sha256(generation_payload).hexdigest()
    generation_blob = str(generation_config["huggingFaceBlobID"])
    generation_git_sha1 = hashlib.sha1(
        f"blob {len(generation_payload)}\0".encode("ascii") + generation_payload
    ).hexdigest()
    if (
        len(generation_payload) != generation_config["sizeBytes"]
        or generation_sha256 != generation_config["sha256"]
        or (len(generation_blob) == 64 and generation_sha256 != generation_blob)
        or (len(generation_blob) == 40 and generation_git_sha1 != generation_blob)
    ):
        raise ValueError("Private generation config content drifted")
    model_signatures.append(
        {
            "path": "generation_config.json",
            "kind": "private_regular_generation_config",
            "device": generation_before.st_dev,
            "inode": generation_before.st_ino,
            "mode": stat.S_IMODE(generation_before.st_mode),
            "sizeBytes": generation_before.st_size,
            "mtimeNS": generation_before.st_mtime_ns,
            "ctimeNS": generation_before.st_ctime_ns,
        }
    )

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Private conversion snapshot index must be a regular file")
    index_descriptor = os.open(
        index_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
    )
    try:
        index_before = os.fstat(index_descriptor)
        if (
            not stat.S_ISREG(index_before.st_mode)
            or index_before.st_uid != os.geteuid()
            or stat.S_IMODE(index_before.st_mode) not in {0o400, 0o600}
        ):
            raise ValueError("Private conversion snapshot index mode drifted")
        chunks: list[bytes] = []
        offset = 0
        while True:
            chunk = os.pread(index_descriptor, 1 << 20, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        index_payload = b"".join(chunks)
        index_after = os.fstat(index_descriptor)
        index_rebound = index_path.stat(follow_symlinks=False)
        if (
            _stat_identity(index_before) != _stat_identity(index_after)
            or _stat_identity(index_before) != _stat_identity(index_rebound)
        ):
            raise ValueError("Private conversion snapshot index changed")
    finally:
        os.close(index_descriptor)
    if hashlib.sha256(index_payload).hexdigest() != model_index_digest:
        raise ValueError("Private conversion snapshot index digest drifted")
    try:
        parsed_index = json.loads(index_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Private conversion snapshot index is invalid") from exc
    referenced = sorted(set((parsed_index.get("weight_map") or {}).values()))
    if referenced != shard_names:
        raise ValueError("Private conversion snapshot index shard set drifted")
    model_signatures.append(
        {
            "path": "model.safetensors.index.json",
            "kind": "private_regular_file",
            "device": index_before.st_dev,
            "inode": index_before.st_ino,
            "mode": stat.S_IMODE(index_before.st_mode),
            "sizeBytes": index_before.st_size,
            "mtimeNS": index_before.st_mtime_ns,
            "ctimeNS": index_before.st_ctime_ns,
        }
    )

    for expected in normalized_shards:
        filename = expected["filename"]
        link = snapshot / filename
        if link.is_symlink() or not link.is_file():
            raise ValueError(
                f"Private conversion snapshot shard is not a private regular file: {filename}"
            )
        descriptor = os.open(
            link,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
            ):
                raise ValueError(
                    f"Private conversion snapshot shard ownership or mode drifted: {filename}"
                )
            digest = hashlib.sha256()
            offset = 0
            while True:
                chunk = os.pread(descriptor, 1 << 20, offset)
                if not chunk:
                    break
                digest.update(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
            rebound = link.stat(follow_symlinks=False)
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(before) != _stat_identity(rebound)
            ):
                raise ValueError(
                    f"Private conversion snapshot shard changed: {filename}"
                )
        finally:
            os.close(descriptor)
        if before.st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(
                f"Private conversion snapshot shard content drifted: {filename}"
            )
        model_signatures.append(
            {
                "path": filename,
                "kind": "private_regular_weight_shard",
                "device": before.st_dev,
                "inode": before.st_ino,
                "mode": stat.S_IMODE(before.st_mode),
                "sizeBytes": before.st_size,
                "mtimeNS": before.st_mtime_ns,
                "ctimeNS": before.st_ctime_ns,
            }
        )

    payload = {
        "schemaVersion": (
            PRIVATE_BASE_MODEL_CONVERSION_SNAPSHOT_VERIFICATION_SCHEMA_VERSION
        ),
        "baseModelID": base_model_id,
        "baseModelRevision": base_model_revision,
        "baseModelIndexDigest": model_index_digest,
        "baseModelIndexReferencedShardNames": shard_names,
        "baseModelIndexShardBindingSHA256": index_shard_binding_sha256,
        "baseModelArtifactDigest": model_artifact_digest,
        "baseModelWeightShards": normalized_shards,
        "baseModelGenerationConfigFile": generation_config,
        "baseModelTokenizerDigest": tokenizer_digest,
        "baseModelTokenizerFiles": tokenizer_verification[
            "baseModelTokenizerFiles"
        ],
        "baseModelTokenizerClosureSHA256": tokenizer_closure_sha256,
        "snapshotPath": str(snapshot),
        "tokenizerSnapshotVerification": tokenizer_verification,
        "modelFileStabilitySignatures": model_signatures,
        "verificationMethod": "private_regular_full_model_snapshot",
    }
    return {
        **payload,
        "snapshotVerificationSHA256": canonical_sha256(payload),
    }


def private_base_model_runtime_snapshot_required_bytes(
    *,
    weight_shards: Any,
    tokenizer_files: Any,
    generation_config_file: Any,
) -> int:
    try:
        weight_bytes = sum(int(item["size"]) for item in weight_shards)
        tokenizer_bytes = sum(int(item["sizeBytes"]) for item in tokenizer_files)
        generation_bytes = int(generation_config_file["sizeBytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Private base-model snapshot sizes are invalid") from exc
    if weight_bytes <= 0 or tokenizer_bytes <= 0 or generation_bytes <= 0:
        raise ValueError("Private base-model snapshot sizes must be positive")
    return weight_bytes + tokenizer_bytes + generation_bytes + (1 << 30)


def create_private_base_model_runtime_snapshot(
    *,
    source_snapshot_dir: Path,
    private_tokenizer_snapshot_dir: Path,
    destination: Path,
    base_model_id: str,
    base_model_name: str,
    base_model_revision: str,
    tokenizer_files: Any,
    tokenizer_digest: str,
    tokenizer_closure_sha256: str,
    generation_config_file: Any,
    model_index_digest: str,
    index_referenced_shard_names: Any,
    index_shard_binding_sha256: str,
    model_artifact_digest: str,
    weight_shards: Any,
) -> dict[str, Any]:
    """Atomically copy one immutable, process-owned full model snapshot."""

    if destination.exists() or destination.is_symlink():
        raise ValueError("Private base-model runtime snapshot already exists")
    tokenizer_verification = verify_private_base_model_tokenizer_snapshot(
        private_tokenizer_snapshot_dir,
        base_model_id=base_model_id,
        base_model_name=base_model_name,
        base_model_revision=base_model_revision,
        tokenizer_files=tokenizer_files,
        tokenizer_digest=tokenizer_digest,
        tokenizer_closure_sha256=tokenizer_closure_sha256,
    )
    source = Path(source_snapshot_dir).resolve(strict=True)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    required_bytes = private_base_model_runtime_snapshot_required_bytes(
        weight_shards=weight_shards,
        tokenizer_files=tokenizer_files,
        generation_config_file=generation_config_file,
    )
    if shutil.disk_usage(destination.parent).free < required_bytes:
        raise ValueError(
            "Insufficient free space for private base-model runtime snapshot"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        staging.chmod(0o700)
        for expected in tokenizer_verification["baseModelTokenizerFiles"]:
            filename = expected["path"]
            target = staging / filename
            shutil.copyfile(
                Path(private_tokenizer_snapshot_dir) / filename,
                target,
                follow_symlinks=False,
            )
            target.chmod(0o400)
            descriptor = os.open(target, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        index_target = staging / "model.safetensors.index.json"
        shutil.copyfile(
            source / "model.safetensors.index.json",
            index_target,
            follow_symlinks=True,
        )
        index_target.chmod(0o400)
        descriptor = os.open(index_target, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        generation_target = staging / "generation_config.json"
        shutil.copyfile(
            source / "generation_config.json",
            generation_target,
            follow_symlinks=True,
        )
        generation_target.chmod(0o400)
        descriptor = os.open(generation_target, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        for shard in weight_shards:
            filename = shard["filename"]
            target = staging / filename
            shutil.copyfile(
                (source / filename).resolve(strict=True),
                target,
                follow_symlinks=False,
            )
            target.chmod(0o400)
            descriptor = os.open(target, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        verify_private_base_model_conversion_snapshot(
            staging,
            base_model_id=base_model_id,
            base_model_name=base_model_name,
            base_model_revision=base_model_revision,
            tokenizer_files=tokenizer_files,
            tokenizer_digest=tokenizer_digest,
            tokenizer_closure_sha256=tokenizer_closure_sha256,
            generation_config_file=generation_config_file,
            model_index_digest=model_index_digest,
            index_referenced_shard_names=index_referenced_shard_names,
            index_shard_binding_sha256=index_shard_binding_sha256,
            model_artifact_digest=model_artifact_digest,
            weight_shards=weight_shards,
        )
        if verify_private_base_model_tokenizer_snapshot(
            private_tokenizer_snapshot_dir,
            base_model_id=base_model_id,
            base_model_name=base_model_name,
            base_model_revision=base_model_revision,
            tokenizer_files=tokenizer_files,
            tokenizer_digest=tokenizer_digest,
            tokenizer_closure_sha256=tokenizer_closure_sha256,
        ) != tokenizer_verification:
            raise ValueError(
                "Private tokenizer snapshot changed during runtime snapshot copy"
            )
        directory_descriptor = os.open(
            staging,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.replace(staging, destination)
        staging = None
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return verify_private_base_model_conversion_snapshot(
            destination,
            base_model_id=base_model_id,
            base_model_name=base_model_name,
            base_model_revision=base_model_revision,
            tokenizer_files=tokenizer_files,
            tokenizer_digest=tokenizer_digest,
            tokenizer_closure_sha256=tokenizer_closure_sha256,
            generation_config_file=generation_config_file,
            model_index_digest=model_index_digest,
            index_referenced_shard_names=index_referenced_shard_names,
            index_shard_binding_sha256=index_shard_binding_sha256,
            model_artifact_digest=model_artifact_digest,
            weight_shards=weight_shards,
        )
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _space_front_matter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Space README must begin with YAML front matter")
    try:
        closing = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("Space README front matter is not terminated") from exc

    values: dict[str, str] = {}
    for raw_line in lines[1:closing]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("Space README front matter must use scalar key/value fields")
        key, raw_value = (part.strip() for part in line.split(":", 1))
        if key in values:
            raise ValueError(f"Duplicate Space README front-matter field: {key}")
        if key not in _SPACE_FRONT_MATTER_KEYS:
            raise ValueError(f"Unsupported Space README front-matter field: {key}")
        if not raw_value or raw_value[0] in "[{>|&*!":
            raise ValueError(f"Space README front-matter field must be scalar: {key}")
        if raw_value[0] in {'\"', "'"}:
            if len(raw_value) < 2 or raw_value[-1] != raw_value[0]:
                raise ValueError(f"Unterminated Space README front-matter scalar: {key}")
            value = raw_value[1:-1]
        else:
            value = raw_value
        values[key] = value
    return values


def build_space_configuration(readme_path: Path) -> dict[str, Any]:
    front_matter = _space_front_matter(readme_path)
    if set(front_matter) != _SPACE_FRONT_MATTER_KEYS:
        missing = sorted(_SPACE_FRONT_MATTER_KEYS - set(front_matter))
        raise ValueError(
            "Space README front matter is missing required fields: "
            + ", ".join(missing)
        )
    payload = {
        "schemaVersion": SPACE_CONFIGURATION_SCHEMA_VERSION,
        "sdk": front_matter["sdk"],
        "appFile": front_matter["app_file"],
        "pythonVersion": front_matter["python_version"],
        # ZeroGPU hardware is requested through the Hub API; README metadata
        # must not silently select a distinct runtime.
        "suggestedHardware": None,
    }
    return {**payload, "spaceConfigurationSHA256": canonical_sha256(payload)}


def verify_space_configuration(
    configuration: Mapping[str, Any],
    *,
    readme_path: Path | None = None,
) -> str:
    payload = {
        "schemaVersion": SPACE_CONFIGURATION_SCHEMA_VERSION,
        "sdk": "gradio",
        "appFile": "app.py",
        "pythonVersion": DEFAULT_PYTHON_VERSION,
        "suggestedHardware": None,
    }
    if set(configuration) != {*payload, "spaceConfigurationSHA256"}:
        raise ValueError("Invalid Space configuration contract")
    if any(configuration.get(key) != value for key, value in payload.items()):
        raise ValueError("Space runtime configuration drifted from the supported contract")
    digest = canonical_sha256(payload)
    if configuration.get("spaceConfigurationSHA256") != digest:
        raise ValueError("spaceConfigurationSHA256 does not match the Space configuration")
    if readme_path is not None and build_space_configuration(readme_path) != dict(
        configuration
    ):
        raise ValueError("Deployed Space README runtime configuration drifted")
    return digest


def _safe_logical_path(value: str) -> str:
    logical = PurePosixPath(value)
    if (
        not value
        or logical.is_absolute()
        or value != logical.as_posix()
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise ValueError(f"Unsafe training-code logical path: {value!r}")
    return value


def _normalize_closure_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "includedExtensions",
        "coveredLogicalPaths",
        "excludedLogicalPaths",
        "excludedDirectoryNames",
        "coverDeployedRoot",
        "rejectUnlistedBehaviorFiles",
    }
    if set(value) != expected_keys:
        raise ValueError("Training-code closure policy contains unsupported fields")
    extensions = value.get("includedExtensions")
    covered_paths = value.get("coveredLogicalPaths")
    excluded_paths = value.get("excludedLogicalPaths")
    excluded_directories = value.get("excludedDirectoryNames")
    if (
        not isinstance(extensions, list)
        or not extensions
        or not isinstance(covered_paths, list)
        or not covered_paths
        or not isinstance(excluded_paths, list)
        or not isinstance(excluded_directories, list)
        or type(value.get("coverDeployedRoot")) is not bool
        or value.get("rejectUnlistedBehaviorFiles") is not True
    ):
        raise ValueError("Invalid training-code closure policy")

    normalized_extensions: list[str] = []
    for extension in extensions:
        if (
            not isinstance(extension, str)
            or not extension.startswith(".")
            or extension != extension.lower()
            or any(character in extension for character in ("/", "\\"))
        ):
            raise ValueError("Invalid training-code included extension")
        normalized_extensions.append(extension)
    if len(normalized_extensions) != len(set(normalized_extensions)):
        raise ValueError("Duplicate training-code included extension")

    normalized_covered = [_safe_logical_path(str(path)) for path in covered_paths]
    normalized_excluded = [_safe_logical_path(str(path)) for path in excluded_paths]
    normalized_directories: list[str] = []
    for name in excluded_directories:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("Invalid volatile training-code directory name")
        normalized_directories.append(name)

    for values, label in (
        (normalized_covered, "covered path"),
        (normalized_excluded, "excluded path"),
        (normalized_directories, "excluded directory"),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate training-code {label}")
        if values != sorted(values):
            raise ValueError(f"Training-code {label}s must be sorted")
    if normalized_extensions != sorted(normalized_extensions):
        raise ValueError("Training-code included extensions must be sorted")

    return {
        "includedExtensions": normalized_extensions,
        "coveredLogicalPaths": normalized_covered,
        "excludedLogicalPaths": normalized_excluded,
        "excludedDirectoryNames": normalized_directories,
        "coverDeployedRoot": value["coverDeployedRoot"],
        "rejectUnlistedBehaviorFiles": True,
    }


def _default_closure_policy(files: Mapping[str, Path]) -> dict[str, Any]:
    return _normalize_closure_policy(
        {
            "includedExtensions": list(_TRAINING_CODE_EXTENSIONS),
            # Generic callers cover exactly the files they supplied. The repository
            # bundle below deliberately covers whole deployed trees instead.
            "coveredLogicalPaths": sorted(files),
            "excludedLogicalPaths": [],
            "excludedDirectoryNames": list(_TRAINING_CODE_VOLATILE_DIRECTORIES),
            "coverDeployedRoot": False,
            "rejectUnlistedBehaviorFiles": True,
        }
    )


def deployed_training_code_closure_policy() -> dict[str, Any]:
    return _normalize_closure_policy(
        {
            "includedExtensions": list(_TRAINING_CODE_EXTENSIONS),
            "coveredLogicalPaths": list(_DEPLOYED_TRAINING_CODE_PATHS),
            "excludedLogicalPaths": list(_DEPLOYED_TRAINING_CODE_EXCLUSIONS),
            "excludedDirectoryNames": list(_TRAINING_CODE_VOLATILE_DIRECTORIES),
            "coverDeployedRoot": True,
            "rejectUnlistedBehaviorFiles": True,
        }
    )


def _is_logical_path_within(path: str, parent: str) -> bool:
    logical = PurePosixPath(path)
    ancestor = PurePosixPath(parent)
    return logical == ancestor or ancestor in logical.parents


def _is_excluded_logical_path(path: str, policy: Mapping[str, Any]) -> bool:
    logical = PurePosixPath(path)
    if any(part in policy["excludedDirectoryNames"] for part in logical.parts):
        return True
    return any(
        _is_logical_path_within(path, excluded)
        for excluded in policy["excludedLogicalPaths"]
    )


def _policy_covers_logical_path(path: str, policy: Mapping[str, Any]) -> bool:
    return policy["coverDeployedRoot"] or any(
        _is_logical_path_within(path, covered)
        for covered in policy["coveredLogicalPaths"]
    )


def _discover_covered_training_code_files(
    root: Path,
    policy: Mapping[str, Any],
) -> set[str]:
    resolved_root = root.resolve()
    discovered: set[str] = set()
    extensions = set(policy["includedExtensions"])
    covered_candidates = (
        [("<deployed-root>", resolved_root)]
        if policy["coverDeployedRoot"]
        else [
            (covered, (resolved_root / covered).resolve())
            for covered in policy["coveredLogicalPaths"]
        ]
    )
    for covered, candidate in covered_candidates:
        if candidate != resolved_root and resolved_root not in candidate.parents:
            raise ValueError("Training-code closure path escapes the deployed code root")
        if not candidate.exists():
            raise ValueError(f"Missing covered training-code path: {covered}")
        paths = [candidate] if candidate.is_file() else candidate.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved_root not in resolved.parents:
                raise ValueError("Training-code closure contains an escaping symlink")
            logical = resolved.relative_to(resolved_root).as_posix()
            if _is_excluded_logical_path(logical, policy):
                continue
            if resolved.suffix.lower() in extensions:
                discovered.add(logical)
    return discovered


def build_training_code_manifest(
    *,
    phase: str,
    files: Mapping[str, Path],
    closure_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in _PHASES:
        raise ValueError(f"Unsupported training phase: {phase}")
    if not files:
        raise ValueError("Training-code manifest must contain at least one file")

    policy = _normalize_closure_policy(
        closure_policy if closure_policy is not None else _default_closure_policy(files)
    )
    entries: list[dict[str, Any]] = []
    for logical_path, source_path in sorted(files.items()):
        logical = _safe_logical_path(logical_path)
        if _is_excluded_logical_path(logical, policy) or not _policy_covers_logical_path(
            logical, policy
        ):
            raise ValueError(f"Training-code file is outside the closure policy: {logical}")
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing training-code file: {source}")
        entries.append(
            {
                "path": logical,
                "sizeBytes": source.stat().st_size,
                "sha256": file_sha256(source),
            }
        )
    payload = {
        "schemaVersion": TRAINING_CODE_MANIFEST_SCHEMA_VERSION,
        "phase": phase,
        "closurePolicy": policy,
        "files": entries,
    }
    return {**payload, "trainingCodeSHA256": canonical_sha256(payload)}


def verify_training_code_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> str:
    phase = manifest.get("phase")
    files = manifest.get("files")
    closure_policy = manifest.get("closurePolicy")
    if (
        manifest.get("schemaVersion") != TRAINING_CODE_MANIFEST_SCHEMA_VERSION
        or phase not in _PHASES
        or not isinstance(files, list)
        or not files
        or not isinstance(closure_policy, Mapping)
    ):
        raise ValueError("Invalid training-code manifest contract")
    policy = _normalize_closure_policy(closure_policy)

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("Training-code file entries must be objects")
        path = _safe_logical_path(str(entry.get("path") or ""))
        size = entry.get("sizeBytes")
        digest = entry.get("sha256")
        if (
            path in seen
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("Invalid training-code file entry")
        seen.add(path)
        if _is_excluded_logical_path(path, policy) or not _policy_covers_logical_path(
            path, policy
        ):
            raise ValueError("Training-code file is outside the closure policy")
        normalized.append({"path": path, "sizeBytes": size, "sha256": digest})

        if root is not None:
            candidate = (root / path).resolve()
            resolved_root = root.resolve()
            if candidate.parent != resolved_root and resolved_root not in candidate.parents:
                raise ValueError("Training-code path escapes the deployed code root")
            if (
                not candidate.is_file()
                or candidate.stat().st_size != size
                or file_sha256(candidate) != digest
            ):
                raise ValueError(f"Deployed training-code drift: {path}")

    if root is not None:
        discovered = _discover_covered_training_code_files(root, policy)
        unexpected = sorted(discovered - seen)
        missing = sorted(seen - discovered)
        if missing:
            raise ValueError(
                "Declared training-code files are outside the deployed closure: "
                + ", ".join(missing)
            )
        if unexpected:
            raise ValueError(
                "Unlisted behavior-affecting training-code files: "
                + ", ".join(unexpected)
            )

    if normalized != sorted(normalized, key=lambda item: item["path"]):
        raise ValueError("Training-code manifest files must be sorted")
    payload = {
        "schemaVersion": TRAINING_CODE_MANIFEST_SCHEMA_VERSION,
        "phase": phase,
        "closurePolicy": policy,
        "files": normalized,
    }
    digest = canonical_sha256(payload)
    if manifest.get("trainingCodeSHA256") != digest:
        raise ValueError("trainingCodeSHA256 does not match the code manifest")
    if set(manifest) != {*payload, "trainingCodeSHA256"}:
        raise ValueError("Training-code manifest contains unsupported fields")
    return digest


def build_training_code_bundle(
    phase_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(phase_manifests) != set(_PHASES):
        raise ValueError(f"Training-code bundle phases must be exactly {_PHASES}")
    phases: dict[str, dict[str, Any]] = {}
    for phase in _PHASES:
        manifest = dict(phase_manifests[phase])
        verify_training_code_manifest(manifest)
        if manifest.get("phase") != phase:
            raise ValueError("Training-code bundle phase key does not match manifest")
        phases[phase] = manifest
    payload = {
        "schemaVersion": TRAINING_CODE_BUNDLE_SCHEMA_VERSION,
        "phases": phases,
    }
    return {**payload, "trainingCodeSHA256": canonical_sha256(payload)}


def verify_training_code_bundle(
    bundle: Mapping[str, Any],
    *,
    deployed_root: Path | None = None,
) -> str:
    phases = bundle.get("phases")
    if (
        bundle.get("schemaVersion") != TRAINING_CODE_BUNDLE_SCHEMA_VERSION
        or not isinstance(phases, Mapping)
        or set(phases) != set(_PHASES)
    ):
        raise ValueError("Invalid training-code bundle contract")
    normalized: dict[str, dict[str, Any]] = {}
    for phase in _PHASES:
        manifest = phases.get(phase)
        if not isinstance(manifest, Mapping):
            raise ValueError("Training-code phase manifest must be an object")
        verify_training_code_manifest(manifest, root=deployed_root)
        normalized[phase] = dict(manifest)
    payload = {
        "schemaVersion": TRAINING_CODE_BUNDLE_SCHEMA_VERSION,
        "phases": normalized,
    }
    digest = canonical_sha256(payload)
    if bundle.get("trainingCodeSHA256") != digest:
        raise ValueError("trainingCodeSHA256 does not match the phase bundle")
    if set(bundle) != {*payload, "trainingCodeSHA256"}:
        raise ValueError("Training-code bundle contains unsupported fields")
    return digest


def repository_training_code_bundle(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    common: dict[str, Path] = {
        "app.py": root / "tools/hf_zerogpu/space_template/app.py",
        "requirements.txt": root / "tools/hf_zerogpu/space_template/requirements.txt",
        "lumen_training/__init__.py": root
        / "tools/fine_tuning/unsloth/lumen_training/__init__.py",
        "lumen_training/adapter_artifact.py": root
        / "tools/fine_tuning/unsloth/adapter_artifact.py",
        "lumen_training/train_dpo.py": root
        / "tools/fine_tuning/unsloth/train_dpo.py",
        "lumen_training/train_sft.py": root
        / "tools/fine_tuning/unsloth/train_sft.py",
        "lumen_training/training_lineage.py": root
        / "tools/fine_tuning/unsloth/training_lineage.py",
    }
    crawler_root = (
        root / "tools/lumen_manifest_crawler/lumen_manifest_crawler"
    )
    for source in sorted(crawler_root.rglob("*")):
        if (
            source.is_file()
            and source.suffix.lower() in _TRAINING_CODE_EXTENSIONS
            and not any(
                part in _TRAINING_CODE_VOLATILE_DIRECTORIES
                for part in source.relative_to(crawler_root).parts
            )
        ):
            logical = (
                PurePosixPath("lumen_manifest_crawler")
                / source.relative_to(crawler_root).as_posix()
            ).as_posix()
            common[logical] = source
    policy = deployed_training_code_closure_policy()
    return build_training_code_bundle(
        {
            phase: build_training_code_manifest(
                phase=phase,
                files=common,
                closure_policy=policy,
            )
            for phase in _PHASES
        }
    )


def deployed_training_code_bundle(deployed_root: Path) -> dict[str, Any]:
    """Rebuild the phase manifests from an already assembled Space bundle."""

    root = deployed_root.resolve()
    policy = deployed_training_code_closure_policy()
    logical_paths = _discover_covered_training_code_files(root, policy)
    files = {logical: root / logical for logical in sorted(logical_paths)}
    return build_training_code_bundle(
        {
            phase: build_training_code_manifest(
                phase=phase,
                files=files,
                closure_policy=policy,
            )
            for phase in _PHASES
        }
    )


def _requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith(("-r", "--")):
        return None
    match = _REQUIREMENT_NAME_PATTERN.match(stripped)
    if match is None:
        raise ValueError(f"Unsupported requirement line: {line!r}")
    return match.group(0).replace("-", "_").casefold()


def build_training_dependency_lock(
    requirements_path: Path,
    *,
    python_version: str = DEFAULT_PYTHON_VERSION,
    cuda_version: str = DEFAULT_CUDA_VERSION,
    llama_cpp_revision: str = DEFAULT_LLAMA_CPP_REVISION,
) -> dict[str, Any]:
    requirement_lines = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    requirement_names: list[str] = []
    for line in requirement_lines:
        name = _requirement_name(line)
        if name is None:
            raise ValueError("requirements.txt may not include nested or option directives")
        requirement_names.append(name)
    expected_names = {*DEFAULT_PACKAGE_VERSIONS, "unsloth"}
    if len(requirement_names) != len(set(requirement_names)):
        raise ValueError("requirements.txt contains duplicate controlled dependencies")
    if set(requirement_names) != expected_names:
        raise ValueError(
            "requirements.txt direct dependency set drifted from the controlled lock: "
            f"missing={sorted(expected_names - set(requirement_names))}, "
            f"extra={sorted(set(requirement_names) - expected_names)}"
        )
    for line, name in zip(requirement_lines, requirement_names, strict=True):
        if name == "unsloth":
            expected = (
                "unsloth[colab-new] @ "
                "git+https://github.com/unslothai/unsloth.git@"
                f"{DEFAULT_UNSLOTH_REVISION}"
            )
            if line != expected:
                raise ValueError("requirements.txt Unsloth VCS revision drifted")
            continue
        version_match = re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*==([^;\s]+)",
            line,
        )
        if (
            version_match is None
            or version_match.group(1) != DEFAULT_PACKAGE_VERSIONS[name]
        ):
            raise ValueError(
                f"requirements.txt version for {name} drifted from the controlled lock"
            )
    payload = {
        "schemaVersion": TRAINING_DEPENDENCY_LOCK_SCHEMA_VERSION,
        "pythonVersion": python_version,
        "cudaVersion": cuda_version,
        "packageVersions": dict(sorted(DEFAULT_PACKAGE_VERSIONS.items())),
        "vcsPackages": {
            "unsloth": {
                "repository": "https://github.com/unslothai/unsloth.git",
                "revision": DEFAULT_UNSLOTH_REVISION,
            }
        },
        "llamaCppRevision": llama_cpp_revision,
        "requirementsSHA256": requirements_sha256(requirements_path),
    }
    return {**payload, "trainingDependencyLockSHA256": canonical_sha256(payload)}


def verify_training_dependency_lock(
    lock: Mapping[str, Any],
    *,
    requirements_path: Path | None = None,
    installed_versions: Mapping[str, str] | None = None,
    installed_unsloth_revision: str | None = None,
    runtime_python_version: str | None = None,
    runtime_cuda_version: str | None = None,
) -> str:
    packages = lock.get("packageVersions")
    vcs = lock.get("vcsPackages")
    if (
        set(lock)
        != {
            "schemaVersion",
            "pythonVersion",
            "cudaVersion",
            "packageVersions",
            "vcsPackages",
            "llamaCppRevision",
            "requirementsSHA256",
            "trainingDependencyLockSHA256",
        }
        or lock.get("schemaVersion") != TRAINING_DEPENDENCY_LOCK_SCHEMA_VERSION
        or lock.get("pythonVersion") != DEFAULT_PYTHON_VERSION
        or lock.get("cudaVersion") != DEFAULT_CUDA_VERSION
        or not isinstance(packages, Mapping)
        or dict(packages) != DEFAULT_PACKAGE_VERSIONS
        or not isinstance(vcs, Mapping)
        or vcs.get("unsloth")
        != {
            "repository": "https://github.com/unslothai/unsloth.git",
            "revision": DEFAULT_UNSLOTH_REVISION,
        }
        or lock.get("llamaCppRevision") != DEFAULT_LLAMA_CPP_REVISION
        or not isinstance(lock.get("requirementsSHA256"), str)
        or _SHA256_PATTERN.fullmatch(lock["requirementsSHA256"]) is None
    ):
        raise ValueError("Invalid training dependency lock")
    payload = {key: value for key, value in lock.items() if key != "trainingDependencyLockSHA256"}
    digest = canonical_sha256(payload)
    if lock.get("trainingDependencyLockSHA256") != digest:
        raise ValueError("trainingDependencyLockSHA256 does not match the dependency lock")
    if requirements_path is not None:
        rebuilt = build_training_dependency_lock(requirements_path)
        if rebuilt != dict(lock):
            raise ValueError("Deployed requirements.txt drifted from the dependency lock")
    if installed_versions is not None:
        observed_versions = dict(installed_versions)
        canonical_versions = {
            name: canonical_controlled_package_version(
                name,
                str(version),
                expected_version=str(packages.get(name) or ""),
                cuda_version=str(lock["cudaVersion"]),
            )
            for name, version in observed_versions.items()
        }
        if canonical_versions != dict(packages):
            raise ValueError(
                "Installed controlled package versions drifted from the lock: "
                + json.dumps(
                    {"actual": observed_versions, "expected": dict(packages)},
                    sort_keys=True,
                )
            )
    if (
        installed_unsloth_revision is not None
        and installed_unsloth_revision != DEFAULT_UNSLOTH_REVISION
    ):
        raise ValueError("Installed Unsloth revision drifted from the dependency lock")
    if (
        runtime_python_version is not None
        and runtime_python_version != lock["pythonVersion"]
    ):
        raise ValueError("Runtime Python version drifted from the dependency lock")
    if runtime_cuda_version is not None and runtime_cuda_version != lock["cudaVersion"]:
        raise ValueError("Runtime CUDA version drifted from the dependency lock")
    return digest


def canonical_controlled_package_version(
    name: str,
    installed_version: str,
    *,
    expected_version: str,
    cuda_version: str,
) -> str:
    """Canonicalize only the CUDA local tag already bound by the lock.

    PyTorch's CUDA wheel index records versions such as ``2.9.1+cu128`` while
    the direct requirement remains ``torch==2.9.1``. The dependency lock binds
    CUDA separately as ``12.8``, so the exact matching ``+cu128`` local tag is
    equivalent for the three PyTorch distributions. Every other local tag or
    package version remains unchanged and therefore fails the caller's exact
    lock comparison.
    """

    if installed_version == expected_version:
        return expected_version
    cuda_tag = "cu" + cuda_version.replace(".", "")
    if (
        name in _CUDA_WHEEL_LOCAL_VERSION_PACKAGES
        and installed_version == f"{expected_version}+{cuda_tag}"
    ):
        return expected_version
    return installed_version


def installed_controlled_package_versions(lock: Mapping[str, Any]) -> dict[str, str]:
    packages = lock.get("packageVersions")
    if not isinstance(packages, Mapping):
        raise ValueError("trainingDependencyLock.packageVersions must be an object")
    return {
        name: importlib_metadata.version(name)
        for name in sorted(packages)
    }


def _normalized_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value).casefold()
    if not normalized or re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized) is None:
        raise ValueError(f"Invalid installed distribution name: {value!r}")
    return normalized


def _normalized_direct_url(
    value: str | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Installed distribution has malformed direct_url.json") from exc
    else:
        parsed = value
    if not isinstance(parsed, Mapping) or set(parsed) - {
        "url",
        "vcs_info",
        "archive_info",
        "dir_info",
        "subdirectory",
    }:
        raise ValueError("Installed distribution has unsupported direct-url provenance")
    raw_url = parsed.get("url")
    if not isinstance(raw_url, str) or not raw_url:
        raise ValueError("Installed distribution direct URL is missing")
    url = urlsplit(raw_url)
    if (
        url.scheme not in {"https", "git+https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
    ):
        raise ValueError("Installed distribution direct URL is not immutable and secret-safe")
    if "dir_info" in parsed:
        raise ValueError("Installed distribution directory provenance is mutable")
    vcs_info = parsed.get("vcs_info")
    archive_info = parsed.get("archive_info")
    if (vcs_info is None) == (archive_info is None):
        raise ValueError("Installed distribution direct URL lacks immutable provenance")
    sanitized: dict[str, Any] = {
        "url": urlunsplit((url.scheme, url.netloc, url.path, "", "")),
    }
    if vcs_info is not None:
        if (
            not isinstance(vcs_info, Mapping)
            or set(vcs_info) - {"vcs", "commit_id", "requested_revision"}
            or vcs_info.get("vcs") != "git"
            or not isinstance(vcs_info.get("commit_id"), str)
            or _REVISION_PATTERN.fullmatch(vcs_info["commit_id"]) is None
        ):
            raise ValueError("Installed distribution VCS provenance is not immutable")
        requested_revision = vcs_info.get("requested_revision")
        if requested_revision is not None and (
            not isinstance(requested_revision, str)
            or _REVISION_PATTERN.fullmatch(requested_revision) is None
        ):
            raise ValueError("Installed distribution VCS provenance is malformed")
        sanitized_vcs = {
            "vcs": "git",
            "commit_id": vcs_info["commit_id"],
        }
        if requested_revision is not None:
            sanitized_vcs["requested_revision"] = requested_revision
        sanitized["vcs_info"] = sanitized_vcs
    else:
        if not isinstance(archive_info, Mapping) or set(archive_info) - {
            "hash",
            "hashes",
        }:
            raise ValueError("Installed distribution archive provenance is malformed")
        hashes = archive_info.get("hashes")
        legacy_hash = archive_info.get("hash")
        archive_sha256 = (
            hashes.get("sha256") if isinstance(hashes, Mapping) else None
        )
        if archive_sha256 is None and isinstance(legacy_hash, str):
            match = re.fullmatch(r"sha256=([0-9a-f]{64})", legacy_hash)
            archive_sha256 = match.group(1) if match is not None else None
        if (
            not isinstance(archive_sha256, str)
            or _SHA256_PATTERN.fullmatch(archive_sha256) is None
        ):
            raise ValueError("Installed distribution archive lacks a SHA-256 digest")
        sanitized["archive_info"] = {"hashes": {"sha256": archive_sha256}}
    subdirectory = parsed.get("subdirectory")
    if subdirectory is not None:
        if not isinstance(subdirectory, str):
            raise ValueError("Installed distribution subdirectory is malformed")
        logical = PurePosixPath(subdirectory)
        if (
            not subdirectory
            or logical.is_absolute()
            or any(part in {"", ".", ".."} for part in logical.parts)
            or subdirectory != logical.as_posix()
        ):
            raise ValueError("Installed distribution subdirectory is unsafe")
        sanitized["subdirectory"] = subdirectory
    return sanitized


def _validated_installer(value: Any) -> str | None:
    if value is not None and (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value) is None
    ):
        raise ValueError("Installed distribution has invalid installer metadata")
    return value


def _record_sha256_and_size(
    name: str,
    declared_hash: str,
    declared_size: str,
) -> tuple[str, int]:
    try:
        algorithm, encoded_digest = declared_hash.split("=", 1)
        padding = "=" * (-len(encoded_digest) % 4)
        declared_digest = base64.urlsafe_b64decode(encoded_digest + padding).hex()
        size = int(declared_size)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Installed distribution {name} has malformed RECORD hashes"
        ) from exc
    if algorithm != "sha256" or _SHA256_PATTERN.fullmatch(declared_digest) is None:
        raise ValueError(
            f"Installed distribution {name} does not use SHA-256 RECORD hashes"
        )
    if size < 0:
        raise ValueError(f"Installed distribution {name} has a RECORD size mismatch")
    return declared_digest, size


class _InstalledRecordRow(NamedTuple):
    logical_path: str
    logical: PurePosixPath
    declared_hash: str
    declared_size: str
    installed_path: Path
    installed_logical_path: str
    installed_within_distribution: bool


def _installed_distribution_entry(
    distribution: Any,
    *,
    statistics: dict[str, int] | None = None,
) -> dict[str, Any]:
    raw_name = distribution.metadata.get("Name")
    version = distribution.version
    if not isinstance(raw_name, str) or not isinstance(version, str) or not version:
        raise ValueError("Installed distribution lacks a name or version")
    name = _normalized_distribution_name(raw_name)
    record_text = distribution.read_text("RECORD")
    if not isinstance(record_text, str) or not record_text:
        raise ValueError(f"Installed distribution {name} lacks a RECORD manifest")
    rows = list(csv.reader(io.StringIO(record_text)))
    if not rows:
        raise ValueError(f"Installed distribution {name} has an empty RECORD manifest")
    distribution_root = Path(distribution.locate_file("")).resolve()
    environment_roots: list[Path] = []
    cursor = distribution_root
    while cursor != cursor.parent:
        if cursor.name.casefold() in {"lib", "lib64"}:
            environment_roots.append(cursor.parent)
            break
        cursor = cursor.parent
    environment_roots.append(Path(sys.prefix).resolve())
    environment_roots = list(dict.fromkeys(environment_roots))
    rows_by_installed_path: dict[Path, list[_InstalledRecordRow]] = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"Installed distribution {name} has malformed RECORD data")
        logical_path = row[0].replace("\\", "/")
        logical = PurePosixPath(logical_path)
        if (
            not logical_path
            or (logical.parts and ":" in logical.parts[0])
            or "\x00" in logical_path
        ):
            raise ValueError(f"Installed distribution {name} has unsafe RECORD data")
        declared_hash, declared_size = row[1], row[2]
        installed_path = Path(distribution.locate_file(logical_path)).resolve()
        installed_within_distribution = True
        try:
            installed_relative = installed_path.relative_to(distribution_root)
            installed_logical_path = (
                PurePosixPath("distribution") / PurePosixPath(installed_relative.as_posix())
            ).as_posix()
        except ValueError:
            installed_within_distribution = False
            installed_logical_path = ""
            for environment_root in environment_roots:
                try:
                    installed_relative = installed_path.relative_to(environment_root)
                except ValueError:
                    continue
                installed_logical_path = (
                    PurePosixPath("environment")
                    / PurePosixPath(installed_relative.as_posix())
                ).as_posix()
                break
            if not installed_logical_path:
                raise ValueError(
                    f"Installed distribution {name} has unsafe RECORD data"
                )
        resolved_row = _InstalledRecordRow(
            logical_path=logical_path,
            logical=logical,
            declared_hash=declared_hash,
            declared_size=declared_size,
            installed_path=installed_path,
            installed_logical_path=installed_logical_path,
            installed_within_distribution=installed_within_distribution,
        )
        rows_by_installed_path.setdefault(installed_path, []).append(resolved_row)

    def is_generated_bytecode(row: _InstalledRecordRow) -> bool:
        return (
            row.installed_path.parent.name == "__pycache__"
            and row.installed_path.suffix == ".pyc"
            and "__pycache__" in row.logical.parts
        )

    # pip can recompile a wheel-supplied .pyc and add an unhashed generated-file
    # row without removing the wheel's now-stale hashed row. Recognize exactly
    # that canonical two-row shape, then hash the actual installed bytecode.
    # Every other duplicate target remains invalid, including path aliases.
    regenerated_bytecode_paths: set[Path] = set()
    for installed_path, matching_rows in rows_by_installed_path.items():
        if len(matching_rows) == 1:
            continue
        unattested_rows = [
            row
            for row in matching_rows
            if not row.declared_hash and not row.declared_size
        ]
        attested_rows = [
            row
            for row in matching_rows
            if row.declared_hash and row.declared_size
        ]
        if (
            len(matching_rows) != 2
            or len(unattested_rows) != 1
            or len(attested_rows) != 1
            or not all(is_generated_bytecode(row) for row in matching_rows)
            or not all(row.installed_within_distribution for row in matching_rows)
        ):
            raise ValueError(f"Installed distribution {name} has unsafe RECORD data")
        _record_sha256_and_size(
            name,
            attested_rows[0].declared_hash,
            attested_rows[0].declared_size,
        )
        regenerated_bytecode_paths.add(installed_path)

    def require_attested_bytecode_source(row: _InstalledRecordRow) -> None:
        match = re.fullmatch(
            r"(?P<source>.+?)\.[^.]+(?:\.opt-[0-9]+)?\.pyc",
            row.installed_path.name,
        )
        if match is None or not is_generated_bytecode(row):
            raise ValueError(
                f"Installed distribution {name} has unsafe generated bytecode data"
            )
        source_path = (
            row.installed_path.parent.parent / f"{match.group('source')}.py"
        ).resolve()
        try:
            source_path.relative_to(distribution_root)
        except ValueError as exc:
            raise ValueError(
                f"Installed distribution {name} has unsafe generated bytecode source"
            ) from exc
        source_rows = rows_by_installed_path.get(source_path, [])
        if (
            len(source_rows) != 1
            or not source_rows[0].installed_within_distribution
            or not source_rows[0].declared_hash
            or not source_rows[0].declared_size
        ):
            raise ValueError(
                f"Installed distribution {name} has generated bytecode without "
                "an attested source"
            )
        _record_sha256_and_size(
            name,
            source_rows[0].declared_hash,
            source_rows[0].declared_size,
        )

    files: list[dict[str, Any]] = []
    for installed_path, matching_rows in rows_by_installed_path.items():
        row = matching_rows[0]
        is_regenerated_bytecode = installed_path in regenerated_bytecode_paths
        is_unattested_bytecode = (
            len(matching_rows) == 1
            and not row.declared_hash
            and not row.declared_size
            and is_generated_bytecode(row)
        )
        if is_regenerated_bytecode or is_unattested_bytecode:
            if not row.installed_within_distribution:
                raise ValueError(
                    f"Installed distribution {name} has unsafe generated bytecode data"
                )
            require_attested_bytecode_source(row)
            if not installed_path.is_file():
                raise ValueError(
                    f"Installed distribution {name} is missing generated bytecode"
                )
            size = installed_path.stat().st_size
            declared_digest = file_sha256(installed_path)
        elif len(matching_rows) == 1 and not row.declared_hash:
            is_self_record = row.logical_path.endswith(".dist-info/RECORD")
            if not is_self_record or row.declared_size:
                raise ValueError(
                    f"Installed distribution {name} has an unattested RECORD file"
                )
            if not row.installed_within_distribution or not installed_path.is_file():
                raise ValueError(
                    f"Installed distribution {name} has unsafe excluded RECORD data"
                )
            continue
        else:
            if len(matching_rows) != 1:
                raise ValueError(
                    f"Installed distribution {name} has duplicate installed RECORD data"
                )
            if not installed_path.is_file():
                raise ValueError(
                    f"Installed distribution {name} is missing RECORD file "
                    f"{row.logical_path}"
                )
            declared_digest, size = _record_sha256_and_size(
                name,
                row.declared_hash,
                row.declared_size,
            )
            if installed_path.stat().st_size != size:
                raise ValueError(
                    f"Installed distribution {name} has a RECORD size mismatch"
                )
            if file_sha256(installed_path) != declared_digest:
                raise ValueError(
                    f"Installed distribution {name} has a RECORD content mismatch"
                )
        files.append(
            {
                "path": row.installed_logical_path,
                "size": size,
                "sha256": declared_digest,
            }
        )
    files.sort(key=lambda item: item["path"])
    if statistics is not None:
        statistics["installedFileCount"] = statistics.get(
            "installedFileCount", 0
        ) + len(files)
        statistics["totalHashedBytes"] = statistics.get(
            "totalHashedBytes", 0
        ) + sum(item["size"] for item in files)
    installer = _validated_installer(
        (distribution.read_text("INSTALLER") or "").strip() or None
    )
    payload = {
        "name": name,
        "version": version,
        "directURL": _normalized_direct_url(
            distribution.read_text("direct_url.json")
        ),
        "installer": installer,
        "recordSHA256": hashlib.sha256(record_text.encode("utf-8")).hexdigest(),
        "installedFileCount": len(files),
        "installedContentSHA256": canonical_sha256(files),
    }
    return {**payload, "distributionSHA256": canonical_sha256(payload)}


def build_resolved_training_environment(
    distributions: Any | None = None,
) -> dict[str, Any]:
    """Attest every installed distribution and all files declared by its RECORD.

    This complements, rather than replaces, the direct requirements lock. The
    owning runtime computes it before training so transitive wheels and
    platform-provided distributions participate in resume and comparison
    lineage; ZeroGPU reuses an authenticated startup scan outside the GPU lease.
    """

    environment, _ = build_resolved_training_environment_snapshot(distributions)
    return environment


def build_resolved_training_environment_snapshot(
    distributions: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one complete environment attestation plus non-secret scan metrics."""

    started = time.perf_counter()
    installed = list(
        importlib_metadata.distributions() if distributions is None else distributions
    )
    statistics = {"installedFileCount": 0, "totalHashedBytes": 0}
    entries = [
        _installed_distribution_entry(item, statistics=statistics)
        for item in installed
    ]
    entries.sort(key=lambda item: item["name"])
    names = [item["name"] for item in entries]
    if not entries or len(names) != len(set(names)):
        raise ValueError("Resolved training environment has missing or duplicate distributions")
    payload = {
        "schemaVersion": RESOLVED_TRAINING_ENVIRONMENT_SCHEMA_VERSION,
        "recordPolicy": RESOLVED_TRAINING_ENVIRONMENT_RECORD_POLICY,
        "distributions": entries,
    }
    environment = {
        **payload,
        "resolvedTrainingEnvironmentSHA256": canonical_sha256(payload),
    }
    scan = {
        "schemaVersion": RESOLVED_TRAINING_ENVIRONMENT_CACHE_SCHEMA_VERSION,
        "resolvedTrainingEnvironmentSHA256": environment[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "durationMilliseconds": max(
            0,
            int(round((time.perf_counter() - started) * 1000)),
        ),
        "distributionCount": len(entries),
        "installedFileCount": statistics["installedFileCount"],
        "totalHashedBytes": statistics["totalHashedBytes"],
    }
    return environment, scan


def sign_resolved_training_environment_cache(
    environment: Mapping[str, Any],
    scan: Mapping[str, Any],
    *,
    key: bytes,
    startup_id: str,
) -> dict[str, Any]:
    """Authenticate a Space-startup scan for child trainer processes.

    The HMAC key is process-local and must be inherited only by trainer
    subprocesses. It is never written into configs, reports, or summaries.
    """

    if len(key) < 32:
        raise ValueError("Resolved-environment cache key must contain 32 bytes")
    digest = verify_resolved_training_environment(environment)
    if (
        not isinstance(startup_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", startup_id) is None
        or set(scan) != {
            "schemaVersion",
            "resolvedTrainingEnvironmentSHA256",
            "durationMilliseconds",
            "distributionCount",
            "installedFileCount",
            "totalHashedBytes",
        }
        or scan.get("schemaVersion")
        != RESOLVED_TRAINING_ENVIRONMENT_CACHE_SCHEMA_VERSION
        or scan.get("resolvedTrainingEnvironmentSHA256") != digest
        or any(
            type(scan.get(field)) is not int or scan[field] < 0
            for field in (
                "durationMilliseconds",
                "distributionCount",
                "installedFileCount",
                "totalHashedBytes",
            )
        )
        or scan.get("distributionCount") != len(environment["distributions"])
    ):
        raise ValueError("Invalid resolved-environment startup scan")
    payload = {
        "schemaVersion": RESOLVED_TRAINING_ENVIRONMENT_CACHE_SCHEMA_VERSION,
        "verificationMode": "space_startup_full_scan",
        "startupID": startup_id,
        "resolvedTrainingEnvironmentSHA256": digest,
        "scan": dict(scan),
    }
    signature = hmac.new(
        key,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {**payload, "cacheHMACSHA256": signature}


def verify_resolved_training_environment_cache(
    environment: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    key: bytes,
) -> dict[str, Any]:
    if set(attestation) != {
        "schemaVersion",
        "verificationMode",
        "startupID",
        "resolvedTrainingEnvironmentSHA256",
        "scan",
        "cacheHMACSHA256",
    }:
        raise ValueError("Invalid resolved-environment cache attestation")
    expected = sign_resolved_training_environment_cache(
        environment,
        attestation.get("scan") if isinstance(attestation.get("scan"), Mapping) else {},
        key=key,
        startup_id=str(attestation.get("startupID") or ""),
    )
    if (
        attestation.get("schemaVersion") != expected["schemaVersion"]
        or attestation.get("verificationMode") != expected["verificationMode"]
        or attestation.get("resolvedTrainingEnvironmentSHA256")
        != expected["resolvedTrainingEnvironmentSHA256"]
        or not isinstance(attestation.get("cacheHMACSHA256"), str)
        or not hmac.compare_digest(
            attestation["cacheHMACSHA256"],
            expected["cacheHMACSHA256"],
        )
    ):
        raise ValueError("Resolved-environment cache attestation is not authentic")
    return dict(expected["scan"])


def verify_resolved_training_environment(
    environment: Mapping[str, Any],
    *,
    distributions: Any | None = None,
    verify_installed: bool = False,
) -> str:
    if set(environment) != {
        "schemaVersion",
        "recordPolicy",
        "distributions",
        "resolvedTrainingEnvironmentSHA256",
    } or environment.get("schemaVersion") != RESOLVED_TRAINING_ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("Invalid resolved training environment")
    if environment.get("recordPolicy") != RESOLVED_TRAINING_ENVIRONMENT_RECORD_POLICY:
        raise ValueError("Invalid resolved training environment RECORD policy")
    entries = environment.get("distributions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Resolved training environment must contain distributions")
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "name",
            "version",
            "directURL",
            "installer",
            "recordSHA256",
            "installedFileCount",
            "installedContentSHA256",
            "distributionSHA256",
        }:
            raise ValueError("Invalid resolved distribution entry")
        unsigned = {
            key: value for key, value in entry.items() if key != "distributionSHA256"
        }
        direct_url = entry.get("directURL")
        normalized_direct_url = _normalized_direct_url(direct_url)
        installer = _validated_installer(entry.get("installer"))
        if (
            not isinstance(entry.get("name"), str)
            or entry["name"] != _normalized_distribution_name(entry["name"])
            or not isinstance(entry.get("version"), str)
            or not entry["version"]
            or normalized_direct_url != direct_url
            or installer != entry.get("installer")
            or not isinstance(entry.get("recordSHA256"), str)
            or _SHA256_PATTERN.fullmatch(entry["recordSHA256"]) is None
            or type(entry.get("installedFileCount")) is not int
            or entry["installedFileCount"] <= 0
            or not isinstance(entry.get("installedContentSHA256"), str)
            or _SHA256_PATTERN.fullmatch(entry["installedContentSHA256"]) is None
            or entry.get("distributionSHA256") != canonical_sha256(unsigned)
        ):
            raise ValueError("Invalid resolved distribution identity")
        names.append(entry["name"])
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("Resolved distributions must be uniquely name-sorted")
    payload = {
        "schemaVersion": RESOLVED_TRAINING_ENVIRONMENT_SCHEMA_VERSION,
        "recordPolicy": RESOLVED_TRAINING_ENVIRONMENT_RECORD_POLICY,
        "distributions": entries,
    }
    digest = canonical_sha256(payload)
    if environment.get("resolvedTrainingEnvironmentSHA256") != digest:
        raise ValueError(
            "resolvedTrainingEnvironmentSHA256 does not match the distribution manifest"
        )
    if verify_installed:
        rebuilt = build_resolved_training_environment(distributions)
        if rebuilt != dict(environment):
            raise ValueError("Installed resolved training environment drifted")
    return digest


def validate_runtime_source(*, kind: Any, revision: Any) -> tuple[str, str]:
    if kind not in {"git", "huggingface_space"}:
        raise ValueError("runtimeSourceKind must be git or huggingface_space")
    if not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("runtimeSourceRevision must be a full lowercase commit SHA")
    return kind, revision


def validate_runtime_source_audit(
    value: Mapping[str, Any],
    *,
    observed_local_revision: str | None = None,
) -> dict[str, Any]:
    """Validate runtime-source audit semantics without overstating trust.

    A local Git source is accepted only when the caller independently observed
    the checkout's current HEAD and it agrees with every recorded revision.
    Hugging Face Space repository-head evidence remains supplemental: the Hub
    does not attest that the matching repository revision is the executing
    container revision, so that contract can never claim a verified binding.
    """

    if not isinstance(value, Mapping):
        raise ValueError("Runtime source audit must be an object")
    kind, expected_revision = validate_runtime_source(
        kind=value.get("runtimeSourceKind"),
        revision=value.get("expectedRuntimeSourceRevision"),
    )
    if value.get("runtimeSourceRevision") != expected_revision:
        raise ValueError("runtimeSourceRevision must equal the expected revision")

    observed_repository_revision = value.get("observedRepositoryRevision")
    observed_runtime_revision = value.get("observedRuntimeRevision")
    for label, revision in (
        ("observedRepositoryRevision", observed_repository_revision),
        ("observedRuntimeRevision", observed_runtime_revision),
    ):
        if revision is not None and (
            not isinstance(revision, str)
            or _REVISION_PATTERN.fullmatch(revision) is None
        ):
            raise ValueError(f"{label} must be null or a full lowercase commit SHA")

    status = value.get("runtimeSourceBindingStatus")
    method = value.get("runtimeSourceBindingMethod")
    if kind == "huggingface_space":
        if observed_runtime_revision is not None:
            raise ValueError(
                "Hugging Face Space runtime revision must remain unobserved without "
                "platform attestation"
            )
        if status != RUNTIME_SOURCE_BINDING_SPACE_UNVERIFIED:
            raise ValueError(
                "Hugging Face Space runtime source binding must remain "
                "operator_declared_unverified"
            )
        if method == RUNTIME_SOURCE_BINDING_SPACE_DECLARATION:
            if observed_repository_revision is not None:
                raise ValueError(
                    "Operator-declared Space source cannot include repository-head "
                    "evidence"
                )
        elif method == RUNTIME_SOURCE_BINDING_SPACE_REPOSITORY_HEAD:
            if observed_repository_revision != expected_revision:
                raise ValueError(
                    "Supplemental Space repository head must equal the expected "
                    "revision"
                )
        else:
            raise ValueError("Unsupported Hugging Face Space source binding method")
    else:
        if (
            not isinstance(observed_local_revision, str)
            or _REVISION_PATTERN.fullmatch(observed_local_revision) is None
        ):
            raise ValueError(
                "Local Git runtime source requires an independently observed HEAD"
            )
        if observed_local_revision != expected_revision:
            raise ValueError(
                "Observed local Git HEAD does not match the expected runtime source"
            )
        if (
            observed_repository_revision != observed_local_revision
            or observed_runtime_revision != observed_local_revision
        ):
            raise ValueError(
                "Local runtime-source observations must equal the current Git HEAD"
            )
        if (status, method) not in {
            (
                RUNTIME_SOURCE_BINDING_LOCAL,
                RUNTIME_SOURCE_BINDING_LOCAL_METHOD,
            ),
            (
                RUNTIME_SOURCE_BINDING_ATTESTED,
                RUNTIME_SOURCE_BINDING_ATTESTED_METHOD,
            ),
        }:
            raise ValueError("Unsupported local Git runtime-source binding evidence")

    return {
        field: value.get(field)
        for field in RUNTIME_SOURCE_AUDIT_FIELDS
    }
