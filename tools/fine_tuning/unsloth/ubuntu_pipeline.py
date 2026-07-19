from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

try:
    from lumen_manifest_crawler.dataset.chat_template_contract import (
        PINNED_QWEN3_CHAT_TEMPLATE_SHA256,
    )
except ImportError:  # pragma: no cover - repository-root module execution.
    from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.chat_template_contract import (
        PINNED_QWEN3_CHAT_TEMPLATE_SHA256,
    )

try:
    from lumen_manifest_crawler.dataset.optimization_policy import (
        EXPERIMENT_VARIANT_SCHEMA_VERSION,
        VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS as _SHARED_VARIANT_OPTIMIZATION_CONFIG_FIELDS,
        invariant_training_config as _normalized_invariant_training_config,
    )
except ImportError:  # pragma: no cover - repository-root module execution.
    from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.optimization_policy import (
        EXPERIMENT_VARIANT_SCHEMA_VERSION,
        VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS as _SHARED_VARIANT_OPTIMIZATION_CONFIG_FIELDS,
        invariant_training_config as _normalized_invariant_training_config,
    )

from tools.fine_tuning.unsloth.training_lineage import (
    DEFAULT_LLAMA_CPP_REVISION,
    TRAINING_VARIANT_ATTESTATION_SCHEMA,
)
from tools.fine_tuning.unsloth.ubuntu_source_integrity import (
    SOURCE_INTEGRITY_ENV,
    attest_repository,
    load_verified_attestation,
    validate_attestation_record,
)


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
VARIANTS = (
    "internal_only",
    "internal_plus_public_baseline",
    "internal_plus_public_optimized",
)
DATASET_FILES = (
    "train_sft.jsonl",
    "val_sft.jsonl",
    "train_dpo.jsonl",
    "val_dpo.jsonl",
)
PREPARED_INPUT_MAX_ENTRIES = 10_000
PREPARED_INPUT_MAX_DEPTH = 64
PREPARED_INPUT_MAX_LOGICAL_BYTES = 8 * 1024 * 1024 * 1024
PREPARED_INPUT_FD_RESERVE = 32
RUNTIME_SOURCE_FIELDS = (
    "runtimeSourceKind",
    "runtimeSourceRevision",
    "expectedRuntimeSourceRevision",
    "observedRepositoryRevision",
    "observedRuntimeRevision",
    "runtimeSourceBindingStatus",
    "runtimeSourceBindingMethod",
)
UBUNTU_SOURCE_INTEGRITY_FIELDS = (
    "workingTreeDigest",
    "ubuntuOrchestrationCodeSHA256",
    "ubuntuSourceIntegritySHA256",
    "ubuntuSourceIntegrity",
)
PRIVATE_TOKENIZER_SNAPSHOT_CONFIG_PROOF_FIELDS = (
    ("base_model_name", "baseModelID"),
    ("baseModelID", "baseModelID"),
    ("baseModelRevision", "baseModelRevision"),
    ("baseModelTokenizerDigest", "baseModelTokenizerDigest"),
    ("baseModelTokenizerFiles", "baseModelTokenizerFiles"),
    ("baseModelTokenizerClosureSHA256", "baseModelTokenizerClosureSHA256"),
    ("baseModelTokenizerSnapshotPath", "snapshotPath"),
)
PRIVATE_RUNTIME_SNAPSHOT_CONFIG_PROOF_FIELDS = (
    ("base_model_name", "baseModelID"),
    ("baseModelID", "baseModelID"),
    ("baseModelRevision", "baseModelRevision"),
    ("baseModelIndexDigest", "baseModelIndexDigest"),
    (
        "baseModelIndexReferencedShardNames",
        "baseModelIndexReferencedShardNames",
    ),
    (
        "baseModelIndexShardBindingSHA256",
        "baseModelIndexShardBindingSHA256",
    ),
    ("baseModelArtifactDigest", "baseModelArtifactDigest"),
    ("baseModelWeightShards", "baseModelWeightShards"),
    ("baseModelGenerationConfigFile", "baseModelGenerationConfigFile"),
    ("baseModelTokenizerDigest", "baseModelTokenizerDigest"),
    ("baseModelTokenizerFiles", "baseModelTokenizerFiles"),
    ("baseModelTokenizerClosureSHA256", "baseModelTokenizerClosureSHA256"),
    ("baseModelRuntimeSnapshotPath", "snapshotPath"),
)
RUNTIME_CONFIG_FIELDS = {
    "trainingContainerImageDigest",
    "trainingContainerImageDigestSource",
    "trainingRuntimeImageBindingStatus",
    "trainingRuntimeImageBindingVerified",
    "trainingEnvironmentSHA256",
    "resolvedTrainingEnvironment",
    "resolvedTrainingEnvironmentSHA256",
    "resolvedTrainingEnvironmentScanAudit",
    "resolvedTrainingEnvironmentCacheAttestation",
    "spaceConfigurationSHA256",
    "zeroGPUSize",
    "zeroGPUDurationSeconds",
    "observedAccelerator",
    *RUNTIME_SOURCE_FIELDS,
    *UBUNTU_SOURCE_INTEGRITY_FIELDS,
}
NON_CONTROLLED_CONFIG_FIELDS = {
    "adapterExport",
    "adapter_gguf_output_path",
    "adapter_output_dir",
    "dataset_dir",
    "dpo_output_dir",
    "gguf_output_dir",
    "gguf_repo_id",
    "mergeExport",
    "output_dir",
    "runExecutionPlan",
    "variant",
    "variantAttestation",
    "variantManifestSHA256",
    *RUNTIME_CONFIG_FIELDS,
}
VARIANT_OPTIMIZATION_CONFIG_FIELDS = (
    _SHARED_VARIANT_OPTIMIZATION_CONFIG_FIELDS
)
_MINIMUM_EFFECTIVE_STEPS_BY_LANE = {
    "sft": {
        "executor": 40,
        "mouth": 24,
        "mimicry": 20,
        "rem": 20,
        "fleet": 24,
    },
    "dpo": {
        "executor": 8,
        "mouth": 9,
        "mimicry": 8,
        "rem": 8,
        "fleet": 8,
    },
}
_MAXIMUM_NON_CORTEX_EPOCHS = 8
GGUF_FIXED_HEADER_SIZE = 24
GGUF_SUPPORTED_VERSIONS = frozenset({2, 3})
GGUF_CONVERTER_RELATIVE_PATH = Path("convert_lora_to_gguf.py")
GGUF_READER_RELATIVE_PATH = Path("gguf-py/gguf/scripts/gguf_dump.py")
GGUF_READER_TIMEOUT_SECONDS = 120
EXPECTED_ADAPTER_GGUF_ARCHITECTURE = "qwen3"
EXPECTED_ADAPTER_GGUF_GENERAL_TYPE = "adapter"
EXPECTED_ADAPTER_GGUF_ADAPTER_TYPE = "lora"
EXPECTED_ADAPTER_GGUF_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
EXPECTED_ADAPTER_GGUF_BASE_MODEL_REPO_URL = (
    f"https://huggingface.co/{EXPECTED_ADAPTER_GGUF_BASE_MODEL_ID}"
)
ADAPTER_GGUF_SEMANTIC_FIELDS = (
    "adapterGGUFArchitecture",
    "adapterGGUFType",
    "adapterGGUFAdapterType",
    "adapterGGUFBaseModelID",
    "adapterGGUFBaseModelRepoURL",
    "adapterGGUFChatTemplateSource",
    "adapterGGUFChatTemplateSHA256",
)
GGUF_CONVERSION_RECEIPT_SCHEMA_VERSION = (
    "lumen.gguf-conversion-receipt/1.3.0"
)
GGUF_BASE_SNAPSHOT_VERIFICATION_FILENAME = (
    "base_model_conversion_snapshot_verification.json"
)
GGUF_CONVERSION_QUALIFICATION = "attested_converter_execution"
GGUF_TENSOR_EQUIVALENCE_STATUS = "not_independently_verified"
GGUF_CONVERSION_SUMMARY_FIELDS = (
    "adapterGGUFConversionReceipt",
    "adapterGGUFConversionReceiptSHA256",
    "adapterGGUFConversionQualification",
    "adapterGGUFTensorEquivalenceStatus",
    "adapterGGUFRuntimeModelBindingSHA256",
    "adapterGGUFRuntimeTokenizerBindingSHA256",
)
GGUF_CONVERSION_RECEIPT_FIELDS = (
    "schema",
    "agent",
    "qualification",
    "tensorEquivalenceStatus",
    "adapterGGUF",
    "adapterGGUFSHA256",
    "adapterGGUFSizeBytes",
    *ADAPTER_GGUF_SEMANTIC_FIELDS,
    "preferenceAdapter",
    "preferenceAdapterSHA256",
    "preferenceAdapterManifestFileSHA256",
    "preferenceFinalizedVariantManifest",
    "preferenceFinalizedVariantManifestSHA256",
    "preferenceFinalizedVariantManifestFileSHA256",
    "runtimeModelBindingSHA256",
    "runtimeTokenizerBindingSHA256",
    "config",
    "configSHA256",
    "baseModelID",
    "baseModelRevision",
    "baseModelIndexDigest",
    "baseModelIndexShardBindingSHA256",
    "baseModelArtifactDigest",
    "baseModelTokenizerDigest",
    "baseModelTokenizerFiles",
    "baseModelTokenizerClosureSHA256",
    "baseModelTokenizerSnapshotPath",
    "baseModelTokenizerSnapshotVerification",
    "baseModelConversionSnapshotVerification",
    "trainingContainerImageDigest",
    "ubuntuOrchestrationCodeSHA256",
    "ubuntuSourceIntegritySHA256",
    "llamaCppRevision",
    "converterPath",
    "converterGitBlobSHA1",
    "converterFileSHA256",
    "readerPath",
    "readerGitBlobSHA1",
    "readerFileSHA256",
    "conversionReceiptSHA256",
)
SUMMARY_SCHEMA_VERSION = "lumen.ubuntu-training-summary/3.6.0"
UPLOAD_SCHEMA_VERSION = "lumen.ubuntu-training-upload/2.7.0"
UPLOAD_INTENT_SCHEMA_VERSION = "lumen.ubuntu-upload-intent/1.4.0"
UPLOAD_ATTEMPT_SCHEMA_VERSION = "lumen.ubuntu-upload-attempt/1.0.0"
UPLOAD_COMMIT_SCHEMA_VERSION = "lumen.ubuntu-upload-commit/1.0.0"
UPLOAD_INTENT_FILENAME = ".lumen-upload-intent.json"
UPLOAD_ATTEMPT_FILENAME = ".lumen-upload-attempt.json"
UPLOAD_COMMIT_FILENAME = ".lumen-upload-commit.json"
UPLOAD_REMOTE_MARKER_FILENAME = ".lumen-upload-intent.json"
PREPARATION_OWNER_SCHEMA_VERSION = "lumen.ubuntu-preparation-owner/1.1.0"
PREPARATION_OWNER_FILENAME = ".lumen-preparation-owner.json"
EXECUTION_PLAN_SCHEMA_VERSION = "lumen.ubuntu-training-execution-plan/1.0.0"
RUN_SCHEMA_VERSION = "lumen.ubuntu-training-run/3.3.0"
PHASE_RUNTIME_EVIDENCE_FIELDS = (
    "trainingReportFileSHA256",
    "runtimeModelBindingSHA256",
    "runtimeTokenizerBindingSHA256",
    "peftBaseModelIdentitySHA256",
    "adapterTokenizerBindingSHA256",
    "baseModelTokenizerSnapshotVerificationSHA256",
    "baseModelRuntimeSnapshotVerificationSHA256",
)
RUNTIME_BINDING_SMOKE_SUMMARY_FIELDS = (
    "runtimeBindingSmokeReport",
    "runtimeBindingSmokeReportFileSHA256",
    "runtimeBindingSmokeGateSHA256",
    "runtimeBindingSmokeContractEvidence",
    "runtimeBindingSmokeBindingsByAgent",
)
_GGUF_READER_FD_BOOTSTRAP = """
import os
import sys

reader_fd = int(sys.argv[1])
reader_path = sys.argv[2]
artifact_path = sys.argv[3]
chunks = []
offset = 0
while True:
    chunk = os.pread(reader_fd, 1 << 20, offset)
    if not chunk:
        break
    chunks.append(chunk)
    offset += len(chunk)
source = b"".join(chunks)
sys.argv = [reader_path, artifact_path, "--json"]
sys.path[0] = os.path.dirname(os.path.abspath(reader_path))
namespace = {
    "__name__": "__main__",
    "__file__": reader_path,
    "__package__": None,
    "__cached__": None,
}
exec(compile(source, reader_path, "exec"), namespace, namespace)
"""


@dataclass(frozen=True)
class _VerifiedGGUFReaderScript:
    path: Path
    git_blob_sha1: str
    file_sha256: str


@dataclass(frozen=True)
class _UploadInputContract:
    relative_path: str
    remote_path: str
    expected_sha256: str | None = None
    expected_size: int | None = None
    expected_json: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _SnapshottedUploadInput:
    path: Path
    remote_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _PreparedInputDirectory:
    relative_path: str
    descriptor: int
    signature: tuple[int, ...]


@dataclass(frozen=True)
class _PreparedInputFile:
    relative_path: str
    descriptor: int
    signature: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class _MountIdentity:
    mount_id: int
    parent_id: int
    device: str
    root: str
    mount_point: str
    mount_options: tuple[str, ...]
    filesystem_type: str
    mount_source: str
    super_options: tuple[str, ...]


class _PreparedInputClosure:
    """Retained, recursively attested prepare-only run inputs."""

    def __init__(
        self,
        *,
        run_root: Path,
        directories: Sequence[_PreparedInputDirectory],
        files: Sequence[_PreparedInputFile],
        require_exact_readonly_mount: bool,
        mount_identity: _MountIdentity | None,
    ) -> None:
        self.run_root = run_root
        self.directories = tuple(directories)
        self.files = tuple(files)
        self.require_exact_readonly_mount = require_exact_readonly_mount
        self.mount_identity = mount_identity
        self.inventory = tuple(
            sorted(
                (
                    *(
                        {
                            "path": item.relative_path,
                            "kind": "directory",
                            "signature": list(item.signature),
                        }
                        for item in self.directories
                    ),
                    *(
                        {
                            "path": item.relative_path,
                            "kind": "file",
                            "signature": list(item.signature),
                            "sha256": item.sha256,
                        }
                        for item in self.files
                    ),
                ),
                key=lambda item: (item["path"], item["kind"]),
            )
        )
        self.inventory_sha256 = canonical_sha256(self.inventory)
        self._closed = False

    def _verify_mount_boundary(self) -> None:
        nested = _mounted_descendants(self.run_root)
        if nested:
            raise RuntimeError(
                "Prepare-only input closure contains nested mounts: "
                + ", ".join(str(item) for item in sorted(nested))
            )
        if not self.require_exact_readonly_mount:
            return
        observed = _exact_readonly_mount_identity(self.run_root)
        if observed != self.mount_identity:
            raise RuntimeError(
                "Prepare-only input read-only mount identity changed during "
                "verification"
            )

    @property
    def mount_identity_sha256(self) -> str | None:
        if self.mount_identity is None:
            return None
        return canonical_sha256(_mount_identity_payload(self.mount_identity))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for item in reversed((*self.files, *self.directories)):
            try:
                os.close(item.descriptor)
            except OSError:
                pass

    def _verify_retained_descriptors(self) -> None:
        if self._closed:
            raise RuntimeError("Prepare-only input closure is already closed")
        for item in self.directories:
            try:
                observed = _file_stability_signature(os.fstat(item.descriptor))
            except OSError as exc:
                raise RuntimeError(
                    "Prepare-only input directory descriptor became unavailable: "
                    f"{item.relative_path}"
                ) from exc
            if observed != item.signature:
                raise RuntimeError(
                    "Prepare-only input directory changed during verification: "
                    f"{item.relative_path}"
                )
        for item in self.files:
            try:
                observed = _file_stability_signature(os.fstat(item.descriptor))
                observed_sha256 = _descriptor_sha256(item.descriptor)
                stable = _file_stability_signature(os.fstat(item.descriptor))
            except OSError as exc:
                raise RuntimeError(
                    "Prepare-only input file descriptor became unavailable: "
                    f"{item.relative_path}"
                ) from exc
            if (
                observed != item.signature
                or stable != item.signature
                or observed_sha256 != item.sha256
            ):
                raise RuntimeError(
                    "Prepare-only input file changed during verification: "
                    f"{item.relative_path}"
                )

    def verify_unchanged(self) -> None:
        """Fail if any retained input or its current path binding changed."""

        self._verify_mount_boundary()
        _require_prepared_input_fd_headroom(
            len(self.directories) + len(self.files) + PREPARED_INPUT_FD_RESERVE
        )
        self._verify_retained_descriptors()
        observed = _acquire_prepared_input_closure_impl(
            self.run_root,
            require_exact_readonly_mount=self.require_exact_readonly_mount,
        )
        try:
            observed._verify_retained_descriptors()
            if (
                observed.inventory_sha256 != self.inventory_sha256
                or observed.inventory != self.inventory
            ):
                raise RuntimeError(
                    "Prepare-only input inventory or path binding changed during "
                    "verification"
                )
            # Recheck both descriptor sets after the path walk so a mutation
            # during the comparison cannot be accepted as a stable closure.
            observed._verify_retained_descriptors()
            self._verify_retained_descriptors()
            observed._verify_mount_boundary()
            self._verify_mount_boundary()
        finally:
            observed.close()


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key is not allowed: {key}")
        value[key] = item
    return value


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_nonfinite_json_constant(value)
    return parsed


def _file_stability_signature(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_descriptor_bytes(handle: BinaryIO) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        chunk = os.pread(handle.fileno(), 1 << 20, offset)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1 << 20, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _open_regular_readonly(path: Path, *, label: str) -> tuple[BinaryIO, os.stat_result]:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeError(f"{label} verification requires O_NOFOLLOW support")
    try:
        descriptor = os.open(path, flags | nofollow)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"{label} must be a regular file: {path}")
        return os.fdopen(descriptor, "rb", closefd=True), file_stat
    except BaseException:
        os.close(descriptor)
        raise


def _require_stable_descriptor(
    handle: BinaryIO,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    if _file_stability_signature(os.fstat(handle.fileno())) != _file_stability_signature(
        expected
    ):
        raise RuntimeError(f"{label} changed while it was being verified")


def _require_path_matches_descriptor(
    path: Path,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"{label} changed while it was being verified") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != expected.st_dev
        or current.st_ino != expected.st_ino
    ):
        raise RuntimeError(f"{label} changed while it was being verified")


def _open_descriptor_count() -> int:
    descriptor_root = Path("/proc/self/fd")
    if descriptor_root.is_dir():
        try:
            return len(os.listdir(descriptor_root))
        except OSError:
            pass
    return PREPARED_INPUT_FD_RESERVE


def _require_prepared_input_fd_headroom(additional_descriptors: int) -> None:
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        return
    observed = _open_descriptor_count()
    if observed + additional_descriptors > soft_limit:
        raise RuntimeError(
            "Prepare-only input closure lacks file-descriptor headroom: "
            f"open={observed}, additionalRequired={additional_descriptors}, "
            f"softLimit={soft_limit}"
        )


def _acquire_prepared_input_closure_impl(
    run_root: Path,
    *,
    require_exact_readonly_mount: bool,
) -> _PreparedInputClosure:
    """Open and hash the complete prepare-only input tree without links."""

    run_root = Path(os.path.abspath(run_root))
    nested_mounts = _mounted_descendants(run_root)
    if nested_mounts:
        raise RuntimeError(
            "Prepare-only input closure contains nested mounts: "
            + ", ".join(str(item) for item in sorted(nested_mounts))
        )
    mount_identity = (
        _exact_readonly_mount_identity(run_root)
        if require_exact_readonly_mount
        else None
    )
    _require_prepared_input_fd_headroom(PREPARED_INPUT_FD_RESERVE)

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if nofollow == 0 or directory == 0:
        raise RuntimeError(
            "Prepare-only input closure requires O_NOFOLLOW and O_DIRECTORY support"
        )
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    directory_flags = common_flags | directory
    owned_descriptors: list[int] = []
    directories: list[_PreparedInputDirectory] = []
    files: list[_PreparedInputFile] = []
    entry_count = 0
    logical_bytes = 0

    def checked_open(
        name_or_path: str | os.PathLike[str],
        flags: int,
        *,
        dir_fd: int | None = None,
        label: str,
    ) -> int:
        try:
            descriptor = os.open(name_or_path, flags, dir_fd=dir_fd)
        except OSError as exc:
            raise RuntimeError(
                "Prepare-only input could not be opened without following links: "
                f"{label}: {exc.strerror or type(exc).__name__} "
                f"(errno {exc.errno})"
            ) from exc
        owned_descriptors.append(descriptor)
        return descriptor

    def child_path(parent: str, name: str) -> str:
        return name if parent == "." else f"{parent}/{name}"

    try:
        root_descriptor = checked_open(
            run_root,
            directory_flags,
            label=".",
        )
        root_stat = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise RuntimeError("Prepare-only input root must be a regular directory")
        root_device = root_stat.st_dev

        def capture_directory(
            descriptor: int,
            relative_path: str,
            *,
            depth: int,
        ) -> None:
            nonlocal entry_count, logical_bytes
            if depth > PREPARED_INPUT_MAX_DEPTH:
                raise RuntimeError(
                    "Prepare-only input closure exceeds maximum directory depth: "
                    f"{PREPARED_INPUT_MAX_DEPTH}"
                )
            entry_count += 1
            if entry_count > PREPARED_INPUT_MAX_ENTRIES:
                raise RuntimeError(
                    "Prepare-only input closure exceeds maximum entry count: "
                    f"{PREPARED_INPUT_MAX_ENTRIES}"
                )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(before.st_mode)
                or before.st_dev != root_device
            ):
                raise RuntimeError(
                    "Prepare-only input closure contains an unsafe directory: "
                    f"{relative_path}"
                )
            signature = _file_stability_signature(before)
            directories.append(
                _PreparedInputDirectory(
                    relative_path=relative_path,
                    descriptor=descriptor,
                    signature=signature,
                )
            )
            try:
                with os.scandir(descriptor) as scanner:
                    entries = sorted(scanner, key=lambda item: item.name)
            except OSError as exc:
                raise RuntimeError(
                    "Prepare-only input directory could not be enumerated: "
                    f"{relative_path}"
                ) from exc

            for entry in entries:
                try:
                    entry.name.encode("utf-8", errors="strict")
                except UnicodeEncodeError as exc:
                    raise RuntimeError(
                        "Prepare-only input closure contains a non-UTF-8 path"
                    ) from exc
                relative_child = child_path(relative_path, entry.name)
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise RuntimeError(
                        "Prepare-only input changed while its inventory was acquired: "
                        f"{relative_child}"
                    ) from exc
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise RuntimeError(
                        "Prepare-only input closure contains a symbolic link: "
                        f"{relative_child}"
                    )
                if entry_stat.st_dev != root_device:
                    raise RuntimeError(
                        "Prepare-only input closure crosses a filesystem boundary: "
                        f"{relative_child}"
                    )
                if stat.S_ISDIR(entry_stat.st_mode):
                    child_descriptor = checked_open(
                        entry.name,
                        directory_flags,
                        dir_fd=descriptor,
                        label=relative_child,
                    )
                    opened_stat = os.fstat(child_descriptor)
                    if (
                        not stat.S_ISDIR(opened_stat.st_mode)
                        or opened_stat.st_dev != entry_stat.st_dev
                        or opened_stat.st_ino != entry_stat.st_ino
                    ):
                        raise RuntimeError(
                            "Prepare-only input directory path binding changed while "
                            f"it was opened: {relative_child}"
                        )
                    capture_directory(
                        child_descriptor,
                        relative_child,
                        depth=depth + 1,
                    )
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise RuntimeError(
                        "Prepare-only input closure contains a special file: "
                        f"{relative_child}"
                    )
                entry_count += 1
                if entry_count > PREPARED_INPUT_MAX_ENTRIES:
                    raise RuntimeError(
                        "Prepare-only input closure exceeds maximum entry count: "
                        f"{PREPARED_INPUT_MAX_ENTRIES}"
                    )
                logical_bytes += entry_stat.st_size
                if logical_bytes > PREPARED_INPUT_MAX_LOGICAL_BYTES:
                    raise RuntimeError(
                        "Prepare-only input closure exceeds maximum logical size: "
                        f"{PREPARED_INPUT_MAX_LOGICAL_BYTES} bytes"
                    )
                file_descriptor = checked_open(
                    entry.name,
                    common_flags,
                    dir_fd=descriptor,
                    label=relative_child,
                )
                opened_stat = os.fstat(file_descriptor)
                if (
                    not stat.S_ISREG(opened_stat.st_mode)
                    or opened_stat.st_dev != entry_stat.st_dev
                    or opened_stat.st_ino != entry_stat.st_ino
                ):
                    raise RuntimeError(
                        "Prepare-only input file path binding changed while it was "
                        f"opened: {relative_child}"
                    )
                file_signature = _file_stability_signature(opened_stat)
                file_digest = _descriptor_sha256(file_descriptor)
                after_hash = os.fstat(file_descriptor)
                try:
                    rebound_stat = os.stat(
                        entry.name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        "Prepare-only input file path binding changed while it was "
                        f"hashed: {relative_child}"
                    ) from exc
                if (
                    _file_stability_signature(after_hash) != file_signature
                    or not stat.S_ISREG(rebound_stat.st_mode)
                    or rebound_stat.st_dev != opened_stat.st_dev
                    or rebound_stat.st_ino != opened_stat.st_ino
                ):
                    raise RuntimeError(
                        "Prepare-only input file changed while it was hashed: "
                        f"{relative_child}"
                    )
                files.append(
                    _PreparedInputFile(
                        relative_path=relative_child,
                        descriptor=file_descriptor,
                        signature=file_signature,
                        sha256=file_digest,
                    )
                )

            after = os.fstat(descriptor)
            if _file_stability_signature(after) != signature:
                raise RuntimeError(
                    "Prepare-only input directory changed while its inventory was "
                    f"acquired: {relative_path}"
                )

        capture_directory(root_descriptor, ".", depth=0)
        try:
            rebound_root = os.stat(run_root, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                "Prepare-only input root path binding changed while its inventory "
                "was acquired"
            ) from exc
        if (
            not stat.S_ISDIR(rebound_root.st_mode)
            or rebound_root.st_dev != root_stat.st_dev
            or rebound_root.st_ino != root_stat.st_ino
        ):
            raise RuntimeError(
                "Prepare-only input root path binding changed while its inventory "
                "was acquired"
            )
        closure = _PreparedInputClosure(
            run_root=run_root,
            directories=directories,
            files=files,
            require_exact_readonly_mount=require_exact_readonly_mount,
            mount_identity=mount_identity,
        )
        closure._verify_mount_boundary()
        _require_prepared_input_fd_headroom(
            len(directories) + len(files) + PREPARED_INPUT_FD_RESERVE
        )
        owned_descriptors.clear()
        return closure
    except BaseException:
        for descriptor in reversed(owned_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _acquire_prepared_input_closure(run_root: Path) -> _PreparedInputClosure:
    """Acquire production closure from an exact read-only mount point."""

    return _acquire_prepared_input_closure_impl(
        run_root,
        require_exact_readonly_mount=True,
    )


def _acquire_prepared_input_closure_test_only(
    run_root: Path,
) -> _PreparedInputClosure:
    """Acquire the closure on a mutable test filesystem."""

    return _acquire_prepared_input_closure_impl(
        run_root,
        require_exact_readonly_mount=False,
    )


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_plan(
    *,
    evaluation_scope: str,
    evaluation_max_examples: int | None,
    gguf_requested: bool,
) -> dict[str, Any]:
    if evaluation_scope not in {"full", "smoke", "none"}:
        raise RuntimeError(f"Unsupported evaluation scope: {evaluation_scope}")
    if evaluation_scope == "smoke":
        if type(evaluation_max_examples) is not int or evaluation_max_examples <= 0:
            raise RuntimeError("Smoke evaluation requires a positive maximum")
    elif evaluation_max_examples is not None:
        raise RuntimeError(
            "An evaluation maximum is valid only for smoke evaluation"
        )
    if type(gguf_requested) is not bool:
        raise RuntimeError("GGUF execution-plan state must be boolean")
    value: dict[str, Any] = {
        "schema": EXECUTION_PLAN_SCHEMA_VERSION,
        "evaluationScope": evaluation_scope,
        "evaluationMaxExamples": evaluation_max_examples,
        "ggufRequested": gguf_requested,
    }
    value["executionPlanSHA256"] = canonical_sha256(value)
    return value


def _verified_execution_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Prepared run lacks an immutable execution plan")
    plan = dict(value)
    if set(plan) != {
        "schema",
        "evaluationScope",
        "evaluationMaxExamples",
        "ggufRequested",
        "executionPlanSHA256",
    }:
        raise RuntimeError("Prepared execution plan has an invalid field set")
    declared = plan.pop("executionPlanSHA256", None)
    expected = execution_plan(
        evaluation_scope=str(plan.get("evaluationScope") or ""),
        evaluation_max_examples=plan.get("evaluationMaxExamples"),
        gguf_requested=plan.get("ggufRequested"),
    )
    if (
        plan.get("schema") != EXECUTION_PLAN_SCHEMA_VERSION
        or declared != expected["executionPlanSHA256"]
        or dict(value) != expected
    ):
        raise RuntimeError("Prepared execution plan failed integrity verification")
    return expected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_upload_relative_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"{label} is not a canonical relative path: {value}")
    return path


def _open_regular_beneath(
    root_descriptor: int,
    relative_path: str,
    *,
    label: str,
) -> tuple[BinaryIO, os.stat_result]:
    path = _validated_upload_relative_path(relative_path, label=label)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if nofollow == 0 or directory == 0:
        raise RuntimeError(f"{label} requires O_NOFOLLOW and O_DIRECTORY support")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    parent_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        for component in path.parts[:-1]:
            child_descriptor = os.open(
                component,
                common_flags | directory,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor
        file_descriptor = os.open(
            path.parts[-1],
            common_flags,
            dir_fd=parent_descriptor,
        )
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"{label} must be a regular file: {relative_path}")
        handle = os.fdopen(file_descriptor, "rb", closefd=True)
        file_descriptor = None
        return handle, file_stat
    except OSError as exc:
        raise RuntimeError(
            f"{label} could not be opened without following links: {relative_path}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(parent_descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Short write while snapshotting an upload input")
        offset += written


def _snapshot_verified_upload_inputs(
    run_root: Path,
    contracts: Sequence[_UploadInputContract],
    snapshot_root: Path,
) -> list[_SnapshottedUploadInput]:
    if not contracts:
        raise RuntimeError("Upload requires at least one verified input")
    snapshot_stat = os.stat(snapshot_root, follow_symlinks=False)
    if (
        not stat.S_ISDIR(snapshot_stat.st_mode)
        or snapshot_stat.st_uid != os.geteuid()
        or stat.S_IMODE(snapshot_stat.st_mode) & 0o077
    ):
        raise RuntimeError("Upload snapshot root must be a private owned directory")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if nofollow == 0 or directory == 0:
        raise RuntimeError("Upload snapshotting requires Linux no-follow support")
    try:
        root_descriptor = os.open(
            run_root,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | directory,
        )
    except OSError as exc:
        raise RuntimeError("Upload run root is not a regular no-follow directory") from exc

    local_paths: set[str] = set()
    remote_paths: set[str] = set()
    snapshotted: list[_SnapshottedUploadInput] = []
    try:
        for index, contract in enumerate(contracts):
            relative = _validated_upload_relative_path(
                contract.relative_path,
                label="Upload input path",
            )
            remote = _validated_upload_relative_path(
                contract.remote_path,
                label="Upload remote path",
            )
            if relative.as_posix() in local_paths:
                raise RuntimeError("Upload file contract contains duplicate local paths")
            if remote.as_posix() in remote_paths:
                raise RuntimeError("Upload file contract contains duplicate remote paths")
            local_paths.add(relative.as_posix())
            remote_paths.add(remote.as_posix())
            if contract.expected_sha256 is None and contract.expected_json is None:
                raise RuntimeError("Upload input lacks a verified content contract")
            if contract.expected_sha256 is not None and re.fullmatch(
                r"[0-9a-f]{64}", contract.expected_sha256
            ) is None:
                raise RuntimeError("Upload input has an invalid expected digest")
            if contract.expected_size is not None and (
                type(contract.expected_size) is not int or contract.expected_size < 0
            ):
                raise RuntimeError("Upload input has an invalid expected size")

            source_handle, source_stat = _open_regular_beneath(
                root_descriptor,
                relative.as_posix(),
                label="Upload input",
            )
            destination = snapshot_root / f"{index:05d}.upload"
            destination_descriptor: int | None = None
            digest = hashlib.sha256()
            copied_size = 0
            try:
                destination_descriptor = os.open(
                    destination,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | nofollow,
                    0o400,
                )
                offset = 0
                while True:
                    chunk = os.pread(source_handle.fileno(), 1 << 20, offset)
                    if not chunk:
                        break
                    digest.update(chunk)
                    _write_all(destination_descriptor, chunk)
                    copied_size += len(chunk)
                    offset += len(chunk)
                os.fsync(destination_descriptor)
                destination_stat = os.fstat(destination_descriptor)
                if (
                    not stat.S_ISREG(destination_stat.st_mode)
                    or destination_stat.st_size != copied_size
                ):
                    raise RuntimeError("Private upload snapshot is not a stable regular file")
                _require_stable_descriptor(
                    source_handle,
                    source_stat,
                    label="Upload input",
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Unable to snapshot verified upload input: {relative.as_posix()}"
                ) from exc
            finally:
                source_handle.close()
                if destination_descriptor is not None:
                    os.close(destination_descriptor)

            observed_sha256 = digest.hexdigest()
            if (
                contract.expected_sha256 is not None
                and observed_sha256 != contract.expected_sha256
            ) or (
                contract.expected_size is not None
                and copied_size != contract.expected_size
            ):
                raise RuntimeError(
                    f"Upload input drifted from its verified contract: {relative.as_posix()}"
                )
            if contract.expected_json is not None and read_object(destination) != dict(
                contract.expected_json
            ):
                raise RuntimeError(
                    f"Upload JSON drifted from its verified contract: {relative.as_posix()}"
                )
            snapshotted.append(
                _SnapshottedUploadInput(
                    path=destination,
                    remote_path=remote.as_posix(),
                    sha256=observed_sha256,
                    size_bytes=copied_size,
                )
            )
    finally:
        os.close(root_descriptor)
    return snapshotted


def _git_output(checkout: Path, *arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(checkout), *arguments],
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or ""
        raise RuntimeError(
            f"Unable to verify pinned llama.cpp checkout: {detail.strip()}"
        ) from exc


def _verified_pinned_gguf_script(
    run_root: Path,
    *,
    relative_path: Path,
    label: str,
) -> _VerifiedGGUFReaderScript:
    checkout = run_root / "llama.cpp"
    script = checkout / relative_path
    if checkout.is_symlink() or not checkout.is_dir():
        raise RuntimeError(
            f"Missing regular pinned llama.cpp checkout for {label}: {checkout}"
        )
    if script.is_symlink() or not script.is_file():
        raise RuntimeError(f"Missing regular pinned llama.cpp {label}: {script}")
    if checkout.resolve() not in script.resolve().parents:
        raise RuntimeError(f"Pinned {label} escapes its checkout: {script}")
    head = _git_output(checkout, "rev-parse", "HEAD")
    if head != DEFAULT_LLAMA_CPP_REVISION:
        raise RuntimeError(f"llama.cpp {label} revision drifted")
    if _git_output(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        raise RuntimeError(f"llama.cpp {label} checkout is dirty")
    relative_script = relative_path.as_posix()
    expected_blob = _git_output(
        checkout,
        "rev-parse",
        f"HEAD:{relative_script}",
    )
    if re.fullmatch(r"[0-9a-f]{40}", expected_blob) is None:
        raise RuntimeError(f"llama.cpp {label} has an invalid pinned blob identity")
    handle, script_stat = _open_regular_readonly(script, label=f"Pinned {label}")
    try:
        payload = _read_descriptor_bytes(handle)
        if _git_blob_sha1(payload) != expected_blob:
            raise RuntimeError(f"llama.cpp {label} drifted from the pinned revision")
        _require_stable_descriptor(handle, script_stat, label=f"Pinned {label}")
        _require_path_matches_descriptor(script, script_stat, label=f"Pinned {label}")
    finally:
        handle.close()
    return _VerifiedGGUFReaderScript(
        path=script,
        git_blob_sha1=expected_blob,
        file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _verified_pinned_gguf_reader_script(
    run_root: Path,
) -> _VerifiedGGUFReaderScript:
    return _verified_pinned_gguf_script(
        run_root,
        relative_path=GGUF_READER_RELATIVE_PATH,
        label="GGUF reader",
    )


def _verified_pinned_gguf_converter_script(
    run_root: Path,
) -> _VerifiedGGUFReaderScript:
    return _verified_pinned_gguf_script(
        run_root,
        relative_path=GGUF_CONVERTER_RELATIVE_PATH,
        label="LoRA-to-GGUF converter",
    )


def _gguf_scalar_metadata_value(
    metadata: Mapping[str, Any],
    key: str,
    *,
    expected_type: str,
    path: Path,
) -> Any:
    field = metadata.get(key)
    if (
        not isinstance(field, Mapping)
        or set(field) != {"index", "type", "offset", "value"}
        or type(field.get("index")) is not int
        or field["index"] < 0
        or field.get("type") != expected_type
        or type(field.get("offset")) is not int
        or field["offset"] < 0
    ):
        raise RuntimeError(
            "Pinned llama.cpp GGUF reader returned invalid scalar metadata "
            f"for {key}: {path}"
        )
    return field["value"]


def _verified_adapter_gguf_semantics(
    metadata: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    required_strings = {
        "general.architecture": EXPECTED_ADAPTER_GGUF_ARCHITECTURE,
        "general.type": EXPECTED_ADAPTER_GGUF_GENERAL_TYPE,
        "adapter.type": EXPECTED_ADAPTER_GGUF_ADAPTER_TYPE,
        "general.base_model.0.repo_url": (
            EXPECTED_ADAPTER_GGUF_BASE_MODEL_REPO_URL
        ),
    }
    verified_strings: dict[str, str] = {}
    for key, expected in required_strings.items():
        observed = _gguf_scalar_metadata_value(
            metadata,
            key,
            expected_type="STRING",
            path=path,
        )
        if not isinstance(observed, str) or observed != expected:
            raise RuntimeError(
                "Adapter GGUF semantic metadata drifted from the pinned "
                f"Qwen3 contract for {key}: {path}"
            )
        verified_strings[key] = observed

    base_model_count = _gguf_scalar_metadata_value(
        metadata,
        "general.base_model.count",
        expected_type="UINT32",
        path=path,
    )
    if type(base_model_count) is not int or base_model_count != 1:
        raise RuntimeError(
            "Adapter GGUF semantic metadata must bind exactly one base model: "
            f"{path}"
        )

    template_field = metadata.get("tokenizer.chat_template")
    if template_field is None:
        chat_template_source = "shared_base"
        chat_template_sha256: str | None = None
    else:
        chat_template = _gguf_scalar_metadata_value(
            metadata,
            "tokenizer.chat_template",
            expected_type="STRING",
            path=path,
        )
        if not isinstance(chat_template, str) or not chat_template:
            raise RuntimeError(
                f"Adapter GGUF contains an invalid chat template: {path}"
            )
        chat_template_sha256 = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()
        if chat_template_sha256 != PINNED_QWEN3_CHAT_TEMPLATE_SHA256:
            raise RuntimeError(
                "Adapter GGUF chat template drifted from the pinned Qwen3 "
                f"contract: {path}"
            )
        chat_template_source = "adapter_gguf"

    return {
        "adapterGGUFArchitecture": verified_strings["general.architecture"],
        "adapterGGUFType": verified_strings["general.type"],
        "adapterGGUFAdapterType": verified_strings["adapter.type"],
        "adapterGGUFBaseModelID": EXPECTED_ADAPTER_GGUF_BASE_MODEL_ID,
        "adapterGGUFBaseModelRepoURL": verified_strings[
            "general.base_model.0.repo_url"
        ],
        "adapterGGUFChatTemplateSource": chat_template_source,
        "adapterGGUFChatTemplateSHA256": chat_template_sha256,
    }


def _verify_gguf_with_reader(
    path: Path,
    *,
    artifact_handle: BinaryIO,
    reader_script: Path | _VerifiedGGUFReaderScript,
    tensor_count: int,
    metadata_kv_count: int,
) -> dict[str, Any]:
    if isinstance(reader_script, _VerifiedGGUFReaderScript):
        reader_path = reader_script.path
        expected_reader_blob = reader_script.git_blob_sha1
    else:
        reader_path = reader_script
        expected_reader_blob = None
    if reader_path.is_symlink() or not reader_path.is_file():
        raise RuntimeError(f"Pinned llama.cpp GGUF reader is unavailable: {reader_path}")
    reader_handle, reader_stat = _open_regular_readonly(
        reader_path,
        label="Pinned llama.cpp GGUF reader",
    )
    try:
        if (
            expected_reader_blob is not None
            and _git_blob_sha1(_read_descriptor_bytes(reader_handle))
            != expected_reader_blob
        ):
            raise RuntimeError(
                "llama.cpp GGUF reader drifted from the pinned revision"
            )
        reader_fd = reader_handle.fileno()
        artifact_fd = artifact_handle.fileno()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _GGUF_READER_FD_BOOTSTRAP,
                    str(reader_fd),
                    str(reader_path),
                    f"/proc/self/fd/{artifact_fd}",
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=GGUF_READER_TIMEOUT_SECONDS,
                pass_fds=(reader_fd, artifact_fd),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Pinned llama.cpp GGUF reader could not inspect artifact: {path}"
            ) from exc
        _require_stable_descriptor(
            reader_handle,
            reader_stat,
            label="Pinned llama.cpp GGUF reader",
        )
        _require_path_matches_descriptor(
            reader_path,
            reader_stat,
            label="Pinned llama.cpp GGUF reader",
        )
    finally:
        reader_handle.close()
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f" ({detail[-1]})" if detail else ""
        raise RuntimeError(
            f"Pinned llama.cpp GGUF reader rejected artifact: {path}{suffix}"
        )
    try:
        result = json.loads(
            completed.stdout,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Pinned llama.cpp GGUF reader returned invalid evidence: {path}"
        ) from exc
    metadata = result.get("metadata") if isinstance(result, Mapping) else None
    tensors = result.get("tensors") if isinstance(result, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or set(result) != {"filename", "endian", "metadata", "tensors"}
        or Path(str(result.get("filename") or "")).resolve() != path.resolve()
        or result.get("endian") not in {"LITTLE", "BIG"}
        or not isinstance(metadata, Mapping)
        or not isinstance(tensors, Mapping)
        or not {
            "GGUF.version",
            "GGUF.tensor_count",
            "GGUF.kv_count",
        }.issubset(metadata)
        or len(metadata) != metadata_kv_count + 3
        or len(tensors) != tensor_count
        or not all(isinstance(item, Mapping) for item in metadata.values())
        or not all(isinstance(item, Mapping) for item in tensors.values())
    ):
        raise RuntimeError(
            f"Pinned llama.cpp GGUF reader evidence mismatches the fixed header: {path}"
        )
    return _verified_adapter_gguf_semantics(metadata, path=path)


def verify_gguf_artifact(
    path: Path,
    *,
    reader_script: Path | _VerifiedGGUFReaderScript,
) -> dict[str, Any]:
    """Verify one regular GGUF file with the pinned llama.cpp reader."""

    if path.is_symlink():
        raise RuntimeError(f"GGUF artifact must not be a symlink: {path}")
    handle, file_stat = _open_regular_readonly(path, label="GGUF artifact")
    try:
        if file_stat.st_size <= GGUF_FIXED_HEADER_SIZE:
            raise RuntimeError(
                "GGUF artifact must be a regular file larger than its fixed "
                f"header: {path}"
            )
        header = handle.read(GGUF_FIXED_HEADER_SIZE)
        if header[:4] != b"GGUF":
            raise RuntimeError(f"GGUF artifact has invalid magic bytes: {path}")
        version = int.from_bytes(header[4:8], byteorder="little", signed=False)
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"GGUF artifact has unsupported version {version}: {path}"
            )
        tensor_count = int.from_bytes(
            header[8:16], byteorder="little", signed=False
        )
        metadata_kv_count = int.from_bytes(
            header[16:24], byteorder="little", signed=False
        )
        if tensor_count <= 0:
            raise RuntimeError(f"GGUF artifact has no tensors: {path}")
        if metadata_kv_count <= 0:
            raise RuntimeError(f"GGUF artifact has no metadata key-values: {path}")
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        _require_stable_descriptor(
            handle,
            file_stat,
            label="GGUF artifact",
        )
        semantic_verification = _verify_gguf_with_reader(
            path,
            artifact_handle=handle,
            reader_script=reader_script,
            tensor_count=tensor_count,
            metadata_kv_count=metadata_kv_count,
        )
        _require_stable_descriptor(
            handle,
            file_stat,
            label="GGUF artifact",
        )
        _require_path_matches_descriptor(
            path,
            file_stat,
            label="GGUF artifact",
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to read GGUF artifact: {path}") from exc
    finally:
        handle.close()
    return {
        "adapterGGUF": str(path),
        "adapterGGUFSHA256": digest.hexdigest(),
        "adapterGGUFSizeBytes": file_stat.st_size,
        **semantic_verification,
    }


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Unable to read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _fsync_directory(path: Path, *, label: str) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RuntimeError(f"Unable to durably commit {label}") from exc


def write_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent, label=f"JSON object {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Unable to read dataset: {path}") from exc
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON at {path}:{lineno}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected a JSON object at {path}:{lineno}")
        rows.append(value)
    return rows


def parse_agents(value: str) -> tuple[str, ...]:
    raw = value.split(",")
    if not raw or any(not item.strip() for item in raw):
        raise RuntimeError("Agent list must be a comma-separated list without empty entries")
    agents = tuple(item.strip().lower() for item in raw)
    unknown = sorted(set(agents) - set(AGENTS))
    if unknown:
        raise RuntimeError(f"Unsupported agents: {', '.join(unknown)}")
    if len(set(agents)) != len(agents):
        raise RuntimeError("Agent list must not contain duplicates")
    return agents


def require_variant(value: str) -> str:
    if value not in VARIANTS:
        raise RuntimeError(
            f"Unsupported experiment variant: {value}. Expected one of: {', '.join(VARIANTS)}"
        )
    return value


def require_container_digest(value: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise RuntimeError(
            "Container image digest must be sha256 followed by 64 lowercase hex characters"
        )
    return value


def git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to observe the repository Git revision") from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError("Repository HEAD is not an immutable Git commit SHA")
    return value


def current_source_integrity(root: Path) -> dict[str, Any]:
    if SOURCE_INTEGRITY_ENV in os.environ:
        return load_verified_attestation(
            root,
            Path(os.environ[SOURCE_INTEGRITY_ENV]),
        )
    return attest_repository(root)


def source_integrity_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "workingTreeDigest": record["workingTreeDigest"],
        "ubuntuOrchestrationCodeSHA256": record[
            "ubuntuOrchestrationCodeSHA256"
        ],
        "ubuntuSourceIntegritySHA256": record["sourceIntegritySHA256"],
        "ubuntuSourceIntegrity": dict(record),
    }


def verify_embedded_source_integrity(value: Mapping[str, Any]) -> dict[str, Any]:
    record = value.get("ubuntuSourceIntegrity")
    if not isinstance(record, Mapping):
        raise RuntimeError("Ubuntu source-integrity record is missing")
    verified = validate_attestation_record(record)
    if any(
        value.get(field) != expected
        for field, expected in source_integrity_fields(verified).items()
    ):
        raise RuntimeError("Ubuntu source-integrity digest fields drifted")
    return verified


def local_runtime_source(
    root: Path,
    *,
    source_integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    integrity = (
        dict(source_integrity)
        if source_integrity is not None
        else current_source_integrity(root)
    )
    revision = str(integrity["baseCommit"])
    return {
        "runtimeSourceKind": "git",
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": revision,
        "observedRuntimeRevision": revision,
        "runtimeSourceBindingStatus": "verified_clean_snapshot",
        "runtimeSourceBindingMethod": (
            "git_clean_worktree_plus_ubuntu_orchestration_manifest"
        ),
    }


def validate_run_root(run_root: Path, *, allowed_parent: Path) -> Path:
    if run_root.is_symlink():
        raise RuntimeError(f"Run root must not be a symlink: {run_root}")
    resolved = run_root.expanduser().resolve()
    parent = allowed_parent.expanduser().resolve()
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(
            f"Run root must be a child of the allowed output parent {parent}: {resolved}"
        )
    if resolved == Path("/") or len(resolved.parts) < 3:
        raise RuntimeError(f"Unsafe run root: {resolved}")
    return resolved


def _private_directory_identity(
    path: Path,
    *,
    label: str,
    require_parent_device: bool = True,
) -> str:
    """Return a stable identity for one private process-owned directory.

    The identity is deliberately made from the inode metadata Docker preserves
    across a bind mount.  Callers can therefore bind the host-side directory to
    the launch contract and have the inner launcher prove it received that
    exact inode, rather than merely a directory with the expected spelling.
    """

    try:
        observed = os.stat(path, follow_symlinks=False)
        parent = os.stat(path.parent, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_gid != os.getegid()
        or stat.S_IMODE(observed.st_mode) != 0o700
        or (require_parent_device and observed.st_dev != parent.st_dev)
    ):
        raise RuntimeError(
            f"{label} must be a same-filesystem process-owned mode-0700 "
            f"regular directory: {path}"
        )
    return (
        f"{observed.st_dev}:{observed.st_ino}:{observed.st_uid}:"
        f"{observed.st_gid}:{stat.S_IMODE(observed.st_mode):04o}"
    )


def initialize_bind_root(
    run_root: Path,
    *,
    allowed_parent: Path,
    create_if_missing: bool,
) -> dict[str, Any]:
    """Exclusively reserve and durably commit an exact host bind root."""

    resolved = validate_run_root(run_root, allowed_parent=allowed_parent)
    parent = allowed_parent.expanduser().resolve()
    if resolved.parent != parent:
        raise RuntimeError("Bind root must be an exact child of its allowed parent")
    try:
        parent_stat = os.stat(parent, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"Bind-root parent is unavailable: {parent}") from exc
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) & 0o022
    ):
        raise RuntimeError(
            "Bind-root parent must be a process-owned directory that is not "
            "group/world writable"
        )
    created = False
    if not resolved.exists() and not resolved.is_symlink():
        if not create_if_missing:
            raise RuntimeError(f"Exact bind root does not exist: {resolved}")
        try:
            os.mkdir(resolved, mode=0o700)
        except OSError as exc:
            raise RuntimeError(f"Unable to reserve exact bind root: {resolved}") from exc
        os.chmod(resolved, 0o700, follow_symlinks=False)
        _fsync_directory(parent, label="the precreated bind root")
        created = True
    identity = _private_directory_identity(
        resolved,
        label="Exact bind root",
    )
    return {
        "schema": "lumen.ubuntu-bind-root/1.0.0",
        "status": "bind_root_ready",
        "runRoot": str(resolved),
        "rootIdentity": identity,
        "created": created,
    }


def verify_bind_root(
    run_root: Path,
    *,
    allowed_parent: Path,
    expected_identity: str,
    mounted_bind: bool = False,
) -> dict[str, Any]:
    if mounted_bind:
        resolved = validate_run_root(run_root, allowed_parent=allowed_parent)
        if resolved.parent != allowed_parent.expanduser().resolve():
            raise RuntimeError("Mounted bind root escaped its exact container parent")
        observed = {
            "schema": "lumen.ubuntu-bind-root/1.0.0",
            "status": "bind_root_ready",
            "runRoot": str(resolved),
            "rootIdentity": _private_directory_identity(
                resolved,
                label="Mounted bind root",
                require_parent_device=False,
            ),
            "created": False,
        }
    else:
        observed = initialize_bind_root(
            run_root,
            allowed_parent=allowed_parent,
            create_if_missing=False,
        )
    if observed["rootIdentity"] != expected_identity:
        raise RuntimeError("Exact bind-root device/inode/ownership/mode changed")
    return {
        **observed,
        "status": "bind_root_identity_verified",
    }


def _require_dataset_contract(
    manifest: Mapping[str, Any],
    key: str,
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
) -> None:
    datasets = manifest.get("datasets")
    contract = datasets.get(key) if isinstance(datasets, Mapping) else None
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"Variant manifest is missing datasets.{key}: {manifest_path}")
    if type(contract.get("count")) is not int or contract["count"] != len(records):
        raise RuntimeError(f"Dataset count mismatch for datasets.{key}: {manifest_path}")
    if contract.get("sha256") != canonical_sha256(list(records)):
        raise RuntimeError(f"Dataset digest mismatch for datasets.{key}: {manifest_path}")


def _load_adapter_evaluation_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    crawler_root = repo_root / "tools" / "lumen_manifest_crawler"
    if not crawler_root.is_dir():
        raise RuntimeError(
            f"Contamination validators are unavailable: {crawler_root}"
        )
    if str(crawler_root) not in sys.path:
        sys.path.insert(0, str(crawler_root))
    try:
        from lumen_manifest_crawler.dataset import adapter_evaluation
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Unable to load the contamination validators") from exc
    return adapter_evaluation


def _require_clean_contamination_report(
    *,
    report_path: Path,
    manifest: Mapping[str, Any],
    training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
) -> None:
    if report_path.is_symlink() or not report_path.is_file():
        raise RuntimeError(
            f"Missing regular controlled contamination report: {report_path}"
        )
    report = read_object(report_path)
    evaluation_module = _load_adapter_evaluation_module()
    valid_report = getattr(evaluation_module, "_valid_contamination_report", None)
    matches_variant = getattr(
        evaluation_module,
        "_contamination_matches_variant",
        None,
    )
    upgrade_record = getattr(evaluation_module, "upgrade_evaluation_record", None)
    build_public_bundle = getattr(
        evaluation_module,
        "build_public_adapter_eval_fingerprint_bundle",
        None,
    )
    if not all(
        callable(item)
        for item in (
            valid_report,
            matches_variant,
            upgrade_record,
            build_public_bundle,
        )
    ):
        raise RuntimeError("Contamination validators are incomplete")
    if not valid_report(report):
        raise RuntimeError(
            f"Contamination report integrity check failed: {report_path}"
        )
    if not matches_variant(report, manifest):
        raise RuntimeError(
            f"Contamination report is not bound to its variant manifest: {report_path}"
        )

    contamination = manifest.get("contamination")
    if not isinstance(contamination, Mapping):
        raise RuntimeError(
            f"Variant manifest lacks contamination lineage: {report_path}"
        )
    bound_fields = (
        "contaminated",
        "matchCount",
        "reportSHA256",
        "trainingRecordsSHA256",
        "evaluationRecordsSHA256",
        "publicEvaluationBundleSHA256",
        "publicEvaluationRowCount",
    )
    if any(report.get(field) != contamination.get(field) for field in bound_fields):
        raise RuntimeError(
            f"Contamination report lineage mismatches the variant manifest: {report_path}"
        )
    try:
        public_bundle = build_public_bundle()
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Public evaluation fingerprints could not be verified: {report_path}"
        ) from exc
    if (
        not isinstance(public_bundle, Mapping)
        or report.get("publicEvaluationBundleSHA256")
        != public_bundle.get("bundleSHA256")
        or report.get("publicEvaluationRowCount") != public_bundle.get("rowCount")
    ):
        raise RuntimeError(
            f"Contamination report public-evaluation binding mismatch: {report_path}"
        )

    try:
        upgraded_evaluation = [upgrade_record(record) for record in evaluation_records]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Frozen evaluation records could not be normalized: {report_path}"
        ) from exc
    if (
        report.get("trainingRecordCount") != len(training_records)
        or report.get("trainingRecordsSHA256")
        != canonical_sha256(list(training_records))
    ):
        raise RuntimeError(
            f"Contamination report training-dataset binding mismatch: {report_path}"
        )
    if (
        report.get("evaluationRecordCount") != len(upgraded_evaluation)
        or report.get("evaluationRecordsSHA256")
        != canonical_sha256(upgraded_evaluation)
    ):
        raise RuntimeError(
            f"Contamination report evaluation-dataset binding mismatch: {report_path}"
        )
    if (
        report.get("contaminated") is not False
        or report.get("matchCount") != 0
        or report.get("matches") != []
    ):
        raise RuntimeError(
            f"Controlled training variant is contaminated: {report_path}"
        )


def _expected_variant_optimization_policy(
    *,
    agent: str,
    sft_train_record_count: int,
    dpo_train_record_count: int,
) -> dict[str, Any]:
    batch_size = 1 if agent in {"cortex", "fleet"} else 2
    gradient_accumulation_steps = {
        "cortex": 16,
        "executor": 8,
        "mouth": 4,
        "mimicry": 2,
        "rem": 4,
        "fleet": 8,
    }[agent]
    high_reasoning = agent in {"cortex", "executor", "rem"}
    base_epochs = {
        "sft": (
            3
            if agent in {"cortex", "fleet"}
            else 2
            if high_reasoning
            else 1
        ),
        "dpo": (
            1
            if agent == "cortex"
            else 2
            if high_reasoning
            else 1
        ),
    }
    maximum_epochs = None if agent == "cortex" else _MAXIMUM_NON_CORTEX_EPOCHS
    lanes: dict[str, dict[str, Any]] = {}
    for lane, record_count in (
        ("sft", sft_train_record_count),
        ("dpo", dpo_train_record_count),
    ):
        if type(record_count) is not int or record_count <= 0:
            raise RuntimeError(
                f"{agent} {lane} optimization record count is invalid"
            )
        micro_batches = (record_count + batch_size - 1) // batch_size
        steps_per_epoch = (
            micro_batches + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        minimum_steps = (
            None
            if agent == "cortex"
            else _MINIMUM_EFFECTIVE_STEPS_BY_LANE[lane][agent]
        )
        if agent == "cortex":
            selected_epochs = base_epochs[lane]
        else:
            if steps_per_epoch <= 0 or minimum_steps is None:
                raise RuntimeError(
                    f"{agent} {lane} optimization lane cannot satisfy its minimum"
                )
            selected_epochs = max(
                base_epochs[lane],
                (minimum_steps + steps_per_epoch - 1) // steps_per_epoch,
            )
            if selected_epochs > _MAXIMUM_NON_CORTEX_EPOCHS:
                raise RuntimeError(
                    f"{agent} {lane} optimization lane exceeds the epoch cap"
                )
        projected_steps = steps_per_epoch * selected_epochs
        lanes[lane] = {
            "trainRecordCount": record_count,
            "baseEpochs": base_epochs[lane],
            "selectedEpochs": selected_epochs,
            "effectiveStepsPerEpoch": steps_per_epoch,
            "minimumEffectiveSteps": minimum_steps,
            "projectedEffectiveSteps": projected_steps,
            "minimumSatisfied": (
                True
                if minimum_steps is None
                else projected_steps >= minimum_steps
            ),
        }
    return {
        "schemaVersion": "lumen.adapter-effective-steps/1.0.0",
        "mode": (
            "cortex_empirical_fixed"
            if agent == "cortex"
            else "non_cortex_minimum_effective_steps"
        ),
        "batchSize": batch_size,
        "gradientAccumulationSteps": gradient_accumulation_steps,
        "sft": lanes["sft"],
        "dpo": lanes["dpo"],
        "maximumEpochs": maximum_epochs,
    }


def _variant_effective_training_config(
    *,
    agent: str,
    base_config: Mapping[str, Any],
    controlled_config: Mapping[str, Any],
    train_sft_record_count: int,
    train_dpo_record_count: int,
) -> dict[str, Any]:
    base_controlled_keys = set(base_config) - NON_CONTROLLED_CONFIG_FIELDS
    if set(controlled_config) != base_controlled_keys:
        raise RuntimeError(
            "Variant controlled training config fields drifted from the base config"
        )
    if not VARIANT_OPTIMIZATION_CONFIG_FIELDS.issubset(controlled_config):
        raise RuntimeError(
            "Variant controlled training config lacks optimizer-policy fields"
        )
    for key in sorted(base_controlled_keys - VARIANT_OPTIMIZATION_CONFIG_FIELDS):
        if canonical_sha256({"value": controlled_config[key]}) != canonical_sha256(
            {"value": base_config[key]}
        ):
            raise RuntimeError(
                f"Variant controlled training config changed non-variant field: {key}"
            )

    base_policy = base_config.get("optimizationStepPolicy")
    if not isinstance(base_policy, Mapping):
        raise RuntimeError("Base config lacks an optimization-step policy")
    base_sft = base_policy.get("sft")
    base_dpo = base_policy.get("dpo")
    if not isinstance(base_sft, Mapping) or not isinstance(base_dpo, Mapping):
        raise RuntimeError("Base optimization-step policy lacks lane state")
    base_expected = _expected_variant_optimization_policy(
        agent=agent,
        sft_train_record_count=base_sft.get("trainRecordCount"),
        dpo_train_record_count=base_dpo.get("trainRecordCount"),
    )
    if (
        canonical_sha256(dict(base_policy)) != canonical_sha256(base_expected)
        or type(base_config.get("batch_size")) is not int
        or base_config.get("batch_size") != base_expected["batchSize"]
        or type(base_config.get("gradient_accumulation_steps")) is not int
        or base_config.get("gradient_accumulation_steps")
        != base_expected["gradientAccumulationSteps"]
        or type(base_config.get("num_train_epochs")) is not int
        or base_config.get("num_train_epochs")
        != base_expected["sft"]["selectedEpochs"]
        or type(base_config.get("dpo_num_train_epochs")) is not int
        or base_config.get("dpo_num_train_epochs")
        != base_expected["dpo"]["selectedEpochs"]
    ):
        raise RuntimeError("Base optimization-step policy is internally inconsistent")

    variant_expected = _expected_variant_optimization_policy(
        agent=agent,
        sft_train_record_count=train_sft_record_count,
        dpo_train_record_count=train_dpo_record_count,
    )
    if (
        canonical_sha256(controlled_config.get("optimizationStepPolicy"))
        != canonical_sha256(variant_expected)
        or type(controlled_config.get("num_train_epochs")) is not int
        or controlled_config.get("num_train_epochs")
        != variant_expected["sft"]["selectedEpochs"]
        or type(controlled_config.get("dpo_num_train_epochs")) is not int
        or controlled_config.get("dpo_num_train_epochs")
        != variant_expected["dpo"]["selectedEpochs"]
    ):
        raise RuntimeError(
            "Variant optimization-step policy does not match its training lanes"
        )

    effective = dict(base_config)
    for key in VARIANT_OPTIMIZATION_CONFIG_FIELDS:
        effective[key] = controlled_config[key]
    return effective


def _variant_invariant_training_config(
    config: Mapping[str, Any],
    *,
    agent: str | None = None,
    sft_train_record_count: int | None = None,
    dpo_train_record_count: int | None = None,
) -> dict[str, Any]:
    try:
        return _normalized_invariant_training_config(
            config,
            agent=agent,
            sft_train_record_count=sft_train_record_count,
            dpo_train_record_count=dpo_train_record_count,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Variant invariant training config is invalid") from exc


def validate_variant(
    source_root: Path,
    *,
    agent: str,
    variant: str,
    seed: int,
    base_model_override: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    from tools.fine_tuning.unsloth.train_dpo import (
        _validate_preference_training_config,
    )
    from tools.fine_tuning.unsloth.train_sft import _resolve_training_precision

    agent_root = source_root / agent
    variant_root = agent_root / "experiments" / variant
    for filename in DATASET_FILES:
        if not (variant_root / filename).is_file():
            raise RuntimeError(f"Missing controlled dataset: {variant_root / filename}")
    manifest_path = variant_root / "variant_manifest.json"
    config_path = agent_root / "unsloth_config.json"
    manifest = read_object(manifest_path)
    config = read_object(config_path)
    if manifest.get("schemaVersion") != EXPERIMENT_VARIANT_SCHEMA_VERSION:
        raise RuntimeError(f"Variant manifest schema is unsupported: {manifest_path}")
    if manifest.get("agent") != agent or manifest.get("variant") != variant:
        raise RuntimeError(f"Variant manifest identity mismatch: {manifest_path}")
    declared_manifest_digest = manifest.get("variantManifestSHA256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("variantManifestSHA256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(declared_manifest_digest or "")) is None
        or canonical_sha256(unsigned_manifest) != declared_manifest_digest
    ):
        raise RuntimeError(f"Variant manifest integrity check failed: {manifest_path}")

    lanes = {
        filename.removesuffix(".jsonl"): read_jsonl(variant_root / filename)
        for filename in DATASET_FILES
    }
    _require_dataset_contract(manifest, "trainSFT", lanes["train_sft"], manifest_path=manifest_path)
    _require_dataset_contract(
        manifest,
        "validationSFT",
        lanes["val_sft"],
        manifest_path=manifest_path,
    )
    datasets = manifest.get("datasets")
    if isinstance(datasets, Mapping) and (
        "trainDPO" in datasets or "validationDPO" in datasets
    ):
        _require_dataset_contract(
            manifest,
            "trainDPO",
            lanes["train_dpo"],
            manifest_path=manifest_path,
        )
        _require_dataset_contract(
            manifest,
            "validationDPO",
            lanes["val_dpo"],
            manifest_path=manifest_path,
        )
    else:
        _require_dataset_contract(
            manifest,
            "dpo",
            [*lanes["train_dpo"], *lanes["val_dpo"]],
            manifest_path=manifest_path,
        )
    training_corpus = [
        *lanes["train_sft"],
        *lanes["val_sft"],
        *lanes["train_dpo"],
        *lanes["val_dpo"],
    ]
    if manifest.get("trainingCorpusSHA256") != canonical_sha256(training_corpus):
        raise RuntimeError(f"Training-corpus digest mismatch: {manifest_path}")
    evaluation_path = agent_root / "eval.jsonl"
    if evaluation_path.is_symlink() or not evaluation_path.is_file():
        raise RuntimeError(f"Missing regular frozen evaluation dataset: {evaluation_path}")
    evaluation_records = read_jsonl(evaluation_path)
    _require_clean_contamination_report(
        report_path=variant_root / "contamination_report.json",
        manifest=manifest,
        training_records=training_corpus,
        evaluation_records=evaluation_records,
    )

    controlled = manifest.get("controlledTrainingConfig")
    if not isinstance(controlled, Mapping):
        raise RuntimeError(f"Variant manifest lacks controlledTrainingConfig: {manifest_path}")
    if manifest.get("trainingConfigSHA256") != canonical_sha256(dict(controlled)):
        raise RuntimeError(
            f"Variant controlled training-config digest drifted: {manifest_path}"
        )
    if (
        type(manifest.get("seed")) is not int
        or type(controlled.get("seed")) is not int
        or manifest.get("seed") != controlled.get("seed")
    ):
        raise RuntimeError(f"Variant seed contract drifted: {manifest_path}")
    invariant_digest = canonical_sha256(
        _variant_invariant_training_config(
            controlled,
            agent=agent,
            sft_train_record_count=len(lanes["train_sft"]),
            dpo_train_record_count=len(lanes["train_dpo"]),
        )
    )
    if manifest.get("trainingConfigInvariantSHA256") != invariant_digest:
        raise RuntimeError(
            f"Variant invariant training-config digest drifted: {manifest_path}"
        )
    base_controlled = {
        key: value
        for key, value in config.items()
        if key not in NON_CONTROLLED_CONFIG_FIELDS
    }
    if canonical_sha256(
        _variant_invariant_training_config(base_controlled, agent=agent)
    ) != invariant_digest:
        raise RuntimeError(
            f"Variant invariant training config differs from the base config: {config_path}"
        )
    config = _variant_effective_training_config(
        agent=agent,
        base_config=config,
        controlled_config=controlled,
        train_sft_record_count=len(lanes["train_sft"]),
        train_dpo_record_count=len(lanes["train_dpo"]),
    )
    _resolve_training_precision(config)
    _validate_preference_training_config(config)
    if config.get("agent") != agent:
        raise RuntimeError(f"Generated config agent mismatch: {config_path}")
    public_corpus_contract = (
        _pipeline_validated_public_corpus_loss_share_contract(
            config.get("publicCorpusLossShareContract"),
            config=config,
        )
    )
    for lane_name, rows in lanes.items():
        for row_index, row in enumerate(rows):
            try:
                _pipeline_public_corpus_row_classification(
                    row,
                    contract=public_corpus_contract,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Public-corpus row-metadata preflight rejected "
                    f"{lane_name} row {row_index}"
                ) from exc
    if agent == "fleet":
        fleet_contract = _pipeline_validated_fleet_loss_share_contract(
            config.get("fleetLossShareContract"),
            config=config,
        )
        for lane_name, rows in lanes.items():
            for row_index, row in enumerate(rows):
                try:
                    _pipeline_fleet_source_role(row, contract=fleet_contract)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Fleet source-role preflight rejected "
                        f"{lane_name} row {row_index}"
                    ) from exc
    elif config.get("fleetLossShareContract") is not None:
        raise RuntimeError(
            f"Non-Fleet generated config contains Fleet loss-share state: {config_path}"
        )
    if (
        type(seed) is not int
        or type(manifest.get("seed")) is not int
        or manifest.get("seed") != seed
        or type(config.get("seed")) is not int
        or config.get("seed") != seed
    ):
        raise RuntimeError(f"Seed {seed} would break the controlled variant for {agent}")
    base_model = base_model_override or str(config.get("base_model_name") or "")
    if not base_model or manifest.get("baseModelID") != base_model:
        raise RuntimeError(f"Base model would break the controlled variant for {agent}")
    for split in (lanes["train_sft"], lanes["val_sft"]):
        for row in split:
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise RuntimeError(f"Invalid messages array in {agent}/{variant}")
            assistant = next(
                (
                    message.get("content", "")
                    for message in messages
                    if isinstance(message, Mapping)
                    and message.get("role") == "assistant"
                ),
                "",
            )
            if not str(assistant).strip() or str(assistant).strip().lower() in {"null", "none"}:
                raise RuntimeError(f"Empty/null assistant output in {agent}/{variant}")
    return config, manifest, variant_root


def _verify_smoke_plan_against_frozen_suites(
    dataset_source: Path,
    agents: Sequence[str],
    plan: Mapping[str, Any],
) -> None:
    verified_plan = _verified_execution_plan(plan)
    if verified_plan["evaluationScope"] != "smoke":
        return
    max_examples = verified_plan["evaluationMaxExamples"]
    for agent in agents:
        evaluation_path = dataset_source / agent / "eval.jsonl"
        if evaluation_path.is_symlink() or not evaluation_path.is_file():
            raise RuntimeError(
                f"Missing regular frozen evaluation dataset: {evaluation_path}"
            )
        frozen_case_count = len(read_jsonl(evaluation_path))
        if max_examples >= frozen_case_count:
            raise RuntimeError(
                "Smoke evaluation size must be smaller than the frozen suite for "
                f"{agent}: requested {max_examples}, available {frozen_case_count}"
            )


def static_preflight(
    *,
    root: Path,
    dataset_source: Path,
    agents: Sequence[str],
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
    run_root: Path,
    allowed_parent: Path,
    expected_run_id: str | None = None,
    evaluation_scope: str = "full",
    evaluation_max_examples: int | None = None,
    gguf_requested: bool = True,
    precreated_bind_root: bool = False,
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth import evaluate_adapter

    require_variant(variant)
    require_container_digest(container_digest)
    resolved_run_root = validate_run_root(run_root, allowed_parent=allowed_parent)
    if expected_run_id is not None:
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", expected_run_id)
            is None
            or resolved_run_root != allowed_parent.expanduser().resolve() / expected_run_id
        ):
            raise RuntimeError("Run root must be the exact expected child of its allowed parent")
    if precreated_bind_root:
        _private_directory_identity(
            resolved_run_root,
            label="Precreated bind root",
            require_parent_device=False,
        )
        entries = list(resolved_run_root.iterdir())
        if entries and not any(
            entry.name in {PREPARATION_OWNER_FILENAME, "aio_run_manifest.json"}
            and entry.is_file()
            and not entry.is_symlink()
            for entry in entries
        ):
            raise RuntimeError(
                "Precreated bind root contains unexpected state without an "
                "ownership record"
            )
    if not dataset_source.is_dir():
        raise RuntimeError(f"Dataset source does not exist: {dataset_source}")
    runtime_manifest = read_object(dataset_source / "adapter_runtime_manifest.json")
    behavior_manifest_path = (
        root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    )
    read_object(behavior_manifest_path)
    evaluation_module = evaluate_adapter._load_evaluation_module()
    tool_contracts, allowed_slots, _ = evaluate_adapter.load_behavior_contract(
        behavior_manifest_path
    )
    prepared_execution_plan = execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=evaluation_max_examples,
        gguf_requested=gguf_requested,
    )
    checked: list[dict[str, Any]] = []
    for agent in agents:
        _, manifest, _ = validate_variant(
            dataset_source,
            agent=agent,
            variant=variant,
            seed=seed,
            base_model_override=base_model_override,
        )
        evaluation_path = dataset_source / agent / "eval.jsonl"
        evaluation_records, evaluation_sha256 = (
            evaluate_adapter.load_evaluation_records(
                evaluation_path,
                agent=agent,
                evaluation_module=evaluation_module,
            )
        )
        evaluate_adapter.validate_scoring_contracts(
            evaluation_records,
            tool_contracts=tool_contracts,
            allowed_slots=allowed_slots,
        )
        prompt_preflight = evaluate_adapter.evaluation_prompt_preflight(
            evaluation_records,
            agent=agent,
            tool_contracts=tool_contracts,
        )
        checked.append(
            {
                "agent": agent,
                "variantManifestSHA256": manifest["variantManifestSHA256"],
                "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
                "evaluationSHA256": evaluation_sha256,
                "evaluationPromptPreflight": prompt_preflight,
            }
        )
    _verify_smoke_plan_against_frozen_suites(
        dataset_source,
        agents,
        prepared_execution_plan,
    )
    return {
        "schema": "lumen.ubuntu-training-static-preflight/2.1.0",
        "status": "static_ready",
        "trainingReady": False,
        "unchecked": ["python_environment", "cuda_runtime", "accelerator", "network"],
        "variant": variant,
        "executionPlan": prepared_execution_plan,
        "agents": checked,
        "runRoot": str(run_root.resolve()),
        "runRootInitializationMode": (
            "precreated_bind_root"
            if precreated_bind_root
            else "atomic_sibling_promotion"
        ),
        "adapterRepoID": runtime_manifest.get("adapterRepoID"),
    }


def _runtime_lineage(
    *,
    root: Path,
    source_config: Mapping[str, Any],
    container_digest: str,
    source_integrity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.fine_tuning.unsloth.train_sft import (
        _training_environment,
        _training_runtime_lineage,
    )

    integrity = (
        dict(source_integrity)
        if source_integrity is not None
        else current_source_integrity(root)
    )
    config = dict(source_config)
    config.update(local_runtime_source(root, source_integrity=integrity))
    config.update(source_integrity_fields(integrity))
    config["trainingContainerImageDigest"] = container_digest
    config["trainingContainerImageDigestSource"] = "operator_declared"
    config["trainingRuntimeImageBindingStatus"] = "manual_validation_required"
    config["trainingRuntimeImageBindingVerified"] = False
    config["trainingEnvironmentSHA256"] = None
    config["resolvedTrainingEnvironment"] = None
    config["resolvedTrainingEnvironmentSHA256"] = None
    config["resolvedTrainingEnvironmentScanAudit"] = None
    config["resolvedTrainingEnvironmentCacheAttestation"] = None
    config["spaceConfigurationSHA256"] = None
    config["zeroGPUSize"] = None
    config["zeroGPUDurationSeconds"] = None
    config["observedAccelerator"] = None
    lineage = _training_runtime_lineage(config, phase="sft")
    config.update(
        {
            key: lineage[key]
            for key in (
                "resolvedTrainingEnvironment",
                "resolvedTrainingEnvironmentSHA256",
                "resolvedTrainingEnvironmentScanAudit",
                "spaceConfigurationSHA256",
                "zeroGPUSize",
                "zeroGPUDurationSeconds",
                "observedAccelerator",
            )
        }
    )
    environment = _training_environment(config, runtime_lineage=lineage)
    return lineage, environment


def runtime_preflight(
    *,
    root: Path,
    dataset_source: Path,
    agent: str,
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
) -> dict[str, Any]:
    config, _, _ = validate_variant(
        dataset_source,
        agent=agent,
        variant=variant,
        seed=seed,
        base_model_override=base_model_override,
    )
    lineage, environment = _runtime_lineage(
        root=root,
        source_config=config,
        container_digest=container_digest,
    )
    accelerator = lineage.get("observedAccelerator")
    if not isinstance(accelerator, Mapping) or accelerator.get("backend") != "cuda":
        raise RuntimeError("Runtime preflight did not observe a CUDA accelerator")
    return {
        "schema": "lumen.ubuntu-training-runtime-preflight/1.0.0",
        "status": "training_ready",
        "trainingReady": True,
        "pythonVersion": environment["environmentLock"]["pythonVersion"],
        "cudaVersion": environment["environmentLock"]["cudaVersion"],
        "observedAccelerator": accelerator,
        "resolvedTrainingEnvironmentSHA256": lineage[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "trainingEnvironmentSHA256": environment["trainingEnvironmentSHA256"],
        "runtimeSourceRevision": lineage["runtimeSourceRevision"],
    }


def _training_attestation(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    prepared_execution_plan = _verified_execution_plan(
        config.get("runExecutionPlan")
    )
    controlled = manifest.get("controlledTrainingConfig")
    if not isinstance(controlled, Mapping):
        raise RuntimeError("Variant manifest lacks controlled training config")
    missing_controlled = set(controlled) - set(config)
    if missing_controlled:
        raise RuntimeError(
            "Effective training config lacks controlled fields: "
            + ", ".join(sorted(missing_controlled))
        )
    effective = {key: config[key] for key in controlled}
    effective_digest = canonical_sha256(effective)
    controlled_digest = canonical_sha256(dict(controlled))
    if (
        effective_digest != controlled_digest
        or effective_digest != manifest.get("trainingConfigSHA256")
    ):
        raise RuntimeError("Effective training config drifted from the controlled variant")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise RuntimeError("Variant manifest lacks dataset lineage")
    return {
        "schema": TRAINING_VARIANT_ATTESTATION_SCHEMA,
        "variant": manifest["variant"],
        "variantManifestSHA256": manifest["variantManifestSHA256"],
        "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
        "laneHashes": {
            name: contract["sha256"]
            for name, contract in sorted(datasets.items())
            if isinstance(contract, Mapping) and isinstance(contract.get("sha256"), str)
        },
        "effectiveTrainingConfigSHA256": effective_digest,
        "trainingConfigInvariantSHA256": manifest[
            "trainingConfigInvariantSHA256"
        ],
        "baseModelRevision": manifest["baseModelRevision"],
        "baseModelIndexDigest": manifest["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": manifest[
            "baseModelIndexReferencedShardNames"
        ],
        "baseModelIndexShardBindingSHA256": manifest[
            "baseModelIndexShardBindingSHA256"
        ],
        "baseModelArtifactDigest": manifest["baseModelArtifactDigest"],
        "baseModelWeightShards": manifest["baseModelWeightShards"],
        "baseModelTokenizerDigest": manifest["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": manifest["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": manifest[
            "baseModelTokenizerClosureSHA256"
        ],
        "trainingEnvironmentLockSHA256": manifest[
            "trainingEnvironmentLockSHA256"
        ],
        "trainingEnvironmentSHA256": config["trainingEnvironmentSHA256"],
        "trainingCodeSHA256": config["trainingCodeSHA256"],
        "trainingDependencyLockSHA256": config[
            "trainingDependencyLockSHA256"
        ],
        "requirementsSHA256": config["requirementsSHA256"],
        "executionPlanSHA256": prepared_execution_plan["executionPlanSHA256"],
        "runtimeImageBindingStatus": config[
            "trainingRuntimeImageBindingStatus"
        ],
        "runtimeImageBindingVerified": config[
            "trainingRuntimeImageBindingVerified"
        ],
        **{field: config[field] for field in RUNTIME_SOURCE_FIELDS},
        **{field: config[field] for field in UBUNTU_SOURCE_INTEGRITY_FIELDS},
    }


def _preparation_owner_record(
    *,
    root: Path,
    dataset_source: Path,
    run_root: Path,
    agents: Sequence[str],
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
    prepared_execution_plan: Mapping[str, Any],
    source_integrity: Mapping[str, Any],
    precreated_bind_root: bool = False,
) -> dict[str, Any]:
    runtime_source = local_runtime_source(
        root,
        source_integrity=source_integrity,
    )
    unsigned: dict[str, Any] = {
        "schema": PREPARATION_OWNER_SCHEMA_VERSION,
        "runID": run_root.name,
        "runRoot": str(run_root),
        "sourceDatasetRoot": str(dataset_source.resolve()),
        "agents": list(agents),
        "variant": variant,
        "seed": seed,
        "baseModelOverride": base_model_override,
        "containerImageDigest": container_digest,
        "executionPlanSHA256": prepared_execution_plan["executionPlanSHA256"],
        "runRootInitializationMode": (
            "precreated_bind_root"
            if precreated_bind_root
            else "atomic_sibling_promotion"
        ),
        **runtime_source,
        **source_integrity_fields(source_integrity),
    }
    return {
        **unsigned,
        "preparationOwnerSHA256": canonical_sha256(unsigned),
    }


def _initialize_preparation_root(
    run_root: Path,
    owner: Mapping[str, Any],
    *,
    precreated_bind_root: bool = False,
) -> None:
    if precreated_bind_root:
        identity = _private_directory_identity(
            run_root,
            label="Precreated preparation bind root",
            require_parent_device=False,
        )
        if any(run_root.iterdir()):
            raise RuntimeError(
                "Precreated preparation bind root must be empty before its owner "
                "record is committed"
            )
        write_object(run_root / PREPARATION_OWNER_FILENAME, owner)
        if (
            _private_directory_identity(
                run_root,
                label="Precreated preparation bind root",
                require_parent_device=False,
            )
            != identity
        ):
            raise RuntimeError("Precreated preparation bind root changed during initialization")
        _fsync_directory(run_root, label="the precreated preparation owner")
        return
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(f"Run root already exists: {run_root}")
    parent = run_root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(f"Run-root parent is not a regular directory: {parent}")
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{run_root.name}.prepare-",
            dir=parent,
        )
    )
    try:
        write_object(staging / PREPARATION_OWNER_FILENAME, owner)
        os.replace(staging, run_root)
        staging = None
        _fsync_directory(parent, label="the initialized run root")
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _make_private_owned_directory(path: Path) -> None:
    """Create one destination directory with recovery-safe permissions."""

    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Private snapshot directory already exists: {path}")
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700, follow_symlinks=False)
    directory_stat = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.getuid()
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
    ):
        raise RuntimeError(
            f"Private snapshot directory is not process-owned mode 0700: {path}"
        )


def _copy_private_regular_file(source: Path, destination: Path) -> None:
    """Copy one regular file without carrying read-only source metadata.

    A partial destination deliberately remains mode 0600 if copying is
    interrupted. The preparation-owner recovery path can therefore remove the
    incomplete tree without first trusting or changing source-derived modes.
    """

    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Private snapshot destination already exists: {destination}")
    source_handle, source_stat = _open_regular_readonly(
        source,
        label="Prepared snapshot source file",
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        source_handle.close()
        raise RuntimeError("Prepared snapshot copying requires O_NOFOLLOW support")
    destination_descriptor: int | None = None
    try:
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
            0o600,
        )
        offset = 0
        while True:
            chunk = os.pread(source_handle.fileno(), 1 << 20, offset)
            if not chunk:
                break
            written_offset = 0
            while written_offset < len(chunk):
                written = os.write(
                    destination_descriptor,
                    chunk[written_offset:],
                )
                if written <= 0:
                    raise OSError("Short write while copying a prepared snapshot")
                written_offset += written
            offset += len(chunk)
        os.fchmod(destination_descriptor, 0o600)
        os.fsync(destination_descriptor)
        destination_stat = os.fstat(destination_descriptor)
        current_destination = os.stat(destination, follow_symlinks=False)
        if (
            not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_uid != os.getuid()
            or stat.S_IMODE(destination_stat.st_mode) != 0o600
            or current_destination.st_dev != destination_stat.st_dev
            or current_destination.st_ino != destination_stat.st_ino
        ):
            raise RuntimeError(
                f"Prepared snapshot destination is not a private regular file: {destination}"
            )
        _require_stable_descriptor(
            source_handle,
            source_stat,
            label="Prepared snapshot source file",
        )
        _require_path_matches_descriptor(
            source,
            source_stat,
            label="Prepared snapshot source file",
        )
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        source_handle.close()


def _copy_private_regular_tree(source: Path, destination: Path) -> None:
    """Snapshot a same-device regular tree into private writable paths.

    Unlike ``shutil.copytree``, this never applies source directory metadata to
    the destination. Image-baked inputs are intentionally read-only, but every
    partially copied run directory must remain removable after a process kill.
    """

    try:
        source_root_stat = os.stat(source, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"Prepared snapshot source is unavailable: {source}") from exc
    if source.is_symlink() or not stat.S_ISDIR(source_root_stat.st_mode):
        raise RuntimeError(
            f"Prepared snapshot source must be a regular directory: {source}"
        )
    source_device = source_root_stat.st_dev
    _make_private_owned_directory(destination)

    def copy_directory(current_source: Path, current_destination: Path) -> None:
        before = os.stat(current_source, follow_symlinks=False)
        if (
            not stat.S_ISDIR(before.st_mode)
            or before.st_dev != source_device
            or current_source.is_symlink()
        ):
            raise RuntimeError(
                f"Prepared snapshot contains an unsafe directory: {current_source}"
            )
        try:
            with os.scandir(current_source) as scanner:
                entries = sorted(scanner, key=lambda item: item.name)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to enumerate prepared snapshot source: {current_source}"
            ) from exc
        for entry in entries:
            source_child = current_source / entry.name
            destination_child = current_destination / entry.name
            child_stat = os.stat(source_child, follow_symlinks=False)
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(
                    f"Prepared snapshot source contains a symlink: {source_child}"
                )
            if child_stat.st_dev != source_device:
                raise RuntimeError(
                    f"Prepared snapshot source crosses a filesystem boundary: {source_child}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                _make_private_owned_directory(destination_child)
                copy_directory(source_child, destination_child)
            elif stat.S_ISREG(child_stat.st_mode):
                _copy_private_regular_file(source_child, destination_child)
            else:
                raise RuntimeError(
                    f"Prepared snapshot source contains a special file: {source_child}"
                )
        after = os.stat(current_source, follow_symlinks=False)
        if _file_stability_signature(after) != _file_stability_signature(before):
            raise RuntimeError(
                f"Prepared snapshot source changed while copying: {current_source}"
            )
        _fsync_directory(
            current_destination,
            label=f"prepared snapshot directory {current_destination}",
        )

    copy_directory(source, destination)
    _fsync_directory(destination.parent, label=f"prepared snapshot {destination}")


def _verified_preparation_owner(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Incomplete run lacks a regular preparation owner record")
    observed = read_object(path)
    declared = observed.get("preparationOwnerSHA256")
    unsigned = dict(observed)
    unsigned.pop("preparationOwnerSHA256", None)
    if (
        set(observed) != {*expected, "preparationOwnerSHA256"}
        or observed.get("schema") != PREPARATION_OWNER_SCHEMA_VERSION
        or re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) is None
        or canonical_sha256(unsigned) != declared
        or unsigned != dict(expected)
    ):
        raise RuntimeError("Incomplete preparation owner record failed verification")
    return observed


def _assert_incomplete_preparation_has_no_progress(
    run_root: Path,
    *,
    agents: Sequence[str],
) -> None:
    """Permit deletion only for inputs that `prepare_run` itself can create.

    The preparation-owner record is deliberately insufficient authorization to
    remove a directory: a stale record must never turn loss of the final run
    manifest into loss of checkpoints or completed artifacts.
    """

    stat_result = run_root.stat(follow_symlinks=False)
    if stat_result.st_uid != os.getuid() or stat.S_IMODE(stat_result.st_mode) != 0o700:
        raise RuntimeError(
            "Incomplete preparation root is not privately owned by this process user"
        )
    def is_owned_atomic_temp(entry: Path, targets: set[str]) -> bool:
        matching_target = next(
            (
                target
                for target in targets
                if entry.name.startswith(f".{target}.")
                and entry.name.endswith(".tmp")
                and len(entry.name) > len(target) + len("..tmp")
            ),
            None,
        )
        if matching_target is None or entry.is_symlink() or not entry.is_file():
            return False
        entry_stat = entry.stat(follow_symlinks=False)
        return (
            entry_stat.st_uid == os.getuid()
            and stat.S_IMODE(entry_stat.st_mode) & 0o077 == 0
        )

    allowed_top_level = {
        PREPARATION_OWNER_FILENAME,
        "generated",
        "configs",
        "checkpoint_lineage",
        "logs",
        "training",
        "models",
        "evaluation",
        "training_environment.json",
    }
    entries = list(run_root.iterdir())
    top_level_temp_targets = {
        "aio_run_manifest.json",
        "training_environment.json",
    }
    unexpected = sorted(
        entry.name
        for entry in entries
        if entry.name not in allowed_top_level
        and not is_owned_atomic_temp(entry, top_level_temp_targets)
    )
    if unexpected:
        raise RuntimeError(
            "Refusing to remove incomplete preparation with unexpected or progressed "
            f"state: {', '.join(unexpected)}"
        )
    if any(entry.is_symlink() for entry in entries):
        raise RuntimeError("Incomplete preparation root contains a top-level symlink")

    for name in ("logs", "evaluation"):
        directory = run_root / name
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                f"Refusing to remove incomplete preparation with {name} progress"
            )

    training = run_root / "training"
    if training.exists():
        from tools.fine_tuning.unsloth.training_lineage import (
            BASE_MODEL_TOKENIZER_REQUIRED_PATHS,
        )

        tokenizer_names = set(BASE_MODEL_TOKENIZER_REQUIRED_PATHS)
        runtime_names = {
            *tokenizer_names,
            "generation_config.json",
            "model.safetensors.index.json",
        }
        snapshot_targets = {
            GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME: tokenizer_names,
            "base_model_runtime_snapshot": runtime_names,
        }

        def snapshot_target(entry: Path) -> str | None:
            if entry.name in snapshot_targets:
                return entry.name
            for target in snapshot_targets:
                prefix = f".{target}."
                suffix = entry.name.removeprefix(prefix)
                if (
                    entry.name.startswith(prefix)
                    and re.fullmatch(r"[A-Za-z0-9_]{6,}", suffix) is not None
                ):
                    return target
            return None

        def is_preparation_snapshot(entry: Path) -> bool:
            target = snapshot_target(entry)
            if target is None or entry.is_symlink() or not entry.is_dir():
                return False
            entry_stat = entry.stat(follow_symlinks=False)
            if (
                entry_stat.st_uid != os.geteuid()
                or stat.S_IMODE(entry_stat.st_mode) != 0o700
                or entry_stat.st_dev != training.stat(follow_symlinks=False).st_dev
            ):
                return False
            allowed_names = snapshot_targets[target]
            for child in entry.iterdir():
                child_stat = child.stat(follow_symlinks=False)
                is_runtime_shard = (
                    target == "base_model_runtime_snapshot"
                    and re.fullmatch(
                        r"model-[0-9]{5}-of-[0-9]{5}\.safetensors",
                        child.name,
                    )
                    is not None
                )
                if (
                    child.is_symlink()
                    or not child.is_file()
                    or child_stat.st_uid != os.geteuid()
                    or child_stat.st_dev != entry_stat.st_dev
                    or stat.S_IMODE(child_stat.st_mode) not in {0o400, 0o600, 0o644}
                    or (child.name not in allowed_names and not is_runtime_shard)
                ):
                    return False
            return True

        training_entries = list(training.iterdir())
        if any(not is_preparation_snapshot(entry) for entry in training_entries):
            raise RuntimeError(
                "Refusing to remove incomplete preparation with training progress"
            )

    models = run_root / "models"
    allowed_model_directories = {
        "lora_qwen3_bootstrap",
        "lora_qwen3_dpo",
        "lora_qwen3_gguf",
        "lora_qwen3_gguf_receipts",
    }
    if models.exists():
        model_entries = list(models.iterdir())
        if any(
            entry.name not in allowed_model_directories
            or entry.is_symlink()
            or not entry.is_dir()
            or any(entry.iterdir())
            for entry in model_entries
        ):
            raise RuntimeError(
                "Refusing to remove incomplete preparation with model or GGUF progress"
            )

    configs = run_root / "configs"
    allowed_configs = {f"{agent}.json" for agent in agents}
    if configs.exists() and any(
        (
            entry.name not in allowed_configs
            and not is_owned_atomic_temp(entry, allowed_configs)
        )
        or entry.is_symlink()
        or not entry.is_file()
        for entry in configs.iterdir()
    ):
        raise RuntimeError(
            "Refusing to remove incomplete preparation with unexpected config state"
        )

    lineage = run_root / "checkpoint_lineage"
    allowed_lineage = {
        *(f"{agent}.sft.json" for agent in agents),
        *(f"{agent}.preference.json" for agent in agents),
    }
    if lineage.exists():
        lineage_entries = list(lineage.iterdir())
        if any(
            (
                entry.name not in allowed_lineage
                and not is_owned_atomic_temp(entry, allowed_lineage)
            )
            or entry.is_symlink()
            or not entry.is_file()
            for entry in lineage_entries
        ):
            raise RuntimeError(
                "Refusing to remove incomplete preparation with unexpected checkpoint state"
            )
        from tools.fine_tuning.unsloth.train_dpo import (
            _initial_preference_checkpoint_lineage,
        )
        from tools.fine_tuning.unsloth.train_sft import _initial_sft_checkpoint_lineage

        for entry in (
            candidate
            for candidate in lineage_entries
            if candidate.name in allowed_lineage
        ):
            suffix = ".preference.json" if entry.name.endswith(".preference.json") else ".sft.json"
            agent = entry.name[: -len(suffix)]
            config_path = configs / f"{agent}.json"
            if not config_path.is_file() or config_path.is_symlink():
                raise RuntimeError(
                    "Refusing to remove checkpoint state without its prepared config"
                )
            config = read_object(config_path)
            expected = (
                _initial_preference_checkpoint_lineage(config, cfg_path=config_path)
                if suffix == ".preference.json"
                else _initial_sft_checkpoint_lineage(config, cfg_path=config_path)
            )
            if read_object(entry) != expected:
                raise RuntimeError(
                    "Refusing to remove progressed or drifted checkpoint lineage"
                )


def _durably_remove_preparation_owner(run_root: Path) -> None:
    owner_path = run_root / PREPARATION_OWNER_FILENAME
    if owner_path.is_symlink() or not owner_path.is_file():
        raise RuntimeError("Prepared run lost its regular preparation owner record")
    owner_path.unlink()
    _fsync_directory(run_root, label="the preparation-owner removal")


def _mountinfo_records() -> tuple[_MountIdentity, ...]:
    """Return canonical records from the current Linux mount namespace."""

    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return ()

    def decode_mount_path(value: str) -> str:
        return re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            value,
        )

    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("Unable to inspect mount boundaries") from exc
    records: list[_MountIdentity] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as exc:
            raise RuntimeError("Malformed Linux mountinfo record") from exc
        if separator < 6 or len(fields) < separator + 4:
            raise RuntimeError("Malformed Linux mountinfo record")
        try:
            mount_id = int(fields[0])
            parent_id = int(fields[1])
        except ValueError as exc:
            raise RuntimeError("Malformed Linux mountinfo identity") from exc
        records.append(
            _MountIdentity(
                mount_id=mount_id,
                parent_id=parent_id,
                device=fields[2],
                root=decode_mount_path(fields[3]),
                mount_point=decode_mount_path(fields[4]),
                mount_options=tuple(sorted(fields[5].split(","))),
                filesystem_type=fields[separator + 1],
                mount_source=decode_mount_path(fields[separator + 2]),
                super_options=tuple(sorted(fields[separator + 3].split(","))),
            )
        )
    return tuple(records)


def _mount_identity_payload(identity: _MountIdentity) -> dict[str, Any]:
    return {
        "mountID": identity.mount_id,
        "parentID": identity.parent_id,
        "device": identity.device,
        "root": identity.root,
        "mountPoint": identity.mount_point,
        "mountOptions": list(identity.mount_options),
        "filesystemType": identity.filesystem_type,
        "mountSource": identity.mount_source,
        "superOptions": list(identity.super_options),
    }


def _exact_readonly_mount_identity(path: Path) -> _MountIdentity:
    root = os.path.abspath(path)
    matches = tuple(
        record for record in _mountinfo_records() if record.mount_point == root
    )
    if len(matches) != 1:
        raise RuntimeError(
            "Prepare-only input root must be an exact mount point in the current "
            "namespace"
        )
    identity = matches[0]
    if "ro" not in identity.mount_options or "rw" in identity.mount_options:
        raise RuntimeError(
            "Prepare-only input root must be mounted read-only in the current "
            "namespace"
        )
    return identity


def _mounted_descendants(path: Path) -> list[Path]:
    """Return Linux mount points strictly below ``path`` without resolving them."""

    root = Path(os.path.abspath(path))
    descendants: list[Path] = []
    for record in _mountinfo_records():
        candidate = Path(record.mount_point)
        if candidate != root and root in candidate.parents:
            descendants.append(candidate)
    return descendants


def _clear_private_owned_directory_contents(path: Path, *, label: str) -> None:
    """Clear an owned directory in place without following links or mounts."""

    identity = _private_directory_identity(
        path,
        label=label,
        require_parent_device=False,
    )
    mounted = _mounted_descendants(path)
    if mounted:
        raise RuntimeError(
            f"Refusing to clear {label} with nested mounts: "
            + ", ".join(str(item) for item in sorted(mounted))
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeError(f"Clearing {label} requires O_NOFOLLOW support")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    root_descriptor = os.open(path, flags)
    root_stat = os.fstat(root_descriptor)

    def clear_descriptor(descriptor: int, *, device: int) -> None:
        entries = list(os.scandir(descriptor))
        for entry in entries:
            entry_stat = os.stat(
                entry.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if entry_stat.st_uid != os.geteuid():
                raise RuntimeError(
                    f"Refusing to clear non-owned entry from {label}: {entry.name}"
                )
            if stat.S_ISLNK(entry_stat.st_mode) or stat.S_ISREG(entry_stat.st_mode):
                os.unlink(entry.name, dir_fd=descriptor)
                continue
            if not stat.S_ISDIR(entry_stat.st_mode) or entry_stat.st_dev != device:
                raise RuntimeError(
                    f"Refusing to clear special or cross-device entry from {label}: "
                    f"{entry.name}"
                )
            child_descriptor = os.open(entry.name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child_descriptor)
                current = os.stat(
                    entry.name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    opened.st_dev != entry_stat.st_dev
                    or opened.st_ino != entry_stat.st_ino
                    or current.st_dev != entry_stat.st_dev
                    or current.st_ino != entry_stat.st_ino
                ):
                    raise RuntimeError(
                        f"Directory entry changed while clearing {label}: {entry.name}"
                    )
                clear_descriptor(child_descriptor, device=device)
                os.fsync(child_descriptor)
            finally:
                os.close(child_descriptor)
            os.rmdir(entry.name, dir_fd=descriptor)

    try:
        clear_descriptor(root_descriptor, device=root_stat.st_dev)
        os.fsync(root_descriptor)
        current = os.stat(path, follow_symlinks=False)
        if current.st_dev != root_stat.st_dev or current.st_ino != root_stat.st_ino:
            raise RuntimeError(f"{label} changed while it was being cleared")
    finally:
        os.close(root_descriptor)
    if _private_directory_identity(
        path,
        label=label,
        require_parent_device=False,
    ) != identity:
        raise RuntimeError(f"{label} identity changed after it was cleared")


def reset_owned_run_root(run_root: Path, *, variant: str) -> dict[str, Any]:
    """Verify a completed run's ownership, then clear only its mounted contents."""

    owned = verify_owned_run(run_root, variant=variant)
    if owned.get("runRootInitializationMode") != "precreated_bind_root":
        raise RuntimeError(
            "Only an explicitly precreated bind root may be reset in place"
        )
    _clear_private_owned_directory_contents(
        run_root,
        label="owned precreated run root",
    )
    return {
        "status": "owned_bind_root_reset",
        "runRoot": str(run_root),
        "previousRunManifestSHA256": owned["runManifestSHA256"],
    }


def recover_incomplete_preparation(
    *,
    root: Path,
    dataset_source: Path,
    run_root: Path,
    allowed_parent: Path,
    agents: Sequence[str],
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
    evaluation_scope: str,
    evaluation_max_examples: int | None,
    gguf_requested: bool,
    precreated_bind_root: bool = False,
) -> dict[str, Any]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError("Recoverable preparation root must be a regular directory")
    allowed_parent = allowed_parent.resolve()
    if allowed_parent.is_symlink() or run_root.parent != allowed_parent:
        raise RuntimeError("Recoverable preparation root escaped its allowed parent")
    if (run_root / "aio_run_manifest.json").exists():
        raise RuntimeError("Refusing to reset a run that has a prepared manifest")
    prepared_execution_plan = execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=evaluation_max_examples,
        gguf_requested=gguf_requested,
    )
    source_integrity = current_source_integrity(root)
    expected = _preparation_owner_record(
        root=root,
        dataset_source=dataset_source,
        run_root=run_root,
        agents=agents,
        variant=variant,
        seed=seed,
        base_model_override=base_model_override,
        container_digest=container_digest,
        prepared_execution_plan=prepared_execution_plan,
        source_integrity=source_integrity,
        precreated_bind_root=precreated_bind_root,
    )
    expected_unsigned = dict(expected)
    expected_unsigned.pop("preparationOwnerSHA256")
    _verified_preparation_owner(
        run_root / PREPARATION_OWNER_FILENAME,
        expected_unsigned,
    )
    _reject_managed_symlinks(run_root)
    _assert_incomplete_preparation_has_no_progress(run_root, agents=agents)
    if precreated_bind_root:
        _clear_private_owned_directory_contents(
            run_root,
            label="incomplete precreated preparation root",
        )
        status = "incomplete_preparation_cleared"
    else:
        shutil.rmtree(run_root)
        _fsync_directory(allowed_parent, label="the incomplete preparation removal")
        status = "incomplete_preparation_removed"
    return {
        "status": status,
        "runRoot": str(run_root),
    }


def prepare_run(
    *,
    root: Path,
    dataset_source: Path,
    run_root: Path,
    agents: Sequence[str],
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
    evaluation_scope: str = "full",
    evaluation_max_examples: int | None = None,
    gguf_requested: bool = True,
    precreated_bind_root: bool = False,
) -> dict[str, Any]:
    if run_root.exists() and not precreated_bind_root:
        raise RuntimeError(f"Run root already exists: {run_root}")
    if precreated_bind_root:
        _private_directory_identity(
            run_root,
            label="Precreated preparation bind root",
            require_parent_device=False,
        )
    prepared_execution_plan = execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=evaluation_max_examples,
        gguf_requested=gguf_requested,
    )
    _verify_smoke_plan_against_frozen_suites(
        dataset_source,
        agents,
        prepared_execution_plan,
    )
    source_config, _, _ = validate_variant(
        dataset_source,
        agent=agents[0],
        variant=variant,
        seed=seed,
        base_model_override=base_model_override,
    )
    prepared_tokenizer_closure = _validated_base_model_tokenizer_closure(
        source_config
    )
    source_integrity = current_source_integrity(root)
    runtime_lineage, runtime_environment = _runtime_lineage(
        root=root,
        source_config=source_config,
        container_digest=container_digest,
        source_integrity=source_integrity,
    )
    owner = _preparation_owner_record(
        root=root,
        dataset_source=dataset_source,
        run_root=run_root,
        agents=agents,
        variant=variant,
        seed=seed,
        base_model_override=base_model_override,
        container_digest=container_digest,
        prepared_execution_plan=prepared_execution_plan,
        source_integrity=source_integrity,
        precreated_bind_root=precreated_bind_root,
    )
    _initialize_preparation_root(
        run_root,
        owner,
        precreated_bind_root=precreated_bind_root,
    )
    initialized_entries = list(run_root.iterdir())
    if (
        len(initialized_entries) != 1
        or initialized_entries[0].name != PREPARATION_OWNER_FILENAME
        or initialized_entries[0].is_symlink()
        or not initialized_entries[0].is_file()
    ):
        raise RuntimeError(
            "Preparation root gained unexpected contents before input snapshotting"
        )
    snapshot_root = run_root / "generated" / "fine_tuning"
    generated_root = run_root / "generated"
    _make_private_owned_directory(generated_root)
    _copy_private_regular_tree(dataset_source, snapshot_root)
    behavior_manifest_source = (
        root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    )
    behavior_manifest_snapshot = (
        run_root
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    _make_private_owned_directory(behavior_manifest_snapshot.parent)
    _copy_private_regular_file(
        behavior_manifest_source,
        behavior_manifest_snapshot,
    )
    _fsync_directory(
        behavior_manifest_snapshot.parent,
        label="the frozen behavior-manifest snapshot",
    )
    _fsync_directory(
        generated_root,
        label="the prepared generated-input snapshots",
    )
    for directory in (
        run_root / "configs",
        run_root / "checkpoint_lineage",
        run_root / "logs",
        run_root / "training",
        run_root / "models" / "lora_qwen3_bootstrap",
        run_root / "models" / "lora_qwen3_dpo",
        run_root / "models" / "lora_qwen3_gguf",
        run_root / "models" / "lora_qwen3_gguf_receipts",
        run_root / "evaluation",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    tokenizer_snapshot_path = (
        run_root / "training" / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
    )
    _create_global_tokenizer_snapshot(
        snapshot_dir=tokenizer_snapshot_path,
        config=source_config,
    )
    from tools.fine_tuning.unsloth.training_lineage import (
        DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE,
        create_private_base_model_runtime_snapshot,
        private_base_model_runtime_snapshot_required_bytes,
        verify_private_base_model_tokenizer_snapshot,
    )

    tokenizer_snapshot_verification = (
        verify_private_base_model_tokenizer_snapshot(
            tokenizer_snapshot_path,
            base_model_id=str(source_config["baseModelID"]),
            base_model_name=str(source_config["base_model_name"]),
            base_model_revision=str(source_config["baseModelRevision"]),
            tokenizer_files=source_config["baseModelTokenizerFiles"],
            tokenizer_digest=str(source_config["baseModelTokenizerDigest"]),
            tokenizer_closure_sha256=str(
                source_config["baseModelTokenizerClosureSHA256"]
            ),
        )
    )
    if source_config.get("baseModelID") != EXPECTED_ADAPTER_GGUF_BASE_MODEL_ID:
        raise RuntimeError(
            "Private runtime snapshot registry supports only the pinned Qwen base"
        )
    required_runtime_bytes = private_base_model_runtime_snapshot_required_bytes(
        weight_shards=source_config["baseModelWeightShards"],
        tokenizer_files=source_config["baseModelTokenizerFiles"],
        generation_config_file=DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE,
    )
    if shutil.disk_usage(run_root / "training").free < required_runtime_bytes:
        raise RuntimeError(
            "Insufficient free space for private base-model runtime snapshot"
        )
    from huggingface_hub import snapshot_download  # type: ignore

    cached_model_snapshot = Path(
        snapshot_download(
            repo_id=str(source_config["baseModelID"]),
            revision=str(source_config["baseModelRevision"]),
        )
    )
    model_runtime_snapshot_path = (
        run_root / "training" / "base_model_runtime_snapshot"
    )
    model_runtime_snapshot_verification = (
        create_private_base_model_runtime_snapshot(
            source_snapshot_dir=cached_model_snapshot,
            private_tokenizer_snapshot_dir=tokenizer_snapshot_path,
            destination=model_runtime_snapshot_path,
            base_model_id=str(source_config["baseModelID"]),
            base_model_name=str(source_config["base_model_name"]),
            base_model_revision=str(source_config["baseModelRevision"]),
            tokenizer_files=source_config["baseModelTokenizerFiles"],
            tokenizer_digest=str(source_config["baseModelTokenizerDigest"]),
            tokenizer_closure_sha256=str(
                source_config["baseModelTokenizerClosureSHA256"]
            ),
            generation_config_file=(
                DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE
            ),
            model_index_digest=str(source_config["baseModelIndexDigest"]),
            index_referenced_shard_names=source_config[
                "baseModelIndexReferencedShardNames"
            ],
            index_shard_binding_sha256=str(
                source_config["baseModelIndexShardBindingSHA256"]
            ),
            model_artifact_digest=str(
                source_config["baseModelArtifactDigest"]
            ),
            weight_shards=source_config["baseModelWeightShards"],
        )
    )

    runtime_manifest = read_object(snapshot_root / "adapter_runtime_manifest.json")
    base_by_agent = {
        item["agent"]: item.get("baseModelID")
        for item in runtime_manifest.get("adapters", [])
        if isinstance(item, Mapping) and isinstance(item.get("agent"), str)
    }
    prepared: list[dict[str, Any]] = []
    runtime_source = local_runtime_source(
        root,
        source_integrity=source_integrity,
    )
    integrity_fields = source_integrity_fields(source_integrity)
    for agent in agents:
        config, manifest, variant_root = validate_variant(
            snapshot_root,
            agent=agent,
            variant=variant,
            seed=seed,
            base_model_override=base_model_override,
        )
        for field in (
            "trainingEnvironmentLock",
            "trainingCodeManifest",
            "trainingCodeSHA256",
            "trainingDependencyLock",
            "trainingDependencyLockSHA256",
            "requirementsSHA256",
            "baseModelTokenizerFiles",
            "baseModelTokenizerClosureSHA256",
        ):
            if config.get(field) != source_config.get(field):
                raise RuntimeError(f"Shared training lineage differs for {agent}: {field}")
        if (
            _validated_base_model_tokenizer_closure(config)
            != prepared_tokenizer_closure
            or manifest.get("baseModelTokenizerFiles")
            != prepared_tokenizer_closure["files"]
            or manifest.get("baseModelTokenizerClosureSHA256")
            != prepared_tokenizer_closure[
                "baseModelTokenizerClosureSHA256"
            ]
        ):
            raise RuntimeError(
                f"Shared tokenizer closure differs for {agent}"
            )
        base_model = (
            base_model_override
            or str(base_by_agent.get(agent) or "")
            or str(config.get("base_model_name") or "")
        )
        adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
        preference_adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
        training_dir = run_root / "training" / agent
        sft_checkpoint_lineage_path = (
            run_root / "checkpoint_lineage" / f"{agent}.sft.json"
        )
        preference_checkpoint_lineage_path = (
            run_root / "checkpoint_lineage" / f"{agent}.preference.json"
        )
        gguf_path = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        config["base_model_name"] = base_model
        config["baseModelID"] = base_model
        config["baseModelTokenizerSnapshotPath"] = str(
            tokenizer_snapshot_path
        )
        config["baseModelTokenizerSnapshotVerification"] = (
            tokenizer_snapshot_verification
        )
        config["baseModelGenerationConfigFile"] = (
            DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE
        )
        config["baseModelRuntimeSnapshotPath"] = str(
            model_runtime_snapshot_path
        )
        config["baseModelRuntimeSnapshotVerification"] = (
            model_runtime_snapshot_verification
        )
        config["trainingContainerImageDigest"] = container_digest
        config["trainingContainerImageDigestSource"] = "operator_declared"
        config["trainingRuntimeImageBindingStatus"] = "manual_validation_required"
        config["trainingRuntimeImageBindingVerified"] = False
        config["runExecutionPlan"] = prepared_execution_plan
        config.update(runtime_source)
        config.update(integrity_fields)
        # A ZeroGPU cache HMAC is an ephemeral remote authorization artifact,
        # never local Ubuntu runtime lineage. Persist an explicit null so the
        # exact loader contract is complete without reusing remote authority.
        config["resolvedTrainingEnvironmentCacheAttestation"] = None
        for key in (
            "resolvedTrainingEnvironment",
            "resolvedTrainingEnvironmentSHA256",
            "resolvedTrainingEnvironmentScanAudit",
            "spaceConfigurationSHA256",
            "zeroGPUSize",
            "zeroGPUDurationSeconds",
            "observedAccelerator",
        ):
            config[key] = runtime_lineage[key]
        config["trainingEnvironmentSHA256"] = None
        from tools.fine_tuning.unsloth.train_sft import _training_environment

        environment = _training_environment(config, runtime_lineage=runtime_lineage)
        if environment["trainingEnvironmentSHA256"] != runtime_environment[
            "trainingEnvironmentSHA256"
        ]:
            raise RuntimeError(f"Training environment differs across agent configs: {agent}")
        config["trainingEnvironmentSHA256"] = environment[
            "trainingEnvironmentSHA256"
        ]
        config["dataset_dir"] = str(variant_root)
        config["variant"] = variant
        config["variantManifestSHA256"] = manifest["variantManifestSHA256"]
        config["output_dir"] = str(training_dir)
        config["adapter_output_dir"] = str(adapter_dir)
        config["dpo_output_dir"] = str(preference_adapter_dir)
        config["sftCheckpointLineagePath"] = str(sft_checkpoint_lineage_path)
        config["preferenceCheckpointLineagePath"] = str(
            preference_checkpoint_lineage_path
        )
        config["sftTokenLengthPreflightPath"] = str(
            training_dir / "sft_token_length_preflight.json"
        )
        config["preferenceTokenLengthPreflightPath"] = str(
            training_dir / "dpo" / "token_length_preflight.json"
        )
        config["adapter_gguf_output_path"] = str(gguf_path)
        config["gguf_output_dir"] = str(
            run_root
            / "models"
            / "gguf_release_bake_qwen3_bootstrap"
            / f"{agent}_merged_gguf"
        )
        config["seed"] = seed
        config["merge_adapters_by_default"] = False
        config["release_bake_enabled_by_default"] = False
        export = config.setdefault("adapterExport", {})
        if not isinstance(export, dict):
            raise RuntimeError(f"adapterExport must be an object for {agent}")
        export.update(
            {
                "trainBaseModelWeights": False,
                "mergeAdaptersByDefault": False,
                "adapterArtifact": str(adapter_dir),
                "adapterDirectory": str(adapter_dir),
                "adapterGGUFArtifact": str(gguf_path),
            }
        )
        config["variantAttestation"] = _training_attestation(config, manifest)
        config_path = run_root / "configs" / f"{agent}.json"
        write_object(config_path, config)
        from tools.fine_tuning.unsloth.train_sft import (
            _initial_sft_checkpoint_lineage,
        )
        from tools.fine_tuning.unsloth.train_dpo import (
            _initial_preference_checkpoint_lineage,
            _validate_preference_training_config,
        )

        preference_config = _validate_preference_training_config(config)

        write_object(
            sft_checkpoint_lineage_path,
            _initial_sft_checkpoint_lineage(
                config,
                cfg_path=config_path,
            ),
        )
        write_object(
            preference_checkpoint_lineage_path,
            _initial_preference_checkpoint_lineage(
                config,
                cfg_path=config_path,
            ),
        )
        prepared.append(
            {
                "agent": agent,
                "config": str(config_path),
                "configSHA256": file_sha256(config_path),
                "datasetDir": str(variant_root),
                "variantManifestSHA256": manifest["variantManifestSHA256"],
                "sftAdapterDir": str(adapter_dir),
                "sftFinalizedVariantManifest": str(
                    training_dir / "finalized_variant_manifest.json"
                ),
                "sftCheckpointLineagePath": str(sft_checkpoint_lineage_path),
                "sftTokenLengthPreflight": str(
                    training_dir / "sft_token_length_preflight.json"
                ),
                "preferenceTrainer": preference_config["preferenceTrainer"],
                "preferenceAdapterDir": str(preference_adapter_dir),
                "preferenceCheckpointLineagePath": str(
                    preference_checkpoint_lineage_path
                ),
                "preferenceTokenLengthPreflight": str(
                    training_dir / "dpo" / "token_length_preflight.json"
                ),
                "preferenceFinalizedVariantManifest": str(
                    training_dir / "dpo" / "finalized_variant_manifest.json"
                ),
                "adapterGGUF": str(gguf_path),
            }
        )
    run_manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA_VERSION,
        "runID": run_root.name,
        "runRoot": str(run_root),
        "runRootInitializationMode": (
            "precreated_bind_root"
            if precreated_bind_root
            else "atomic_sibling_promotion"
        ),
        "adapterFirst": True,
        "trainBaseModelWeights": False,
        "freshRun": True,
        "variant": variant,
        "seed": seed,
        "containerImageDigest": container_digest,
        "sourceDatasetRoot": str(dataset_source.resolve()),
        "snapshotDatasetRoot": str(snapshot_root),
        "adapterRepoID": runtime_manifest.get("adapterRepoID"),
        "adapterRuntimeManifestFileSHA256": file_sha256(
            snapshot_root / "adapter_runtime_manifest.json"
        ),
        "behaviorManifest": str(behavior_manifest_snapshot),
        "behaviorManifestFileSHA256": file_sha256(behavior_manifest_snapshot),
        "trainingEnvironment": runtime_environment,
        "executionPlan": prepared_execution_plan,
        "baseModelID": prepared_tokenizer_closure["baseModelID"],
        "baseModelRevision": prepared_tokenizer_closure[
            "baseModelRevision"
        ],
        "baseModelTokenizerDigest": next(
            item["sha256"]
            for item in prepared_tokenizer_closure["files"]
            if item["path"] == "tokenizer.json"
        ),
        "baseModelTokenizerFiles": prepared_tokenizer_closure["files"],
        "baseModelTokenizerClosureSHA256": prepared_tokenizer_closure[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelTokenizerSnapshotPath": str(tokenizer_snapshot_path),
        "baseModelTokenizerSnapshotVerification": (
            tokenizer_snapshot_verification
        ),
        "baseModelGenerationConfigFile": (
            DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE
        ),
        "baseModelRuntimeSnapshotPath": str(model_runtime_snapshot_path),
        "baseModelRuntimeSnapshotVerification": (
            model_runtime_snapshot_verification
        ),
        **runtime_source,
        **integrity_fields,
        "agents": prepared,
    }
    run_manifest["runManifestSHA256"] = canonical_sha256(run_manifest)
    write_object(run_root / "training_environment.json", runtime_environment)
    expected_precommit_entries = {
        PREPARATION_OWNER_FILENAME,
        "generated",
        "configs",
        "checkpoint_lineage",
        "logs",
        "training",
        "models",
        "evaluation",
        "training_environment.json",
    }
    observed_precommit_entries = {entry.name for entry in run_root.iterdir()}
    if observed_precommit_entries != expected_precommit_entries:
        raise RuntimeError(
            "Preparation root contains unexpected state before manifest commit"
        )
    _reject_managed_symlinks(run_root)
    # The self-hashed manifest is the preparation commit record and is written
    # last. Its presence therefore proves every earlier prepared input exists.
    write_object(run_root / "aio_run_manifest.json", run_manifest)
    _durably_remove_preparation_owner(run_root)
    return run_manifest


def _verified_run_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "aio_run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"Missing regular Lumen run manifest: {manifest_path}")
    manifest = read_object(manifest_path)
    declared = manifest.get("runManifestSHA256")
    unsigned = dict(manifest)
    unsigned.pop("runManifestSHA256", None)
    if canonical_sha256(unsigned) != declared:
        raise RuntimeError("Prepared run manifest integrity check failed")
    verify_embedded_source_integrity(manifest)
    manifest_agents = manifest.get("agents")
    prepared_execution_plan = _verified_execution_plan(
        manifest.get("executionPlan")
    )
    if (
        manifest.get("schema") != RUN_SCHEMA_VERSION
        or manifest.get("adapterFirst") is not True
        or manifest.get("trainBaseModelWeights") is not False
        or manifest.get("runID") != run_root.name
        or manifest.get("runRoot") != str(run_root)
        or manifest.get("runRootInitializationMode")
        not in {"atomic_sibling_promotion", "precreated_bind_root"}
        or not isinstance(manifest_agents, list)
        or not manifest_agents
        or any(
            not isinstance(item, Mapping) or item.get("agent") not in AGENTS
            for item in manifest_agents
        )
        or len({item["agent"] for item in manifest_agents}) != len(manifest_agents)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}",
            str(manifest.get("runID") or ""),
        )
        is None
    ):
        raise RuntimeError("Prepared run manifest does not own this exact run root")
    shared_tokenizer_snapshot_verification: dict[str, Any] | None = None
    shared_runtime_snapshot_verification: dict[str, Any] | None = None
    for prepared_agent in manifest_agents:
        agent = str(prepared_agent["agent"])
        config_path = run_root / "configs" / f"{agent}.json"
        if config_path.is_symlink() or not config_path.is_file():
            raise RuntimeError(
                f"Prepared run config is not a regular file for {agent}: {config_path}"
            )
        if (
            prepared_agent.get("config") != str(config_path)
            or prepared_agent.get("configSHA256") != file_sha256(config_path)
        ):
            raise RuntimeError(
                f"Prepared run config binding failed verification for {agent}"
            )
        prepared_config = read_object(config_path)
        prepared_tokenizer_closure = _validated_base_model_tokenizer_closure(
            prepared_config
        )
        if shared_tokenizer_snapshot_verification is None:
            shared_tokenizer_snapshot_verification = (
                _verified_private_tokenizer_snapshot_binding(prepared_config)
            )
            shared_runtime_snapshot_verification = (
                _verified_private_base_model_runtime_snapshot_binding(
                    prepared_config
                )
            )
        tokenizer_snapshot_verification = shared_tokenizer_snapshot_verification
        runtime_snapshot_verification = shared_runtime_snapshot_verification
        if runtime_snapshot_verification is None:  # pragma: no cover - paired above.
            raise RuntimeError("Prepared shared runtime snapshot was not verified")
        _assert_config_matches_private_snapshot_proof(
            prepared_config,
            tokenizer_snapshot_verification,
            field_bindings=PRIVATE_TOKENIZER_SNAPSHOT_CONFIG_PROOF_FIELDS,
            label="private-tokenizer snapshot",
        )
        _assert_config_matches_private_snapshot_proof(
            prepared_config,
            runtime_snapshot_verification,
            field_bindings=PRIVATE_RUNTIME_SNAPSHOT_CONFIG_PROOF_FIELDS,
            label="base-model runtime snapshot",
        )
        expected_tokenizer_snapshot_path = str(
            run_root / "training" / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
        )
        expected_runtime_snapshot_path = str(
            run_root / "training" / "base_model_runtime_snapshot"
        )
        if (
            manifest.get("baseModelID")
            != prepared_tokenizer_closure["baseModelID"]
            or manifest.get("baseModelRevision")
            != prepared_tokenizer_closure["baseModelRevision"]
            or manifest.get("baseModelTokenizerDigest")
            != next(
                item["sha256"]
                for item in prepared_tokenizer_closure["files"]
                if item["path"] == "tokenizer.json"
            )
            or manifest.get("baseModelTokenizerFiles")
            != prepared_tokenizer_closure["files"]
            or manifest.get("baseModelTokenizerClosureSHA256")
            != prepared_tokenizer_closure[
                "baseModelTokenizerClosureSHA256"
            ]
            or prepared_config.get("baseModelTokenizerSnapshotPath")
            != expected_tokenizer_snapshot_path
            or manifest.get("baseModelTokenizerSnapshotPath")
            != expected_tokenizer_snapshot_path
            or prepared_config.get(
                "baseModelTokenizerSnapshotVerification"
            )
            != tokenizer_snapshot_verification
            or manifest.get("baseModelTokenizerSnapshotVerification")
            != tokenizer_snapshot_verification
            or prepared_config.get("baseModelRuntimeSnapshotPath")
            != expected_runtime_snapshot_path
            or manifest.get("baseModelRuntimeSnapshotPath")
            != expected_runtime_snapshot_path
            or prepared_config.get("baseModelRuntimeSnapshotVerification")
            != runtime_snapshot_verification
            or manifest.get("baseModelRuntimeSnapshotVerification")
            != runtime_snapshot_verification
            or prepared_config.get("baseModelGenerationConfigFile")
            != manifest.get("baseModelGenerationConfigFile")
        ):
            raise RuntimeError(
                f"Prepared run tokenizer closure drifted for {agent}"
            )
        from tools.fine_tuning.unsloth.train_sft import (
            _validate_sft_checkpoint_lineage_static,
        )
        from tools.fine_tuning.unsloth.train_dpo import (
            _validate_preference_checkpoint_lineage_static,
        )

        _validate_sft_checkpoint_lineage_static(
            prepared_config,
            cfg_path=config_path,
        )
        _validate_preference_checkpoint_lineage_static(
            prepared_config,
            cfg_path=config_path,
        )
        variant_attestation = prepared_config.get("variantAttestation")
        if (
            prepared_config.get("runExecutionPlan") != prepared_execution_plan
            or not isinstance(variant_attestation, Mapping)
            or variant_attestation.get("executionPlanSHA256")
            != prepared_execution_plan["executionPlanSHA256"]
            or variant_attestation.get("baseModelTokenizerFiles")
            != prepared_tokenizer_closure["files"]
            or variant_attestation.get(
                "baseModelTokenizerClosureSHA256"
            )
            != prepared_tokenizer_closure[
                "baseModelTokenizerClosureSHA256"
            ]
        ):
            raise RuntimeError(
                f"Prepared run execution plan drifted from the config for {agent}"
            )
        expected_variant_root = (
            run_root
            / "generated"
            / "fine_tuning"
            / agent
            / "experiments"
            / str(manifest.get("variant") or "")
        )
        if prepared_agent.get("datasetDir") != str(expected_variant_root):
            raise RuntimeError(
                f"Prepared variant path drifted for {agent}"
            )
        variant_manifest_path = expected_variant_root / "variant_manifest.json"
        variant_manifest = _verify_manifest_integrity(variant_manifest_path)
        expected_variant_attestation = _training_attestation(
            prepared_config,
            variant_manifest,
        )
        if (
            variant_attestation != expected_variant_attestation
            or expected_variant_attestation.get(
                "effectiveTrainingConfigSHA256"
            )
            != variant_manifest.get("trainingConfigSHA256")
            or expected_variant_attestation.get(
                "trainingConfigInvariantSHA256"
            )
            != variant_manifest.get("trainingConfigInvariantSHA256")
            or variant_manifest.get("baseModelTokenizerFiles")
            != prepared_tokenizer_closure["files"]
            or variant_manifest.get("baseModelTokenizerClosureSHA256")
            != prepared_tokenizer_closure[
                "baseModelTokenizerClosureSHA256"
            ]
        ):
            raise RuntimeError(
                f"Prepared variant tokenizer closure drifted for {agent}"
            )
    return manifest


def verify_owned_run(run_root: Path, *, variant: str) -> dict[str, Any]:
    manifest = _verified_run_manifest(run_root)
    if manifest.get("variant") != variant:
        raise RuntimeError("Existing run directory belongs to another experiment variant")
    return {
        "status": "owned_run",
        "runID": manifest["runID"],
        "variant": variant,
        "runManifestSHA256": manifest["runManifestSHA256"],
        "runRootInitializationMode": manifest["runRootInitializationMode"],
    }


def _reject_managed_symlinks(run_root: Path) -> None:
    for name in (
        "configs",
        "checkpoint_lineage",
        "logs",
        "training",
        "models",
        "evaluation",
        "generated",
    ):
        managed = run_root / name
        if managed.is_symlink() or (managed.exists() and not managed.is_dir()):
            raise RuntimeError(f"Managed run path is not a regular directory: {managed}")
        if managed.is_dir():
            for entry in managed.rglob("*"):
                if entry.is_symlink():
                    raise RuntimeError(f"Managed run path contains a symlink: {entry}")


def _expected_agent_paths(run_root: Path, agent: str) -> dict[str, Path]:
    return {
        "config": run_root / "configs" / f"{agent}.json",
        "dataset_dir": (
            run_root
            / "generated"
            / "fine_tuning"
            / agent
            / "experiments"
        ),
        "output_dir": run_root / "training" / agent,
        "adapter_output_dir": run_root / "models" / "lora_qwen3_bootstrap" / agent,
        "dpo_output_dir": run_root / "models" / "lora_qwen3_dpo" / agent,
        "sftCheckpointLineagePath": (
            run_root / "checkpoint_lineage" / f"{agent}.sft.json"
        ),
        "preferenceCheckpointLineagePath": (
            run_root / "checkpoint_lineage" / f"{agent}.preference.json"
        ),
        "sftTokenLengthPreflightPath": (
            run_root / "training" / agent / "sft_token_length_preflight.json"
        ),
        "preferenceTokenLengthPreflightPath": (
            run_root / "training" / agent / "dpo" / "token_length_preflight.json"
        ),
        "adapter_gguf_output_path": (
            run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        ),
        "gguf_output_dir": (
            run_root
            / "models"
            / "gguf_release_bake_qwen3_bootstrap"
            / f"{agent}_merged_gguf"
        ),
    }


def _verify_prepared_agent_entry(
    entry: Mapping[str, Any],
    *,
    run_root: Path,
    agent: str,
    config_sha256: str,
    variant_root: Path,
    variant_manifest_sha256: str,
    preference_trainer: str,
) -> None:
    paths = _expected_agent_paths(run_root, agent)
    expected = {
        "agent": agent,
        "config": str(paths["config"]),
        "configSHA256": config_sha256,
        "datasetDir": str(variant_root),
        "variantManifestSHA256": variant_manifest_sha256,
        "sftAdapterDir": str(paths["adapter_output_dir"]),
        "sftFinalizedVariantManifest": str(
            paths["output_dir"] / "finalized_variant_manifest.json"
        ),
        "sftCheckpointLineagePath": str(paths["sftCheckpointLineagePath"]),
        "sftTokenLengthPreflight": str(paths["sftTokenLengthPreflightPath"]),
        "preferenceTrainer": preference_trainer,
        "preferenceAdapterDir": str(paths["dpo_output_dir"]),
        "preferenceCheckpointLineagePath": str(
            paths["preferenceCheckpointLineagePath"]
        ),
        "preferenceTokenLengthPreflight": str(
            paths["preferenceTokenLengthPreflightPath"]
        ),
        "preferenceFinalizedVariantManifest": str(
            paths["output_dir"] / "dpo" / "finalized_variant_manifest.json"
        ),
        "adapterGGUF": str(paths["adapter_gguf_output_path"]),
    }
    if dict(entry) != expected:
        drifted = sorted(
            key
            for key in set(entry) | set(expected)
            if entry.get(key) != expected.get(key)
            or key not in entry
            or key not in expected
        )
        raise RuntimeError(
            f"Prepared agent ownership entry drifted for {agent}: "
            + ", ".join(drifted)
        )


def validate_prepared_runtime(
    *,
    root: Path,
    run_root: Path,
    agents: Sequence[str],
    variant: str,
    container_digest: str,
    evaluation_scope: str = "full",
    evaluation_max_examples: int | None = None,
    gguf_requested: bool = True,
    observe_runtime: bool = True,
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth import evaluate_adapter
    from tools.fine_tuning.unsloth.train_dpo import (
        _validate_preference_training_config,
    )

    manifest = _verified_run_manifest(run_root)
    requested_execution_plan = execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=evaluation_max_examples,
        gguf_requested=gguf_requested,
    )
    _reject_managed_symlinks(run_root)
    manifest_agents = manifest.get("agents")
    if (
        manifest.get("variant") != variant
        or manifest.get("containerImageDigest") != container_digest
        or manifest.get("executionPlan") != requested_execution_plan
        or not isinstance(manifest_agents, list)
        or any(not isinstance(item, Mapping) for item in manifest_agents)
        or [item.get("agent") for item in manifest_agents] != list(agents)
    ):
        raise RuntimeError("Resume request does not match the prepared run manifest")
    prepared_by_agent = {str(item["agent"]): item for item in manifest_agents}
    current_integrity = current_source_integrity(root)
    current_runtime_source = local_runtime_source(
        root,
        source_integrity=current_integrity,
    )
    current_integrity_fields = source_integrity_fields(current_integrity)
    if any(
        manifest.get(field) != current_runtime_source[field]
        for field in RUNTIME_SOURCE_FIELDS
    ) or any(
        manifest.get(field) != current_integrity_fields[field]
        for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
    ):
        raise RuntimeError("Resume source revision drifted from the prepared run")
    snapshot_root = run_root / "generated" / "fine_tuning"
    runtime_manifest_path = snapshot_root / "adapter_runtime_manifest.json"
    training_environment_path = run_root / "training_environment.json"
    if (
        runtime_manifest_path.is_symlink()
        or not runtime_manifest_path.is_file()
        or file_sha256(runtime_manifest_path)
        != manifest.get("adapterRuntimeManifestFileSHA256")
        or read_object(runtime_manifest_path).get("adapterRepoID")
        != manifest.get("adapterRepoID")
    ):
        raise RuntimeError("Prepared adapter runtime manifest drifted")
    if (
        training_environment_path.is_symlink()
        or not training_environment_path.is_file()
        or read_object(training_environment_path) != manifest.get("trainingEnvironment")
    ):
        raise RuntimeError("Prepared training environment record drifted")
    behavior_manifest_path = (
        run_root
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    if (
        behavior_manifest_path.is_symlink()
        or not behavior_manifest_path.is_file()
        or manifest.get("behaviorManifest") != str(behavior_manifest_path)
        or manifest.get("behaviorManifestFileSHA256")
        != file_sha256(behavior_manifest_path)
    ):
        raise RuntimeError("Prepared behavior manifest drifted")
    evaluation_module = evaluate_adapter._load_evaluation_module()
    tool_contracts, allowed_slots, _ = evaluate_adapter.load_behavior_contract(
        behavior_manifest_path
    )
    evaluation_prompt_preflights: dict[str, dict[str, Any]] = {}
    seed = manifest.get("seed")
    if type(seed) is not int:
        raise RuntimeError("Prepared run manifest has an invalid seed")
    # _verified_run_manifest() has already live-reverified each shared private
    # snapshot once for this command and proved that every prepared config is
    # bound to the same exact evidence.  Reuse that command-local evidence here
    # instead of hashing the multi-gigabyte runtime snapshot once per agent.
    tokenizer_snapshot_verification = manifest.get(
        "baseModelTokenizerSnapshotVerification"
    )
    runtime_snapshot_verification = manifest.get(
        "baseModelRuntimeSnapshotVerification"
    )
    if not isinstance(tokenizer_snapshot_verification, Mapping) or not isinstance(
        runtime_snapshot_verification, Mapping
    ):
        raise RuntimeError("Prepared run lacks verified private snapshot evidence")
    for agent in agents:
        paths = _expected_agent_paths(run_root, agent)
        config_path = paths["config"]
        prepared_entry = prepared_by_agent[agent]
        if (
            config_path.is_symlink()
            or not config_path.is_file()
            or prepared_entry.get("config") != str(config_path)
            or prepared_entry.get("configSHA256") != file_sha256(config_path)
        ):
            raise RuntimeError(f"Prepared config integrity check failed for {agent}")
        prepared_config = read_object(config_path)
        prepared_tokenizer_closure = _validated_base_model_tokenizer_closure(
            prepared_config
        )
        expected_tokenizer_snapshot_path = str(
            run_root / "training" / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
        )
        expected_runtime_snapshot_path = str(
            run_root / "training" / "base_model_runtime_snapshot"
        )
        preference_config = _validate_preference_training_config(
            prepared_config
        )
        verify_embedded_source_integrity(prepared_config)
        pending_config, pending_manifest, variant_root = validate_variant(
            snapshot_root,
            agent=agent,
            variant=variant,
            seed=seed,
            base_model_override=str(prepared_config.get("base_model_name") or ""),
        )
        controlled = pending_manifest.get("controlledTrainingConfig")
        if (
            not isinstance(controlled, Mapping)
            or any(
                prepared_config.get(field) != pending_config.get(field)
                or pending_config.get(field) != value
                for field, value in controlled.items()
            )
        ):
            raise RuntimeError(
                f"Prepared controlled training config drifted for {agent}"
            )
        expected_variant_attestation = _training_attestation(
            prepared_config,
            pending_manifest,
        )
        if (
            expected_variant_attestation.get("effectiveTrainingConfigSHA256")
            != pending_manifest.get("trainingConfigSHA256")
            or prepared_config.get("variantAttestation")
            != expected_variant_attestation
        ):
            raise RuntimeError(
                f"Prepared variant attestation drifted for {agent}"
            )
        evaluation_path = snapshot_root / agent / "eval.jsonl"
        evaluation_records, evaluation_sha256 = (
            evaluate_adapter.load_evaluation_records(
                evaluation_path,
                agent=agent,
                evaluation_module=evaluation_module,
            )
        )
        contamination = pending_manifest.get("contamination")
        if (
            not isinstance(contamination, Mapping)
            or evaluation_sha256 != pending_manifest.get("frozenEvaluationSHA256")
            or evaluation_sha256
            != contamination.get("evaluationRecordsSHA256")
        ):
            raise RuntimeError(
                f"Prepared frozen evaluation binding drifted for {agent}"
            )
        evaluate_adapter.validate_scoring_contracts(
            evaluation_records,
            tool_contracts=tool_contracts,
            allowed_slots=allowed_slots,
        )
        evaluation_prompt_preflights[agent] = (
            evaluate_adapter.evaluation_prompt_preflight(
                evaluation_records,
                agent=agent,
                tool_contracts=tool_contracts,
            )
        )
        _verify_prepared_agent_entry(
            prepared_entry,
            run_root=run_root,
            agent=agent,
            config_sha256=file_sha256(config_path),
            variant_root=variant_root,
            variant_manifest_sha256=str(
                pending_manifest.get("variantManifestSHA256") or ""
            ),
            preference_trainer=preference_config["preferenceTrainer"],
        )
        expected_paths = {
            **paths,
            "dataset_dir": variant_root,
        }
        path_drift = [
            field
            for field, expected in expected_paths.items()
            if field != "config"
            and prepared_config.get(field) != str(expected)
        ]
        export = prepared_config.get("adapterExport")
        if not isinstance(export, Mapping):
            path_drift.append("adapterExport")
        elif (
            export.get("adapterArtifact") != str(paths["adapter_output_dir"])
            or export.get("adapterDirectory") != str(paths["adapter_output_dir"])
            or export.get("adapterGGUFArtifact")
            != str(paths["adapter_gguf_output_path"])
            or export.get("trainBaseModelWeights") is not False
            or export.get("mergeAdaptersByDefault") is not False
        ):
            path_drift.append("adapterExport")
        if (
            prepared_config.get("dataset_dir") != str(variant_root)
            or prepared_config.get("variantManifestSHA256")
            != pending_manifest.get("variantManifestSHA256")
            or pending_manifest.get("baseModelTokenizerFiles")
            != prepared_tokenizer_closure["files"]
            or pending_manifest.get("baseModelTokenizerClosureSHA256")
            != prepared_tokenizer_closure[
                "baseModelTokenizerClosureSHA256"
            ]
            or manifest.get("baseModelTokenizerFiles")
            != prepared_tokenizer_closure["files"]
            or manifest.get("baseModelTokenizerClosureSHA256")
            != prepared_tokenizer_closure[
                "baseModelTokenizerClosureSHA256"
            ]
            or prepared_config.get("baseModelTokenizerSnapshotPath")
            != expected_tokenizer_snapshot_path
            or manifest.get("baseModelTokenizerSnapshotPath")
            != expected_tokenizer_snapshot_path
            or prepared_config.get(
                "baseModelTokenizerSnapshotVerification"
            )
            != tokenizer_snapshot_verification
            or manifest.get("baseModelTokenizerSnapshotVerification")
            != tokenizer_snapshot_verification
            or prepared_config.get("baseModelRuntimeSnapshotPath")
            != expected_runtime_snapshot_path
            or manifest.get("baseModelRuntimeSnapshotPath")
            != expected_runtime_snapshot_path
            or prepared_config.get("baseModelRuntimeSnapshotVerification")
            != runtime_snapshot_verification
            or manifest.get("baseModelRuntimeSnapshotVerification")
            != runtime_snapshot_verification
            or prepared_config.get("baseModelGenerationConfigFile")
            != manifest.get("baseModelGenerationConfigFile")
            or prepared_config.get("trainingContainerImageDigest")
            != container_digest
            or prepared_config.get("runExecutionPlan")
            != requested_execution_plan
            or not isinstance(manifest.get("trainingEnvironment"), Mapping)
            or prepared_config.get("trainingEnvironmentSHA256")
            != manifest["trainingEnvironment"].get("trainingEnvironmentSHA256")
            or any(
                prepared_config.get(field) != current_runtime_source[field]
                for field in RUNTIME_SOURCE_FIELDS
            )
            or any(
                prepared_config.get(field) != current_integrity_fields[field]
                for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
            )
            or path_drift
        ):
            detail = f": {', '.join(path_drift)}" if path_drift else ""
            raise RuntimeError(
                f"Prepared config or dataset snapshot drifted for {agent}{detail}"
            )
    _validated_global_tokenizer_resume_state(
        run_root=run_root,
        agents=agents,
    )
    if not observe_runtime:
        training_environment = manifest.get("trainingEnvironment")
        if not isinstance(training_environment, Mapping):
            raise RuntimeError("Prepared training environment is unavailable")
        environment_digest = training_environment.get("trainingEnvironmentSHA256")
        observed_accelerator = training_environment.get("observedAccelerator")
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(environment_digest or "")) is None
            or not isinstance(observed_accelerator, Mapping)
        ):
            raise RuntimeError("Prepared training environment evidence is invalid")
        return {
            "status": "postexit_artifacts_ready",
            "trainingEnvironmentSHA256": environment_digest,
            "observedAccelerator": dict(observed_accelerator),
            "evaluationPromptPreflights": evaluation_prompt_preflights,
        }

    config = read_object(run_root / "configs" / f"{agents[0]}.json")
    lineage, environment = _runtime_lineage(
        root=root,
        source_config=config,
        container_digest=container_digest,
        source_integrity=current_integrity,
    )
    if environment["trainingEnvironmentSHA256"] != config.get(
        "trainingEnvironmentSHA256"
    ):
        raise RuntimeError("Current runtime drifted from the prepared training environment")
    return {
        "status": "resume_ready",
        "trainingEnvironmentSHA256": environment["trainingEnvironmentSHA256"],
        "observedAccelerator": lineage["observedAccelerator"],
        "evaluationPromptPreflights": evaluation_prompt_preflights,
    }


def _gguf_artifact_name(agent: str) -> str:
    return f"lumen-{agent}-lora.gguf"


def _gguf_conversion_receipt_name(agent: str) -> str:
    return f"lumen-{agent}-lora.conversion.json"


def _gguf_owned_paths(run_root: Path, agent: str) -> tuple[Path, Path]:
    return (
        run_root / "models" / "lora_qwen3_gguf" / _gguf_artifact_name(agent),
        run_root
        / "models"
        / "lora_qwen3_gguf_receipts"
        / _gguf_conversion_receipt_name(agent),
    )


def _prepared_gguf_agents(run_root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    prepared_run = _verified_run_manifest(run_root)
    prepared_entries = prepared_run.get("agents")
    if not isinstance(prepared_entries, list) or any(
        not isinstance(item, Mapping) for item in prepared_entries
    ):
        raise RuntimeError("Prepared run lacks exact agent ownership")
    prepared_agents = tuple(str(item.get("agent") or "") for item in prepared_entries)
    if (
        not prepared_agents
        or any(agent not in AGENTS for agent in prepared_agents)
        or len(set(prepared_agents)) != len(prepared_agents)
    ):
        raise RuntimeError("Prepared run has invalid GGUF agent ownership")
    return prepared_run, prepared_agents


def _verify_gguf_inventory(
    run_root: Path,
    agents: Sequence[str],
    *,
    require_all: bool,
) -> dict[str, Path]:
    gguf_dir = run_root / "models" / "lora_qwen3_gguf"
    if gguf_dir.is_symlink() or not gguf_dir.is_dir():
        raise RuntimeError(f"Missing regular GGUF artifact directory: {gguf_dir}")
    receipt_dir = run_root / "models" / "lora_qwen3_gguf_receipts"
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise RuntimeError(
            f"Missing regular GGUF conversion-receipt directory: {receipt_dir}"
        )
    expected = {
        _gguf_artifact_name(agent): gguf_dir / _gguf_artifact_name(agent)
        for agent in agents
    }
    expected_receipts = {
        _gguf_conversion_receipt_name(agent): (
            receipt_dir / _gguf_conversion_receipt_name(agent)
        )
        for agent in agents
    }
    entries = list(gguf_dir.iterdir())
    unexpected = sorted(entry.name for entry in entries if entry.name not in expected)
    if unexpected:
        raise RuntimeError(
            "GGUF artifact directory contains unexpected entries: "
            + ", ".join(unexpected)
        )
    unsafe = sorted(
        entry.name
        for entry in entries
        if entry.is_symlink() or not entry.is_file()
    )
    if unsafe:
        raise RuntimeError(
            "GGUF artifact directory contains non-regular entries: "
            + ", ".join(unsafe)
        )
    receipt_entries = list(receipt_dir.iterdir())
    unexpected_receipts = sorted(
        entry.name for entry in receipt_entries if entry.name not in expected_receipts
    )
    if unexpected_receipts:
        raise RuntimeError(
            "GGUF conversion-receipt directory contains unexpected entries: "
            + ", ".join(unexpected_receipts)
        )
    unsafe_receipts = sorted(
        entry.name
        for entry in receipt_entries
        if entry.is_symlink() or not entry.is_file()
    )
    if unsafe_receipts:
        raise RuntimeError(
            "GGUF conversion-receipt directory contains non-regular entries: "
            + ", ".join(unsafe_receipts)
        )
    observed = {entry.name for entry in entries}
    observed_receipts = {entry.name for entry in receipt_entries}
    observed_agents = {
        agent for agent in agents if _gguf_artifact_name(agent) in observed
    }
    receipt_agents = {
        agent
        for agent in agents
        if _gguf_conversion_receipt_name(agent) in observed_receipts
    }
    if observed_agents != receipt_agents:
        raise RuntimeError(
            "GGUF artifacts and conversion receipts are not a complete per-agent pair"
        )
    if require_all and observed != set(expected):
        missing = sorted(set(expected) - observed)
        raise RuntimeError(
            "GGUF artifact directory is missing required entries: "
            + ", ".join(missing)
        )
    return {name: path for name, path in expected.items() if name in observed}


def _validated_base_model_conversion_snapshot_verification(
    value: Any,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.training_lineage import (
        PRIVATE_BASE_MODEL_CONVERSION_SNAPSHOT_VERIFICATION_SCHEMA_VERSION,
        PRIVATE_BASE_MODEL_TOKENIZER_SNAPSHOT_VERIFICATION_SCHEMA_VERSION,
    )

    if not isinstance(value, Mapping):
        raise RuntimeError("GGUF conversion lacks private base-snapshot proof")
    record = dict(value)
    declared = record.get("snapshotVerificationSHA256")
    unsigned = dict(record)
    unsigned.pop("snapshotVerificationSHA256", None)
    tokenizer_record = record.get("tokenizerSnapshotVerification")
    tokenizer_unsigned = (
        dict(tokenizer_record)
        if isinstance(tokenizer_record, Mapping)
        else {}
    )
    tokenizer_declared = tokenizer_unsigned.pop(
        "snapshotVerificationSHA256",
        None,
    )
    if (
        record.get("schemaVersion")
        != PRIVATE_BASE_MODEL_CONVERSION_SNAPSHOT_VERIFICATION_SCHEMA_VERSION
        or canonical_sha256(unsigned) != declared
        or not isinstance(tokenizer_record, Mapping)
        or tokenizer_record.get("schemaVersion")
        != PRIVATE_BASE_MODEL_TOKENIZER_SNAPSHOT_VERIFICATION_SCHEMA_VERSION
        or canonical_sha256(tokenizer_unsigned) != tokenizer_declared
        or record.get("baseModelID") != config.get("baseModelID")
        or record.get("baseModelRevision") != config.get("baseModelRevision")
        or record.get("baseModelIndexDigest")
        != config.get("baseModelIndexDigest")
        or record.get("baseModelIndexReferencedShardNames")
        != config.get("baseModelIndexReferencedShardNames")
        or record.get("baseModelIndexShardBindingSHA256")
        != config.get("baseModelIndexShardBindingSHA256")
        or record.get("baseModelArtifactDigest")
        != config.get("baseModelArtifactDigest")
        or record.get("baseModelWeightShards")
        != sorted(
            config.get("baseModelWeightShards") or [],
            key=lambda item: item.get("filename", ""),
        )
        or record.get("baseModelGenerationConfigFile")
        != config.get("baseModelGenerationConfigFile")
        or record.get("baseModelTokenizerDigest")
        != config.get("baseModelTokenizerDigest")
        or record.get("baseModelTokenizerFiles")
        != _validated_base_model_tokenizer_closure(config)["files"]
        or record.get("baseModelTokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
        or tokenizer_record.get("baseModelTokenizerFiles")
        != record.get("baseModelTokenizerFiles")
        or tokenizer_record.get("baseModelTokenizerClosureSHA256")
        != record.get("baseModelTokenizerClosureSHA256")
        or not isinstance(record.get("snapshotPath"), str)
        or not Path(record["snapshotPath"]).is_absolute()
        or record.get("snapshotPath")
        != config.get("baseModelRuntimeSnapshotPath")
        or record != config.get("baseModelRuntimeSnapshotVerification")
    ):
        raise RuntimeError("GGUF private base-snapshot proof drifted")
    return record


def _gguf_conversion_receipt_payload(
    run_root: Path,
    agent: str,
    artifact_path: Path,
    *,
    conversion_snapshot_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prepared_run, prepared_agents = _prepared_gguf_agents(run_root)
    if agent not in prepared_agents:
        raise RuntimeError(f"Prepared run does not own agent {agent}")
    prepared_entry = next(
        item for item in prepared_run["agents"] if item.get("agent") == agent
    )
    config_path = run_root / "configs" / f"{agent}.json"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or prepared_entry.get("config") != str(config_path)
    ):
        raise RuntimeError(f"Prepared GGUF config path drifted for {agent}")
    config_sha256 = file_sha256(config_path)
    if prepared_entry.get("configSHA256") != config_sha256:
        raise RuntimeError(f"Prepared GGUF config digest drifted for {agent}")
    config = read_object(config_path)
    if conversion_snapshot_verification is None:
        proof_path = (
            artifact_path.parent / GGUF_BASE_SNAPSHOT_VERIFICATION_FILENAME
        )
        if proof_path.is_symlink() or not proof_path.is_file():
            raise RuntimeError(
                "GGUF conversion snapshot verification proof is missing"
            )
        conversion_snapshot_verification = read_object(proof_path)
    conversion_verification = (
        _validated_base_model_conversion_snapshot_verification(
            conversion_snapshot_verification,
            config=config,
        )
    )

    preference = verify_preference(run_root, agent)
    preference_adapter = run_root / "models" / "lora_qwen3_dpo" / agent
    preference_manifest = preference_adapter / "adapter_artifact_manifest.json"
    finalized_path = (
        run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    )
    finalized = _verify_manifest_integrity(finalized_path)
    if (
        preference_adapter.is_symlink()
        or not preference_adapter.is_dir()
        or preference_manifest.is_symlink()
        or not preference_manifest.is_file()
        or finalized_path.is_symlink()
        or finalized.get("variantManifestSHA256")
        != preference.get("finalizedVariantManifestSHA256")
    ):
        raise RuntimeError(f"Current preference lineage drifted before GGUF conversion: {agent}")

    base_model_id = config.get("base_model_name")
    base_revision = config.get("baseModelRevision")
    base_tokenizer_closure = _validated_base_model_tokenizer_closure(config)
    container_digest = config.get("trainingContainerImageDigest")
    base_digest_fields = {
        field: config.get(field)
        for field in (
            "baseModelIndexDigest",
            "baseModelIndexShardBindingSHA256",
            "baseModelArtifactDigest",
            "baseModelTokenizerDigest",
        )
    }
    if (
        base_model_id != EXPECTED_ADAPTER_GGUF_BASE_MODEL_ID
        or re.fullmatch(r"[0-9a-f]{40}", str(base_revision or "")) is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is None
            for value in base_digest_fields.values()
        )
        or container_digest != prepared_run.get("containerImageDigest")
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(container_digest or ""))
        is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(prepared_run.get(field) or ""))
            is None
            or config.get(field) != prepared_run.get(field)
            for field in (
                "ubuntuOrchestrationCodeSHA256",
                "ubuntuSourceIntegritySHA256",
            )
        )
    ):
        raise RuntimeError(f"Prepared base or runtime lineage is invalid for GGUF: {agent}")

    converter = _verified_pinned_gguf_converter_script(run_root)
    reader = _verified_pinned_gguf_reader_script(run_root)
    artifact = verify_gguf_artifact(artifact_path, reader_script=reader)
    final_artifact, _ = _gguf_owned_paths(run_root, agent)
    receipt: dict[str, Any] = {
        "schema": GGUF_CONVERSION_RECEIPT_SCHEMA_VERSION,
        "agent": agent,
        "qualification": GGUF_CONVERSION_QUALIFICATION,
        "tensorEquivalenceStatus": GGUF_TENSOR_EQUIVALENCE_STATUS,
        "adapterGGUF": str(final_artifact),
        "adapterGGUFSHA256": artifact["adapterGGUFSHA256"],
        "adapterGGUFSizeBytes": artifact["adapterGGUFSizeBytes"],
        **{field: artifact[field] for field in ADAPTER_GGUF_SEMANTIC_FIELDS},
        "preferenceAdapter": str(preference_adapter),
        "preferenceAdapterSHA256": preference["adapterSHA256"],
        "preferenceAdapterManifestFileSHA256": file_sha256(preference_manifest),
        "preferenceFinalizedVariantManifest": str(finalized_path),
        "preferenceFinalizedVariantManifestSHA256": preference[
            "finalizedVariantManifestSHA256"
        ],
        "preferenceFinalizedVariantManifestFileSHA256": file_sha256(finalized_path),
        "runtimeModelBindingSHA256": preference["runtimeModelBindingSHA256"],
        "runtimeTokenizerBindingSHA256": preference[
            "runtimeTokenizerBindingSHA256"
        ],
        "config": str(config_path),
        "configSHA256": config_sha256,
        "baseModelID": base_model_id,
        "baseModelRevision": base_revision,
        **base_digest_fields,
        "baseModelTokenizerFiles": base_tokenizer_closure["files"],
        "baseModelTokenizerClosureSHA256": base_tokenizer_closure[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelTokenizerSnapshotPath": config[
            "baseModelTokenizerSnapshotPath"
        ],
        "baseModelTokenizerSnapshotVerification": config[
            "baseModelTokenizerSnapshotVerification"
        ],
        "baseModelConversionSnapshotVerification": conversion_verification,
        "trainingContainerImageDigest": container_digest,
        "ubuntuOrchestrationCodeSHA256": prepared_run[
            "ubuntuOrchestrationCodeSHA256"
        ],
        "ubuntuSourceIntegritySHA256": prepared_run[
            "ubuntuSourceIntegritySHA256"
        ],
        "llamaCppRevision": DEFAULT_LLAMA_CPP_REVISION,
        "converterPath": GGUF_CONVERTER_RELATIVE_PATH.as_posix(),
        "converterGitBlobSHA1": converter.git_blob_sha1,
        "converterFileSHA256": converter.file_sha256,
        "readerPath": GGUF_READER_RELATIVE_PATH.as_posix(),
        "readerGitBlobSHA1": reader.git_blob_sha1,
        "readerFileSHA256": reader.file_sha256,
    }
    receipt["conversionReceiptSHA256"] = canonical_sha256(receipt)
    return receipt


def _verified_gguf_conversion_receipt(
    run_root: Path,
    agent: str,
    *,
    artifact_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError(
            f"Missing regular GGUF conversion receipt for {agent}: {receipt_path}"
        )
    receipt = read_object(receipt_path)
    declared = receipt.get("conversionReceiptSHA256")
    unsigned = dict(receipt)
    unsigned.pop("conversionReceiptSHA256", None)
    if (
        set(receipt) != set(GGUF_CONVERSION_RECEIPT_FIELDS)
        or receipt.get("schema") != GGUF_CONVERSION_RECEIPT_SCHEMA_VERSION
        or receipt.get("agent") != agent
        or re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) is None
        or canonical_sha256(unsigned) != declared
    ):
        raise RuntimeError(f"GGUF conversion receipt failed integrity checks: {receipt_path}")
    expected = _gguf_conversion_receipt_payload(
        run_root,
        agent,
        artifact_path,
        conversion_snapshot_verification=receipt.get(
            "baseModelConversionSnapshotVerification"
        ),
    )
    if receipt != expected:
        raise RuntimeError(
            f"GGUF conversion receipt drifted from current lineage: {receipt_path}"
        )
    return receipt


def _gguf_verification_evidence(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
) -> dict[str, Any]:
    return {
        "adapterGGUF": receipt["adapterGGUF"],
        "adapterGGUFSHA256": receipt["adapterGGUFSHA256"],
        "adapterGGUFSizeBytes": receipt["adapterGGUFSizeBytes"],
        **{field: receipt[field] for field in ADAPTER_GGUF_SEMANTIC_FIELDS},
        "adapterGGUFConversionReceipt": str(receipt_path),
        "adapterGGUFConversionReceiptSHA256": receipt[
            "conversionReceiptSHA256"
        ],
        "adapterGGUFConversionQualification": receipt["qualification"],
        "adapterGGUFTensorEquivalenceStatus": receipt[
            "tensorEquivalenceStatus"
        ],
        "adapterGGUFRuntimeModelBindingSHA256": receipt[
            "runtimeModelBindingSHA256"
        ],
        "adapterGGUFRuntimeTokenizerBindingSHA256": receipt[
            "runtimeTokenizerBindingSHA256"
        ],
    }


def _verify_private_gguf_staging(
    run_root: Path,
    agent: str,
    *,
    expected_names: set[str],
) -> Path:
    staging_root = run_root / ".gguf-staging"
    staging_dir = staging_root / agent
    for path, label in (
        (staging_root, "GGUF staging root"),
        (staging_dir, "per-agent GGUF staging directory"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"{label} must be a regular directory: {path}")
        directory_stat = path.stat(follow_symlinks=False)
        if (
            directory_stat.st_uid != os.getuid()
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            raise RuntimeError(f"{label} must be private and process-owned: {path}")
    entries = list(staging_dir.iterdir())
    if {entry.name for entry in entries} != expected_names or any(
        entry.is_symlink()
        or not entry.is_file()
        or entry.stat(follow_symlinks=False).st_uid != os.getuid()
        or stat.S_IMODE(entry.stat(follow_symlinks=False).st_mode) & 0o077 != 0
        for entry in entries
    ):
        raise RuntimeError(
            "Per-agent GGUF staging directory has unsafe or unexpected entries: "
            f"{staging_dir}"
        )
    return staging_dir


def write_gguf_conversion_receipt(
    run_root: Path,
    agent: str,
    staging_path: Path,
) -> dict[str, Any]:
    """Bind one staged conversion to the current adapter and toolchain lineage."""

    _, prepared_agents = _prepared_gguf_agents(run_root)
    if agent not in prepared_agents:
        raise RuntimeError(f"Prepared run does not own agent {agent}")
    _reject_managed_symlinks(run_root)
    staging_dir = run_root / ".gguf-staging" / agent
    expected_name = _gguf_artifact_name(agent)
    if staging_path != staging_dir / expected_name:
        raise RuntimeError(
            f"Staged GGUF is not at the owned per-agent path: {staging_path}"
        )
    proof_path = staging_dir / GGUF_BASE_SNAPSHOT_VERIFICATION_FILENAME
    _verify_private_gguf_staging(
        run_root,
        agent,
        expected_names={expected_name, proof_path.name},
    )
    final_artifact, final_receipt = _gguf_owned_paths(run_root, agent)
    staged_receipt = staging_dir / "conversion_receipt.json"
    if any(
        path.exists() or path.is_symlink()
        for path in (final_artifact, final_receipt, staged_receipt)
    ):
        raise RuntimeError(f"Refusing to replace existing GGUF lineage for {agent}")
    _verify_gguf_inventory(run_root, prepared_agents, require_all=False)
    receipt = _gguf_conversion_receipt_payload(run_root, agent, staging_path)
    proof_path.unlink()
    _fsync_directory(staging_dir, label="the GGUF conversion staging directory")
    write_object(staged_receipt, receipt)
    _verify_private_gguf_staging(
        run_root,
        agent,
        expected_names={expected_name, staged_receipt.name},
    )
    if (
        _verified_gguf_conversion_receipt(
            run_root,
            agent,
            artifact_path=staging_path,
            receipt_path=staged_receipt,
        )
        != receipt
    ):
        raise RuntimeError(f"Staged GGUF conversion receipt drifted for {agent}")
    return {
        "agent": agent,
        "adapterGGUF": str(final_artifact),
        "adapterGGUFConversionReceipt": str(staged_receipt),
        "adapterGGUFConversionReceiptSHA256": receipt[
            "conversionReceiptSHA256"
        ],
        "adapterGGUFConversionQualification": receipt["qualification"],
        "adapterGGUFTensorEquivalenceStatus": receipt[
            "tensorEquivalenceStatus"
        ],
    }


def verify_gguf_file(run_root: Path, path: Path) -> dict[str, Any]:
    _, prepared_agents = _prepared_gguf_agents(run_root)
    inventory = _verify_gguf_inventory(
        run_root,
        prepared_agents,
        require_all=False,
    )
    expected_path = (
        run_root / "models" / "lora_qwen3_gguf" / path.name
    ).resolve()
    if (
        path.resolve() != expected_path
        or path.name not in inventory
        or path.name not in {_gguf_artifact_name(agent) for agent in prepared_agents}
    ):
        raise RuntimeError(f"GGUF artifact is not owned by the prepared run: {path}")
    agent = next(
        agent for agent in prepared_agents if _gguf_artifact_name(agent) == path.name
    )
    _, receipt_path = _gguf_owned_paths(run_root, agent)
    receipt = _verified_gguf_conversion_receipt(
        run_root,
        agent,
        artifact_path=path,
        receipt_path=receipt_path,
    )
    return _gguf_verification_evidence(receipt, receipt_path=receipt_path)


def install_gguf_file(
    run_root: Path,
    agent: str,
    staging_path: Path,
) -> dict[str, Any]:
    """Verify and durably install one staged GGUF without summary evidence."""

    _, prepared_agents = _prepared_gguf_agents(run_root)
    if agent not in prepared_agents:
        raise RuntimeError(f"Prepared run does not own agent {agent}")

    _reject_managed_symlinks(run_root)
    staging_root = run_root / ".gguf-staging"
    staging_dir = staging_root / agent
    expected_name = _gguf_artifact_name(agent)
    expected_staging_path = staging_dir / expected_name
    staged_receipt = staging_dir / "conversion_receipt.json"
    final_path, final_receipt = _gguf_owned_paths(run_root, agent)
    final_dir = final_path.parent
    final_receipt_dir = final_receipt.parent
    if staging_path != expected_staging_path:
        raise RuntimeError(
            f"Staged GGUF is not at the owned per-agent path: {staging_path}"
        )
    _verify_private_gguf_staging(
        run_root,
        agent,
        expected_names={expected_name, staged_receipt.name},
    )
    for path, label in (
        (final_dir, "GGUF artifact directory"),
        (final_receipt_dir, "GGUF conversion-receipt directory"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"{label} must be regular: {path}")
    _verify_gguf_inventory(run_root, prepared_agents, require_all=False)
    if any(
        path.exists() or path.is_symlink() for path in (final_path, final_receipt)
    ):
        raise RuntimeError(f"Refusing to replace existing GGUF lineage for {agent}")

    staged_receipt_value = _verified_gguf_conversion_receipt(
        run_root,
        agent,
        artifact_path=staging_path,
        receipt_path=staged_receipt,
    )
    for path, label in (
        (staging_path, "Staged GGUF artifact"),
        (staged_receipt, "Staged GGUF conversion receipt"),
    ):
        staged_handle, staged_stat = _open_regular_readonly(path, label=label)
        try:
            os.fsync(staged_handle.fileno())
            _require_stable_descriptor(staged_handle, staged_stat, label=label)
            _require_path_matches_descriptor(path, staged_stat, label=label)
        except OSError as exc:
            raise RuntimeError(f"Unable to durably commit {label}: {path}") from exc
        finally:
            staged_handle.close()
    if any(
        path.exists() or path.is_symlink() for path in (final_path, final_receipt)
    ):
        raise RuntimeError(f"Refusing to replace existing GGUF lineage for {agent}")
    os.replace(staging_path, final_path)
    _fsync_directory(staging_dir, label=f"staged GGUF removal {staging_path}")
    _fsync_directory(final_dir, label=f"installed GGUF artifact {final_path}")
    os.replace(staged_receipt, final_receipt)
    _fsync_directory(
        staging_dir,
        label=f"staged GGUF receipt removal {staged_receipt}",
    )
    _fsync_directory(
        final_receipt_dir,
        label=f"installed GGUF conversion receipt {final_receipt}",
    )

    installed_evidence = verify_gguf_file(run_root, final_path)
    expected_evidence = _gguf_verification_evidence(
        staged_receipt_value,
        receipt_path=final_receipt,
    )
    if installed_evidence != expected_evidence:
        raise RuntimeError(
            f"Installed GGUF evidence drifted during atomic promotion: {final_path}"
        )
    staging_dir.rmdir()
    _fsync_directory(staging_root, label="per-agent GGUF staging cleanup")
    if not any(staging_root.iterdir()):
        staging_root.rmdir()
    _fsync_directory(run_root, label="GGUF staging cleanup")
    return {"agent": agent, **installed_evidence}


def verify_gguf(run_root: Path, agent: str) -> dict[str, Any]:
    _, prepared_agents = _prepared_gguf_agents(run_root)
    if agent not in prepared_agents:
        raise RuntimeError(f"Prepared run does not own agent {agent}")
    summary = _verified_completed_summary(run_root, prepared_agents)
    if summary.get("ggufStatus") != "verified":
        raise RuntimeError(
            "Existing GGUF reuse requires a verified canonical GGUF inventory"
        )
    agent_summary = summary["agents"].get(agent)
    if (
        not isinstance(agent_summary, Mapping)
        or agent_summary.get("adapterGGUFExists") is not True
    ):
        raise RuntimeError(f"Existing summary lacks verified GGUF evidence for {agent}")
    path, _ = _gguf_owned_paths(run_root, agent)
    expected_digest = agent_summary.get("adapterGGUFSHA256")
    expected_size = agent_summary.get("adapterGGUFSizeBytes")
    gguf = verify_gguf_file(run_root, path)
    if (
        type(expected_size) is not int
        or expected_size <= 0
        or gguf["adapterGGUFSizeBytes"] != expected_size
        or re.fullmatch(r"[0-9a-f]{64}", str(expected_digest or "")) is None
        or gguf["adapterGGUFSHA256"] != expected_digest
        or any(
            agent_summary.get(field) != gguf[field]
            for field in (
                *ADAPTER_GGUF_SEMANTIC_FIELDS,
                *GGUF_CONVERSION_SUMMARY_FIELDS,
            )
        )
    ):
        raise RuntimeError(f"Existing GGUF does not match the completed summary: {path}")
    return {
        "agent": agent,
        **gguf,
    }


def _verify_manifest_integrity(path: Path) -> dict[str, Any]:
    value = read_object(path)
    digest = value.get("variantManifestSHA256")
    unsigned = dict(value)
    unsigned.pop("variantManifestSHA256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
        or canonical_sha256(unsigned) != digest
    ):
        raise RuntimeError(f"Finalized variant manifest integrity check failed: {path}")
    return value


def _reconstructed_training_precision(config: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reconstruct the canonical precision contract."""

    bf16 = config.get("bf16")
    fp16 = config.get("fp16")
    if type(bf16) is not bool or type(fp16) is not bool or bf16 == fp16:
        raise RuntimeError(
            "Training config must enable exactly one explicit precision mode"
        )
    return {
        "schemaVersion": "lumen.training-precision/1.0.0",
        "bf16": bf16,
        "fp16": fp16,
        "dtype": "bfloat16" if bf16 else "float16",
    }


def _verify_training_report(
    path: Path,
    *,
    phase: str,
    expected: Mapping[str, Any],
    configured_num_train_epochs: float,
    per_device_train_batch_size: int,
    configured_gradient_accumulation_steps: int,
    expected_precision: Mapping[str, Any],
) -> dict[str, Any]:
    if dict(expected_precision) != _reconstructed_training_precision(
        expected_precision
    ):
        raise RuntimeError(f"{phase} expected precision contract is not canonical")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Missing regular {phase} training report: {path}")
    report = read_object(path)
    drifted = [
        field for field, value in expected.items() if report.get(field) != value
    ]
    if (
        drifted
        or not isinstance(report.get("metrics"), Mapping)
        or not isinstance(report.get("evaluation_metrics"), Mapping)
        or type(report.get("train_records")) is not int
        or report["train_records"] <= 0
        or type(report.get("val_records")) is not int
        or report["val_records"] <= 0
    ):
        detail = f": {', '.join(drifted)}" if drifted else ""
        raise RuntimeError(f"{phase} training report failed lineage validation{detail}")

    def verify_metrics(
        value: Mapping[str, Any],
        *,
        name: str,
        required: frozenset[str],
    ) -> None:
        if not value:
            raise RuntimeError(f"{phase} {name} are empty")
        missing = sorted(required - set(value))
        if missing:
            raise RuntimeError(
                f"{phase} {name} lack required values: {', '.join(missing)}"
            )

        def verify_metric(metric: Any, metric_path: str) -> None:
            if isinstance(metric, Mapping):
                if not metric:
                    raise RuntimeError(f"{phase} metric {metric_path} is empty")
                for key, nested in metric.items():
                    if not isinstance(key, str) or not key:
                        raise RuntimeError(f"{phase} metric name is invalid")
                    verify_metric(nested, f"{metric_path}.{key}")
                return
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
            ):
                raise RuntimeError(
                    f"{phase} metric {metric_path} is not finite numeric evidence"
                )

        verify_metric(value, name)

    metrics = report["metrics"]
    evaluation_metrics = report["evaluation_metrics"]
    verify_metrics(
        metrics,
        name="training metrics",
        required=frozenset({"train_loss", "epoch"}),
    )
    verify_metrics(
        evaluation_metrics,
        name="evaluation metrics",
        required=frozenset({"eval_loss", "epoch"}),
    )
    completion = report.get("trainingCompletion")
    completion_fields = {
        "schema",
        "status",
        "globalStep",
        "maxSteps",
        "expectedMaxSteps",
        "trainResultGlobalStep",
        "configuredNumTrainEpochs",
        "observedEpoch",
        "trainRecordCount",
        "perDeviceTrainBatchSize",
        "gradientAccumulationSteps",
        "worldSize",
        "trainDataloaderBatchCount",
        "updateStepsPerEpoch",
        "trainMetricsVerified",
        "evaluationMetricsVerified",
        "resolvedPrecision",
    }
    if not isinstance(completion, Mapping) or set(completion) != completion_fields:
        raise RuntimeError(f"{phase} training completion evidence is not canonical")
    global_step = completion.get("globalStep")
    max_steps = completion.get("maxSteps")
    train_result_global_step = completion.get("trainResultGlobalStep")
    configured_epochs = completion.get("configuredNumTrainEpochs")
    observed_epoch = completion.get("observedEpoch")
    per_device_batch_size = completion.get("perDeviceTrainBatchSize")
    gradient_accumulation_steps = completion.get("gradientAccumulationSteps")
    if (
        type(per_device_batch_size) is int
        and per_device_batch_size > 0
        and type(gradient_accumulation_steps) is int
        and gradient_accumulation_steps > 0
    ):
        expected_batch_count = math.ceil(
            report["train_records"] / per_device_batch_size
        )
        expected_updates_per_epoch = max(
            math.ceil(expected_batch_count / gradient_accumulation_steps),
            1,
        )
        expected_max_steps = math.ceil(
            float(configured_num_train_epochs) * expected_updates_per_epoch
        )
    else:
        expected_batch_count = None
        expected_updates_per_epoch = None
        expected_max_steps = None
    if (
        completion.get("schema") != "lumen.training_completion/1.1.0"
        or completion.get("status") != "completed"
        or completion.get("trainMetricsVerified") is not True
        or completion.get("evaluationMetricsVerified") is not True
        or report.get("precision") != dict(expected_precision)
        or completion.get("resolvedPrecision") != dict(expected_precision)
        or type(global_step) is not int
        or global_step <= 0
        or type(max_steps) is not int
        or max_steps != global_step
        or completion.get("expectedMaxSteps") != expected_max_steps
        or max_steps != expected_max_steps
        or type(train_result_global_step) is not int
        or train_result_global_step != global_step
        or isinstance(configured_epochs, bool)
        or not isinstance(configured_epochs, (int, float))
        or isinstance(observed_epoch, bool)
        or not isinstance(observed_epoch, (int, float))
        or not math.isfinite(float(configured_epochs))
        or not math.isfinite(float(observed_epoch))
        or not math.isclose(
            float(configured_epochs),
            float(configured_num_train_epochs),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or completion.get("trainRecordCount") != report["train_records"]
        or per_device_batch_size != per_device_train_batch_size
        or gradient_accumulation_steps
        != configured_gradient_accumulation_steps
        or completion.get("worldSize") != 1
        or completion.get("trainDataloaderBatchCount") != expected_batch_count
        or completion.get("updateStepsPerEpoch")
        != expected_updates_per_epoch
        or not math.isclose(
            float(observed_epoch),
            float(configured_num_train_epochs),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(metrics["epoch"]),
            float(observed_epoch),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(evaluation_metrics["epoch"]),
            float(observed_epoch),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise RuntimeError(f"{phase} training completion evidence is false or incomplete")
    return report


def _verify_dpo_reference_log_prob_report(
    *,
    run_root: Path,
    agent: str,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    parent_sft_adapter_sha256: str,
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.train_dpo import (
        DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME,
        _read_preference_checkpoint_lineage,
        _read_reference_log_prob_evidence,
        _reference_log_prob_static_contract,
        _verify_reference_log_prob_lineage,
        row_to_preference,
    )

    output_dir = run_root / "training" / agent / "dpo"
    evidence_path = output_dir / DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME
    config_path = run_root / "configs" / f"{agent}.json"
    dataset_dir = Path(str(config.get("dataset_dir") or "")).resolve()
    source_rows_by_split = {
        "train": [
            row_to_preference(row)
            for row in read_jsonl(dataset_dir / "train_dpo.jsonl")
        ],
        "validation": [
            row_to_preference(row)
            for row in read_jsonl(dataset_dir / "val_dpo.jsonl")
        ],
    }
    evidence = _read_reference_log_prob_evidence(evidence_path)
    expected = _reference_log_prob_static_contract(
        config,
        cfg_path=config_path,
        parent_sft_adapter_sha256=parent_sft_adapter_sha256,
        source_rows_by_split=source_rows_by_split,
    )
    _verify_reference_log_prob_lineage(evidence, expected=expected)
    lineage_path = Path(
        str(config.get("preferenceCheckpointLineagePath") or "")
    ).resolve()
    checkpoint_record = _read_preference_checkpoint_lineage(lineage_path)
    digest = evidence["referenceLogProbEvidenceSHA256"]
    report_evidence = report.get("reference_log_prob_evidence")
    expected_report_fields = {
        "path",
        "referenceLogProbEvidenceSHA256",
        "fileSHA256",
        "reusedFromCheckpointLineage",
        "trainRowCount",
        "validationRowCount",
    }
    if (
        report.get("reference_log_probs_precomputed")
        != {"train": True, "evaluation": True}
        or report.get("checkpoint_adapter_contract")
        != {
            "referenceAdapterRemovedAfterPrecompute": True,
            "checkpointAdapterNames": ["default"],
        }
        or not isinstance(report_evidence, Mapping)
        or set(report_evidence) != expected_report_fields
        or report_evidence.get("path") != str(evidence_path)
        or report_evidence.get("referenceLogProbEvidenceSHA256") != digest
        or report_evidence.get("fileSHA256") != file_sha256(evidence_path)
        or type(report_evidence.get("reusedFromCheckpointLineage")) is not bool
        or report_evidence.get("trainRowCount")
        != evidence["splits"]["train"]["rowCount"]
        or report_evidence.get("validationRowCount")
        != evidence["splits"]["validation"]["rowCount"]
        or report_evidence.get("trainRowCount") != report.get("train_records")
        or report_evidence.get("validationRowCount") != report.get("val_records")
        or checkpoint_record.get("referenceLogProbEvidencePath")
        != str(evidence_path)
        or checkpoint_record.get("referenceLogProbEvidenceSHA256") != digest
        or checkpoint_record.get("parentSFTAdapterSHA256")
        != parent_sft_adapter_sha256
        or checkpoint_record.get("referenceSFTAdapterSHA256")
        != parent_sft_adapter_sha256
    ):
        raise RuntimeError(
            "DPO reference log-probability qualification evidence is incomplete"
        )
    return {
        "path": str(evidence_path),
        "referenceLogProbEvidenceSHA256": digest,
        "fileSHA256": file_sha256(evidence_path),
        "trainRowCount": evidence["splits"]["train"]["rowCount"],
        "validationRowCount": evidence["splits"]["validation"]["rowCount"],
    }


def _verify_finalized_variant_binding(
    config: Mapping[str, Any],
    finalized: Mapping[str, Any],
    *,
    agent: str,
    phase: str,
    expected_training_code_sha256: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from tools.fine_tuning.unsloth.train_dpo import (
        _shared_finalized_variant_validator,
    )

    validator = _shared_finalized_variant_validator()
    if not validator(
        finalized,
        agent=agent,
        expected_variant=str(config.get("variant") or ""),
        require_trained_artifact=True,
    ):
        raise RuntimeError(f"{phase} finalized variant manifest is structurally invalid")

    attestation = config.get("variantAttestation")
    artifact = finalized.get("artifact")
    datasets = finalized.get("datasets")
    if (
        not isinstance(attestation, Mapping)
        or not isinstance(artifact, Mapping)
        or not isinstance(datasets, Mapping)
    ):
        raise RuntimeError(f"{phase} finalized manifest lacks controlled lineage")

    lane_hashes = {
        name: contract.get("sha256")
        for name, contract in sorted(datasets.items())
        if isinstance(contract, Mapping) and isinstance(contract.get("sha256"), str)
    }
    expected_static = {
        "agent": agent,
        "variant": config.get("variant"),
        "sourceVariantManifestSHA256": config.get("variantManifestSHA256"),
        "baseModelID": config.get("baseModelID", config.get("base_model_name")),
        "trainingCorpusSHA256": attestation.get("trainingCorpusSHA256"),
        "trainingConfigSHA256": attestation.get(
            "effectiveTrainingConfigSHA256"
        ),
        "trainingConfigInvariantSHA256": attestation.get(
            "trainingConfigInvariantSHA256"
        ),
        "baseModelRevision": attestation.get("baseModelRevision"),
        "baseModelIndexDigest": attestation.get("baseModelIndexDigest"),
        "baseModelIndexReferencedShardNames": attestation.get(
            "baseModelIndexReferencedShardNames"
        ),
        "baseModelIndexShardBindingSHA256": attestation.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": attestation.get("baseModelArtifactDigest"),
        "baseModelWeightShards": attestation.get("baseModelWeightShards"),
        "baseModelTokenizerDigest": attestation.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": attestation.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": attestation.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "trainingEnvironmentLockSHA256": attestation.get(
            "trainingEnvironmentLockSHA256"
        ),
    }
    drifted = [
        field
        for field, expected in expected_static.items()
        if finalized.get(field) != expected
    ]
    if drifted:
        raise RuntimeError(
            f"{phase} finalized manifest drifted from controlled lineage: "
            + ", ".join(drifted)
        )
    if lane_hashes != attestation.get("laneHashes"):
        raise RuntimeError(f"{phase} finalized dataset lane hashes drifted")

    expected_runtime = {
        "trainingCodeSHA256": expected_training_code_sha256,
        "trainingDependencyLockSHA256": config.get(
            "trainingDependencyLockSHA256"
        ),
        "requirementsSHA256": config.get("requirementsSHA256"),
        "resolvedTrainingEnvironment": config.get("resolvedTrainingEnvironment"),
        "resolvedTrainingEnvironmentSHA256": config.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "spaceConfigurationSHA256": config.get("spaceConfigurationSHA256"),
        "zeroGPUSize": config.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": config.get("zeroGPUDurationSeconds"),
        "observedAccelerator": config.get("observedAccelerator"),
        **{field: config.get(field) for field in RUNTIME_SOURCE_FIELDS},
    }
    runtime_drifted = [
        field
        for field, expected in expected_runtime.items()
        if finalized.get(field) != expected
    ]
    if runtime_drifted:
        raise RuntimeError(
            f"{phase} finalized runtime lineage drifted: "
            + ", ".join(runtime_drifted)
        )

    environment = finalized.get("trainingEnvironment")
    if not isinstance(environment, Mapping):
        raise RuntimeError(f"{phase} finalized manifest lacks training environment")
    declared_environment_sha = finalized.get("trainingEnvironmentSHA256")
    if (
        canonical_sha256(dict(environment)) != declared_environment_sha
        or environment.get("containerImageDigest")
        != config.get("trainingContainerImageDigest")
        or environment.get("environmentLock") != config.get("trainingEnvironmentLock")
        or environment.get("effectiveSeed") != config.get("seed")
        or any(
            environment.get(field) != expected
            for field, expected in expected_runtime.items()
            if field
            in {
                "trainingCodeSHA256",
                "trainingDependencyLockSHA256",
                "requirementsSHA256",
                "resolvedTrainingEnvironment",
                "resolvedTrainingEnvironmentSHA256",
                "zeroGPUSize",
                "zeroGPUDurationSeconds",
                "observedAccelerator",
            }
        )
    ):
        raise RuntimeError(f"{phase} training environment binding failed")

    artifact_expected = {
        "adapterManifestSHA256": artifact.get("adapterSHA256"),
        "effectiveSeed": config.get("seed"),
        "trainingCodeSHA256": expected_training_code_sha256,
        "trainingDependencyLockSHA256": config.get(
            "trainingDependencyLockSHA256"
        ),
        "requirementsSHA256": config.get("requirementsSHA256"),
        "resolvedTrainingEnvironmentSHA256": config.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "spaceConfigurationSHA256": config.get("spaceConfigurationSHA256"),
        "zeroGPUSize": config.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": config.get("zeroGPUDurationSeconds"),
        "observedAccelerator": config.get("observedAccelerator"),
        **{field: config.get(field) for field in RUNTIME_SOURCE_FIELDS},
    }
    artifact_drifted = [
        field
        for field, expected in artifact_expected.items()
        if artifact.get(field) != expected
    ]
    if artifact_drifted:
        raise RuntimeError(
            f"{phase} adapter artifact lineage drifted: "
            + ", ".join(artifact_drifted)
        )
    return artifact, attestation


def _valid_token_length_statistics(value: Any, *, require_positive: bool) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"min", "p50", "p95", "max"}:
        return False
    values = [value[field] for field in ("min", "p50", "p95", "max")]
    minimum = 1 if require_positive else 0
    return (
        all(type(item) is int and item >= minimum for item in values)
        and values == sorted(values)
    )


GLOBAL_TOKENIZER_PREFLIGHT_SCHEMA = (
    "lumen.global-tokenizer-preflight/2.2.0"
)
GLOBAL_TOKENIZER_PREFLIGHT_FILENAME = "global_tokenizer_preflight.json"
GLOBAL_TOKENIZER_SNAPSHOT_SCHEMA = "lumen.global-tokenizer-snapshot/1.1.0"
GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME = "global_tokenizer_snapshot"
GLOBAL_TOKENIZER_SNAPSHOT_FILES = (
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _validated_base_model_tokenizer_closure(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from lumen_manifest_crawler.dataset.adapter_evaluation import (
        DEFAULT_BASE_MODEL_ID,
        DEFAULT_BASE_MODEL_REVISION,
        DEFAULT_BASE_MODEL_TOKENIZER_CLOSURE_SHA256,
        DEFAULT_BASE_MODEL_TOKENIZER_FILES,
        canonical_base_model_tokenizer_closure,
    )

    if (
        not isinstance(config.get("baseModelID"), str)
        or not config.get("baseModelID")
        or config.get("baseModelID") != config.get("base_model_name")
    ):
        raise RuntimeError(
            "baseModelID must exactly match base_model_name"
        )
    files = config.get("baseModelTokenizerFiles")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise RuntimeError("Base-model tokenizer file closure is missing")
    try:
        closure = canonical_base_model_tokenizer_closure(
            base_model_id=config.get("baseModelID"),
            base_model_revision=config.get("baseModelRevision"),
            files=files,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Base-model tokenizer file closure is invalid"
        ) from exc
    declared = config.get("baseModelTokenizerClosureSHA256")
    tokenizer_json = next(
        item for item in closure["files"]
        if item["path"] == "tokenizer.json"
    )
    if (
        canonical_sha256(closure) != declared
        or tokenizer_json["sha256"]
        != config.get("baseModelTokenizerDigest")
    ):
        raise RuntimeError("Base-model tokenizer closure digest drifted")
    if (
        closure["baseModelID"] == DEFAULT_BASE_MODEL_ID
        and closure["baseModelRevision"] == DEFAULT_BASE_MODEL_REVISION
        and (
            closure["files"] != DEFAULT_BASE_MODEL_TOKENIZER_FILES
            or declared != DEFAULT_BASE_MODEL_TOKENIZER_CLOSURE_SHA256
        )
    ):
        raise RuntimeError(
            "Pinned Qwen tokenizer closure drifted from the trusted registry"
        )
    return {
        **closure,
        "baseModelTokenizerClosureSHA256": declared,
    }


def _verified_private_tokenizer_snapshot_binding(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.training_lineage import (
        verify_private_base_model_tokenizer_snapshot,
    )

    raw_path = config.get("baseModelTokenizerSnapshotPath")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise RuntimeError("Private tokenizer snapshot path is invalid")
    observed = verify_private_base_model_tokenizer_snapshot(
        Path(raw_path),
        base_model_id=str(config.get("baseModelID") or ""),
        base_model_name=str(config.get("base_model_name") or ""),
        base_model_revision=str(config.get("baseModelRevision") or ""),
        tokenizer_files=config.get("baseModelTokenizerFiles"),
        tokenizer_digest=str(config.get("baseModelTokenizerDigest") or ""),
        tokenizer_closure_sha256=str(
            config.get("baseModelTokenizerClosureSHA256") or ""
        ),
    )
    if observed != config.get("baseModelTokenizerSnapshotVerification"):
        raise RuntimeError("Private tokenizer snapshot verification drifted")
    return observed


def _assert_config_matches_private_snapshot_proof(
    config: Mapping[str, Any],
    proof: Mapping[str, Any],
    *,
    field_bindings: Sequence[tuple[str, str]],
    label: str,
) -> None:
    drifted = [
        config_field
        for config_field, proof_field in field_bindings
        if config.get(config_field) != proof.get(proof_field)
    ]
    if drifted:
        raise RuntimeError(
            f"Prepared config {label} contract drifted from the shared "
            f"observed proof: {', '.join(drifted)}"
        )


def _verified_private_base_model_runtime_snapshot_binding(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.training_lineage import (
        verify_private_base_model_conversion_snapshot,
    )

    raw_path = config.get("baseModelRuntimeSnapshotPath")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise RuntimeError("Private base-model runtime snapshot path is invalid")
    observed = verify_private_base_model_conversion_snapshot(
        Path(raw_path),
        base_model_id=str(config.get("baseModelID") or ""),
        base_model_name=str(config.get("base_model_name") or ""),
        base_model_revision=str(config.get("baseModelRevision") or ""),
        tokenizer_files=config.get("baseModelTokenizerFiles"),
        tokenizer_digest=str(config.get("baseModelTokenizerDigest") or ""),
        tokenizer_closure_sha256=str(
            config.get("baseModelTokenizerClosureSHA256") or ""
        ),
        generation_config_file=config.get("baseModelGenerationConfigFile"),
        model_index_digest=str(config.get("baseModelIndexDigest") or ""),
        index_referenced_shard_names=config.get(
            "baseModelIndexReferencedShardNames"
        ),
        index_shard_binding_sha256=str(
            config.get("baseModelIndexShardBindingSHA256") or ""
        ),
        model_artifact_digest=str(config.get("baseModelArtifactDigest") or ""),
        weight_shards=config.get("baseModelWeightShards"),
    )
    if observed != config.get("baseModelRuntimeSnapshotVerification"):
        raise RuntimeError("Private base-model runtime snapshot verification drifted")
    return observed


def _verified_runtime_model_binding(
    value: Any,
    *,
    config: Mapping[str, Any],
    snapshot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.train_sft import RUNTIME_MODEL_BINDING_SCHEMA

    expected_keys = {
        "schemaVersion",
        "baseModelID",
        "baseModelRevision",
        "baseModelIndexDigest",
        "baseModelIndexShardBindingSHA256",
        "baseModelArtifactDigest",
        "baseModelTokenizerClosureSHA256",
        "baseModelGenerationConfigFile",
        "runtimeSnapshotVerificationSHA256",
        "runtimeSnapshotPath",
        "modelConfigSHA256",
        "modelConfigVerificationStatus",
        "sourceGenerationConfigSHA256",
        "generationConfigSHA256",
        "generationConfigSource",
        "allowedGenerationConfigTransformations",
        "runtimeLoadMaterialization",
        "localFilesOnly",
        "runtimeModelBindingSHA256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("Runtime model binding has an invalid schema")
    binding = dict(value)
    declared = binding.pop("runtimeModelBindingSHA256")
    runtime_snapshot_path = Path(
        str(config.get("baseModelRuntimeSnapshotPath") or "")
    )
    runtime_model_config = read_object(runtime_snapshot_path / "config.json")
    configured_max_length = config.get("max_seq_length")
    max_position_embeddings = runtime_model_config.get("max_position_embeddings")
    if (
        type(configured_max_length) is not int
        or configured_max_length <= 0
        or type(max_position_embeddings) is not int
        or max_position_embeddings <= 0
    ):
        raise RuntimeError("Runtime generation binding inputs are invalid")
    expected_runtime_max_length = max(
        configured_max_length,
        max_position_embeddings,
    )
    from transformers import GenerationConfig  # type: ignore

    source_generation = GenerationConfig.from_pretrained(
        str(runtime_snapshot_path),
        local_files_only=True,
    ).to_dict()
    original_generation_max_length = source_generation.get("max_length")
    if (
        not isinstance(source_generation, Mapping)
        or type(original_generation_max_length) is not int
        or original_generation_max_length <= 0
    ):
        raise RuntimeError("Private generation configuration is not canonical")
    expected_source_generation_sha256 = canonical_sha256(
        dict(source_generation)
    )
    expected_runtime_generation = dict(source_generation)
    expected_runtime_generation["max_length"] = expected_runtime_max_length
    expected_runtime_generation_sha256 = canonical_sha256(
        expected_runtime_generation
    )
    from tools.fine_tuning.unsloth.runtime_binding_smoke_gate import (
        verify_runtime_load_materialization_evidence,
    )

    verified_materialization = verify_runtime_load_materialization_evidence(
        binding.get("runtimeLoadMaterialization", {}),
        config,
        runtime_model_config,
    )
    after_generation_read = (
        _verified_private_base_model_runtime_snapshot_binding(config)
    )
    if after_generation_read != dict(snapshot_verification):
        raise RuntimeError(
            "Private base-model runtime snapshot changed while reconstructing "
            "generation configuration"
        )
    expected_static = {
        "schemaVersion": RUNTIME_MODEL_BINDING_SCHEMA,
        "baseModelID": config.get("baseModelID"),
        "baseModelRevision": config.get("baseModelRevision"),
        "baseModelIndexDigest": config.get("baseModelIndexDigest"),
        "baseModelIndexShardBindingSHA256": config.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": config.get("baseModelArtifactDigest"),
        "baseModelTokenizerClosureSHA256": config.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": config.get(
            "baseModelGenerationConfigFile"
        ),
        "runtimeSnapshotVerificationSHA256": snapshot_verification.get(
            "snapshotVerificationSHA256"
        ),
        "runtimeSnapshotPath": config.get("baseModelRuntimeSnapshotPath"),
        "modelConfigVerificationStatus": (
            "attested_runtime_observation_not_independently_reconstructed"
        ),
        "sourceGenerationConfigSHA256": expected_source_generation_sha256,
        "generationConfigSHA256": expected_runtime_generation_sha256,
        "generationConfigSource": "verified_private_generation_config_file",
        "allowedGenerationConfigTransformations": {
            "maxLength": {
                "source": "verified_runtime_model.config.max_position_embeddings",
                "sourceValue": expected_runtime_max_length,
                "originalValue": original_generation_max_length,
                "runtimeValue": expected_runtime_max_length,
            }
        },
        "runtimeLoadMaterialization": verified_materialization,
        "localFilesOnly": True,
    }
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) is None
        or canonical_sha256(binding) != declared
        or any(binding.get(field) != expected for field, expected in expected_static.items())
        or re.fullmatch(
            r"[0-9a-f]{64}", str(binding.get("modelConfigSHA256") or "")
        )
        is None
    ):
        raise RuntimeError("Runtime model binding drifted from the private base view")
    return dict(value)


def _verified_runtime_tokenizer_binding(
    value: Any,
    *,
    config: Mapping[str, Any],
    snapshot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.train_sft import RUNTIME_TOKENIZER_BINDING_SCHEMA

    expected_keys = {
        "schemaVersion",
        "baseModelID",
        "baseModelRevision",
        "baseModelTokenizerClosureSHA256",
        "runtimeSnapshotVerificationSHA256",
        "runtimeSnapshotPath",
        "backendContractSHA256",
        "allowedRuntimeTransformations",
        "runtimeTokenizerBindingSHA256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("Runtime tokenizer binding has an invalid schema")
    binding = dict(value)
    declared = binding.pop("runtimeTokenizerBindingSHA256")
    transformations = binding.get("allowedRuntimeTransformations")
    runtime_model_config_path = (
        Path(str(config.get("baseModelRuntimeSnapshotPath") or ""))
        / "config.json"
    )
    runtime_model_config = read_object(runtime_model_config_path)
    max_position_embeddings = runtime_model_config.get("max_position_embeddings")
    configured_max_length = config.get("max_seq_length")
    if (
        type(configured_max_length) is not int
        or configured_max_length <= 0
        or type(max_position_embeddings) is not int
        or max_position_embeddings <= 0
    ):
        raise RuntimeError("Runtime tokenizer maximum-length inputs are invalid")
    expected_runtime_max_length = max(
        configured_max_length,
        max_position_embeddings,
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) is None
        or canonical_sha256(binding) != declared
        or binding.get("schemaVersion") != RUNTIME_TOKENIZER_BINDING_SCHEMA
        or binding.get("baseModelID") != config.get("baseModelID")
        or binding.get("baseModelRevision") != config.get("baseModelRevision")
        or binding.get("baseModelTokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
        or binding.get("runtimeSnapshotVerificationSHA256")
        != snapshot_verification.get("snapshotVerificationSHA256")
        or binding.get("runtimeSnapshotPath")
        != config.get("baseModelRuntimeSnapshotPath")
        or re.fullmatch(
            r"[0-9a-f]{64}", str(binding.get("backendContractSHA256") or "")
        )
        is None
        or not isinstance(transformations, Mapping)
        or set(transformations)
        != {"modelMaxLength", "paddingSide", "truncationSide"}
        or transformations.get("modelMaxLength") != expected_runtime_max_length
        or transformations.get("paddingSide") != "left"
        or transformations.get("truncationSide") != "right"
    ):
        raise RuntimeError(
            "Runtime tokenizer binding drifted from the private tokenizer closure"
        )
    return dict(value)


def _verified_peft_base_model_evidence(
    value: Any,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schemaVersion",
        "baseModelID",
        "baseModelRevision",
        "adapterNames",
        "privateRuntimePathPersisted",
        "peftBaseModelIdentitySHA256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("PEFT base-model identity evidence has an invalid schema")
    evidence = dict(value)
    declared = evidence.pop("peftBaseModelIdentitySHA256")
    names = evidence.get("adapterNames")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) is None
        or canonical_sha256(evidence) != declared
        or evidence.get("schemaVersion")
        != "lumen.peft-base-model-identity/1.0.0"
        or evidence.get("baseModelID") != config.get("baseModelID")
        or evidence.get("baseModelRevision") != config.get("baseModelRevision")
        or names != ["default"]
        or evidence.get("privateRuntimePathPersisted") is not False
    ):
        raise RuntimeError("PEFT base-model identity evidence drifted")
    return dict(value)


def _verified_adapter_tokenizer_evidence(
    value: Any,
    *,
    config: Mapping[str, Any],
    adapter_dir: Path,
    snapshot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.train_sft import (
        ADAPTER_BASE_TOKENIZER_FILES,
        ADAPTER_DERIVED_TOKENIZER_FILES,
    )

    expected_keys = {
        "schemaVersion",
        "baseModelTokenizerClosureSHA256",
        "runtimeSnapshotVerificationSHA256",
        "files",
        "transformation",
        "adapterTokenizerBindingSHA256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("Adapter tokenizer binding has an invalid schema")
    evidence = dict(value)
    declared = evidence.pop("adapterTokenizerBindingSHA256")
    closure_by_path = {
        item.get("path"): item
        for item in config.get("baseModelTokenizerFiles", [])
        if isinstance(item, Mapping)
    }
    expected_files: list[dict[str, Any]] = []
    for filename in ADAPTER_BASE_TOKENIZER_FILES:
        source = closure_by_path.get(filename)
        path = adapter_dir / filename
        if (
            not isinstance(source, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or path.stat(follow_symlinks=False).st_size != source.get("sizeBytes")
            or file_sha256(path) != source.get("sha256")
        ):
            raise RuntimeError("Adapter tokenizer files drifted from the base closure")
        expected_files.append(
            {
                "path": filename,
                "sizeBytes": source["sizeBytes"],
                "sha256": source["sha256"],
            }
        )
    if any(
        (adapter_dir / filename).exists() or (adapter_dir / filename).is_symlink()
        for filename in ADAPTER_DERIVED_TOKENIZER_FILES
    ):
        raise RuntimeError("Adapter contains unapproved derived tokenizer files")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) is None
        or canonical_sha256(evidence) != declared
        or evidence.get("schemaVersion")
        != "lumen.adapter-base-tokenizer-binding/1.0.0"
        or evidence.get("baseModelTokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
        or evidence.get("runtimeSnapshotVerificationSHA256")
        != snapshot_verification.get("snapshotVerificationSHA256")
        or evidence.get("files") != expected_files
        or evidence.get("transformation")
        != "exact_byte_subset_no_derived_tokenizer"
    ):
        raise RuntimeError("Adapter tokenizer binding drifted")
    return dict(value)


def _verify_phase_runtime_evidence(
    *,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    adapter_dir: Path,
) -> dict[str, str]:
    tokenizer_verification = _verified_private_tokenizer_snapshot_binding(config)
    runtime_verification = _verified_private_base_model_runtime_snapshot_binding(config)
    expected_report_values = {
        "baseModelTokenizerDigest": config.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": config.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": config.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": config.get(
            "baseModelGenerationConfigFile"
        ),
        "baseModelTokenizerSnapshotPath": config.get(
            "baseModelTokenizerSnapshotPath"
        ),
        "baseModelTokenizerSnapshotVerification": tokenizer_verification,
        "baseModelRuntimeSnapshotPath": config.get("baseModelRuntimeSnapshotPath"),
        "baseModelRuntimeSnapshotVerification": runtime_verification,
    }
    drifted = [
        field
        for field, expected in expected_report_values.items()
        if report.get(field) != expected
    ]
    if drifted:
        raise RuntimeError(
            "Training report private-runtime evidence drifted: "
            + ", ".join(drifted)
        )
    runtime_model_binding = _verified_runtime_model_binding(
        report.get("runtimeModelBinding"),
        config=config,
        snapshot_verification=runtime_verification,
    )
    runtime_tokenizer_binding = _verified_runtime_tokenizer_binding(
        report.get("runtimeTokenizerBinding"),
        config=config,
        snapshot_verification=runtime_verification,
    )
    peft_base_model_identity = _verified_peft_base_model_evidence(
        report.get("peftBaseModelIdentity"),
        config=config,
    )
    adapter_tokenizer_binding = _verified_adapter_tokenizer_evidence(
        report.get("adapterTokenizerBinding"),
        config=config,
        adapter_dir=adapter_dir,
        snapshot_verification=runtime_verification,
    )
    evidence = {
        "runtimeModelBindingSHA256": runtime_model_binding.get(
            "runtimeModelBindingSHA256"
        ),
        "runtimeTokenizerBindingSHA256": runtime_tokenizer_binding.get(
            "runtimeTokenizerBindingSHA256"
        ),
        "peftBaseModelIdentitySHA256": peft_base_model_identity.get(
            "peftBaseModelIdentitySHA256"
        ),
        "adapterTokenizerBindingSHA256": adapter_tokenizer_binding.get(
            "adapterTokenizerBindingSHA256"
        ),
        "baseModelTokenizerSnapshotVerificationSHA256": (
            tokenizer_verification.get("snapshotVerificationSHA256")
        ),
        "baseModelRuntimeSnapshotVerificationSHA256": (
            runtime_verification.get("snapshotVerificationSHA256")
        ),
    }
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is None
        for value in evidence.values()
    ):
        raise RuntimeError("Training report runtime evidence lacks exact digests")
    return {field: str(value) for field, value in evidence.items()}


def _global_tokenizer_snapshot_stability_signatures(
    snapshot_dir: Path,
) -> dict[str, tuple[int, ...]]:
    signatures = {
        ".": _file_stability_signature(
            os.stat(snapshot_dir, follow_symlinks=False)
        )
    }
    for filename in GLOBAL_TOKENIZER_SNAPSHOT_FILES:
        signatures[filename] = _file_stability_signature(
            os.stat(snapshot_dir / filename, follow_symlinks=False)
        )
    return signatures


def _global_tokenizer_snapshot_contract(
    *,
    snapshot_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the exact local-only tokenizer closure through stable descriptors."""

    base_closure = _validated_base_model_tokenizer_closure(config)
    expected_files = base_closure["files"]
    expected_by_path = {item["path"]: item for item in expected_files}

    try:
        directory_stat = os.stat(snapshot_dir, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("Global tokenizer snapshot is unavailable") from exc
    if (
        snapshot_dir.is_symlink()
        or not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.getuid()
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
    ):
        raise RuntimeError(
            "Global tokenizer snapshot must be process-owned mode 0700"
        )
    observed_names = sorted(item.name for item in os.scandir(snapshot_dir))
    if observed_names != sorted(GLOBAL_TOKENIZER_SNAPSHOT_FILES):
        raise RuntimeError("Global tokenizer snapshot file closure drifted")

    files: list[dict[str, Any]] = []
    for filename in GLOBAL_TOKENIZER_SNAPSHOT_FILES:
        path = snapshot_dir / filename
        handle, file_stat = _open_regular_readonly(
            path,
            label=f"Global tokenizer snapshot {filename}",
        )
        try:
            if (
                file_stat.st_uid != os.getuid()
                or stat.S_IMODE(file_stat.st_mode) != 0o400
            ):
                raise RuntimeError(
                    f"Global tokenizer snapshot {filename} must be process-owned mode 0400"
                )
            payload = _read_descriptor_bytes(handle)
            _require_stable_descriptor(
                handle,
                file_stat,
                label=f"Global tokenizer snapshot {filename}",
            )
            _require_path_matches_descriptor(
                path,
                file_stat,
                label=f"Global tokenizer snapshot {filename}",
            )
        finally:
            handle.close()
        observed = {
            "path": filename,
            "sizeBytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        expected = expected_by_path[filename]
        if observed != {
            key: expected[key] for key in ("path", "sizeBytes", "sha256")
        }:
            raise RuntimeError(
                f"Pinned tokenizer snapshot binding failed for {filename}"
            )
        files.append(dict(expected))

    unsigned = {
        "schemaVersion": GLOBAL_TOKENIZER_SNAPSHOT_SCHEMA,
        "baseModelID": config.get("base_model_name"),
        "baseModelRevision": config.get("baseModelRevision"),
        "baseModelTokenizerClosureSHA256": base_closure[
            "baseModelTokenizerClosureSHA256"
        ],
        "files": files,
    }
    return {
        **unsigned,
        "tokenizerClosureSHA256": canonical_sha256(unsigned),
    }


def _injected_global_tokenizer_snapshot_contract(
    config: Mapping[str, Any],
    tokenizer_file_sha256: str,
) -> dict[str, Any]:
    """Build an unmistakable unit-test seam that production verification rejects."""

    unsigned = {
        "schemaVersion": "lumen.global-tokenizer-snapshot/injected-test-double",
        "baseModelID": config.get("base_model_name"),
        "baseModelRevision": config.get("baseModelRevision"),
        "baseModelTokenizerClosureSHA256": config.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "files": [
            {
                "path": "tokenizer.json",
                "sizeBytes": 0,
                "sha256": tokenizer_file_sha256,
            }
        ],
    }
    return {
        **unsigned,
        "tokenizerClosureSHA256": canonical_sha256(unsigned),
    }


def _validated_global_tokenizer_closure_record(
    value: Any,
    *,
    config: Mapping[str, Any],
    allow_injected_test_double: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schemaVersion",
        "baseModelID",
        "baseModelRevision",
        "baseModelTokenizerClosureSHA256",
        "files",
        "tokenizerClosureSHA256",
    }:
        raise RuntimeError("Global tokenizer closure has an invalid schema")
    record = dict(value)
    schema = record.get("schemaVersion")
    expected_paths = (
        list(GLOBAL_TOKENIZER_SNAPSHOT_FILES)
        if schema == GLOBAL_TOKENIZER_SNAPSHOT_SCHEMA
        else ["tokenizer.json"]
        if allow_injected_test_double
        and schema == "lumen.global-tokenizer-snapshot/injected-test-double"
        else None
    )
    base_closure = _validated_base_model_tokenizer_closure(config)
    files = record.get("files")
    expected_file_keys = (
        {"path", "sizeBytes", "sha256", "huggingFaceBlobID"}
        if schema == GLOBAL_TOKENIZER_SNAPSHOT_SCHEMA
        else {"path", "sizeBytes", "sha256"}
    )
    if (
        expected_paths is None
        or record.get("baseModelID") != config.get("base_model_name")
        or record.get("baseModelRevision") != config.get("baseModelRevision")
        or record.get("baseModelTokenizerClosureSHA256")
        != base_closure["baseModelTokenizerClosureSHA256"]
        or not isinstance(files, list)
        or [item.get("path") if isinstance(item, Mapping) else None for item in files]
        != expected_paths
        or any(
            not isinstance(item, Mapping)
            or set(item) != expected_file_keys
            or type(item.get("sizeBytes")) is not int
            or item["sizeBytes"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256") or ""))
            is None
            for item in files
        )
    ):
        raise RuntimeError("Global tokenizer closure binding drifted")
    if (
        schema == GLOBAL_TOKENIZER_SNAPSHOT_SCHEMA
        and [dict(item) for item in files] != base_closure["files"]
    ):
        raise RuntimeError("Global tokenizer closure files drifted")
    tokenizer_files = [item for item in files if item["path"] == "tokenizer.json"]
    unsigned = dict(record)
    declared = unsigned.pop("tokenizerClosureSHA256")
    if (
        len(tokenizer_files) != 1
        or tokenizer_files[0]["sha256"]
        != config.get("baseModelTokenizerDigest")
        or re.fullmatch(r"[0-9a-f]{64}", str(declared or "")) is None
        or canonical_sha256(unsigned) != declared
    ):
        raise RuntimeError("Global tokenizer closure integrity drifted")
    return record


def _create_global_tokenizer_snapshot(
    *,
    snapshot_dir: Path,
    config: Mapping[str, Any],
) -> None:
    from huggingface_hub import hf_hub_download  # type: ignore

    model_id = str(config["base_model_name"])
    revision = str(config["baseModelRevision"])
    base_closure = _validated_base_model_tokenizer_closure(config)
    snapshot_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME}.",
            dir=snapshot_dir.parent,
        )
    )
    try:
        os.chmod(staging, 0o700, follow_symlinks=False)
        for expected in base_closure["files"]:
            filename = str(expected["path"])
            cached_link = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                )
            )
            cached = cached_link.resolve(strict=True)
            cache_handle, cache_stat = _open_regular_readonly(
                cached,
                label=f"Pinned Hugging Face tokenizer cache blob {filename}",
            )
            try:
                cache_payload = _read_descriptor_bytes(cache_handle)
                cache_payload_sha256 = hashlib.sha256(cache_payload).hexdigest()
                blob_id = cached.name
                if (
                    blob_id != expected["huggingFaceBlobID"]
                    or len(cache_payload) != expected["sizeBytes"]
                    or cache_payload_sha256 != expected["sha256"]
                    or (
                        len(blob_id) == 64
                        and cache_payload_sha256 != blob_id
                    )
                    or (
                        len(blob_id) == 40
                        and _git_blob_sha1(cache_payload) != blob_id
                    )
                ):
                    raise RuntimeError(
                        f"Pinned tokenizer cache blob identity failed for {filename}"
                    )
                _require_stable_descriptor(
                    cache_handle,
                    cache_stat,
                    label=f"Pinned Hugging Face tokenizer cache blob {filename}",
                )
                _require_path_matches_descriptor(
                    cached,
                    cache_stat,
                    label=f"Pinned Hugging Face tokenizer cache blob {filename}",
                )
            finally:
                cache_handle.close()
            destination = staging / filename
            _copy_private_regular_file(cached, destination)
            if file_sha256(destination) != cache_payload_sha256:
                raise RuntimeError(
                    f"Pinned tokenizer cache blob changed before copying {filename}"
                )
            os.chmod(destination, 0o400, follow_symlinks=False)
        _fsync_directory(staging, label="the global tokenizer snapshot")
        os.chmod(staging, 0o700, follow_symlinks=False)
        if snapshot_dir.exists() or snapshot_dir.is_symlink():
            raise RuntimeError("Global tokenizer snapshot appeared during creation")
        os.replace(staging, snapshot_dir)
        staging = None
        _fsync_directory(
            snapshot_dir.parent,
            label="the global tokenizer snapshot parent",
        )
    finally:
        if staging is not None and staging.exists():
            os.chmod(staging, 0o700, follow_symlinks=False)
            shutil.rmtree(staging)


def _load_verified_global_tokenizer_snapshot(
    *,
    snapshot_dir: Path,
    config: Mapping[str, Any],
    expected_contract: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer  # type: ignore

    before_signatures = _global_tokenizer_snapshot_stability_signatures(
        snapshot_dir
    )
    before = _global_tokenizer_snapshot_contract(
        snapshot_dir=snapshot_dir,
        config=config,
    )
    if expected_contract is not None and before != dict(expected_contract):
        raise RuntimeError("Global tokenizer snapshot audit binding drifted")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    after = _global_tokenizer_snapshot_contract(
        snapshot_dir=snapshot_dir,
        config=config,
    )
    after_signatures = _global_tokenizer_snapshot_stability_signatures(
        snapshot_dir
    )
    if after != before or after_signatures != before_signatures:
        raise RuntimeError("Global tokenizer snapshot changed while loading")
    if getattr(tokenizer, "is_fast", None) is not True:
        raise RuntimeError("Global tokenizer preflight requires the pinned fast tokenizer")
    return tokenizer, before


def _load_exact_global_preflight_tokenizer(
    config: Mapping[str, Any],
    *,
    snapshot_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Snapshot and load the pinned tokenizer without model weights."""

    model_id = config.get("base_model_name")
    revision = config.get("baseModelRevision")
    expected_digest = config.get("baseModelTokenizerDigest")
    if (
        not isinstance(model_id, str)
        or not model_id
        or re.fullmatch(r"[0-9a-f]{40}", str(revision or "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(expected_digest or "")) is None
    ):
        raise RuntimeError("Global tokenizer preflight has invalid model lineage")
    _validated_base_model_tokenizer_closure(config)
    if not snapshot_dir.exists() and not snapshot_dir.is_symlink():
        _create_global_tokenizer_snapshot(
            snapshot_dir=snapshot_dir,
            config=config,
        )
    return _load_verified_global_tokenizer_snapshot(
        snapshot_dir=snapshot_dir,
        config=config,
    )


def global_tokenizer_preflight(
    *,
    run_root: Path,
    agents: Sequence[str],
    tokenizer: Any | None = None,
    tokenizer_file_sha256: str | None = None,
    preference_renderer: Any | None = None,
    chat_contract_verifier: Any | None = None,
) -> dict[str, Any]:
    """Fail all requested agents before the first model/optimizer allocation.

    The trainers intentionally repeat these calculations and bind their own
    authoritative evidence. This global audit exists to surface any exact
    length, margin, metadata, or Fleet loss-share defect before agent one
    starts an expensive optimization run.
    """

    manifest = _verified_run_manifest(run_root)
    prepared_agents = manifest.get("agents")
    if (
        not isinstance(prepared_agents, list)
        or any(not isinstance(item, Mapping) for item in prepared_agents)
        or [item.get("agent") for item in prepared_agents] != list(agents)
    ):
        raise RuntimeError(
            "Global tokenizer preflight agents drifted from the prepared run"
        )
    if not agents:
        raise RuntimeError("Global tokenizer preflight requires at least one agent")

    from tools.fine_tuning.unsloth import train_dpo, train_sft

    first_config = read_object(run_root / "configs" / f"{agents[0]}.json")
    if tokenizer is None:
        if tokenizer_file_sha256 is not None:
            raise RuntimeError("Observed tokenizer digest cannot be supplied without a tokenizer")
        tokenizer, tokenizer_closure = (
            _load_exact_global_preflight_tokenizer(
                first_config,
                snapshot_dir=(
                    run_root
                    / "training"
                    / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
                ),
            )
        )
        observed_tokenizer_digest = next(
            item["sha256"]
            for item in tokenizer_closure["files"]
            if item["path"] == "tokenizer.json"
        )
    else:
        observed_tokenizer_digest = tokenizer_file_sha256
        if observed_tokenizer_digest != first_config.get(
            "baseModelTokenizerDigest"
        ):
            raise RuntimeError("Injected global-preflight tokenizer digest drifted")
        tokenizer_closure = _injected_global_tokenizer_snapshot_contract(
            first_config,
            str(observed_tokenizer_digest),
        )
    if preference_renderer is None:
        from trl.data_utils import maybe_apply_chat_template  # type: ignore

        preference_renderer = maybe_apply_chat_template
    if not callable(preference_renderer):
        raise RuntimeError("Global preference renderer is unavailable")
    if chat_contract_verifier is None:
        chat_contract_verifier = train_sft.verify_chat_template_contract
    if not callable(chat_contract_verifier):
        raise RuntimeError("Global chat-template verifier is unavailable")

    base_binding = {
        "baseModelID": first_config.get("base_model_name"),
        "baseModelRevision": first_config.get("baseModelRevision"),
        "baseModelTokenizerDigest": first_config.get("baseModelTokenizerDigest"),
        "baseModelTokenizerClosureSHA256": first_config.get(
            "baseModelTokenizerClosureSHA256"
        ),
    }
    entries: list[dict[str, Any]] = []
    for agent in agents:
        config_path = run_root / "configs" / f"{agent}.json"
        config = read_object(config_path)
        if (
            config.get("agent") != agent
            or {
                "baseModelID": config.get("base_model_name"),
                "baseModelRevision": config.get("baseModelRevision"),
                "baseModelTokenizerDigest": config.get(
                    "baseModelTokenizerDigest"
                ),
                "baseModelTokenizerClosureSHA256": config.get(
                    "baseModelTokenizerClosureSHA256"
                ),
            }
            != base_binding
        ):
            raise RuntimeError(
                f"Global tokenizer preflight model binding drifted for {agent}"
            )
        chat_contract_verifier(
            config.get("chatTemplateContract"),
            tokenizer=tokenizer,
        )
        dataset_dir = Path(str(config.get("dataset_dir") or "")).resolve()
        expected_dataset_root = (
            run_root
            / "generated"
            / "fine_tuning"
            / agent
            / "experiments"
            / str(config.get("variant") or "")
        ).resolve()
        if dataset_dir != expected_dataset_root:
            raise RuntimeError(
                f"Global tokenizer preflight dataset path drifted for {agent}"
            )
        train_sft_path = dataset_dir / "train_sft.jsonl"
        val_sft_path = dataset_dir / "val_sft.jsonl"
        train_dpo_path = dataset_dir / "train_dpo.jsonl"
        val_dpo_path = dataset_dir / "val_dpo.jsonl"
        train_sft_rows = train_sft._limit_records(
            read_jsonl(train_sft_path),
            config.get("max_train_records"),
        )
        val_sft_rows = train_sft._limit_records(
            read_jsonl(val_sft_path),
            config.get("max_val_records"),
        )
        train_dpo_source = read_jsonl(train_dpo_path)
        val_dpo_source = read_jsonl(val_dpo_path)
        train_preference_rows = [
            train_dpo.row_to_preference(row) for row in train_dpo_source
        ]
        val_preference_rows = [
            train_dpo.row_to_preference(row) for row in val_dpo_source
        ]
        preference_config = train_dpo._validate_preference_training_config(
            config
        )
        max_sequence_length = config.get("max_seq_length")
        max_prompt_length = preference_config["maxPromptLength"]
        sft_preflight = train_sft._preflight_sft_token_lengths(
            {
                "train": (train_sft_rows, train_sft_path),
                "validation": (val_sft_rows, val_sft_path),
            },
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            minimum_sequence_margin_tokens=config.get(
                "sft_minimum_sequence_margin_tokens",
                train_sft.SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            ),
            agent=agent,
            fleet_loss_share_contract=config.get("fleetLossShareContract"),
            public_corpus_loss_share_contract=config.get(
                "publicCorpusLossShareContract"
            ),
            fleet_config=config,
        )
        preference_preflight = train_dpo._preflight_preference_token_lengths(
            {
                "train": train_preference_rows,
                "validation": val_preference_rows,
            },
            tokenizer=tokenizer,
            render_preference=preference_renderer,
            max_prompt_length=max_prompt_length,
            max_sequence_length=max_sequence_length,
            minimum_prompt_margin_tokens=config.get(
                "preference_minimum_prompt_margin_tokens",
                train_dpo.PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS,
            ),
            minimum_sequence_margin_tokens=config.get(
                "preference_minimum_sequence_margin_tokens",
                train_dpo.PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            ),
            source_splits={
                "train": train_dpo_source,
                "validation": val_dpo_source,
            },
            agent=agent,
            fleet_loss_share_contract=config.get("fleetLossShareContract"),
            public_corpus_loss_share_contract=config.get(
                "publicCorpusLossShareContract"
            ),
            fleet_config=config,
        )
        _verify_global_tokenizer_phase_evidence(
            run_root=run_root,
            agent=agent,
            config=config,
            phase="sft",
            evidence=sft_preflight,
            tokenizer=tokenizer,
        )
        _verify_global_tokenizer_phase_evidence(
            run_root=run_root,
            agent=agent,
            config=config,
            phase="preference",
            evidence=preference_preflight,
            tokenizer=tokenizer,
            preference_renderer=preference_renderer,
        )
        sft_dataset_hashes = train_sft._sft_checkpoint_dataset_sha256(config)
        dpo_dataset_hashes = train_dpo._preference_dataset_file_sha256(config)
        phase_code = config.get("trainingCodeSHA256ByPhase")
        if not isinstance(phase_code, Mapping):
            raise RuntimeError(
                f"Global tokenizer preflight lacks phase code lineage for {agent}"
            )
        entries.append(
            {
                "agent": agent,
                "configSHA256": file_sha256(config_path),
                "sftDatasetFileSHA256": sft_dataset_hashes,
                "preferenceDatasetFileSHA256": dpo_dataset_hashes,
                "sftTrainingCodeSHA256": phase_code.get("sft"),
                "preferenceTrainingCodeSHA256": phase_code.get(
                    preference_config["preferenceTrainer"]
                ),
                "sft": sft_preflight,
                "preference": preference_preflight,
            }
        )

    unsigned = {
        "schemaVersion": GLOBAL_TOKENIZER_PREFLIGHT_SCHEMA,
        "status": "passed",
        "runManifestSHA256": manifest.get("runManifestSHA256"),
        **base_binding,
        "observedTokenizerFileSHA256": observed_tokenizer_digest,
        "tokenizerClosure": tokenizer_closure,
        "agents": entries,
    }
    audit = {
        **unsigned,
        "globalPreflightSHA256": canonical_sha256(unsigned),
    }
    audit_path = run_root / "training" / GLOBAL_TOKENIZER_PREFLIGHT_FILENAME
    write_object(audit_path, audit)
    return {
        "status": "global_tokenizer_preflight_passed",
        "path": str(audit_path),
        "globalPreflightSHA256": audit["globalPreflightSHA256"],
        "tokenizerClosureSHA256": tokenizer_closure[
            "tokenizerClosureSHA256"
        ],
        "agents": list(agents),
        **base_binding,
    }


def _verified_global_tokenizer_preflight_impl(
    *,
    run_root: Path,
    agent: str,
    config: Mapping[str, Any],
    phase: str,
    bound_preflight: Mapping[str, Any],
    audit_snapshot: Mapping[str, Any] | None = None,
    allow_injected_test_double: bool,
    tokenizer: Any | None = None,
    preference_renderer: Any | None = None,
) -> dict[str, Any]:
    path = run_root / "training" / GLOBAL_TOKENIZER_PREFLIGHT_FILENAME
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Global tokenizer preflight audit is missing")
    audit = (
        dict(audit_snapshot)
        if audit_snapshot is not None
        else read_object(path)
    )
    digest = audit.get("globalPreflightSHA256")
    unsigned = dict(audit)
    unsigned.pop("globalPreflightSHA256", None)
    manifest = _verified_run_manifest(run_root)
    if (
        set(audit)
        != {
            "schemaVersion",
            "status",
            "runManifestSHA256",
            "baseModelID",
            "baseModelRevision",
            "baseModelTokenizerDigest",
            "baseModelTokenizerClosureSHA256",
            "observedTokenizerFileSHA256",
            "tokenizerClosure",
            "agents",
            "globalPreflightSHA256",
        }
        or audit.get("schemaVersion") != GLOBAL_TOKENIZER_PREFLIGHT_SCHEMA
        or audit.get("status") != "passed"
        or re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
        or canonical_sha256(unsigned) != digest
        or audit.get("runManifestSHA256") != manifest.get("runManifestSHA256")
        or audit.get("baseModelID") != config.get("base_model_name")
        or audit.get("baseModelRevision") != config.get("baseModelRevision")
        or audit.get("baseModelTokenizerDigest")
        != config.get("baseModelTokenizerDigest")
        or audit.get("baseModelTokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
        or audit.get("observedTokenizerFileSHA256")
        != config.get("baseModelTokenizerDigest")
    ):
        raise RuntimeError("Global tokenizer preflight audit binding drifted")
    tokenizer_closure = _validated_global_tokenizer_closure_record(
        audit.get("tokenizerClosure"),
        config=config,
        allow_injected_test_double=allow_injected_test_double,
    )
    tokenizer_file = next(
        item
        for item in tokenizer_closure["files"]
        if item["path"] == "tokenizer.json"
    )
    if tokenizer_file["sha256"] != audit.get("observedTokenizerFileSHA256"):
        raise RuntimeError("Global tokenizer closure file digest drifted")
    entries = audit.get("agents")
    manifest_agents = manifest.get("agents")
    expected_agents = (
        [item.get("agent") for item in manifest_agents]
        if isinstance(manifest_agents, list)
        and all(isinstance(item, Mapping) for item in manifest_agents)
        else None
    )
    if (
        not isinstance(entries, list)
        or expected_agents is None
        or [
            item.get("agent") if isinstance(item, Mapping) else None
            for item in entries
        ]
        != expected_agents
        or any(
            set(item)
            != {
                "agent",
                "configSHA256",
                "sftDatasetFileSHA256",
                "preferenceDatasetFileSHA256",
                "sftTrainingCodeSHA256",
                "preferenceTrainingCodeSHA256",
                "sft",
                "preference",
            }
            for item in entries
            if isinstance(item, Mapping)
        )
    ):
        raise RuntimeError("Global tokenizer preflight agents are invalid")
    matching = [
        item
        for item in entries
        if isinstance(item, Mapping) and item.get("agent") == agent
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Global tokenizer preflight lacks {agent}")
    entry = matching[0]
    config_path = run_root / "configs" / f"{agent}.json"
    dataset_hash_field = (
        "sftDatasetFileSHA256"
        if phase == "sft"
        else "preferenceDatasetFileSHA256"
    )
    training_code_field = (
        "sftTrainingCodeSHA256"
        if phase == "sft"
        else "preferenceTrainingCodeSHA256"
    )
    checkpoint_code = bound_preflight.get("trainingCodeSHA256")
    global_phase = "sft" if phase == "sft" else "preference"
    global_preflight = entry.get(global_phase)
    bound_dataset_hashes = bound_preflight.get("datasetFileSHA256")
    if (
        entry.get("configSHA256") != file_sha256(config_path)
        or entry.get(dataset_hash_field) != bound_dataset_hashes
        or entry.get(training_code_field) != checkpoint_code
        or not isinstance(global_preflight, Mapping)
        or any(
            bound_preflight.get(key) != value
            for key, value in global_preflight.items()
        )
    ):
        raise RuntimeError(
            f"Global {phase} tokenizer preflight drifted from authoritative evidence"
        )
    if tokenizer is None:
        if allow_injected_test_double:
            raise RuntimeError(
                "Injected tokenizer audit verification requires its tokenizer"
            )
        tokenizer, verified_closure = _load_verified_global_tokenizer_snapshot(
            snapshot_dir=(
                run_root
                / "training"
                / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
            ),
            config=config,
            expected_contract=tokenizer_closure,
        )
        if verified_closure != tokenizer_closure:
            raise RuntimeError(
                "Global tokenizer verification closure changed while loading"
            )
    if phase == "preference" and preference_renderer is None:
        from trl.data_utils import maybe_apply_chat_template  # type: ignore

        preference_renderer = maybe_apply_chat_template
    _verify_global_tokenizer_phase_evidence(
        run_root=run_root,
        agent=agent,
        config=config,
        phase=phase,
        evidence=global_preflight,
        tokenizer=tokenizer,
        preference_renderer=preference_renderer,
    )
    return audit


def _verified_global_tokenizer_preflight(
    *,
    run_root: Path,
    agent: str,
    config: Mapping[str, Any],
    phase: str,
    bound_preflight: Mapping[str, Any],
    audit_snapshot: Mapping[str, Any] | None = None,
    tokenizer: Any | None = None,
    preference_renderer: Any | None = None,
) -> dict[str, Any]:
    """Production verifier; injected tokenizer audits are never accepted."""

    return _verified_global_tokenizer_preflight_impl(
        run_root=run_root,
        agent=agent,
        config=config,
        phase=phase,
        bound_preflight=bound_preflight,
        audit_snapshot=audit_snapshot,
        allow_injected_test_double=False,
        tokenizer=tokenizer,
        preference_renderer=preference_renderer,
    )


def _verified_global_tokenizer_preflight_test_only(
    *,
    run_root: Path,
    agent: str,
    config: Mapping[str, Any],
    phase: str,
    bound_preflight: Mapping[str, Any],
    audit_snapshot: Mapping[str, Any],
    tokenizer: Any,
    preference_renderer: Any | None = None,
) -> dict[str, Any]:
    """Narrow unit-test seam for an explicitly injected in-memory tokenizer."""

    return _verified_global_tokenizer_preflight_impl(
        run_root=run_root,
        agent=agent,
        config=config,
        phase=phase,
        bound_preflight=bound_preflight,
        audit_snapshot=audit_snapshot,
        allow_injected_test_double=True,
        tokenizer=tokenizer,
        preference_renderer=preference_renderer,
    )


def _verify_global_tokenizer_phase_evidence(
    *,
    run_root: Path,
    agent: str,
    config: Mapping[str, Any],
    phase: str,
    evidence: Any,
    tokenizer: Any,
    preference_renderer: Any | None = None,
) -> None:
    """Validate raw global evidence without trusting its self-declared hashes."""

    from tools.fine_tuning.unsloth import train_dpo, train_sft

    dataset_dir = Path(str(config.get("dataset_dir") or "")).resolve()
    expected_dataset_dir = (
        run_root
        / "generated"
        / "fine_tuning"
        / agent
        / "experiments"
        / str(config.get("variant") or "")
    ).resolve()
    if dataset_dir != expected_dataset_dir:
        raise RuntimeError(
            f"Global {phase} tokenizer dataset path drifted for {agent}"
        )
    loss_share_fields = {"publicCorpusLossShareEvidence"}
    if agent == "fleet":
        loss_share_fields.add("fleetLossShareEvidence")
    if phase == "sft":
        expected_keys = {
            "schemaVersion",
            "maxSequenceLength",
            "minimumSequenceMarginTokens",
            "percentileMethod",
            "records",
            "totalTokens",
            "assistantTargetTokens",
            "tokenizationTranscriptSHA256",
            "smallestSequenceMarginTokens",
            "truncationRequired",
            "splits",
            *loss_share_fields,
        }
        if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
            raise RuntimeError("Global SFT tokenizer evidence has an invalid schema")
        max_sequence_length = config.get("max_seq_length")
        required_sequence_margin = max(
            train_sft.SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            int(config.get("sft_minimum_sequence_margin_tokens", 0)),
        )
        split_rows = {
            "train": train_sft._limit_records(
                read_jsonl(dataset_dir / "train_sft.jsonl"),
                config.get("max_train_records"),
            ),
            "validation": train_sft._limit_records(
                read_jsonl(dataset_dir / "val_sft.jsonl"),
                config.get("max_val_records"),
            ),
        }
        if (
            evidence.get("schemaVersion")
            != train_sft.SFT_TOKEN_LENGTH_PREFLIGHT_SCHEMA
            or evidence.get("maxSequenceLength") != max_sequence_length
            or evidence.get("minimumSequenceMarginTokens")
            != required_sequence_margin
            or evidence.get("percentileMethod") != "nearest_rank"
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(evidence.get("tokenizationTranscriptSHA256") or ""),
            )
            is None
            or evidence.get("truncationRequired") is not False
            or not _valid_token_length_statistics(
                evidence.get("totalTokens"),
                require_positive=True,
            )
            or not _valid_token_length_statistics(
                evidence.get("assistantTargetTokens"),
                require_positive=True,
            )
            or type(evidence.get("smallestSequenceMarginTokens")) is not int
            or evidence.get("smallestSequenceMarginTokens")
            != max_sequence_length - evidence["totalTokens"]["max"]
            or evidence["smallestSequenceMarginTokens"]
            < required_sequence_margin
        ):
            raise RuntimeError("Global SFT tokenizer evidence controls drifted")
        split_stat_fields = ("totalTokens", "assistantTargetTokens")
        split_margin_fields = ("smallestSequenceMarginTokens",)
    elif phase == "preference":
        expected_keys = {
            "schemaVersion",
            "renderer",
            "addSpecialTokens",
            "completionTokenizationPolicy",
            "appendedEOSTokenID",
            "percentileMethod",
            "maxPromptLength",
            "maxSequenceLength",
            "minimumPromptMarginTokens",
            "minimumSequenceMarginTokens",
            "records",
            "promptTokens",
            "chosenCompletionTokens",
            "rejectedCompletionTokens",
            "chosenTotalTokens",
            "rejectedTotalTokens",
            "maximumTotalTokens",
            "tokenizationTranscriptSHA256",
            "smallestPromptMarginTokens",
            "smallestSequenceMarginTokens",
            "truncationRequired",
            "splits",
            *loss_share_fields,
        }
        if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
            raise RuntimeError(
                "Global preference tokenizer evidence has an invalid schema"
            )
        preference_config = train_dpo._validate_preference_training_config(
            config
        )
        max_prompt_length = preference_config["maxPromptLength"]
        max_sequence_length = config.get("max_seq_length")
        required_prompt_margin = max(
            train_dpo.PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS,
            int(config.get("preference_minimum_prompt_margin_tokens", 0)),
        )
        required_sequence_margin = max(
            train_dpo.PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            int(config.get("preference_minimum_sequence_margin_tokens", 0)),
        )
        split_rows = {
            "train": read_jsonl(dataset_dir / "train_dpo.jsonl"),
            "validation": read_jsonl(dataset_dir / "val_dpo.jsonl"),
        }
        statistic_fields = (
            "promptTokens",
            "chosenCompletionTokens",
            "rejectedCompletionTokens",
            "chosenTotalTokens",
            "rejectedTotalTokens",
            "maximumTotalTokens",
        )
        if (
            evidence.get("schemaVersion")
            != train_dpo.PREFERENCE_TOKEN_LENGTH_PREFLIGHT_SCHEMA
            or not isinstance(evidence.get("renderer"), str)
            or not evidence.get("renderer")
            or evidence.get("addSpecialTokens") is not False
            or evidence.get("completionTokenizationPolicy")
            != train_dpo.DPO_COMPLETION_TOKENIZATION_POLICY
            or type(evidence.get("appendedEOSTokenID")) is not int
            or evidence.get("appendedEOSTokenID") < 0
            or evidence.get("percentileMethod") != "nearest_rank"
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(evidence.get("tokenizationTranscriptSHA256") or ""),
            )
            is None
            or evidence.get("maxPromptLength") != max_prompt_length
            or evidence.get("maxSequenceLength") != max_sequence_length
            or evidence.get("minimumPromptMarginTokens")
            != required_prompt_margin
            or evidence.get("minimumSequenceMarginTokens")
            != required_sequence_margin
            or evidence.get("truncationRequired") is not False
            or any(
                not _valid_token_length_statistics(
                    evidence.get(field),
                    require_positive=True,
                )
                for field in statistic_fields
            )
            or type(evidence.get("smallestPromptMarginTokens")) is not int
            or evidence.get("smallestPromptMarginTokens")
            != max_prompt_length - evidence["promptTokens"]["max"]
            or evidence["smallestPromptMarginTokens"] < required_prompt_margin
            or type(evidence.get("smallestSequenceMarginTokens")) is not int
            or evidence.get("smallestSequenceMarginTokens")
            != max_sequence_length - evidence["maximumTotalTokens"]["max"]
            or evidence["smallestSequenceMarginTokens"]
            < required_sequence_margin
        ):
            raise RuntimeError(
                "Global preference tokenizer evidence controls drifted"
            )
        split_stat_fields = statistic_fields
        split_margin_fields = (
            "smallestPromptMarginTokens",
            "smallestSequenceMarginTokens",
        )
    else:
        raise RuntimeError(f"Unsupported global tokenizer phase: {phase}")

    splits = evidence.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation"}:
        raise RuntimeError(f"Global {phase} tokenizer split evidence is invalid")
    observed_records = 0
    for split in ("train", "validation"):
        split_evidence = splits.get(split)
        expected_records = len(split_rows[split])
        observed_records += expected_records
        expected_split_keys = (
            {"records"}
            if expected_records == 0
            else {"records", *split_stat_fields, *split_margin_fields}
        )
        if (
            not isinstance(split_evidence, Mapping)
            or set(split_evidence) != expected_split_keys
            or split_evidence.get("records") != expected_records
        ):
            raise RuntimeError(
                f"Global {phase} {split} tokenizer split binding drifted"
            )
        if expected_records == 0:
            continue
        if any(
            not _valid_token_length_statistics(
                split_evidence.get(field),
                require_positive=True,
            )
            for field in split_stat_fields
        ):
            raise RuntimeError(
                f"Global {phase} {split} token statistics are invalid"
            )
        if phase == "sft":
            if split_evidence["smallestSequenceMarginTokens"] != (
                max_sequence_length - split_evidence["totalTokens"]["max"]
            ):
                raise RuntimeError("Global SFT split margin drifted")
        elif (
            split_evidence["smallestPromptMarginTokens"]
            != max_prompt_length - split_evidence["promptTokens"]["max"]
            or split_evidence["smallestSequenceMarginTokens"]
            != max_sequence_length
            - split_evidence["maximumTotalTokens"]["max"]
        ):
            raise RuntimeError("Global preference split margin drifted")
    if evidence.get("records") != observed_records:
        raise RuntimeError(f"Global {phase} tokenizer record count drifted")

    _verify_fleet_loss_share_evidence(
        value=evidence.get("fleetLossShareEvidence"),
        config=config,
        phase=phase,
        dataset_dir=dataset_dir,
    )
    _verify_public_corpus_loss_share_evidence(
        value=evidence.get("publicCorpusLossShareEvidence"),
        config=config,
        phase=phase,
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        preference_renderer=preference_renderer,
        require_exact_tokenizer_counts=True,
    )


def _validated_global_tokenizer_resume_state(
    *,
    run_root: Path,
    agents: Sequence[str],
) -> str:
    """Validate every durable global-tokenizer preflight resume state."""

    snapshot_dir = run_root / "training" / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
    audit_path = run_root / "training" / GLOBAL_TOKENIZER_PREFLIGHT_FILENAME
    snapshot_present = snapshot_dir.exists() or snapshot_dir.is_symlink()
    audit_present = audit_path.exists() or audit_path.is_symlink()
    if snapshot_present and (
        snapshot_dir.is_symlink() or not snapshot_dir.is_dir()
    ):
        raise RuntimeError("Global tokenizer resume snapshot is not a regular directory")
    if audit_present and (audit_path.is_symlink() or not audit_path.is_file()):
        raise RuntimeError("Global tokenizer resume audit is not a regular file")
    if audit_present and not snapshot_present:
        raise RuntimeError(
            "Global tokenizer resume audit exists without its verified snapshot"
        )
    if not snapshot_present:
        return "not_started"
    if not agents:
        raise RuntimeError("Global tokenizer resume state requires prepared agents")

    first_config = read_object(run_root / "configs" / f"{agents[0]}.json")
    snapshot_contract = _global_tokenizer_snapshot_contract(
        snapshot_dir=snapshot_dir,
        config=first_config,
    )
    if not audit_present:
        return "verified_snapshot_audit_pending"

    audit = read_object(audit_path)
    closure = _validated_global_tokenizer_closure_record(
        audit.get("tokenizerClosure"),
        config=first_config,
    )
    if closure != snapshot_contract:
        raise RuntimeError(
            "Global tokenizer resume audit drifted from its verified snapshot"
        )
    tokenizer, loaded_closure = _load_verified_global_tokenizer_snapshot(
        snapshot_dir=snapshot_dir,
        config=first_config,
        expected_contract=closure,
    )
    if loaded_closure != closure:
        raise RuntimeError(
            "Global tokenizer resume closure changed while loading"
        )
    entries = audit.get("agents")
    if not isinstance(entries, list):
        raise RuntimeError("Global tokenizer resume audit agents are invalid")
    by_agent = {
        str(entry.get("agent")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if set(by_agent) != set(agents) or len(by_agent) != len(agents):
        raise RuntimeError("Global tokenizer resume audit agent set drifted")
    for agent in agents:
        config = read_object(run_root / "configs" / f"{agent}.json")
        if _validated_base_model_tokenizer_closure(config) != (
            _validated_base_model_tokenizer_closure(first_config)
        ):
            raise RuntimeError(
                f"Global tokenizer resume closure differs for {agent}"
            )
        entry = by_agent[agent]
        for phase, dataset_field, training_code_field in (
            (
                "sft",
                "sftDatasetFileSHA256",
                "sftTrainingCodeSHA256",
            ),
            (
                "preference",
                "preferenceDatasetFileSHA256",
                "preferenceTrainingCodeSHA256",
            ),
        ):
            evidence = entry.get(phase)
            if not isinstance(evidence, Mapping):
                raise RuntimeError(
                    f"Global tokenizer resume audit lacks {agent} {phase}"
                )
            _verified_global_tokenizer_preflight(
                run_root=run_root,
                agent=agent,
                config=config,
                phase=phase,
                bound_preflight={
                    **evidence,
                    "datasetFileSHA256": entry.get(dataset_field),
                    "trainingCodeSHA256": entry.get(training_code_field),
                },
                audit_snapshot=audit,
                tokenizer=tokenizer,
            )
    return "verified_snapshot_and_audit"


def _verified_prepared_global_tokenizer_preflight(
    *,
    run_root: Path,
    agents: tuple[str, ...],
    tokenizer: Any | None = None,
    tokenizer_file_sha256: str | None = None,
    preference_renderer: Any | None = None,
    chat_contract_verifier: Any | None = None,
) -> dict[str, Any]:
    """Retokenize every bound row before accepting prepare-only output."""

    from tools.fine_tuning.unsloth import train_dpo, train_sft

    injected_tokenizer = tokenizer is not None
    path = run_root / "training" / GLOBAL_TOKENIZER_PREFLIGHT_FILENAME
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Prepare-only global tokenizer preflight audit is missing")
    audit = read_object(path)
    entries = audit.get("agents")
    if not isinstance(entries, list):
        raise RuntimeError("Prepare-only global tokenizer preflight agents are invalid")
    by_agent = {
        str(entry.get("agent")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if set(by_agent) != set(agents) or len(by_agent) != len(agents):
        raise RuntimeError("Prepare-only global tokenizer preflight agent set drifted")
    first_config = read_object(run_root / "configs" / f"{agents[0]}.json")
    audit_tokenizer_closure = _validated_global_tokenizer_closure_record(
        audit.get("tokenizerClosure"),
        config=first_config,
        allow_injected_test_double=tokenizer is not None,
    )
    if tokenizer is None:
        if tokenizer_file_sha256 is not None:
            raise RuntimeError(
                "Prepare-only tokenizer digest cannot be injected without a tokenizer"
            )
        if (
            audit_tokenizer_closure.get("schemaVersion")
            != GLOBAL_TOKENIZER_SNAPSHOT_SCHEMA
        ):
            raise RuntimeError(
                "Prepare-only requires a production tokenizer snapshot"
            )
        tokenizer, verified_tokenizer_closure = (
            _load_verified_global_tokenizer_snapshot(
                snapshot_dir=(
                    run_root
                    / "training"
                    / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
                ),
                config=first_config,
                expected_contract=audit_tokenizer_closure,
            )
        )
    else:
        if tokenizer_file_sha256 != first_config.get("baseModelTokenizerDigest"):
            raise RuntimeError("Injected prepare-only tokenizer digest drifted")
        verified_tokenizer_closure = (
            _injected_global_tokenizer_snapshot_contract(
                first_config,
                str(tokenizer_file_sha256),
            )
        )
        if verified_tokenizer_closure != audit_tokenizer_closure:
            raise RuntimeError("Injected prepare-only tokenizer closure drifted")
    if preference_renderer is None:
        from trl.data_utils import maybe_apply_chat_template  # type: ignore

        preference_renderer = maybe_apply_chat_template
    if not callable(preference_renderer):
        raise RuntimeError("Prepare-only preference renderer is unavailable")
    if chat_contract_verifier is None:
        chat_contract_verifier = train_sft.verify_chat_template_contract
    if not callable(chat_contract_verifier):
        raise RuntimeError("Prepare-only chat-template verifier is unavailable")
    for agent in agents:
        entry = by_agent[agent]
        config = read_object(run_root / "configs" / f"{agent}.json")
        preference_config = train_dpo._validate_preference_training_config(
            config
        )
        expected_sft_datasets = train_sft._sft_checkpoint_dataset_sha256(
            config
        )
        expected_preference_datasets = (
            train_dpo._preference_dataset_file_sha256(config)
        )
        expected_sft_code = train_sft._sft_training_code_sha256(config)
        expected_preference_code = train_dpo._preference_training_code_sha256(
            config,
            preference_trainer=preference_config["preferenceTrainer"],
        )
        chat_contract_verifier(
            config.get("chatTemplateContract"),
            tokenizer=tokenizer,
        )
        dataset_dir = Path(str(config.get("dataset_dir") or "")).resolve()
        expected_dataset_dir = (
            run_root
            / "generated"
            / "fine_tuning"
            / agent
            / "experiments"
            / str(config.get("variant") or "")
        ).resolve()
        if dataset_dir != expected_dataset_dir:
            raise RuntimeError(
                f"Prepare-only tokenizer dataset path drifted for {agent}"
            )
        train_sft_path = dataset_dir / "train_sft.jsonl"
        val_sft_path = dataset_dir / "val_sft.jsonl"
        train_dpo_path = dataset_dir / "train_dpo.jsonl"
        val_dpo_path = dataset_dir / "val_dpo.jsonl"
        train_sft_rows = train_sft._limit_records(
            read_jsonl(train_sft_path),
            config.get("max_train_records"),
        )
        val_sft_rows = train_sft._limit_records(
            read_jsonl(val_sft_path),
            config.get("max_val_records"),
        )
        train_dpo_source = read_jsonl(train_dpo_path)
        val_dpo_source = read_jsonl(val_dpo_path)
        recomputed_sft = train_sft._preflight_sft_token_lengths(
            {
                "train": (train_sft_rows, train_sft_path),
                "validation": (val_sft_rows, val_sft_path),
            },
            tokenizer=tokenizer,
            max_sequence_length=config.get("max_seq_length"),
            minimum_sequence_margin_tokens=config.get(
                "sft_minimum_sequence_margin_tokens",
                train_sft.SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            ),
            agent=agent,
            fleet_loss_share_contract=config.get("fleetLossShareContract"),
            public_corpus_loss_share_contract=config.get(
                "publicCorpusLossShareContract"
            ),
            fleet_config=config,
        )
        recomputed_preference = train_dpo._preflight_preference_token_lengths(
            {
                "train": [
                    train_dpo.row_to_preference(row)
                    for row in train_dpo_source
                ],
                "validation": [
                    train_dpo.row_to_preference(row)
                    for row in val_dpo_source
                ],
            },
            tokenizer=tokenizer,
            render_preference=preference_renderer,
            max_prompt_length=preference_config["maxPromptLength"],
            max_sequence_length=config.get("max_seq_length"),
            minimum_prompt_margin_tokens=config.get(
                "preference_minimum_prompt_margin_tokens",
                train_dpo.PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS,
            ),
            minimum_sequence_margin_tokens=config.get(
                "preference_minimum_sequence_margin_tokens",
                train_dpo.PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            ),
            source_splits={
                "train": train_dpo_source,
                "validation": val_dpo_source,
            },
            agent=agent,
            fleet_loss_share_contract=config.get("fleetLossShareContract"),
            public_corpus_loss_share_contract=config.get(
                "publicCorpusLossShareContract"
            ),
            fleet_config=config,
        )
        if (
            entry.get("sft") != recomputed_sft
            or entry.get("preference") != recomputed_preference
        ):
            raise RuntimeError(
                f"Prepare-only exact tokenizer evidence drifted for {agent}"
            )
        for phase, dataset_field, training_code_field, expected_datasets, expected_code in (
            (
                "sft",
                "sftDatasetFileSHA256",
                "sftTrainingCodeSHA256",
                expected_sft_datasets,
                expected_sft_code,
            ),
            (
                "preference",
                "preferenceDatasetFileSHA256",
                "preferenceTrainingCodeSHA256",
                expected_preference_datasets,
                expected_preference_code,
            ),
        ):
            phase_evidence = entry.get(phase)
            if not isinstance(phase_evidence, Mapping):
                raise RuntimeError(
                    f"Prepare-only global tokenizer preflight lacks {agent} {phase}"
                )
            if (
                entry.get(dataset_field) != expected_datasets
                or entry.get(training_code_field) != expected_code
            ):
                raise RuntimeError(
                    f"Prepare-only global tokenizer preflight {agent} {phase} "
                    "input binding drifted"
                )
            reconstructed = {
                **phase_evidence,
                "datasetFileSHA256": expected_datasets,
                "trainingCodeSHA256": expected_code,
            }
            verifier = (
                _verified_global_tokenizer_preflight_test_only
                if injected_tokenizer
                else _verified_global_tokenizer_preflight
            )
            verifier(
                run_root=run_root,
                agent=agent,
                config=config,
                phase=phase,
                bound_preflight=reconstructed,
                audit_snapshot=audit,
                tokenizer=tokenizer,
                preference_renderer=(
                    preference_renderer if phase == "preference" else None
                ),
            )
    if read_object(path) != audit:
        raise RuntimeError("Prepare-only global tokenizer preflight changed during verification")
    if (
        verified_tokenizer_closure.get("schemaVersion")
        == GLOBAL_TOKENIZER_SNAPSHOT_SCHEMA
        and _global_tokenizer_snapshot_contract(
            snapshot_dir=(
                run_root / "training" / GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
            ),
            config=first_config,
        )
        != verified_tokenizer_closure
    ):
        raise RuntimeError(
            "Prepare-only global tokenizer snapshot changed during verification"
        )
    return audit


_FLEET_LOSS_SHARE_FIELD_NAMES = {
    "sft": {
        "denominatorTokenCount": "assistantTargetTokenCount",
        "supplementalNumeratorTokenCount": (
            "supplementalStaticAssistantTargetTokenCount"
        ),
        "publicNumeratorTokenCount": (
            "publicBehavioralAssistantTargetTokenCount"
        ),
        "perSourceFamilyNumeratorTokenCounts": (
            "supplementalStaticAssistantTargetTokenCountsBySourceFamily"
        ),
    },
    "dpo": {
        "denominatorTokenCount": "chosenTargetTokenCount",
        "supplementalNumeratorTokenCount": (
            "supplementalStaticChosenTargetTokenCount"
        ),
        "publicNumeratorTokenCount": (
            "publicBehavioralChosenTargetTokenCount"
        ),
        "perSourceFamilyNumeratorTokenCounts": (
            "supplementalStaticChosenTargetTokenCountsBySourceFamily"
        ),
    },
}
_FLEET_SOURCE_ROLES = (
    "behavioral_primary",
    "public_behavioral",
    "supplemental_static",
)
_FLEET_DPO_TOKENIZATION_POLICY = {
    "trainerImplementation": "trl.DPOTrainer.tokenize_row",
    "trlVersion": "0.24.0",
    "completionTokenization": "add_special_tokens_false",
    "completionSuffix": "append_tokenizer_eos_token_id",
    "appendedEOSTokensPerCompletion": 1,
}
_FLEET_OPTIMIZER_FAMILY_SHARE_SCHEMA = (
    "lumen.fleet-optimizer-family-share/1.0.0"
)
_FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY = "fleet_orchestration_native"
_FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE = {
    "sft": "fleet_orchestration_event_graph",
    "dpo": "fleet_orchestration_event_graph_preference",
}
_FLEET_OPTIMIZER_FAMILY_SHARE_LANES = {
    "sft": {
        "basis": "assistant_mask_non_ignored_token_count",
        "numeratorEvidenceField": (
            "nativeOrchestrationAssistantTargetTokenCount"
        ),
        "denominatorEvidenceField": "assistantTargetTokenCount",
        "minimumBasisPoints": 5_000,
        "maximumBasisPoints": 6_000,
    },
    "dpo": {
        "basis": "preference_pair_count",
        "numeratorEvidenceField": "nativeOrchestrationPreferencePairCount",
        "denominatorEvidenceField": "preferencePairCount",
        "minimumBasisPoints": 1_800,
        "maximumBasisPoints": 2_200,
    },
}
_FLEET_OPTIMIZER_FAMILY_SHARE_COMPARISON_RULES = {
    "minimum": (
        "numeratorCount*basisPointDenominator>="
        "denominatorCount*minimumBasisPoints"
    ),
    "maximum": (
        "numeratorCount*basisPointDenominator<="
        "denominatorCount*maximumBasisPoints"
    ),
}


def _pipeline_exact_mapping(
    value: Any,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{label} has an invalid schema")
    return value


_PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES = {
    "sft": {
        "denominatorTokenCount": "assistantTargetTokenCount",
        "publicNumeratorTokenCount": "publicAssistantTargetTokenCount",
    },
    "dpo": {
        "denominatorTokenCount": "chosenTargetTokenCount",
        "publicNumeratorTokenCount": "publicChosenTargetTokenCount",
    },
}
_PUBLIC_CORPUS_DPO_TOKENIZATION_POLICY = dict(
    _FLEET_DPO_TOKENIZATION_POLICY
)


def _pipeline_validated_public_corpus_loss_share_contract(
    value: Any,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate the all-agent exact public-target cap."""

    contract = _pipeline_exact_mapping(
        value,
        {
            "schemaVersion",
            "enforcementRequired",
            "enforcementPhase",
            "requiredLanes",
            "authoritativeCapEncoding",
            "basisPointDenominator",
            "capBasisPoints",
            "dpoTokenizationPolicy",
            "exactTokenEvidenceContract",
            "failurePolicy",
            "rowMetadataContract",
            "sourceSelectionProxy",
            "tokenizer",
            "tokenAccounting",
        },
        label="Public-corpus loss-share contract",
    )
    if (
        contract.get("schemaVersion")
        != "lumen.public-corpus-loss-share/1.0.0"
        or contract.get("enforcementRequired") is not True
        or contract.get("enforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
        or contract.get("requiredLanes") != ["sft", "dpo"]
        or contract.get("authoritativeCapEncoding")
        != "integer_basis_points"
        or type(contract.get("basisPointDenominator")) is not int
        or contract.get("basisPointDenominator") != 10_000
        or contract.get("failurePolicy") != "abort_before_optimizer"
    ):
        raise RuntimeError("Public-corpus loss-share contract controls drifted")

    caps = _pipeline_exact_mapping(
        contract.get("capBasisPoints"),
        {"requested", "hard"},
        label="Public-corpus loss-share caps",
    )
    requested_cap = caps.get("requested")
    hard_cap = caps.get("hard")
    if (
        type(requested_cap) is not int
        or type(hard_cap) is not int
        or not 0 <= requested_cap <= hard_cap
        or hard_cap != 3_500
    ):
        raise RuntimeError("Public-corpus loss-share caps drifted")

    if contract.get("rowMetadataContract") != {
        "publicSourceFamilyPrefix": "public_adapter_corpus_",
        "publicCorpusField": "publicCorpus",
        "classificationRule": "prefix_and_nonempty_lineage_required",
        "mismatch": "hard_fail",
    }:
        raise RuntimeError("Public-corpus row-metadata contract drifted")

    source_selection_proxy = _pipeline_exact_mapping(
        contract.get("sourceSelectionProxy"),
        {"status", "maximumPublicShareBasisPoints", "contract"},
        label="Public-corpus source-selection proxy",
    )
    source_proxy_contract = _pipeline_exact_mapping(
        source_selection_proxy.get("contract"),
        {
            "schemaVersion",
            "status",
            "strategy",
            "maxCharsPerToken",
            "exactPinnedTokenizerAuthoritative",
            "authoritativeEnforcementPhase",
        },
        label="Public-corpus source-token proxy contract",
    )
    if (
        source_selection_proxy.get("status")
        != "safety_budget_not_exact_token_count"
        or source_selection_proxy.get("maximumPublicShareBasisPoints")
        != min(requested_cap, 3_000)
        or source_proxy_contract.get("schemaVersion")
        != "lumen.source-token-proxy/1.0.0"
        or source_proxy_contract.get("status")
        != "source_side_selection_proxy_not_exact_token_count"
        or source_proxy_contract.get("strategy")
        != "max_whitespace_terms_utf8_byte_ceiling"
        or type(source_proxy_contract.get("maxCharsPerToken")) is not int
        or source_proxy_contract["maxCharsPerToken"] <= 0
        or source_proxy_contract.get("exactPinnedTokenizerAuthoritative")
        is not True
        or source_proxy_contract.get("authoritativeEnforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
    ):
        raise RuntimeError("Public-corpus source-selection proxy drifted")

    dpo_policy = _pipeline_exact_mapping(
        contract.get("dpoTokenizationPolicy"),
        set(_PUBLIC_CORPUS_DPO_TOKENIZATION_POLICY),
        label="Public-corpus DPO tokenization policy",
    )
    if dict(dpo_policy) != _PUBLIC_CORPUS_DPO_TOKENIZATION_POLICY:
        raise RuntimeError("Public-corpus DPO tokenization policy drifted")
    if contract.get("tokenAccounting") != {
        "sft": "assistant_mask_non_ignored_token_count",
        "dpo": (
            "rendered_chosen_completion_tokens_add_special_tokens_false_"
            "plus_one_trl_0_24_0_appended_eos"
        ),
    }:
        raise RuntimeError("Public-corpus token-accounting contract drifted")

    tokenizer = _pipeline_exact_mapping(
        contract.get("tokenizer"),
        {
            "baseModelID",
            "baseModelRevision",
            "tokenizerSHA256",
            "tokenizerClosureSHA256",
        },
        label="Public-corpus tokenizer binding",
    )
    if (
        tokenizer.get("baseModelID") != config.get("base_model_name")
        or tokenizer.get("baseModelRevision")
        != config.get("baseModelRevision")
        or tokenizer.get("tokenizerSHA256")
        != config.get("baseModelTokenizerDigest")
        or tokenizer.get("tokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(tokenizer.get("baseModelRevision") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(tokenizer.get("tokenizerSHA256") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(tokenizer.get("tokenizerClosureSHA256") or ""),
        )
        is None
    ):
        raise RuntimeError("Public-corpus tokenizer binding drifted")

    exact = _pipeline_exact_mapping(
        contract.get("exactTokenEvidenceContract"),
        {
            "required",
            "schemaVersion",
            "statusAtGeneration",
            "tokenizer",
            "comparisonRule",
            "lanes",
        },
        label="Public-corpus exact-token evidence contract",
    )
    if (
        exact.get("required") is not True
        or exact.get("schemaVersion")
        != "lumen.public-corpus-loss-share-evidence/1.0.0"
        or exact.get("statusAtGeneration")
        != "pending_exact_tokenizer_preflight"
        or exact.get("tokenizer") != "pinned_qwen_tokenizer"
        or exact.get("comparisonRule")
        != (
            "numeratorTokenCount*basisPointDenominator<="
            "denominatorTokenCount*capBasisPoints"
        )
    ):
        raise RuntimeError("Public-corpus exact-token evidence contract drifted")
    lanes = _pipeline_exact_mapping(
        exact.get("lanes"),
        {"sft", "dpo"},
        label="Public-corpus exact-token evidence lanes",
    )
    for lane, fields in _PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES.items():
        if lanes.get(lane) != fields:
            raise RuntimeError(
                f"Public-corpus {lane} exact-token fields drifted"
            )
    return dict(contract)


def _pipeline_public_corpus_row_classification(
    row: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> tuple[str, bool]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Public-corpus evidence row has invalid metadata")
    source_family = metadata.get("sourceFamily")
    if (
        not isinstance(source_family, str)
        or not source_family
        or source_family.strip() != source_family
    ):
        raise RuntimeError(
            "Public-corpus evidence row metadata.sourceFamily is not canonical"
        )
    row_contract = contract["rowMetadataContract"]
    has_public_prefix = source_family.startswith(
        row_contract["publicSourceFamilyPrefix"]
    )
    lineage = metadata.get(row_contract["publicCorpusField"])
    has_public_lineage = isinstance(lineage, Mapping) and bool(lineage)
    if has_public_prefix != has_public_lineage:
        raise RuntimeError(
            "Public-corpus evidence prefix and lineage classification disagree"
        )
    return source_family, has_public_prefix


def _pipeline_public_corpus_cap_passes(
    numerator: Any,
    denominator: Any,
    cap: Any,
) -> bool:
    return (
        type(numerator) is int
        and numerator >= 0
        and type(denominator) is int
        and denominator > 0
        and type(cap) is int
        and 0 <= cap <= 10_000
        and numerator * 10_000 <= denominator * cap
    )


def _pipeline_exact_public_target_token_count(
    row: Mapping[str, Any],
    *,
    lane: str,
    tokenizer: Any,
    source_path: Path,
    split: str,
    row_index: int,
    preference_renderer: Any | None,
    appended_eos_token_id: int | None,
) -> int:
    """Recompute the optimizer-bearing target count from the pinned tokenizer."""

    if lane == "sft":
        from tools.fine_tuning.unsloth import train_sft

        messages = train_sft.normalize_chat_messages(
            dict(row),
            row_index=row_index,
            path=source_path,
        )
        tokenized = train_sft.tokenize_assistant_only_row(
            tokenizer,
            messages,
            path=source_path,
            row_index=row_index,
            max_seq_length=None,
        )
        labels = tokenized.get("labels")
        if not isinstance(labels, list):
            raise RuntimeError(
                f"Public-corpus SFT {split} row {row_index} lacks exact labels"
            )
        target_tokens = sum(1 for label in labels if label != -100)
    elif lane == "dpo":
        from tools.fine_tuning.unsloth import train_dpo

        if not callable(preference_renderer):
            raise RuntimeError(
                "Public-corpus DPO verification requires the pinned renderer"
            )
        if type(appended_eos_token_id) is not int:
            raise RuntimeError(
                "Public-corpus DPO verification lacks the pinned EOS token"
            )
        try:
            rendered = preference_renderer(
                train_dpo.row_to_preference(dict(row)),
                tokenizer=tokenizer,
            )
        except Exception as exc:
            raise RuntimeError(
                "Public-corpus DPO verification could not render "
                f"{split} row {row_index}"
            ) from exc
        if not isinstance(rendered, Mapping):
            raise RuntimeError(
                f"Public-corpus DPO {split} row {row_index} rendered invalid data"
            )
        target_tokens = len(
            train_dpo._dpo_completion_token_ids(
                tokenizer,
                rendered.get("chosen"),
                split=split,
                row_index=row_index,
                field="chosen completion",
                appended_eos_token_id=appended_eos_token_id,
            )
        )
    else:
        raise RuntimeError(f"Unsupported public-corpus target lane: {lane}")
    if type(target_tokens) is not int or target_tokens <= 0:
        raise RuntimeError(
            f"Public-corpus {lane} {split} row {row_index} has no target tokens"
        )
    return target_tokens


def _verify_public_corpus_loss_share_evidence(
    *,
    value: Any,
    config: Mapping[str, Any],
    phase: str,
    dataset_dir: Path,
    tokenizer: Any | None = None,
    preference_renderer: Any | None = None,
    require_exact_tokenizer_counts: bool = False,
) -> dict[str, Any]:
    agent = config.get("agent")
    if not isinstance(agent, str) or not agent:
        raise RuntimeError(
            "Public-corpus exact-token evidence requires a controlled agent"
        )
    if phase not in {"sft", "preference"}:
        raise RuntimeError(f"Unsupported public-corpus loss-share phase: {phase}")
    lane = "sft" if phase == "sft" else "dpo"
    contract = _pipeline_validated_public_corpus_loss_share_contract(
        config.get("publicCorpusLossShareContract"),
        config=config,
    )
    evidence = _pipeline_exact_mapping(
        value,
        {
            "schemaVersion",
            "status",
            "lane",
            "enforcementScope",
            "basisPointDenominator",
            "capBasisPoints",
            "tokenizer",
            "tokenAccounting",
            "dpoTokenizationPolicy",
            "contractSHA256",
            "splits",
        },
        label="Public-corpus loss-share evidence",
    )
    if (
        evidence.get("schemaVersion")
        != "lumen.public-corpus-loss-share-evidence/1.0.0"
        or evidence.get("status") != "passed"
        or evidence.get("lane") != lane
        or evidence.get("enforcementScope")
        != "optimizer_train_with_validation_observation"
        or type(evidence.get("basisPointDenominator")) is not int
        or evidence.get("basisPointDenominator") != 10_000
        or evidence.get("capBasisPoints") != contract["capBasisPoints"]
        or evidence.get("tokenizer") != contract["tokenizer"]
        or evidence.get("tokenAccounting")
        != contract["tokenAccounting"][lane]
        or evidence.get("dpoTokenizationPolicy")
        != (contract["dpoTokenizationPolicy"] if lane == "dpo" else None)
        or evidence.get("contractSHA256") != canonical_sha256(contract)
    ):
        raise RuntimeError("Public-corpus loss-share evidence bindings drifted")

    split_values = _pipeline_exact_mapping(
        evidence.get("splits"),
        {"train", "validation"},
        label="Public-corpus loss-share evidence splits",
    )
    filenames = (
        {"train": "train_sft.jsonl", "validation": "val_sft.jsonl"}
        if lane == "sft"
        else {"train": "train_dpo.jsonl", "validation": "val_dpo.jsonl"}
    )
    fields = _PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES[lane]
    if require_exact_tokenizer_counts and tokenizer is None:
        raise RuntimeError(
            "Public-corpus verification requires the exact pinned tokenizer"
        )
    appended_eos_token_id = None
    if lane == "dpo" and tokenizer is not None:
        from tools.fine_tuning.unsloth import train_dpo

        if preference_renderer is None:
            from trl.data_utils import maybe_apply_chat_template  # type: ignore

            preference_renderer = maybe_apply_chat_template
        appended_eos_token_id = train_dpo._dpo_appended_eos_token_id(tokenizer)
    for split in ("train", "validation"):
        split_evidence = _pipeline_exact_mapping(
            split_values.get(split),
            {
                "records",
                "capEnforcementStatus",
                "sourceRowsSHA256",
                "rowTokenEvidence",
                *fields.values(),
            },
            label=f"Public-corpus {lane} {split} loss-share evidence",
        )
        source_rows = read_jsonl(dataset_dir / filenames[split])
        if lane == "sft":
            limit_key = (
                "max_train_records" if split == "train" else "max_val_records"
            )
            limit = int(config.get(limit_key) or 0)
            if limit > 0:
                source_rows = source_rows[:limit]
        expected_enforcement_status = (
            "optimizer_enforced"
            if split == "train"
            else "observed_non_optimizer_split"
        )
        row_values = split_evidence.get("rowTokenEvidence")
        if (
            not source_rows
            or not isinstance(row_values, list)
            or len(row_values) != len(source_rows)
            or split_evidence.get("records") != len(source_rows)
            or split_evidence.get("capEnforcementStatus")
            != expected_enforcement_status
        ):
            raise RuntimeError(
                f"Public-corpus {lane} {split} evidence row count drifted"
            )
        denominator = 0
        public = 0
        row_hashes: list[str] = []
        for index, (source_row, row_value) in enumerate(
            zip(source_rows, row_values)
        ):
            row_evidence = _pipeline_exact_mapping(
                row_value,
                {
                    "rowIndex",
                    "sourceRowSHA256",
                    "sourceFamily",
                    "isPublicCorpus",
                    "targetTokenCount",
                },
                label=f"Public-corpus {lane} {split} row evidence",
            )
            row_hash = canonical_sha256(source_row)
            source_family, is_public = (
                _pipeline_public_corpus_row_classification(
                    source_row,
                    contract=contract,
                )
            )
            target_tokens = row_evidence.get("targetTokenCount")
            exact_target_tokens = (
                _pipeline_exact_public_target_token_count(
                    source_row,
                    lane=lane,
                    tokenizer=tokenizer,
                    source_path=dataset_dir / filenames[split],
                    split=split,
                    row_index=index,
                    preference_renderer=preference_renderer,
                    appended_eos_token_id=appended_eos_token_id,
                )
                if tokenizer is not None
                else None
            )
            if (
                type(row_evidence.get("rowIndex")) is not int
                or row_evidence.get("rowIndex") != index
                or row_evidence.get("sourceRowSHA256") != row_hash
                or row_evidence.get("sourceFamily") != source_family
                or type(row_evidence.get("isPublicCorpus")) is not bool
                or row_evidence.get("isPublicCorpus") is not is_public
                or type(target_tokens) is not int
                or target_tokens <= 0
                or (
                    exact_target_tokens is not None
                    and target_tokens != exact_target_tokens
                )
            ):
                raise RuntimeError(
                    f"Public-corpus {lane} {split} exact-token row evidence drifted"
                )
            row_hashes.append(row_hash)
            denominator += target_tokens
            if is_public:
                public += target_tokens
        if (
            type(split_evidence.get(fields["denominatorTokenCount"])) is not int
            or type(
                split_evidence.get(fields["publicNumeratorTokenCount"])
            )
            is not int
            or split_evidence.get("sourceRowsSHA256")
            != canonical_sha256(row_hashes)
            or split_evidence.get(fields["denominatorTokenCount"])
            != denominator
            or split_evidence.get(fields["publicNumeratorTokenCount"])
            != public
        ):
            raise RuntimeError(
                f"Public-corpus {lane} {split} totals failed reconstruction"
            )
        if split == "train" and any(
            not _pipeline_public_corpus_cap_passes(public, denominator, cap)
            for cap in contract["capBasisPoints"].values()
        ):
            raise RuntimeError(
                f"Public-corpus {lane} {split} exact-token cap failed"
            )
    return dict(evidence)


def _pipeline_validated_fleet_loss_share_contract(
    value: Any,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _pipeline_exact_mapping(
        value,
        {
            "schemaVersion",
            "enforcementRequired",
            "enforcementPhase",
            "requiredLanes",
            "authoritativeCapEncoding",
            "basisPointDenominator",
            "capsBasisPoints",
            "dpoTokenizationPolicy",
            "exactTokenEvidenceContract",
            "failurePolicy",
            "optimizerFamilyShareBands",
            "rowMetadataContract",
            "sourceSelectionProxy",
            "sourceRoleRegistry",
            "tokenizer",
            "tokenAccounting",
        },
        label="Fleet loss-share contract",
    )
    if (
        contract.get("schemaVersion") != "lumen.fleet-loss-share/1.4.0"
        or contract.get("enforcementRequired") is not True
        or contract.get("enforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
        or contract.get("requiredLanes") != ["sft", "dpo"]
        or contract.get("authoritativeCapEncoding") != "integer_basis_points"
        or type(contract.get("basisPointDenominator")) is not int
        or contract.get("basisPointDenominator") != 10_000
        or contract.get("failurePolicy") != "abort_before_optimizer"
    ):
        raise RuntimeError("Fleet loss-share contract controls drifted")
    caps = _pipeline_exact_mapping(
        contract.get("capsBasisPoints"),
        {
            "supplementalStaticTotal",
            "publicBehavioralTotal",
            "eachSupplementalSourceFamily",
        },
        label="Fleet loss-share caps",
    )
    if (
        not all(
            type(value) is int
            for group in caps.values()
            if isinstance(group, Mapping)
            for value in group.values()
        )
        or caps.get("supplementalStaticTotal")
        != {"requested": 2_500, "hard": 3_000}
        or caps.get("publicBehavioralTotal")
        != {"requested": 3_500, "hard": 3_500}
        or caps.get("eachSupplementalSourceFamily") != {"hard": 1_000}
    ):
        raise RuntimeError("Fleet loss-share caps drifted")
    if contract.get("rowMetadataContract") != {
        "requiredCanonicalFields": ["sourceFamily", "taskType"],
        "missingOrUnknown": "hard_fail",
    }:
        raise RuntimeError("Fleet row-metadata contract drifted")
    source_selection_proxy = _pipeline_exact_mapping(
        contract.get("sourceSelectionProxy"),
        {
            "status",
            "maximumPublicBehavioralShareBasisPoints",
            "maximumSupplementalStaticShareBasisPoints",
            "optimizerFamilySafetyBand",
            "contract",
        },
        label="Fleet source-selection proxy",
    )
    source_family_safety_band = _pipeline_exact_mapping(
        source_selection_proxy.get("optimizerFamilySafetyBand"),
        {
            "schemaVersion",
            "lane",
            "basis",
            "sourceFamily",
            "taskType",
            "minimumBasisPoints",
            "maximumBasisPoints",
            "selectionPolicy",
            "authoritativeExactBandBasisPoints",
        },
        label="Fleet optimizer-family source-proxy safety band",
    )
    authoritative_exact_band = _pipeline_exact_mapping(
        source_family_safety_band.get("authoritativeExactBandBasisPoints"),
        {"minimum", "maximum"},
        label="Fleet authoritative optimizer-family band reference",
    )
    source_proxy_contract = _pipeline_exact_mapping(
        source_selection_proxy.get("contract"),
        {
            "schemaVersion",
            "status",
            "strategy",
            "maxCharsPerToken",
            "exactPinnedTokenizerAuthoritative",
            "authoritativeEnforcementPhase",
        },
        label="Fleet source-token proxy contract",
    )
    if (
        source_selection_proxy.get("status")
        != "safety_budget_not_exact_token_count"
        or source_selection_proxy.get(
            "maximumPublicBehavioralShareBasisPoints"
        )
        != 3_000
        or source_selection_proxy.get("maximumSupplementalStaticShareBasisPoints")
        != 1_500
        or type(source_family_safety_band.get("minimumBasisPoints")) is not int
        or type(source_family_safety_band.get("maximumBasisPoints")) is not int
        or type(authoritative_exact_band.get("minimum")) is not int
        or type(authoritative_exact_band.get("maximum")) is not int
        or source_family_safety_band
        != {
            "schemaVersion": (
                "lumen.fleet-optimizer-family-source-proxy/1.0.0"
            ),
            "lane": "sft",
            "basis": "assistant_target_source_token_proxy_count",
            "sourceFamily": "fleet_orchestration_native",
            "taskType": "fleet_orchestration_event_graph",
            "minimumBasisPoints": 5_300,
            "maximumBasisPoints": 6_210,
            "selectionPolicy": (
                "retain_non_public_then_bound_public_behavioral"
            ),
            "authoritativeExactBandBasisPoints": authoritative_exact_band,
        }
        or authoritative_exact_band != {"minimum": 5_000, "maximum": 6_000}
        or source_proxy_contract.get("schemaVersion")
        != "lumen.source-token-proxy/1.0.0"
        or source_proxy_contract.get("status")
        != "source_side_selection_proxy_not_exact_token_count"
        or source_proxy_contract.get("strategy")
        != "max_whitespace_terms_utf8_byte_ceiling"
        or type(source_proxy_contract.get("maxCharsPerToken")) is not int
        or source_proxy_contract["maxCharsPerToken"] <= 0
        or source_proxy_contract.get("exactPinnedTokenizerAuthoritative") is not True
        or source_proxy_contract.get("authoritativeEnforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
    ):
        raise RuntimeError("Fleet source-selection proxy contract drifted")
    dpo_tokenization_policy = _pipeline_exact_mapping(
        contract.get("dpoTokenizationPolicy"),
        set(_FLEET_DPO_TOKENIZATION_POLICY),
        label="Fleet DPO tokenization policy",
    )
    if dict(dpo_tokenization_policy) != _FLEET_DPO_TOKENIZATION_POLICY:
        raise RuntimeError("Fleet DPO tokenization policy drifted")
    family_share = _pipeline_exact_mapping(
        contract.get("optimizerFamilyShareBands"),
        {
            "schemaVersion",
            "enforcementScope",
            "classification",
            "lanes",
            "comparisonRules",
            "failurePolicy",
        },
        label="Fleet optimizer-family share bands",
    )
    classification = _pipeline_exact_mapping(
        family_share.get("classification"),
        {"sourceFamily", "taskTypeByLane"},
        label="Fleet optimizer-family classification",
    )
    task_types = _pipeline_exact_mapping(
        classification.get("taskTypeByLane"),
        {"sft", "dpo"},
        label="Fleet optimizer-family task types",
    )
    family_lanes = _pipeline_exact_mapping(
        family_share.get("lanes"),
        {"sft", "dpo"},
        label="Fleet optimizer-family lane bands",
    )
    if (
        family_share.get("schemaVersion")
        != _FLEET_OPTIMIZER_FAMILY_SHARE_SCHEMA
        or family_share.get("enforcementScope") != "optimizer_train_only"
        or classification.get("sourceFamily")
        != _FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
        or dict(task_types) != _FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE
        or family_share.get("comparisonRules")
        != _FLEET_OPTIMIZER_FAMILY_SHARE_COMPARISON_RULES
        or family_share.get("failurePolicy") != "abort_before_optimizer"
    ):
        raise RuntimeError("Fleet optimizer-family share contract drifted")
    for expected_lane, expected_band in (
        _FLEET_OPTIMIZER_FAMILY_SHARE_LANES.items()
    ):
        actual_band = _pipeline_exact_mapping(
            family_lanes.get(expected_lane),
            set(expected_band),
            label=f"Fleet {expected_lane} optimizer-family share band",
        )
        if any(
            type(actual_band[field]) is not type(expected_value)
            or actual_band[field] != expected_value
            for field, expected_value in expected_band.items()
        ):
            raise RuntimeError(
                f"Fleet {expected_lane} optimizer-family share band drifted"
            )
    if contract.get("tokenAccounting") != {
        "sft": "assistant_mask_non_ignored_token_count",
        "dpo": (
            "rendered_chosen_completion_tokens_add_special_tokens_false_"
            "plus_one_trl_0_24_0_appended_eos"
        ),
    }:
        raise RuntimeError("Fleet token-accounting contract drifted")
    tokenizer = _pipeline_exact_mapping(
        contract.get("tokenizer"),
        {
            "baseModelID",
            "baseModelRevision",
            "tokenizerSHA256",
            "tokenizerClosureSHA256",
        },
        label="Fleet tokenizer binding",
    )
    if (
        tokenizer.get("baseModelID") != config.get("base_model_name")
        or tokenizer.get("baseModelRevision") != config.get("baseModelRevision")
        or tokenizer.get("tokenizerSHA256")
        != config.get("baseModelTokenizerDigest")
        or tokenizer.get("tokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
        or re.fullmatch(r"[0-9a-f]{40}", str(tokenizer.get("baseModelRevision") or ""))
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(tokenizer.get("tokenizerSHA256") or ""))
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(tokenizer.get("tokenizerClosureSHA256") or ""),
        )
        is None
    ):
        raise RuntimeError("Fleet tokenizer binding drifted")

    exact = _pipeline_exact_mapping(
        contract.get("exactTokenEvidenceContract"),
        {
            "required",
            "schemaVersion",
            "statusAtGeneration",
            "tokenizer",
            "comparisonRule",
            "lanes",
        },
        label="Fleet exact-token evidence contract",
    )
    if (
        exact.get("required") is not True
        or exact.get("schemaVersion")
        != "lumen.fleet-loss-share-evidence/1.2.0"
        or exact.get("statusAtGeneration")
        != "pending_exact_tokenizer_preflight"
        or exact.get("tokenizer") != "pinned_qwen_tokenizer"
        or exact.get("comparisonRule")
        != (
            "numeratorTokenCount*basisPointDenominator<="
            "denominatorTokenCount*capBasisPoints"
        )
    ):
        raise RuntimeError("Fleet exact-token evidence contract drifted")
    lanes = _pipeline_exact_mapping(
        exact.get("lanes"),
        {"sft", "dpo"},
        label="Fleet exact-token evidence lanes",
    )
    for lane, fields in _FLEET_LOSS_SHARE_FIELD_NAMES.items():
        if lanes.get(lane) != fields:
            raise RuntimeError(f"Fleet {lane} exact-token fields drifted")

    registry = _pipeline_exact_mapping(
        contract.get("sourceRoleRegistry"),
        {
            "schemaVersion",
            "unknownPairs",
            "categories",
            "registeredPairs",
            "publicBehavioralRule",
        },
        label="Fleet source-role registry",
    )
    if (
        registry.get("schemaVersion") != "lumen.fleet-source-role/1.0.0"
        or registry.get("unknownPairs") != "hard_fail"
        or registry.get("categories") != list(_FLEET_SOURCE_ROLES)
        or registry.get("publicBehavioralRule")
        != {
            "sourceFamilyPrefix": "public_adapter_corpus_",
            "taskType": "public_capability_delegation",
            "requiresPublicCorpusLineage": True,
        }
    ):
        raise RuntimeError("Fleet source-role registry controls drifted")
    registered_pairs = registry.get("registeredPairs")
    if not isinstance(registered_pairs, list) or not registered_pairs:
        raise RuntimeError("Fleet source-role registry is empty")
    observed: set[tuple[str, str]] = set()
    observed_registered_categories: set[str] = set()
    for item in registered_pairs:
        pair = _pipeline_exact_mapping(
            item,
            {"sourceFamily", "taskType", "category"},
            label="Fleet source-role pair",
        )
        source_family = pair.get("sourceFamily")
        task_type = pair.get("taskType")
        category = pair.get("category")
        if (
            not isinstance(source_family, str)
            or not source_family
            or source_family.strip() != source_family
            or not isinstance(task_type, str)
            or not task_type
            or task_type.strip() != task_type
            or category not in _FLEET_SOURCE_ROLES
            or category == "public_behavioral"
            or source_family.startswith("public_adapter_corpus_")
            or (source_family, task_type) in observed
        ):
            raise RuntimeError("Fleet source-role registry contains an invalid pair")
        observed.add((source_family, task_type))
        observed_registered_categories.add(category)
    if observed_registered_categories != {
        "behavioral_primary",
        "supplemental_static",
    }:
        raise RuntimeError(
            "Fleet source-role registry must contain primary and static pairs"
        )
    return dict(contract)


def _pipeline_fleet_source_role(
    row: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> tuple[str, str, str]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Fleet evidence source row has invalid metadata")
    source_family = metadata.get("sourceFamily")
    task_type = metadata.get("taskType")
    if (
        not isinstance(source_family, str)
        or not source_family
        or source_family.strip() != source_family
        or not isinstance(task_type, str)
        or not task_type
        or task_type.strip() != task_type
    ):
        raise RuntimeError("Fleet evidence source row metadata is not canonical")
    registry = contract["sourceRoleRegistry"]
    registered = {
        (item["sourceFamily"], item["taskType"]): item["category"]
        for item in registry["registeredPairs"]
    }
    category = registered.get((source_family, task_type))
    rule = registry["publicBehavioralRule"]
    if category is None and (
        source_family.startswith(rule["sourceFamilyPrefix"])
        and task_type == rule["taskType"]
        and isinstance(metadata.get("publicCorpus"), Mapping)
        and bool(metadata["publicCorpus"])
    ):
        category = "public_behavioral"
    if category not in _FLEET_SOURCE_ROLES:
        raise RuntimeError(
            "Fleet evidence contains an unregistered source-role pair: "
            f"{source_family!r}, {task_type!r}"
        )
    return source_family, task_type, category


def _pipeline_fleet_cap_passes(
    numerator: Any,
    denominator: Any,
    cap: Any,
) -> bool:
    return (
        type(numerator) is int
        and numerator >= 0
        and type(denominator) is int
        and denominator > 0
        and type(cap) is int
        and 0 <= cap <= 10_000
        and numerator * 10_000 <= denominator * cap
    )


def _pipeline_fleet_optimizer_family_band_passes(
    numerator: Any,
    denominator: Any,
    minimum_basis_points: Any,
    maximum_basis_points: Any,
) -> bool:
    return (
        type(numerator) is int
        and numerator >= 0
        and type(denominator) is int
        and denominator > 0
        and type(minimum_basis_points) is int
        and type(maximum_basis_points) is int
        and 0 <= minimum_basis_points <= maximum_basis_points <= 10_000
        and numerator * 10_000 >= denominator * minimum_basis_points
        and numerator * 10_000 <= denominator * maximum_basis_points
    )


def _verify_fleet_loss_share_evidence(
    *,
    value: Any,
    config: Mapping[str, Any],
    phase: str,
    dataset_dir: Path,
) -> dict[str, Any] | None:
    agent = config.get("agent")
    if agent != "fleet":
        if value is not None or config.get("fleetLossShareContract") is not None:
            raise RuntimeError("Non-Fleet training contains Fleet loss-share state")
        return None
    if phase not in {"sft", "preference"}:
        raise RuntimeError(f"Unsupported Fleet loss-share phase: {phase}")
    lane = "sft" if phase == "sft" else "dpo"
    contract = _pipeline_validated_fleet_loss_share_contract(
        config.get("fleetLossShareContract"),
        config=config,
    )
    evidence = _pipeline_exact_mapping(
        value,
        {
            "schemaVersion",
            "status",
            "lane",
            "enforcementScope",
            "basisPointDenominator",
            "capsBasisPoints",
            "tokenizer",
            "tokenAccounting",
            "dpoTokenizationPolicy",
            "optimizerFamilyShareBand",
            "contractSHA256",
            "sourceRoleRegistrySHA256",
            "splits",
        },
        label="Fleet loss-share evidence",
    )
    if (
        evidence.get("schemaVersion")
        != "lumen.fleet-loss-share-evidence/1.2.0"
        or evidence.get("status") != "passed"
        or evidence.get("lane") != lane
        or evidence.get("enforcementScope")
        != "optimizer_train_with_validation_observation"
        or type(evidence.get("basisPointDenominator")) is not int
        or evidence.get("basisPointDenominator") != 10_000
        or evidence.get("capsBasisPoints") != contract["capsBasisPoints"]
        or canonical_sha256(evidence.get("capsBasisPoints"))
        != canonical_sha256(contract["capsBasisPoints"])
        or evidence.get("tokenizer") != contract["tokenizer"]
        or evidence.get("tokenAccounting") != contract["tokenAccounting"][lane]
        or evidence.get("dpoTokenizationPolicy")
        != (contract["dpoTokenizationPolicy"] if lane == "dpo" else None)
        or evidence.get("optimizerFamilyShareBand")
        != contract["optimizerFamilyShareBands"]["lanes"][lane]
        or evidence.get("contractSHA256") != canonical_sha256(contract)
        or evidence.get("sourceRoleRegistrySHA256")
        != canonical_sha256(contract["sourceRoleRegistry"])
    ):
        raise RuntimeError("Fleet loss-share evidence bindings drifted")
    split_values = _pipeline_exact_mapping(
        evidence.get("splits"),
        {"train", "validation"},
        label="Fleet loss-share evidence splits",
    )
    filenames = (
        {"train": "train_sft.jsonl", "validation": "val_sft.jsonl"}
        if lane == "sft"
        else {"train": "train_dpo.jsonl", "validation": "val_dpo.jsonl"}
    )
    fields = _FLEET_LOSS_SHARE_FIELD_NAMES[lane]
    family_share_contract = contract["optimizerFamilyShareBands"]
    selected_family_band = family_share_contract["lanes"][lane]
    native_source_family = family_share_contract["classification"][
        "sourceFamily"
    ]
    native_task_type = family_share_contract["classification"][
        "taskTypeByLane"
    ][lane]
    for split in ("train", "validation"):
        split_evidence = _pipeline_exact_mapping(
            split_values.get(split),
            {
                "records",
                "capEnforcementStatus",
                "sourceRowsSHA256",
                "rowTokenEvidence",
                "targetTokenCountsByCategory",
                "optimizerFamilyBandEnforcementStatus",
                selected_family_band["numeratorEvidenceField"],
                selected_family_band["denominatorEvidenceField"],
                *fields.values(),
            },
            label=f"Fleet {lane} {split} loss-share evidence",
        )
        source_rows = read_jsonl(dataset_dir / filenames[split])
        expected_enforcement_status = (
            "optimizer_enforced"
            if split == "train"
            else "observed_non_optimizer_split"
        )
        row_values = split_evidence.get("rowTokenEvidence")
        if (
            not source_rows
            or not isinstance(row_values, list)
            or len(row_values) != len(source_rows)
            or split_evidence.get("records") != len(source_rows)
            or split_evidence.get("capEnforcementStatus")
            != expected_enforcement_status
        ):
            raise RuntimeError(f"Fleet {lane} {split} evidence row count drifted")
        target_by_category = {category: 0 for category in _FLEET_SOURCE_ROLES}
        supplemental_by_family: dict[str, int] = {}
        native_target_tokens = 0
        native_preference_pairs = 0
        row_hashes: list[str] = []
        for index, (source_row, row_value) in enumerate(zip(source_rows, row_values)):
            row_evidence = _pipeline_exact_mapping(
                row_value,
                {
                    "rowIndex",
                    "sourceRowSHA256",
                    "sourceFamily",
                    "taskType",
                    "category",
                    "targetTokenCount",
                },
                label=f"Fleet {lane} {split} row evidence",
            )
            row_hash = canonical_sha256(source_row)
            source_family, task_type, category = _pipeline_fleet_source_role(
                source_row,
                contract=contract,
            )
            target_tokens = row_evidence.get("targetTokenCount")
            if (
                type(row_evidence.get("rowIndex")) is not int
                or row_evidence.get("rowIndex") != index
                or row_evidence.get("sourceRowSHA256") != row_hash
                or row_evidence.get("sourceFamily") != source_family
                or row_evidence.get("taskType") != task_type
                or row_evidence.get("category") != category
                or type(target_tokens) is not int
                or target_tokens <= 0
            ):
                raise RuntimeError(f"Fleet {lane} {split} row evidence drifted")
            row_hashes.append(row_hash)
            target_by_category[category] += target_tokens
            if (
                source_family == native_source_family
                and task_type == native_task_type
            ):
                native_target_tokens += target_tokens
                native_preference_pairs += 1
            if category == "supplemental_static":
                supplemental_by_family[source_family] = (
                    supplemental_by_family.get(source_family, 0) + target_tokens
                )
        denominator = sum(target_by_category.values())
        supplemental = target_by_category["supplemental_static"]
        public = target_by_category["public_behavioral"]
        family_numerator = (
            native_target_tokens
            if lane == "sft"
            else native_preference_pairs
        )
        family_denominator = denominator if lane == "sft" else len(source_rows)
        observed_categories = split_evidence.get("targetTokenCountsByCategory")
        observed_families = split_evidence.get(
            fields["perSourceFamilyNumeratorTokenCounts"]
        )
        if (
            type(split_evidence.get("records")) is not int
            or not isinstance(observed_categories, Mapping)
            or set(observed_categories) != set(_FLEET_SOURCE_ROLES)
            or any(type(item) is not int for item in observed_categories.values())
            or not isinstance(observed_families, Mapping)
            or any(
                not isinstance(key, str) or type(item) is not int
                for key, item in observed_families.items()
            )
            or any(
                type(split_evidence.get(field_name)) is not int
                for field_name in (
                    fields["denominatorTokenCount"],
                    fields["supplementalNumeratorTokenCount"],
                    fields["publicNumeratorTokenCount"],
                )
            )
            or split_evidence.get("sourceRowsSHA256")
            != canonical_sha256(row_hashes)
            or observed_categories != target_by_category
            or split_evidence.get(fields["denominatorTokenCount"])
            != denominator
            or split_evidence.get(fields["supplementalNumeratorTokenCount"])
            != supplemental
            or split_evidence.get(fields["publicNumeratorTokenCount"]) != public
            or split_evidence.get("optimizerFamilyBandEnforcementStatus")
            != expected_enforcement_status
            or type(
                split_evidence.get(
                    selected_family_band["numeratorEvidenceField"]
                )
            )
            is not int
            or type(
                split_evidence.get(
                    selected_family_band["denominatorEvidenceField"]
                )
            )
            is not int
            or split_evidence.get(
                selected_family_band["numeratorEvidenceField"]
            )
            != family_numerator
            or split_evidence.get(
                selected_family_band["denominatorEvidenceField"]
            )
            != family_denominator
            or observed_families != dict(sorted(supplemental_by_family.items()))
        ):
            raise RuntimeError(
                f"Fleet {lane} {split} loss-share totals failed reconstruction"
            )
        caps = contract["capsBasisPoints"]
        checks = (
            (supplemental, caps["supplementalStaticTotal"]["requested"]),
            (supplemental, caps["supplementalStaticTotal"]["hard"]),
            (public, caps["publicBehavioralTotal"]["requested"]),
            (public, caps["publicBehavioralTotal"]["hard"]),
        )
        if split == "train" and any(
            not _pipeline_fleet_cap_passes(numerator, denominator, cap)
            for numerator, cap in checks
        ):
            raise RuntimeError(f"Fleet {lane} {split} total token cap failed")
        family_cap = caps["eachSupplementalSourceFamily"]["hard"]
        if split == "train" and any(
            not _pipeline_fleet_cap_passes(numerator, denominator, family_cap)
            for numerator in supplemental_by_family.values()
        ):
            raise RuntimeError(
                f"Fleet {lane} {split} per-source-family token cap failed"
            )
        if split == "train" and not (
            _pipeline_fleet_optimizer_family_band_passes(
                family_numerator,
                family_denominator,
                selected_family_band["minimumBasisPoints"],
                selected_family_band["maximumBasisPoints"],
            )
        ):
            raise RuntimeError(
                f"Fleet {lane} {split} optimizer-family share band failed"
            )
    return dict(evidence)


def _verify_training_token_length_preflight(
    *,
    run_root: Path,
    agent: str,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    config_path = run_root / "configs" / f"{agent}.json"
    if phase == "sft":
        from tools.fine_tuning.unsloth.train_sft import (
            SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            SFT_TOKEN_LENGTH_PREFLIGHT_SCHEMA,
            _validate_sft_checkpoint_lineage_static,
        )

        path_field = "sftTokenLengthPreflightPath"
        expected_path = run_root / "training" / agent / "sft_token_length_preflight.json"
        schema = SFT_TOKEN_LENGTH_PREFLIGHT_SCHEMA
        checkpoint_record = _validate_sft_checkpoint_lineage_static(
            config,
            cfg_path=config_path,
        )
        required_sequence_margin = max(
            SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            int(config.get("sft_minimum_sequence_margin_tokens", 0)),
        )
        required_prompt_margin = None
    elif phase == "preference":
        from tools.fine_tuning.unsloth.train_dpo import (
            PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS,
            PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            PREFERENCE_TOKEN_LENGTH_PREFLIGHT_SCHEMA,
            _validate_preference_checkpoint_lineage_static,
        )

        path_field = "preferenceTokenLengthPreflightPath"
        expected_path = run_root / "training" / agent / "dpo" / "token_length_preflight.json"
        schema = PREFERENCE_TOKEN_LENGTH_PREFLIGHT_SCHEMA
        checkpoint_record = _validate_preference_checkpoint_lineage_static(
            config,
            cfg_path=config_path,
        )
        required_prompt_margin = max(
            PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS,
            int(config.get("preference_minimum_prompt_margin_tokens", 0)),
        )
        required_sequence_margin = max(
            PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS,
            int(config.get("preference_minimum_sequence_margin_tokens", 0)),
        )
    else:
        raise RuntimeError(f"Unsupported token-length preflight phase: {phase}")
    declared_path = config.get(path_field)
    if (
        declared_path != str(expected_path)
        or expected_path.is_symlink()
        or not expected_path.is_file()
    ):
        raise RuntimeError(f"Missing controlled {phase} token-length preflight evidence")
    evidence = read_object(expected_path)
    digest = evidence.get("preflightSHA256")
    unsigned = dict(evidence)
    unsigned.pop("preflightSHA256", None)
    dataset_hashes = evidence.get("datasetFileSHA256")
    dataset_dir = Path(str(config.get("dataset_dir") or "")).resolve()
    actual_dataset_hashes = (
        {
            filename: file_sha256(dataset_dir / filename)
            for filename in sorted(dataset_hashes)
        }
        if isinstance(dataset_hashes, Mapping)
        else None
    )
    observed_sequence_margin = evidence.get("smallestSequenceMarginTokens")
    declared_sequence_margin = evidence.get("minimumSequenceMarginTokens")
    common_valid = (
        evidence.get("schemaVersion") == schema
        and re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is not None
        and canonical_sha256(unsigned) == digest
        and checkpoint_record.get("tokenLengthPreflightSHA256") == digest
        and report.get("token_length_preflight") == evidence
        and report.get("token_length_preflight_path") == str(expected_path)
        and report.get("token_length_preflight_sha256") == digest
        and evidence.get("agent") == agent
        and evidence.get("variant") == config.get("variant")
        and evidence.get("configPath") == str(config_path)
        and evidence.get("configSHA256") == file_sha256(config_path)
        and dataset_hashes == checkpoint_record.get("datasetFileSHA256")
        and actual_dataset_hashes == dataset_hashes
        and evidence.get("trainingCodeSHA256")
        == checkpoint_record.get("trainingCodeSHA256")
        and evidence.get("baseModelID") == config.get("base_model_name")
        and evidence.get("baseModelRevision") == config.get("baseModelRevision")
        and evidence.get("baseModelTokenizerDigest")
        == config.get("baseModelTokenizerDigest")
        and evidence.get("baseModelTokenizerClosureSHA256")
        == config.get("baseModelTokenizerClosureSHA256")
        and evidence.get("chatTemplateContract") == config.get("chatTemplateContract")
        and evidence.get("truncationRequired") is False
        and type(observed_sequence_margin) is int
        and observed_sequence_margin >= required_sequence_margin
        and type(declared_sequence_margin) is int
        and declared_sequence_margin >= required_sequence_margin
        and _valid_token_length_statistics(
            evidence.get("totalTokens")
            if phase == "sft"
            else evidence.get("maximumTotalTokens"),
            require_positive=True,
        )
    )
    if not common_valid:
        raise RuntimeError(f"{phase} token-length preflight evidence failed verification")
    if phase == "sft":
        if (
            not _valid_token_length_statistics(
                evidence.get("assistantTargetTokens"),
                require_positive=True,
            )
            or evidence["totalTokens"]["max"]
            + evidence["smallestSequenceMarginTokens"]
            != evidence.get("maxSequenceLength")
        ):
            raise RuntimeError("SFT token-length preflight statistics are inconsistent")
    else:
        observed_prompt_margin = evidence.get("smallestPromptMarginTokens")
        declared_prompt_margin = evidence.get("minimumPromptMarginTokens")
        if (
            type(observed_prompt_margin) is not int
            or observed_prompt_margin < required_prompt_margin
            or type(declared_prompt_margin) is not int
            or declared_prompt_margin < required_prompt_margin
            or not _valid_token_length_statistics(
                evidence.get("promptTokens"),
                require_positive=True,
            )
            or evidence["promptTokens"]["max"]
            + observed_prompt_margin
            != evidence.get("maxPromptLength")
            or evidence["maximumTotalTokens"]["max"]
            + evidence["smallestSequenceMarginTokens"]
            != evidence.get("maxSequenceLength")
        ):
            raise RuntimeError(
                "Preference token-length preflight statistics are inconsistent"
            )
    if agent != "fleet" and "fleetLossShareEvidence" in evidence:
        raise RuntimeError("Non-Fleet token preflight contains Fleet evidence")
    _verify_fleet_loss_share_evidence(
        value=evidence.get("fleetLossShareEvidence"),
        config=config,
        phase=phase,
        dataset_dir=dataset_dir,
    )
    _verify_public_corpus_loss_share_evidence(
        value=evidence.get("publicCorpusLossShareEvidence"),
        config=config,
        phase=phase,
        dataset_dir=dataset_dir,
    )
    _verified_global_tokenizer_preflight(
        run_root=run_root,
        agent=agent,
        config=config,
        phase=phase,
        bound_preflight=evidence,
    )
    return evidence


def _verify_peft_base_model_identity(
    adapter_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    path = adapter_dir / "adapter_config.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Adapter lacks a regular PEFT configuration")
    value = read_object(path)
    revision = value.get("revision")
    if (
        value.get("base_model_name_or_path") != config.get("baseModelID")
        or revision != config.get("baseModelRevision")
        or re.fullmatch(r"[0-9a-f]{40}", str(revision or "")) is None
    ):
        raise RuntimeError("Adapter PEFT base-model identity or revision drifted")
    return value


def verify_sft(run_root: Path, agent: str) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.adapter_artifact import verify_adapter_artifact

    config = read_object(run_root / "configs" / f"{agent}.json")
    precision = _reconstructed_training_precision(config)
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
    _verify_peft_base_model_identity(adapter_dir, config)
    finalized_path = run_root / "training" / agent / "finalized_variant_manifest.json"
    report_path = run_root / "training" / agent / "training_report.json"
    finalized = _verify_manifest_integrity(finalized_path)
    artifact, attestation = _verify_finalized_variant_binding(
        config,
        finalized,
        agent=agent,
        phase="SFT",
        expected_training_code_sha256=config.get("trainingCodeSHA256"),
    )
    if (
        artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft"
        or artifact.get("parentSFTAdapterSHA256") is not None
        or artifact.get("referenceSFTAdapterSHA256") is not None
        or artifact.get("preferenceTrainer") is not None
    ):
        raise RuntimeError(f"SFT identity or lineage mismatch: {finalized_path}")
    if finalized.get("trainingEnvironmentSHA256") != attestation.get(
        "trainingEnvironmentSHA256"
    ):
        raise RuntimeError("SFT training environment drifted from its attestation")
    adapter_manifest = verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=str(artifact.get("adapterSHA256") or ""),
        expected_training_phase="sft",
        expected_base_model=str(config.get("baseModelID") or ""),
        expected_base_revision=str(config.get("baseModelRevision") or ""),
    )
    training_report = _verify_training_report(
        report_path,
        phase="SFT",
        configured_num_train_epochs=float(config.get("num_train_epochs")),
        per_device_train_batch_size=int(config.get("batch_size")),
        configured_gradient_accumulation_steps=int(
            config.get("gradient_accumulation_steps")
        ),
        expected_precision=precision,
        expected={
            "schema": "lumen.train_sft.manifest/1.2.0",
            "agent": agent,
            "baseModelID": config.get("baseModelID"),
            "trainingPhase": "sft",
            "seed": config.get("seed"),
            "config_sha256": file_sha256(
                run_root / "configs" / f"{agent}.json"
            ),
            "output_dir": str(run_root / "training" / agent),
            "adapter_output_dir": str(adapter_dir),
            "adapterSHA256": adapter_manifest["adapterSHA256"],
            "finalizedVariantManifestSHA256": finalized[
                "variantManifestSHA256"
            ],
            "trainingEnvironmentSHA256": finalized[
                "trainingEnvironmentSHA256"
            ],
            "precision": precision,
        },
    )
    phase_runtime_evidence = _verify_phase_runtime_evidence(
        config=config,
        report=training_report,
        adapter_dir=adapter_dir,
    )
    token_length_preflight = _verify_training_token_length_preflight(
        run_root=run_root,
        agent=agent,
        config=config,
        report=training_report,
        phase="sft",
    )
    return {
        "phase": "sft",
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "report": str(report_path),
        "trainingReportFileSHA256": file_sha256(report_path),
        **phase_runtime_evidence,
        "trainingCompletion": training_report["trainingCompletion"],
        "precision": precision,
        "tokenLengthPreflight": str(
            run_root / "training" / agent / "sft_token_length_preflight.json"
        ),
        "tokenLengthPreflightSHA256": token_length_preflight["preflightSHA256"],
        "fleetLossShareEvidence": token_length_preflight.get(
            "fleetLossShareEvidence"
        ),
        "publicCorpusLossShareEvidence": token_length_preflight[
            "publicCorpusLossShareEvidence"
        ],
        "tokenLengthStatistics": {
            field: token_length_preflight[field]
            for field in (
                "totalTokens",
                "assistantTargetTokens",
                "smallestSequenceMarginTokens",
            )
        },
    }


def verify_preference(run_root: Path, agent: str) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.adapter_artifact import verify_adapter_artifact
    from tools.fine_tuning.unsloth.train_dpo import (
        _validate_preference_training_config,
    )

    config = read_object(run_root / "configs" / f"{agent}.json")
    preference_config = _validate_preference_training_config(config)
    precision = _reconstructed_training_precision(config)
    sft = verify_sft(run_root, agent)
    adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
    _verify_peft_base_model_identity(adapter_dir, config)
    finalized_path = run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    report_path = run_root / "training" / agent / "dpo" / "dpo_report.json"
    finalized = _verify_manifest_integrity(finalized_path)
    trainer = preference_config["preferenceTrainer"]
    phase_digests = config.get("trainingCodeSHA256ByPhase")
    if not isinstance(phase_digests, Mapping):
        raise RuntimeError(f"Preference manifest lacks lineage: {finalized_path}")
    artifact, _ = _verify_finalized_variant_binding(
        config,
        finalized,
        agent=agent,
        phase=trainer.upper(),
        expected_training_code_sha256=phase_digests.get(trainer),
    )
    expected_reference = sft["adapterSHA256"] if trainer == "dpo" else None
    if (
        artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft_dpo"
        or artifact.get("parentSFTAdapterSHA256") != sft["adapterSHA256"]
        or artifact.get("referenceSFTAdapterSHA256") != expected_reference
        or artifact.get("preferenceTrainer") != trainer
    ):
        raise RuntimeError(f"Preference identity or parent lineage mismatch: {finalized_path}")
    adapter_manifest = verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=str(artifact.get("adapterSHA256") or ""),
        expected_training_phase="sft_dpo",
        expected_parent_sft_adapter_sha256=sft["adapterSHA256"],
        expected_base_model=str(config.get("baseModelID") or ""),
        expected_base_revision=str(config.get("baseModelRevision") or ""),
    )
    training_report = _verify_training_report(
        report_path,
        phase=trainer.upper(),
        configured_num_train_epochs=preference_config["numTrainEpochs"],
        per_device_train_batch_size=int(config.get("batch_size")),
        configured_gradient_accumulation_steps=int(
            config.get("gradient_accumulation_steps")
        ),
        expected_precision=precision,
        expected={
            "schema": "lumen.train_preference.report/1.0.0",
            "agent": agent,
            "trainer": "ORPOTrainer" if trainer == "orpo" else "DPOTrainer",
            "training_phase": "sft_dpo",
            "seed": config.get("seed"),
            "variantManifestSHA256": config.get("variantManifestSHA256"),
            "config_sha256": file_sha256(
                run_root / "configs" / f"{agent}.json"
            ),
            "preferenceTrainingConfig": preference_config,
            "output_dir": str(run_root / "training" / agent / "dpo"),
            "adapter_output_dir": str(adapter_dir),
            "adapterSHA256": adapter_manifest["adapterSHA256"],
            "parent_sft_adapter_sha256": sft["adapterSHA256"],
            "reference_sft_adapter_sha256": expected_reference,
            "finalized_variant_manifest_sha256": finalized[
                "variantManifestSHA256"
            ],
            "trainingEnvironmentSHA256": finalized[
                "trainingEnvironmentSHA256"
            ],
            "precision": precision,
        },
    )
    phase_runtime_evidence = _verify_phase_runtime_evidence(
        config=config,
        report=training_report,
        adapter_dir=adapter_dir,
    )
    reference_log_prob_evidence = None
    if trainer == "dpo":
        reference_log_prob_evidence = _verify_dpo_reference_log_prob_report(
            run_root=run_root,
            agent=agent,
            config=config,
            report=training_report,
            parent_sft_adapter_sha256=sft["adapterSHA256"],
        )
    elif any(
        training_report.get(field) is not None
        for field in (
            "reference_log_probs_precomputed",
            "reference_log_prob_evidence",
            "checkpoint_adapter_contract",
        )
    ):
        raise RuntimeError("ORPO training report contains invalid DPO reference evidence")
    token_length_preflight = _verify_training_token_length_preflight(
        run_root=run_root,
        agent=agent,
        config=config,
        report=training_report,
        phase="preference",
    )
    return {
        "phase": trainer,
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "parentSFTAdapterSHA256": sft["adapterSHA256"],
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "report": str(report_path),
        "trainingReportFileSHA256": file_sha256(report_path),
        **phase_runtime_evidence,
        "trainingCompletion": training_report["trainingCompletion"],
        "precision": precision,
        "referenceLogProbEvidence": reference_log_prob_evidence,
        "tokenLengthPreflight": str(
            run_root / "training" / agent / "dpo" / "token_length_preflight.json"
        ),
        "tokenLengthPreflightSHA256": token_length_preflight["preflightSHA256"],
        "fleetLossShareEvidence": token_length_preflight.get(
            "fleetLossShareEvidence"
        ),
        "publicCorpusLossShareEvidence": token_length_preflight[
            "publicCorpusLossShareEvidence"
        ],
        "tokenLengthStatistics": {
            field: token_length_preflight[field]
            for field in (
                "promptTokens",
                "maximumTotalTokens",
                "smallestPromptMarginTokens",
                "smallestSequenceMarginTokens",
            )
        },
    }


def _final_evaluation_config_payload(
    run_root: Path,
    agent: str,
    *,
    base_config: Mapping[str, Any],
    finalized: Mapping[str, Any],
    preference: Mapping[str, Any],
    behavior_file_sha: str,
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.train_dpo import (
        _validate_preference_training_config,
    )

    config = dict(base_config)
    finalized_path = (
        run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    )
    trainer = _validate_preference_training_config(config)[
        "preferenceTrainer"
    ]
    phase_manifests = config.get("trainingCodeManifestsByPhase")
    phase_digests = config.get("trainingCodeSHA256ByPhase")
    if not isinstance(phase_manifests, Mapping) or not isinstance(
        phase_digests, Mapping
    ):
        raise RuntimeError("Prepared config lacks phase-specific training code lineage")
    attestation = config.get("variantAttestation")
    if not isinstance(attestation, Mapping):
        raise RuntimeError("Prepared config lacks variant attestation")
    final_attestation = dict(attestation)
    final_attestation["trainingEnvironmentSHA256"] = finalized[
        "trainingEnvironmentSHA256"
    ]
    final_attestation["trainingCodeSHA256"] = phase_digests[trainer]
    for field in RUNTIME_SOURCE_FIELDS:
        final_attestation[field] = finalized[field]
    config.update(
        {
            "adapter_output_dir": str(
                run_root / "models" / "lora_qwen3_dpo" / agent
            ),
            "output_dir": str(run_root / "training" / agent / "dpo"),
            "finalized_variant_manifest": str(finalized_path),
            "adapter_training_phase": "sft_dpo",
            "parent_sft_adapter_sha256": preference["parentSFTAdapterSHA256"],
            "preferenceTokenLengthPreflightSHA256": preference[
                "tokenLengthPreflightSHA256"
            ],
            "preferenceTokenLengthStatistics": preference[
                "tokenLengthStatistics"
            ],
            "trainingEnvironmentSHA256": finalized[
                "trainingEnvironmentSHA256"
            ],
            "trainingCodeManifest": phase_manifests[trainer],
            "trainingCodeSHA256": phase_digests[trainer],
            "resolvedTrainingEnvironment": finalized[
                "resolvedTrainingEnvironment"
            ],
            "resolvedTrainingEnvironmentSHA256": finalized[
                "resolvedTrainingEnvironmentSHA256"
            ],
            "observedAccelerator": finalized.get("observedAccelerator"),
            "zeroGPUSize": finalized.get("zeroGPUSize"),
            "zeroGPUDurationSeconds": finalized.get("zeroGPUDurationSeconds"),
            "variantAttestation": final_attestation,
            "behaviorManifestFileSHA256": behavior_file_sha,
        }
    )
    original_export = config.get("adapterExport")
    if not isinstance(original_export, Mapping):
        raise RuntimeError("Prepared config lacks adapter export lineage")
    export = dict(original_export)
    export.update(
        {
            "adapterArtifact": str(
                run_root / "models" / "lora_qwen3_dpo" / agent
            ),
            "adapterDirectory": str(
                run_root / "models" / "lora_qwen3_dpo" / agent
            ),
            "adapterGGUFArtifact": str(
                run_root
                / "models"
                / "lora_qwen3_gguf"
                / f"lumen-{agent}-lora.gguf"
            ),
            "trainBaseModelWeights": False,
            "mergeAdaptersByDefault": False,
        }
    )
    config["adapterExport"] = export
    for field in RUNTIME_SOURCE_FIELDS:
        config[field] = finalized[field]
    return config


def write_final_config(run_root: Path, agent: str) -> dict[str, Any]:
    preference = verify_preference(run_root, agent)
    run_manifest = read_object(run_root / "aio_run_manifest.json")
    declared_run_sha = run_manifest.get("runManifestSHA256")
    unsigned_run = dict(run_manifest)
    unsigned_run.pop("runManifestSHA256", None)
    if canonical_sha256(unsigned_run) != declared_run_sha:
        raise RuntimeError("Prepared run manifest integrity check failed")
    behavior_manifest_path = (
        run_root
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    behavior_file_sha = file_sha256(behavior_manifest_path)
    if behavior_file_sha != run_manifest.get("behaviorManifestFileSHA256"):
        raise RuntimeError("Frozen behavior manifest drifted from the prepared run")
    base_config = read_object(run_root / "configs" / f"{agent}.json")
    finalized_path = (
        run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    )
    finalized = _verify_manifest_integrity(finalized_path)
    artifact = finalized.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("Preference finalized manifest lacks adapter lineage")
    config = _final_evaluation_config_payload(
        run_root,
        agent,
        base_config=base_config,
        finalized=finalized,
        preference=preference,
        behavior_file_sha=behavior_file_sha,
    )
    final_path = run_root / "configs" / f"{agent}.final.json"
    write_object(final_path, config)
    return {
        "agent": agent,
        "config": str(final_path),
        "adapterSHA256": preference["adapterSHA256"],
        "trainingEnvironmentSHA256": finalized["trainingEnvironmentSHA256"],
    }


def clean_phase(run_root: Path, agent: str, phase: str) -> None:
    if phase not in {"sft", "preference"}:
        raise RuntimeError(f"Unsupported cleanup phase: {phase}")
    manifest = _verified_run_manifest(run_root)
    manifest_agents = manifest.get("agents")
    if (
        not isinstance(manifest_agents, list)
        or agent not in {
            item.get("agent")
            for item in manifest_agents
            if isinstance(item, Mapping)
        }
    ):
        raise RuntimeError(f"Agent is not owned by this prepared run: {agent}")
    _reject_managed_symlinks(run_root)
    _, gguf_receipt = _gguf_owned_paths(run_root, agent)
    targets: list[Path] = [
        run_root / "models" / "lora_qwen3_gguf" / _gguf_artifact_name(agent),
        gguf_receipt,
        run_root / "evaluation" / agent,
    ]
    if phase == "sft":
        targets.extend(
            [
                run_root / "training" / agent,
                run_root / "models" / "lora_qwen3_bootstrap" / agent,
                run_root / "models" / "lora_qwen3_dpo" / agent,
            ]
        )
    else:
        targets.extend(
            [
                run_root / "training" / agent / "dpo",
                run_root / "models" / "lora_qwen3_dpo" / agent,
            ]
        )
    for target in targets:
        if run_root not in target.parents or target.is_symlink():
            raise RuntimeError(f"Refusing unsafe phase cleanup target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    config_path = run_root / "configs" / f"{agent}.json"
    config = read_object(config_path)
    if phase == "sft":
        from tools.fine_tuning.unsloth.train_sft import (
            _reset_sft_checkpoint_lineage,
        )

        _reset_sft_checkpoint_lineage(config, cfg_path=config_path)
    from tools.fine_tuning.unsloth.train_dpo import (
        _reset_preference_checkpoint_lineage,
    )

    _reset_preference_checkpoint_lineage(config, cfg_path=config_path)


def _require_declared_run_file(
    value: Any,
    *,
    run_root: Path,
    expected_path: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must declare an absolute path")
    declared = Path(value)
    root_resolved = run_root.resolve()
    expected_resolved = expected_path.resolve()
    if (
        not declared.is_absolute()
        or expected_path.is_symlink()
        or not expected_path.is_file()
        or root_resolved not in expected_resolved.parents
        or declared != expected_resolved
    ):
        raise RuntimeError(f"{label} is not the expected regular file inside the run root")
    return expected_path


def _require_declared_run_directory(
    value: Any,
    *,
    run_root: Path,
    expected_path: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must declare an absolute path")
    declared = Path(value)
    root_resolved = run_root.resolve()
    expected_resolved = expected_path.resolve()
    if (
        not declared.is_absolute()
        or expected_path.is_symlink()
        or not expected_path.is_dir()
        or root_resolved not in expected_resolved.parents
        or declared != expected_resolved
    ):
        raise RuntimeError(
            f"{label} is not the expected regular directory inside the run root"
        )
    return expected_path


def _verify_evaluation_outputs(
    run_root: Path,
    agent: str,
    *,
    final_phase: Mapping[str, Any],
    require_passing_status: bool = True,
) -> dict[str, Any]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError(f"Evaluation run root is not a regular directory: {run_root}")
    _reject_managed_symlinks(run_root)
    evaluation_dir = run_root / "evaluation" / agent
    expected_names = {
        "candidate_outputs.jsonl",
        "evaluation_report.json",
        "evaluation_run_manifest.json",
    }
    if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
        raise RuntimeError(f"Missing regular evaluation directory: {evaluation_dir}")
    entries = list(evaluation_dir.iterdir())
    if (
        {entry.name for entry in entries} != expected_names
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise RuntimeError(
            f"Evaluation directory must contain exactly the verified evidence trio: {evaluation_dir}"
        )
    candidate_path = evaluation_dir / "candidate_outputs.jsonl"
    report_path = evaluation_dir / "evaluation_report.json"
    run_path = evaluation_dir / "evaluation_run_manifest.json"
    if candidate_path.stat().st_size == 0:
        raise RuntimeError(f"Candidate output evidence is empty: {candidate_path}")
    evaluation_run = read_object(run_path)
    run_digest = evaluation_run.get("runManifestSHA256")
    unsigned_run = dict(evaluation_run)
    unsigned_run.pop("runManifestSHA256", None)
    report = read_object(report_path)
    report_digest = report.get("reportSHA256")
    from tools.fine_tuning.unsloth import evaluate_adapter

    expected_run_keys = {
        "schemaVersion",
        "status",
        "evaluatorCodePath",
        "evaluatorCodeSHA256",
        "agent",
        "variant",
        "configPath",
        "configSHA256",
        "chatTemplateContract",
        "baseModelTokenizerDigest",
        "baseModelTokenizerFiles",
        "baseModelTokenizerClosureSHA256",
        "baseModelGenerationConfigFile",
        "baseModelTokenizerSnapshotPath",
        "baseModelTokenizerSnapshotVerification",
        "baseModelRuntimeSnapshotPath",
        "baseModelRuntimeSnapshotVerification",
        "runtimeModelBinding",
        "runtimeTokenizerBinding",
        "adapterDirectory",
        "adapterSHA256",
        "finalizedVariantManifestPath",
        "finalizedVariantManifestSHA256",
        "evaluationJSONLPath",
        "evaluationSHA256",
        "behaviorManifestPath",
        "behaviorManifestSHA256",
        "candidateOutputsPath",
        "candidateOutputsFileSHA256",
        "candidateOutputsSHA256",
        "evaluationReportPath",
        "evaluationReportFileSHA256",
        "evaluationReportSHA256",
        "fullCaseCount",
        "generatedCaseCount",
        "completeEvaluation",
        "executionPlanSHA256",
        "evaluationScope",
        "evaluationMaxExamples",
        "initialFormatFailureCount",
        "formatRecoveryCount",
        "formatFailureCount",
        "criticalFailureCount",
        "qualityGatePassed",
        "generation",
        "runManifestSHA256",
    }
    source_evidence_keys = set(UBUNTU_SOURCE_INTEGRITY_FIELDS)
    has_source_evidence = set(evaluation_run) == (
        expected_run_keys | source_evidence_keys
    )
    if frozenset(evaluation_run) not in {
        frozenset(expected_run_keys),
        frozenset(expected_run_keys | source_evidence_keys),
    }:
        raise RuntimeError(
            f"Evaluation run manifest has an unexpected evidence schema: {run_path}"
        )

    evaluation_path = run_root / "generated" / "fine_tuning" / agent / "eval.jsonl"
    config_path = run_root / "configs" / f"{agent}.final.json"
    adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
    finalized_path = (
        run_root
        / "training"
        / agent
        / "dpo"
        / "finalized_variant_manifest.json"
    )
    behavior_path = (
        run_root
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    _require_declared_run_file(
        evaluation_run.get("configPath"),
        run_root=run_root,
        expected_path=config_path,
        label="Final evaluation config path",
    )
    _require_declared_run_directory(
        evaluation_run.get("adapterDirectory"),
        run_root=run_root,
        expected_path=adapter_dir,
        label="Final adapter directory",
    )
    _require_declared_run_file(
        evaluation_run.get("finalizedVariantManifestPath"),
        run_root=run_root,
        expected_path=finalized_path,
        label="Finalized variant manifest path",
    )
    _require_declared_run_file(
        evaluation_run.get("evaluationJSONLPath"),
        run_root=run_root,
        expected_path=evaluation_path,
        label="Frozen evaluation path",
    )
    _require_declared_run_file(
        evaluation_run.get("candidateOutputsPath"),
        run_root=run_root,
        expected_path=candidate_path,
        label="Candidate outputs path",
    )
    _require_declared_run_file(
        evaluation_run.get("evaluationReportPath"),
        run_root=run_root,
        expected_path=report_path,
        label="Evaluation report path",
    )
    _require_declared_run_file(
        evaluation_run.get("behaviorManifestPath"),
        run_root=run_root,
        expected_path=behavior_path,
        label="Frozen behavior manifest path",
    )
    evaluation_module = evaluate_adapter._load_evaluation_module()
    try:
        config = evaluate_adapter.load_evaluation_config(config_path)
        evaluate_adapter.verify_chat_template_contract(
            config.get("chatTemplateContract")
        )
        base_config_path = run_root / "configs" / f"{agent}.json"
        if base_config_path.is_symlink() or not base_config_path.is_file():
            raise ValueError("Prepared base config is not a regular file")
        base_config = read_object(base_config_path)
        source_evidence_required = config.get("runtimeSourceBindingMethod") == (
            "git_clean_worktree_plus_ubuntu_orchestration_manifest"
        )
        if has_source_evidence is not source_evidence_required:
            raise ValueError("Evaluation source-integrity evidence is incomplete")
        if source_evidence_required:
            verify_embedded_source_integrity(config)
            if any(
                evaluation_run.get(field) != config.get(field)
                for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
            ):
                raise ValueError("Evaluation source-integrity evidence drifted")
        evaluation_records, evaluation_sha256 = evaluate_adapter.load_evaluation_records(
            evaluation_path,
            agent=agent,
            evaluation_module=evaluation_module,
        )
        tool_contracts, allowed_slots, behavior_sha256 = (
            evaluate_adapter.load_behavior_contract(behavior_path)
        )
        evaluate_adapter.validate_scoring_contracts(
            evaluation_records,
            tool_contracts=tool_contracts,
            allowed_slots=allowed_slots,
        )
        finalized = evaluate_adapter.load_finalized_manifest(
            finalized_path,
            cfg=config,
            evaluation_sha256=evaluation_sha256,
            evaluation_module=evaluation_module,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Frozen evaluation scoring lineage failed verification: {evaluation_path}"
        ) from exc

    tokenizer_snapshot_verification = _verified_private_tokenizer_snapshot_binding(
        config
    )
    runtime_snapshot_verification = (
        _verified_private_base_model_runtime_snapshot_binding(config)
    )
    expected_runtime_evidence = {
        "baseModelTokenizerDigest": config.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": config.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": config.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": config.get(
            "baseModelGenerationConfigFile"
        ),
        "baseModelTokenizerSnapshotPath": config.get(
            "baseModelTokenizerSnapshotPath"
        ),
        "baseModelTokenizerSnapshotVerification": (
            tokenizer_snapshot_verification
        ),
        "baseModelRuntimeSnapshotPath": config.get("baseModelRuntimeSnapshotPath"),
        "baseModelRuntimeSnapshotVerification": runtime_snapshot_verification,
    }
    runtime_evidence_drifted = [
        field
        for field, expected in expected_runtime_evidence.items()
        if evaluation_run.get(field) != expected
    ]
    if runtime_evidence_drifted:
        raise RuntimeError(
            "Evaluation private-runtime evidence drifted: "
            + ", ".join(runtime_evidence_drifted)
        )
    _verified_runtime_model_binding(
        evaluation_run.get("runtimeModelBinding"),
        config=config,
        snapshot_verification=runtime_snapshot_verification,
    )
    _verified_runtime_tokenizer_binding(
        evaluation_run.get("runtimeTokenizerBinding"),
        config=config,
        snapshot_verification=runtime_snapshot_verification,
    )

    prepared_run = _verified_run_manifest(run_root)
    prepared_execution_plan = _verified_execution_plan(
        prepared_run.get("executionPlan")
    )
    prepared_agents = prepared_run.get("agents")
    prepared_agent = next(
        (
            item
            for item in prepared_agents
            if isinstance(item, Mapping) and item.get("agent") == agent
        ),
        None,
    ) if isinstance(prepared_agents, list) else None
    if (
        not isinstance(prepared_agent, Mapping)
        or sum(
            1
            for item in prepared_agents
            if isinstance(item, Mapping) and item.get("agent") == agent
        )
        != 1
        or prepared_run.get("variant") != config.get("variant")
        or prepared_run.get("behaviorManifest") != str(behavior_path.resolve())
        or prepared_run.get("behaviorManifestFileSHA256")
        != file_sha256(behavior_path)
        or prepared_agent.get("config") != str(base_config_path.resolve())
        or prepared_agent.get("configSHA256") != file_sha256(base_config_path)
        or config.get("runExecutionPlan") != prepared_execution_plan
        or evaluation_run.get("executionPlanSHA256")
        != prepared_execution_plan["executionPlanSHA256"]
        or evaluation_run.get("evaluationScope")
        != prepared_execution_plan["evaluationScope"]
        or evaluation_run.get("evaluationMaxExamples")
        != prepared_execution_plan["evaluationMaxExamples"]
        or (
            source_evidence_required
            and any(
                evaluation_run.get(field) != prepared_run.get(field)
                for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
            )
        )
    ):
        raise RuntimeError(
            f"Evaluation inputs drifted from the exact prepared run: {run_root}"
        )

    try:
        expected_config = _final_evaluation_config_payload(
            run_root,
            agent,
            base_config=base_config,
            finalized=finalized,
            preference=final_phase,
            behavior_file_sha=file_sha256(behavior_path),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Final evaluation config could not be reconstructed: {config_path}"
        ) from exc

    config_max_sequence_length = config.get("max_seq_length")
    config_seed = config.get("seed")
    artifact = finalized.get("artifact")
    if (
        config != expected_config
        or config.get("agent") != agent
        or not isinstance(config.get("variant"), str)
        or not config["variant"]
        or config.get("adapter_training_phase") != "sft_dpo"
        or config.get("preference_trainer") != final_phase.get("phase")
        or Path(str(config.get("adapter_output_dir") or "")) != adapter_dir.resolve()
        or Path(str(config.get("output_dir") or ""))
        != (run_root / "training" / agent / "dpo").resolve()
        or Path(str(config.get("finalized_variant_manifest") or ""))
        != finalized_path.resolve()
        or config.get("behaviorManifestFileSHA256") != file_sha256(behavior_path)
        or type(config_max_sequence_length) is not int
        or config_max_sequence_length <= 0
        or type(config_seed) is not int
        or not isinstance(artifact, Mapping)
        or artifact.get("adapterSHA256") != final_phase.get("adapterSHA256")
        or finalized.get("variantManifestSHA256")
        != final_phase.get("finalizedVariantManifestSHA256")
    ):
        raise RuntimeError(
            f"Final evaluation config or adapter lineage failed verification: {config_path}"
        )
    generated_case_count = evaluation_run.get("generatedCaseCount")
    complete_evaluation = evaluation_run.get("completeEvaluation")
    evaluation_scope = prepared_execution_plan["evaluationScope"]
    planned_max_examples = prepared_execution_plan["evaluationMaxExamples"]
    expected_complete_evaluation = evaluation_scope == "full"
    expected_generated_case_count = (
        len(evaluation_records)
        if expected_complete_evaluation
        else planned_max_examples
    )
    if (
        evaluation_scope == "none"
        or type(generated_case_count) is not int
        or generated_case_count <= 0
        or generated_case_count > len(evaluation_records)
        or type(evaluation_run.get("fullCaseCount")) is not int
        or evaluation_run.get("fullCaseCount") != len(evaluation_records)
        or complete_evaluation is not expected_complete_evaluation
        or generated_case_count != expected_generated_case_count
        or (
            evaluation_scope == "smoke"
            and generated_case_count >= len(evaluation_records)
        )
    ):
        raise RuntimeError(f"Evaluation case-count lineage failed verification: {run_path}")
    if complete_evaluation:
        selected_records = list(evaluation_records)
    else:
        try:
            selected_records = evaluate_adapter.select_evaluation_records(
                evaluation_records,
                max_examples=planned_max_examples,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Evaluation smoke cohort failed reconstruction: {evaluation_path}"
            ) from exc
    try:
        candidate_outputs = evaluate_adapter.load_candidate_outputs(
            candidate_path,
            agent=agent,
            evaluation_records=selected_records,
            tool_contracts=tool_contracts,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Candidate output evidence failed verification: {candidate_path}") from exc
    candidate_rows = read_jsonl(candidate_path)
    initial_format_failure_count = sum(
        1
        for row in candidate_rows
        if row["generationAttempts"][0]["formatError"] is not None
    )
    format_recovery_count = sum(
        1
        for row in candidate_rows
        if row["generationAttempts"][0]["formatError"] is not None
        and row["generationAttempts"][-1]["formatError"] is None
    )
    format_failure_count = sum(
        1
        for row in candidate_rows
        if row["generationAttempts"][-1]["formatError"] is not None
    )
    candidate_outputs_sha256 = canonical_sha256(candidate_outputs)
    evaluator_path = Path(evaluate_adapter.__file__).resolve()
    generation = evaluation_run.get("generation")
    try:
        expected_output_mode_contract = (
            evaluate_adapter._evaluation_output_mode_contract(
                selected_records,
                agent=agent,
                tool_contracts=tool_contracts,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Evaluation output-mode contract failed reconstruction: {run_path}"
        ) from exc
    expected_generation_keys = {
        "doSample",
        "numBeams",
        "repetitionPenalty",
        "thinkingEnabled",
        "maxNewTokens",
        "maxSequenceLength",
        "seed",
        "outputModeContract",
    }
    generation_max_new_tokens = (
        generation.get("maxNewTokens") if isinstance(generation, Mapping) else None
    )
    generation_contract_valid = (
        isinstance(generation, Mapping)
        and set(generation) == expected_generation_keys
        and type(generation_max_new_tokens) is int
        and 1 <= generation_max_new_tokens <= 4096
        and type(generation.get("maxSequenceLength")) is int
        and generation.get("maxSequenceLength") == config_max_sequence_length
        and type(generation.get("seed")) is int
        and generation.get("seed") == config_seed
        and generation.get("outputModeContract") == expected_output_mode_contract
        and (
            generation.get("doSample") is False
            and type(generation.get("numBeams")) is int
            and generation.get("numBeams") == 1
            and type(generation.get("repetitionPenalty")) is float
            and generation.get("repetitionPenalty")
            == evaluate_adapter.GENERATION_REPETITION_PENALTY
            and generation.get("thinkingEnabled") is False
        )
    )
    attempt_budgets_valid = generation_contract_valid and bool(candidate_rows)
    for row in (candidate_rows if generation_contract_valid else ()):
        attempts = row.get("generationAttempts")
        if not isinstance(attempts, list) or not attempts:
            attempt_budgets_valid = False
            break
        for expected_index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, Mapping):
                attempt_budgets_valid = False
                break
            input_token_count = attempt.get("inputTokenCount")
            generation_token_budget = attempt.get("generationTokenBudget")
            if (
                type(attempt.get("attemptIndex")) is not int
                or attempt.get("attemptIndex") != expected_index
                or type(input_token_count) is not int
                or input_token_count <= 0
                or type(generation_token_budget) is not int
                or generation_token_budget
                != min(
                    generation_max_new_tokens,
                    config_max_sequence_length - input_token_count,
                )
                or generation_token_budget <= 0
            ):
                attempt_budgets_valid = False
                break
        if not attempt_budgets_valid:
            break

    controlled_lineage_builder = getattr(
        evaluation_module,
        "_variant_controlled_lineage",
        None,
    )
    if controlled_lineage_builder is None:
        raise RuntimeError("Evaluation module lacks controlled-lineage scoring support")
    try:
        recomputed_report = evaluation_module.score_evaluation_suite(
            selected_records,
            candidate_outputs,
            frozen_evaluation_records=evaluation_records,
            tool_contracts=tool_contracts,
            allowed_slots=allowed_slots,
            agent=agent,
            variant=config["variant"],
            controlled_lineage=controlled_lineage_builder(finalized),
            variant_manifest=finalized,
            artifact_sha256=final_phase["adapterSHA256"],
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Evaluation report could not be independently reconstructed: {report_path}"
        ) from exc

    expected_status, quality_gate_passed = evaluate_adapter._evaluation_outcome(
        complete_evaluation=complete_evaluation,
        format_failure_count=format_failure_count,
        report=recomputed_report,
    )
    if (
        canonical_sha256(unsigned_run) != run_digest
        or report != recomputed_report
        or report_digest != recomputed_report.get("reportSHA256")
        or evaluation_run.get("schemaVersion")
        != evaluate_adapter.EVALUATION_RUN_SCHEMA_VERSION
        or evaluation_run.get("agent") != agent
        or evaluation_run.get("variant") != config.get("variant")
        or evaluation_run.get("configSHA256") != file_sha256(config_path)
        or evaluation_run.get("chatTemplateContract")
        != config.get("chatTemplateContract")
        or evaluation_run.get("evaluatorCodePath") != str(evaluator_path)
        or evaluation_run.get("evaluatorCodeSHA256") != file_sha256(evaluator_path)
        or evaluation_run.get("adapterSHA256") != final_phase.get("adapterSHA256")
        or evaluation_run.get("finalizedVariantManifestSHA256")
        != final_phase.get("finalizedVariantManifestSHA256")
        or evaluation_run.get("candidateOutputsFileSHA256")
        != file_sha256(candidate_path)
        or evaluation_run.get("evaluationReportFileSHA256")
        != file_sha256(report_path)
        or evaluation_run.get("evaluationReportSHA256") != report_digest
        or evaluation_run.get("candidateOutputsSHA256")
        != report.get("candidateOutputsSHA256")
        or evaluation_run.get("candidateOutputsSHA256") != candidate_outputs_sha256
        or evaluation_run.get("evaluationSHA256") != evaluation_sha256
        or evaluation_run.get("behaviorManifestSHA256") != behavior_sha256
        or recomputed_report.get("variantLineageBound") is not True
        or recomputed_report.get("frozenCaseCount") != len(evaluation_records)
        or recomputed_report.get("caseCount") != generated_case_count
        or recomputed_report.get("completeEvaluation") is not complete_evaluation
        or recomputed_report.get("promotionEvidenceBound") is not complete_evaluation
        or type(evaluation_run.get("initialFormatFailureCount")) is not int
        or evaluation_run.get("initialFormatFailureCount")
        != initial_format_failure_count
        or type(evaluation_run.get("formatRecoveryCount")) is not int
        or evaluation_run.get("formatRecoveryCount") != format_recovery_count
        or type(evaluation_run.get("formatFailureCount")) is not int
        or evaluation_run.get("formatFailureCount") != format_failure_count
        or evaluation_run.get("criticalFailureCount")
        != recomputed_report.get("criticalFailureCount")
        or type(evaluation_run.get("criticalFailureCount")) is not int
        or evaluation_run.get("qualityGatePassed") is not quality_gate_passed
        or evaluation_run.get("status") != expected_status
        or not generation_contract_valid
        or not attempt_budgets_valid
    ):
        raise RuntimeError(f"Evaluation evidence lineage failed verification: {evaluation_dir}")
    evaluation_status = evaluation_run.get("status")
    passing_statuses = {"quality_gate_passed", "smoke_complete"}
    failed_statuses = {"format_failed", "quality_gate_failed", "smoke_failed"}
    if evaluation_status not in passing_statuses | failed_statuses:
        raise RuntimeError(f"Evaluation has an unsupported terminal status: {run_path}")
    if require_passing_status:
        if evaluation_status not in passing_statuses:
            raise RuntimeError(
                f"Evaluation did not pass or complete a smoke run: {run_path}"
            )
        if evaluation_status == "quality_gate_passed" and (
            complete_evaluation is not True
            or quality_gate_passed is not True
            or format_failure_count != 0
            or recomputed_report.get("promotionEvidenceBound") is not True
        ):
            raise RuntimeError(f"Full evaluation quality gate failed: {run_path}")
        if evaluation_status == "smoke_complete" and (
            complete_evaluation is not False
            or quality_gate_passed is not False
            or format_failure_count != 0
            or recomputed_report.get("promotionEvidenceBound") is not False
            or recomputed_report.get("criticalFailureCount") != 0
            or recomputed_report.get("passedCaseCount")
            != recomputed_report.get("caseCount")
        ):
            raise RuntimeError(f"Evaluation smoke status is inconsistent: {run_path}")
    return evaluation_run


def verify_evaluation(run_root: Path, agent: str) -> dict[str, Any]:
    """Replay and verify one agent's final evaluation before summary creation."""

    final_phase = verify_preference(run_root, agent)
    return _verify_evaluation_outputs(
        run_root,
        agent,
        final_phase=final_phase,
    )


def classify_completed_evaluation(run_root: Path, agent: str) -> dict[str, Any]:
    """Verify a terminal evidence trio without treating failed quality as invalid."""

    final_phase = verify_preference(run_root, agent)
    evaluation = _verify_evaluation_outputs(
        run_root,
        agent,
        final_phase=final_phase,
        require_passing_status=False,
    )
    status = str(evaluation["status"])
    state = (
        "completed_success"
        if status in {"quality_gate_passed", "smoke_complete"}
        else "completed_quality_failure"
    )
    return {
        "agent": agent,
        "state": state,
        "status": status,
        "qualityGatePassed": evaluation["qualityGatePassed"],
        "evaluationRunManifest": str(
            run_root / "evaluation" / agent / "evaluation_run_manifest.json"
        ),
        "evaluationRunManifestSHA256": evaluation["runManifestSHA256"],
    }


def _verified_runtime_binding_smoke_summary_evidence(
    run_root: Path,
    agents: Sequence[str],
) -> dict[str, Any]:
    from tools.fine_tuning.unsloth import runtime_binding_smoke_gate

    report = runtime_binding_smoke_gate.verify_existing_report(run_root, agents)
    report_path = run_root / "training" / runtime_binding_smoke_gate.REPORT_FILENAME
    report_digest = report.get(runtime_binding_smoke_gate.REPORT_HASH_FIELD)
    contracts = report.get("contracts")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(report_digest or "")) is None
        or not isinstance(contracts, list)
        or not contracts
    ):
        raise RuntimeError("Runtime-binding smoke evidence is incomplete")
    compact_contracts: list[dict[str, Any]] = []
    bindings_by_agent: dict[str, dict[str, str]] = {}
    for item in contracts:
        smoke = item.get("smoke") if isinstance(item, Mapping) else None
        model_binding = (
            smoke.get("runtimeModelBinding") if isinstance(smoke, Mapping) else None
        )
        tokenizer_binding = (
            smoke.get("runtimeTokenizerBinding") if isinstance(smoke, Mapping) else None
        )
        contract_agents = item.get("agents") if isinstance(item, Mapping) else None
        compact = {
            "runtimeLoadContractSHA256": (
                item.get("runtimeLoadContractSHA256")
                if isinstance(item, Mapping)
                else None
            ),
            "agents": contract_agents,
            "representativeAgent": (
                item.get("representativeAgent")
                if isinstance(item, Mapping)
                else None
            ),
            "runtimeBindingSmokeSHA256": (
                smoke.get(runtime_binding_smoke_gate.SMOKE_HASH_FIELD)
                if isinstance(smoke, Mapping)
                else None
            ),
            "runtimeModelBindingSHA256": (
                model_binding.get("runtimeModelBindingSHA256")
                if isinstance(model_binding, Mapping)
                else None
            ),
            "runtimeTokenizerBindingSHA256": (
                tokenizer_binding.get("runtimeTokenizerBindingSHA256")
                if isinstance(tokenizer_binding, Mapping)
                else None
            ),
        }
        digest_fields = (
            "runtimeLoadContractSHA256",
            "runtimeBindingSmokeSHA256",
            "runtimeModelBindingSHA256",
            "runtimeTokenizerBindingSHA256",
        )
        if (
            not isinstance(contract_agents, list)
            or not contract_agents
            or any(not isinstance(agent, str) for agent in contract_agents)
            or compact["representativeAgent"] not in contract_agents
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(compact[field] or "")) is None
                for field in digest_fields
            )
        ):
            raise RuntimeError("Runtime-binding smoke contract evidence is invalid")
        compact_contracts.append(compact)
        binding_evidence = {
            "runtimeModelBindingSHA256": str(
                compact["runtimeModelBindingSHA256"]
            ),
            "runtimeTokenizerBindingSHA256": str(
                compact["runtimeTokenizerBindingSHA256"]
            ),
        }
        for agent in contract_agents:
            if agent in bindings_by_agent:
                raise RuntimeError(
                    "Runtime-binding smoke assigns an agent to multiple contracts"
                )
            bindings_by_agent[agent] = binding_evidence
    if set(bindings_by_agent) != set(agents):
        raise RuntimeError("Runtime-binding smoke agent coverage drifted")
    return {
        "runtimeBindingSmokeReport": str(report_path),
        "runtimeBindingSmokeReportFileSHA256": file_sha256(report_path),
        "runtimeBindingSmokeGateSHA256": str(report_digest),
        "runtimeBindingSmokeContractEvidence": compact_contracts,
        "runtimeBindingSmokeBindingsByAgent": bindings_by_agent,
    }


def _require_runtime_bindings_match_smoke(
    *,
    agent: str,
    source: Mapping[str, Any],
    smoke_evidence: Mapping[str, Any],
    label: str,
    model_field: str = "runtimeModelBindingSHA256",
    tokenizer_field: str = "runtimeTokenizerBindingSHA256",
) -> None:
    bindings = smoke_evidence.get("runtimeBindingSmokeBindingsByAgent")
    expected = bindings.get(agent) if isinstance(bindings, Mapping) else None
    if (
        not isinstance(expected, Mapping)
        or source.get(model_field) != expected.get("runtimeModelBindingSHA256")
        or source.get(tokenizer_field)
        != expected.get("runtimeTokenizerBindingSHA256")
    ):
        raise RuntimeError(
            f"{label} runtime bindings drifted from the pre-training smoke gate: {agent}"
        )


def _derived_summary_state(
    *,
    plan: Mapping[str, Any],
    evaluation_statuses: Sequence[str],
    agent_count: int,
    gguf_count: int,
    preference_training: bool = True,
) -> dict[str, Any]:
    if agent_count <= 0:
        raise RuntimeError("Summary state requires at least one prepared agent")
    if type(preference_training) is not bool:
        raise RuntimeError("Summary preference-training state must be boolean")
    verified_plan = _verified_execution_plan(plan)
    evaluation_scope = str(verified_plan["evaluationScope"])
    if not preference_training:
        if (
            evaluation_scope != "none"
            or verified_plan["ggufRequested"] is not False
            or evaluation_statuses
            or gguf_count != 0
        ):
            raise RuntimeError(
                "SFT-only diagnostic summaries cannot include preference, "
                "evaluation, or GGUF evidence"
            )
        return {
            "status": "sft_only_diagnostic_complete",
            "trainingScope": "sft_only",
            "evaluationStatus": "not_run",
            "evaluationScope": "none",
            "ggufStatus": "not_applicable_sft_only",
            "ggufConversionStatus": "not_applicable",
            "ggufTensorEquivalenceStatus": "not_applicable",
            "qualification": "diagnostic_only",
            "promotionEligible": False,
        }
    if len(evaluation_statuses) not in {0, agent_count}:
        raise RuntimeError(
            "Summary contains partial evaluation evidence across prepared agents"
        )
    if evaluation_scope == "none":
        if evaluation_statuses:
            raise RuntimeError(
                "Summary contains evaluation evidence disabled by the execution plan"
            )
        evaluation_status = "not_run"
    elif len(evaluation_statuses) != agent_count:
        raise RuntimeError(
            "Summary is missing evaluation evidence required by the execution plan"
        )
    elif evaluation_scope == "full" and set(evaluation_statuses) == {
        "quality_gate_passed"
    }:
        evaluation_status = "quality_gate_passed"
    elif evaluation_scope == "smoke" and set(evaluation_statuses) == {
        "smoke_complete"
    }:
        evaluation_status = "smoke_complete"
    else:
        raise RuntimeError(
            "Summary evaluation evidence does not match the execution plan"
        )

    if verified_plan["ggufRequested"] is True and gguf_count == agent_count:
        gguf_status = "verified"
        gguf_conversion_status = GGUF_CONVERSION_QUALIFICATION
        gguf_tensor_equivalence_status = GGUF_TENSOR_EQUIVALENCE_STATUS
    elif verified_plan["ggufRequested"] is False and gguf_count == 0:
        gguf_status = "skipped_by_operator"
        gguf_conversion_status = "skipped_by_operator"
        gguf_tensor_equivalence_status = "not_applicable"
    else:
        raise RuntimeError(
            "Summary GGUF inventory does not match the execution plan"
        )

    promotion_eligible = evaluation_status == "quality_gate_passed"
    qualification = (
        "quality_gate_passed" if promotion_eligible else "diagnostic_only"
    )
    if evaluation_status == "quality_gate_passed":
        status = "complete" if gguf_status == "verified" else "complete_without_gguf"
    elif evaluation_status == "smoke_complete":
        status = "smoke_complete"
    else:
        status = "training_complete_without_full_evaluation"
    return {
        "status": status,
        "trainingScope": "sft_preference",
        "evaluationStatus": evaluation_status,
        "evaluationScope": evaluation_scope,
        "ggufStatus": gguf_status,
        "ggufConversionStatus": gguf_conversion_status,
        "ggufTensorEquivalenceStatus": gguf_tensor_equivalence_status,
        "qualification": qualification,
        "promotionEligible": promotion_eligible,
    }


def write_summary(
    *,
    run_root: Path,
    agents: Sequence[str],
    variant: str,
    preference: bool,
    require_gguf: bool,
    require_evaluation: bool,
) -> dict[str, Any]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError(f"Summary run root is not a regular directory: {run_root}")
    _reject_managed_symlinks(run_root)
    run_manifest = _verified_run_manifest(run_root)
    prepared_execution_plan = _verified_execution_plan(
        run_manifest.get("executionPlan")
    )
    manifest_agents = run_manifest.get("agents")
    if (
        run_manifest.get("variant") != variant
        or not isinstance(manifest_agents, list)
        or any(not isinstance(item, Mapping) for item in manifest_agents)
        or [item.get("agent") for item in manifest_agents] != list(agents)
    ):
        raise RuntimeError(
            "Summary agents or variant do not match the exact prepared run"
        )
    if require_gguf is not prepared_execution_plan["ggufRequested"]:
        raise RuntimeError("Summary GGUF request drifted from the execution plan")
    if require_evaluation is not (
        prepared_execution_plan["evaluationScope"] != "none"
    ):
        raise RuntimeError("Summary evaluation request drifted from the execution plan")
    if not preference and (
        prepared_execution_plan["evaluationScope"] != "none"
        or prepared_execution_plan["ggufRequested"] is not False
    ):
        raise RuntimeError(
            "SFT-only diagnostic summaries require evaluation and GGUF to be disabled"
        )
    gguf_inventory = (
        _verify_gguf_inventory(
            run_root,
            agents,
            require_all=require_gguf,
        )
        if preference
        else {}
    )
    runtime_binding_smoke_evidence = (
        _verified_runtime_binding_smoke_summary_evidence(run_root, agents)
    )
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA_VERSION,
        "status": "pending_verification",
        "trainingScope": "pending_verification",
        "evaluationStatus": "pending_verification",
        "evaluationScope": "pending_verification",
        "ggufStatus": "pending_verification",
        "ggufConversionStatus": "pending_verification",
        "ggufTensorEquivalenceStatus": "pending_verification",
        "qualification": "pending_verification",
        "promotionEligible": False,
        "executionPlanSHA256": prepared_execution_plan["executionPlanSHA256"],
        "variant": variant,
        "runRoot": str(run_root),
        "preferenceTraining": preference,
        "baseModelID": run_manifest["baseModelID"],
        "baseModelRevision": run_manifest["baseModelRevision"],
        "baseModelTokenizerDigest": run_manifest[
            "baseModelTokenizerDigest"
        ],
        "baseModelTokenizerFiles": run_manifest["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": run_manifest[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelTokenizerSnapshotPath": run_manifest[
            "baseModelTokenizerSnapshotPath"
        ],
        "baseModelTokenizerSnapshotVerification": run_manifest[
            "baseModelTokenizerSnapshotVerification"
        ],
        "baseModelGenerationConfigFile": run_manifest[
            "baseModelGenerationConfigFile"
        ],
        "baseModelRuntimeSnapshotPath": run_manifest[
            "baseModelRuntimeSnapshotPath"
        ],
        "baseModelRuntimeSnapshotVerification": run_manifest[
            "baseModelRuntimeSnapshotVerification"
        ],
        "runManifestSHA256": run_manifest["runManifestSHA256"],
        **runtime_binding_smoke_evidence,
        **{
            field: run_manifest[field]
            for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
        },
        "agents": {},
    }
    for agent in agents:
        sft = verify_sft(run_root, agent)
        _require_runtime_bindings_match_smoke(
            agent=agent,
            source=sft,
            smoke_evidence=runtime_binding_smoke_evidence,
            label="SFT phase",
        )
        final_phase = verify_preference(run_root, agent) if preference else sft
        if preference:
            _require_runtime_bindings_match_smoke(
                agent=agent,
                source=final_phase,
                smoke_evidence=runtime_binding_smoke_evidence,
                label="Preference phase",
            )
        if not preference:
            summary["agents"][agent] = {
                "sft": sft,
                "finalPhase": final_phase,
            }
            continue
        gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        gguf_metadata = (
            verify_gguf_file(run_root, gguf)
            if gguf.name in gguf_inventory
            else None
        )
        gguf_exists = gguf_metadata is not None
        if require_gguf and not gguf_exists:
            raise RuntimeError(f"Missing required GGUF adapter: {gguf}")
        if gguf_metadata is not None:
            _require_runtime_bindings_match_smoke(
                agent=agent,
                source=gguf_metadata,
                smoke_evidence=runtime_binding_smoke_evidence,
                label="GGUF conversion",
                model_field="adapterGGUFRuntimeModelBindingSHA256",
                tokenizer_field="adapterGGUFRuntimeTokenizerBindingSHA256",
            )
        evaluation_dir = run_root / "evaluation" / agent
        evaluation = evaluation_dir / "evaluation_report.json"
        evaluation_run = evaluation_dir / "evaluation_run_manifest.json"
        if require_evaluation and (
            not evaluation.is_file()
            or evaluation.stat().st_size == 0
            or not evaluation_run.is_file()
        ):
            raise RuntimeError(f"Missing required evaluation report: {evaluation}")
        evaluation_status: dict[str, Any] | None = None
        if evaluation_run.is_file():
            evaluation_status = _verify_evaluation_outputs(
                run_root,
                agent,
                final_phase=final_phase,
            )
            _require_runtime_bindings_match_smoke(
                agent=agent,
                source={
                    "runtimeModelBindingSHA256": (
                        evaluation_status.get("runtimeModelBinding", {}).get(
                            "runtimeModelBindingSHA256"
                        )
                        if isinstance(
                            evaluation_status.get("runtimeModelBinding"), Mapping
                        )
                        else None
                    ),
                    "runtimeTokenizerBindingSHA256": (
                        evaluation_status.get("runtimeTokenizerBinding", {}).get(
                            "runtimeTokenizerBindingSHA256"
                        )
                        if isinstance(
                            evaluation_status.get("runtimeTokenizerBinding"), Mapping
                        )
                        else None
                    ),
                },
                smoke_evidence=runtime_binding_smoke_evidence,
                label="Evaluation",
            )
        summary["agents"][agent] = {
            "sft": sft,
            "finalPhase": final_phase,
            "adapterGGUF": str(gguf),
            "adapterGGUFExists": gguf_exists,
            "adapterGGUFSHA256": (
                gguf_metadata["adapterGGUFSHA256"] if gguf_metadata else None
            ),
            "adapterGGUFSizeBytes": (
                gguf_metadata["adapterGGUFSizeBytes"] if gguf_metadata else 0
            ),
            **{
                field: gguf_metadata[field] if gguf_metadata else None
                for field in ADAPTER_GGUF_SEMANTIC_FIELDS
            },
            **{
                field: gguf_metadata[field] if gguf_metadata else None
                for field in GGUF_CONVERSION_SUMMARY_FIELDS
            },
            "evaluationReport": str(evaluation),
            "evaluationReportExists": evaluation.is_file(),
            "evaluation": evaluation_status,
        }
    evaluations = [
        item["evaluation"]
        for item in summary["agents"].values()
        if isinstance(item.get("evaluation"), Mapping)
    ]
    summary.update(
        _derived_summary_state(
            plan=prepared_execution_plan,
            evaluation_statuses=[str(item.get("status")) for item in evaluations],
            agent_count=len(agents),
            gguf_count=len(gguf_inventory),
            preference_training=preference,
        )
    )
    summary["summarySHA256"] = canonical_sha256(summary)
    write_object(run_root / "aio_summary.json", summary)
    return summary


def _verified_completed_summary(
    run_root: Path,
    agents: Sequence[str],
) -> dict[str, Any]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError(f"Summary run root is not a regular directory: {run_root}")
    _reject_managed_symlinks(run_root)
    run_manifest = _verified_run_manifest(run_root)
    prepared_execution_plan = _verified_execution_plan(
        run_manifest.get("executionPlan")
    )
    manifest_agents = run_manifest.get("agents")
    if (
        not isinstance(manifest_agents, list)
        or any(not isinstance(item, Mapping) for item in manifest_agents)
        or [item.get("agent") for item in manifest_agents] != list(agents)
    ):
        raise RuntimeError("Completed summary agents drifted from the prepared run")
    summary_path = run_root / "aio_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise RuntimeError(f"Missing regular completed summary: {summary_path}")
    summary = read_object(summary_path)
    declared = summary.get("summarySHA256")
    unsigned = dict(summary)
    unsigned.pop("summarySHA256", None)
    summary_agents = summary.get("agents")
    if (
        canonical_sha256(unsigned) != declared
        or set(summary)
        != {
            "schema",
            "status",
            "trainingScope",
            "evaluationStatus",
            "evaluationScope",
            "ggufStatus",
            "ggufConversionStatus",
            "ggufTensorEquivalenceStatus",
            "qualification",
            "promotionEligible",
            "executionPlanSHA256",
            "variant",
            "runRoot",
            "preferenceTraining",
            "baseModelID",
            "baseModelRevision",
            "baseModelTokenizerDigest",
            "baseModelTokenizerFiles",
            "baseModelTokenizerClosureSHA256",
            "baseModelTokenizerSnapshotPath",
            "baseModelTokenizerSnapshotVerification",
            "baseModelGenerationConfigFile",
            "baseModelRuntimeSnapshotPath",
            "baseModelRuntimeSnapshotVerification",
            "runManifestSHA256",
            *RUNTIME_BINDING_SMOKE_SUMMARY_FIELDS,
            *UBUNTU_SOURCE_INTEGRITY_FIELDS,
            "agents",
            "summarySHA256",
        }
        or summary.get("schema") != SUMMARY_SCHEMA_VERSION
        or summary.get("runRoot") != str(run_root)
        or summary.get("variant") != run_manifest.get("variant")
        or type(summary.get("preferenceTraining")) is not bool
        or any(
            summary.get(field) != run_manifest.get(field)
            for field in (
                "baseModelID",
                "baseModelRevision",
                "baseModelTokenizerDigest",
                "baseModelTokenizerFiles",
                "baseModelTokenizerClosureSHA256",
                "baseModelTokenizerSnapshotPath",
                "baseModelTokenizerSnapshotVerification",
                "baseModelGenerationConfigFile",
                "baseModelRuntimeSnapshotPath",
                "baseModelRuntimeSnapshotVerification",
                "runManifestSHA256",
            )
        )
        or any(
            summary.get(field) != run_manifest.get(field)
            for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
        )
        or summary.get("status")
        not in {
            "complete",
            "complete_without_gguf",
            "smoke_complete",
            "training_complete_without_full_evaluation",
            "sft_only_diagnostic_complete",
        }
        or summary.get("evaluationStatus")
        not in {"quality_gate_passed", "smoke_complete", "not_run"}
        or summary.get("evaluationScope") not in {"full", "smoke", "none"}
        or summary.get("ggufStatus")
        not in {"verified", "skipped_by_operator", "not_applicable_sft_only"}
        or summary.get("ggufConversionStatus")
        not in {
            GGUF_CONVERSION_QUALIFICATION,
            "skipped_by_operator",
            "not_applicable",
        }
        or summary.get("ggufTensorEquivalenceStatus")
        not in {GGUF_TENSOR_EQUIVALENCE_STATUS, "not_applicable"}
        or summary.get("qualification")
        not in {"quality_gate_passed", "diagnostic_only"}
        or type(summary.get("promotionEligible")) is not bool
        or summary.get("trainingScope") not in {"sft_preference", "sft_only"}
        or summary.get("executionPlanSHA256")
        != prepared_execution_plan["executionPlanSHA256"]
        or not isinstance(summary_agents, Mapping)
        or set(summary_agents) != set(agents)
    ):
        raise RuntimeError("Completed Ubuntu training summary failed verification")
    runtime_binding_smoke_evidence = (
        _verified_runtime_binding_smoke_summary_evidence(run_root, agents)
    )
    if any(
        summary.get(field) != runtime_binding_smoke_evidence.get(field)
        for field in RUNTIME_BINDING_SMOKE_SUMMARY_FIELDS
    ):
        raise RuntimeError(
            "Completed summary runtime-binding smoke evidence drifted"
        )
    preference_training = bool(summary["preferenceTraining"])
    gguf_inventory = (
        _verify_gguf_inventory(
            run_root,
            agents,
            require_all=False,
        )
        if preference_training
        else {}
    )
    evaluation_statuses: list[str] = []
    for agent in agents:
        item = summary_agents.get(agent)
        expected_agent_fields = (
            {
                "sft",
                "finalPhase",
                "adapterGGUF",
                "adapterGGUFExists",
                "adapterGGUFSHA256",
                "adapterGGUFSizeBytes",
                *ADAPTER_GGUF_SEMANTIC_FIELDS,
                *GGUF_CONVERSION_SUMMARY_FIELDS,
                "evaluationReport",
                "evaluationReportExists",
                "evaluation",
            }
            if preference_training
            else {"sft", "finalPhase"}
        )
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_agent_fields
        ):
            raise RuntimeError(f"Completed summary lacks agent {agent}")
        sft = verify_sft(run_root, agent)
        _require_runtime_bindings_match_smoke(
            agent=agent,
            source=sft,
            smoke_evidence=runtime_binding_smoke_evidence,
            label="SFT phase",
        )
        final_phase = (
            verify_preference(run_root, agent) if preference_training else sft
        )
        if preference_training:
            _require_runtime_bindings_match_smoke(
                agent=agent,
                source=final_phase,
                smoke_evidence=runtime_binding_smoke_evidence,
                label="Preference phase",
            )
        if item.get("sft") != sft or item.get("finalPhase") != final_phase:
            raise RuntimeError(f"Completed summary adapter lineage drifted for {agent}")
        if not preference_training:
            continue
        evaluation_report = run_root / "evaluation" / agent / "evaluation_report.json"
        evaluation_report_exists = (
            not evaluation_report.is_symlink()
            and evaluation_report.is_file()
            and evaluation_report.stat().st_size > 0
        )
        if (
            item.get("evaluationReport") != str(evaluation_report)
            or item.get("evaluationReportExists") is not evaluation_report_exists
        ):
            raise RuntimeError(
                f"Completed summary evaluation-report evidence drifted for {agent}"
            )
        evaluation = item.get("evaluation")
        if evaluation is not None:
            verified_evaluation = _verify_evaluation_outputs(
                run_root,
                agent,
                final_phase=final_phase,
            )
            if evaluation != verified_evaluation:
                raise RuntimeError(f"Completed summary evaluation drifted for {agent}")
            _require_runtime_bindings_match_smoke(
                agent=agent,
                source={
                    "runtimeModelBindingSHA256": (
                        verified_evaluation.get("runtimeModelBinding", {}).get(
                            "runtimeModelBindingSHA256"
                        )
                        if isinstance(
                            verified_evaluation.get("runtimeModelBinding"), Mapping
                        )
                        else None
                    ),
                    "runtimeTokenizerBindingSHA256": (
                        verified_evaluation.get("runtimeTokenizerBinding", {}).get(
                            "runtimeTokenizerBindingSHA256"
                        )
                        if isinstance(
                            verified_evaluation.get("runtimeTokenizerBinding"), Mapping
                        )
                        else None
                    ),
                },
                smoke_evidence=runtime_binding_smoke_evidence,
                label="Evaluation",
            )
            evaluation_statuses.append(str(verified_evaluation.get("status")))
        elif evaluation_report_exists:
            raise RuntimeError(f"Completed summary evaluation flag drifted for {agent}")
        gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        gguf_metadata = (
            verify_gguf_file(run_root, gguf)
            if gguf.name in gguf_inventory
            else None
        )
        gguf_exists = gguf_metadata is not None
        if (
            item.get("adapterGGUF") != str(gguf)
            or item.get("adapterGGUFExists") is not gguf_exists
        ):
            raise RuntimeError(f"Completed summary GGUF flag drifted for {agent}")
        if gguf_exists:
            _require_runtime_bindings_match_smoke(
                agent=agent,
                source=gguf_metadata,
                smoke_evidence=runtime_binding_smoke_evidence,
                label="GGUF conversion",
                model_field="adapterGGUFRuntimeModelBindingSHA256",
                tokenizer_field="adapterGGUFRuntimeTokenizerBindingSHA256",
            )
            if (
                type(item.get("adapterGGUFSizeBytes")) is not int
                or item["adapterGGUFSizeBytes"] <= 0
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(item.get("adapterGGUFSHA256") or ""),
                )
                is None
                or gguf_metadata["adapterGGUFSizeBytes"]
                != item["adapterGGUFSizeBytes"]
                or gguf_metadata["adapterGGUFSHA256"]
                != item.get("adapterGGUFSHA256")
                or any(
                    item.get(field) != gguf_metadata[field]
                    for field in (
                        *ADAPTER_GGUF_SEMANTIC_FIELDS,
                        *GGUF_CONVERSION_SUMMARY_FIELDS,
                    )
                )
            ):
                raise RuntimeError(f"Completed summary GGUF drifted for {agent}")
        elif (
            gguf.exists()
            or gguf.is_symlink()
            or item.get("adapterGGUFSizeBytes") != 0
            or item.get("adapterGGUFSHA256") is not None
            or any(
                item.get(field) is not None
                for field in (
                    *ADAPTER_GGUF_SEMANTIC_FIELDS,
                    *GGUF_CONVERSION_SUMMARY_FIELDS,
                )
            )
        ):
            raise RuntimeError(f"Unbound or unsafe GGUF exists for {agent}")
    expected_state = _derived_summary_state(
        plan=prepared_execution_plan,
        evaluation_statuses=evaluation_statuses,
        agent_count=len(agents),
        gguf_count=len(gguf_inventory),
        preference_training=preference_training,
    )
    if any(summary.get(field) != value for field, value in expected_state.items()):
        raise RuntimeError("Completed summary state does not match its verified evidence")
    return summary


def _compact_phase_runtime_evidence(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Summary phase runtime evidence is missing")
    evidence = {field: value.get(field) for field in PHASE_RUNTIME_EVIDENCE_FIELDS}
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
        for digest in evidence.values()
    ):
        raise RuntimeError("Summary phase runtime evidence lacks exact digests")
    return {field: str(digest) for field, digest in evidence.items()}


def _summary_phase_runtime_evidence(
    summary: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, str]]]:
    summary_agents = summary.get("agents")
    preference_training = summary.get("preferenceTraining")
    if (
        not isinstance(summary_agents, Mapping)
        or not summary_agents
        or type(preference_training) is not bool
    ):
        raise RuntimeError("Upload summary lacks phase runtime evidence")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for agent in sorted(summary_agents):
        item = summary_agents.get(agent)
        if not isinstance(agent, str) or not isinstance(item, Mapping):
            raise RuntimeError("Upload summary phase runtime evidence is invalid")
        sft = item.get("sft")
        final_phase = item.get("finalPhase")
        phases = {"sft": _compact_phase_runtime_evidence(sft)}
        if preference_training:
            phases["preference"] = _compact_phase_runtime_evidence(final_phase)
        elif final_phase != sft:
            raise RuntimeError("SFT-only upload summary has a preference final phase")
        result[agent] = phases
    return result


def _upload_publication_contract(
    summary: Mapping[str, Any],
    *,
    allow_diagnostic_upload: bool,
) -> dict[str, Any]:
    if type(allow_diagnostic_upload) is not bool:
        raise RuntimeError("Diagnostic upload override must be boolean")
    tokenizer_files = summary.get("baseModelTokenizerFiles")
    tokenizer_json_records = (
        [
            item
            for item in tokenizer_files
            if isinstance(item, Mapping)
            and item.get("path") == "tokenizer.json"
        ]
        if isinstance(tokenizer_files, list)
        else []
    )
    tokenizer_digest = summary.get("baseModelTokenizerDigest")
    if (
        len(tokenizer_json_records) != 1
        or re.fullmatch(r"[0-9a-f]{64}", str(tokenizer_digest or ""))
        is None
        or tokenizer_json_records[0].get("sha256") != tokenizer_digest
    ):
        raise RuntimeError(
            "Upload summary tokenizer digest is not bound to tokenizer.json"
        )
    promotion_eligible = summary.get("promotionEligible") is True
    qualification = str(summary.get("qualification") or "")
    preference_training = summary.get("preferenceTraining")
    training_scope = summary.get("trainingScope")
    if type(preference_training) is not bool:
        raise RuntimeError("Upload summary preference-training state is invalid")
    runtime_binding_smoke_evidence = {
        field: summary.get(field)
        for field in RUNTIME_BINDING_SMOKE_SUMMARY_FIELDS
    }
    if (
        not isinstance(
            runtime_binding_smoke_evidence["runtimeBindingSmokeReport"], str
        )
        or any(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(runtime_binding_smoke_evidence[field] or ""),
            )
            is None
            for field in (
                "runtimeBindingSmokeReportFileSHA256",
                "runtimeBindingSmokeGateSHA256",
            )
        )
        or not isinstance(
            runtime_binding_smoke_evidence["runtimeBindingSmokeContractEvidence"],
            list,
        )
        or not isinstance(
            runtime_binding_smoke_evidence["runtimeBindingSmokeBindingsByAgent"],
            Mapping,
        )
    ):
        raise RuntimeError("Upload summary runtime-binding smoke evidence is invalid")
    phase_runtime_evidence = _summary_phase_runtime_evidence(summary)
    for agent, phases in phase_runtime_evidence.items():
        for phase_name, phase_evidence in phases.items():
            _require_runtime_bindings_match_smoke(
                agent=agent,
                source=phase_evidence,
                smoke_evidence=runtime_binding_smoke_evidence,
                label=f"Upload {phase_name} phase",
            )
    if promotion_eligible:
        if (
            preference_training is not True
            or training_scope != "sft_preference"
            or qualification != "quality_gate_passed"
            or summary.get("evaluationStatus") != "quality_gate_passed"
            or summary.get("evaluationScope") != "full"
            or summary.get("status") not in {"complete", "complete_without_gguf"}
        ):
            raise RuntimeError("Upload summary has inconsistent qualification state")
        remote_namespace = "runs"
    else:
        if preference_training:
            diagnostic_state_valid = (
                training_scope == "sft_preference"
                and summary.get("evaluationStatus")
                in {"smoke_complete", "not_run"}
                and summary.get("evaluationScope") in {"smoke", "none"}
                and summary.get("status")
                in {"smoke_complete", "training_complete_without_full_evaluation"}
                and summary.get("ggufStatus")
                in {"verified", "skipped_by_operator"}
            )
            remote_namespace = "diagnostic-runs"
        else:
            diagnostic_state_valid = (
                training_scope == "sft_only"
                and summary.get("evaluationStatus") == "not_run"
                and summary.get("evaluationScope") == "none"
                and summary.get("status") == "sft_only_diagnostic_complete"
                and summary.get("ggufStatus") == "not_applicable_sft_only"
                and summary.get("ggufConversionStatus") == "not_applicable"
                and summary.get("ggufTensorEquivalenceStatus") == "not_applicable"
            )
            remote_namespace = "diagnostic-sft-runs"
        if (
            qualification != "diagnostic_only"
            or not diagnostic_state_valid
        ):
            raise RuntimeError("Upload summary has inconsistent diagnostic state")
        if not allow_diagnostic_upload:
            raise RuntimeError(
                "Diagnostic upload requires --allow-diagnostic-upload"
            )
    return {
        "remoteNamespace": remote_namespace,
        "qualification": qualification,
        "promotionEligible": promotion_eligible,
        "diagnosticUploadOverrideApplied": not promotion_eligible,
        "preferenceTraining": preference_training,
        "trainingScope": training_scope,
        "phaseRuntimeEvidenceByAgent": phase_runtime_evidence,
        **runtime_binding_smoke_evidence,
        "evaluationStatus": summary["evaluationStatus"],
        "evaluationScope": summary["evaluationScope"],
        "ggufStatus": summary["ggufStatus"],
        "ggufConversionStatus": summary["ggufConversionStatus"],
        "ggufTensorEquivalenceStatus": summary[
            "ggufTensorEquivalenceStatus"
        ],
        "executionPlanSHA256": summary["executionPlanSHA256"],
        "baseModelID": summary["baseModelID"],
        "baseModelRevision": summary["baseModelRevision"],
        "baseModelTokenizerDigest": tokenizer_digest,
        "baseModelTokenizerClosureSHA256": summary[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelTokenizerSnapshotPath": summary[
            "baseModelTokenizerSnapshotPath"
        ],
        "baseModelTokenizerSnapshotVerification": summary[
            "baseModelTokenizerSnapshotVerification"
        ],
        "baseModelGenerationConfigFile": summary[
            "baseModelGenerationConfigFile"
        ],
        "baseModelRuntimeSnapshotPath": summary[
            "baseModelRuntimeSnapshotPath"
        ],
        "baseModelRuntimeSnapshotVerification": summary[
            "baseModelRuntimeSnapshotVerification"
        ],
        "runManifestSHA256": summary["runManifestSHA256"],
    }


def _verified_upload_final_phase(
    summary: Mapping[str, Any],
    agent: str,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    summary_agents = summary.get("agents")
    summary_agent = (
        summary_agents.get(agent) if isinstance(summary_agents, Mapping) else None
    )
    final_phase = (
        summary_agent.get("finalPhase")
        if isinstance(summary_agent, Mapping)
        else None
    )
    observed_phase = dict(observed)
    if not isinstance(final_phase, Mapping) or observed_phase != dict(final_phase):
        raise RuntimeError(
            f"Upload adapter lineage drifted from the completed summary for {agent}"
        )
    return observed_phase


def _self_hashed_upload_record(
    payload: Mapping[str, Any],
    *,
    digest_field: str,
) -> dict[str, Any]:
    if digest_field in payload:
        raise RuntimeError(f"Upload record already contains {digest_field}")
    result = dict(payload)
    result[digest_field] = canonical_sha256(result)
    return result


def _verified_self_hashed_upload_record(
    path: Path,
    *,
    schema: str,
    digest_field: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Upload transaction record is not a regular file: {path}")
    record_stat = path.stat(follow_symlinks=False)
    if (
        record_stat.st_uid != os.geteuid()
        or stat.S_IMODE(record_stat.st_mode) != 0o600
    ):
        raise RuntimeError(f"Upload transaction record is not private: {path}")
    record = read_object(path)
    digest = record.get(digest_field)
    unsigned = dict(record)
    unsigned.pop(digest_field, None)
    if (
        record.get("schema") != schema
        or re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
        or canonical_sha256(unsigned) != digest
    ):
        raise RuntimeError(f"Upload transaction record is invalid: {path}")
    return record


def _write_once_upload_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    schema: str,
    digest_field: str,
) -> dict[str, Any]:
    expected = dict(record)
    if path.exists() or path.is_symlink():
        observed = _verified_self_hashed_upload_record(
            path,
            schema=schema,
            digest_field=digest_field,
        )
        if observed != expected:
            raise RuntimeError(f"Upload transaction record drifted: {path}")
        return observed
    write_object(path, expected)
    path.chmod(0o600)
    _fsync_directory(path.parent, label=f"upload transaction record {path}")
    return expected


def _upload_intent_payload(
    *,
    repo_id: str,
    private: bool,
    run_id: str,
    publication_root: str,
    publication: Mapping[str, Any],
    include_gguf: bool,
    summary_status: str,
    snapshotted_files: Sequence[_SnapshottedUploadInput],
    image_source_fields: Mapping[str, Any],
) -> dict[str, Any]:
    prefix = f"{publication_root}/"
    files = [
        {
            "path": item.remote_path,
            "sha256": item.sha256,
            "sizeBytes": item.size_bytes,
        }
        for item in sorted(snapshotted_files, key=lambda item: item.remote_path)
    ]
    return _self_hashed_upload_record(
        {
            "schema": UPLOAD_INTENT_SCHEMA_VERSION,
            "repository": repo_id,
            "private": private,
            "runID": run_id,
            **publication,
            "publicationRoot": publication_root,
            "remotePrefix": prefix,
            "remoteMarkerPath": f"{prefix}{UPLOAD_REMOTE_MARKER_FILENAME}",
            "ggufIncluded": include_gguf,
            "summaryStatus": summary_status,
            "files": files,
            **image_source_fields,
        },
        digest_field="uploadIntentSHA256",
    )


def _upload_attempt_payload(
    intent: Mapping[str, Any],
    *,
    parent_revision: str | None,
) -> dict[str, Any]:
    intent_digest = intent.get("uploadIntentSHA256")
    if re.fullmatch(r"[0-9a-f]{64}", str(intent_digest or "")) is None:
        raise RuntimeError("Upload intent lacks an immutable digest")
    if parent_revision is not None and (
        not isinstance(parent_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", parent_revision) is None
    ):
        raise RuntimeError("Remote parent revision is not immutable")
    return _self_hashed_upload_record(
        {
            "schema": UPLOAD_ATTEMPT_SCHEMA_VERSION,
            "uploadIntentSHA256": intent_digest,
            "repository": intent["repository"],
            "private": intent["private"],
            "remotePrefix": intent["remotePrefix"],
            "parentRevision": parent_revision,
            "commitMessage": f"Lumen upload transaction {intent_digest}",
        },
        digest_field="uploadAttemptSHA256",
    )


def _upload_commit_payload(
    attempt: Mapping[str, Any],
    *,
    commit_oid: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", commit_oid) is None:
        raise RuntimeError("Hugging Face upload did not return an immutable commit OID")
    return _self_hashed_upload_record(
        {
            "schema": UPLOAD_COMMIT_SCHEMA_VERSION,
            "uploadIntentSHA256": attempt["uploadIntentSHA256"],
            "uploadAttemptSHA256": attempt["uploadAttemptSHA256"],
            "parentRevision": attempt["parentRevision"],
            "commitMessage": attempt["commitMessage"],
            "commitOID": commit_oid,
        },
        digest_field="uploadCommitSHA256",
    )


def _upload_marker_snapshot(
    snapshot_root: Path,
    intent: Mapping[str, Any],
) -> _SnapshottedUploadInput:
    marker_path = snapshot_root / "upload-intent.remote.json"
    write_object(marker_path, intent)
    marker_path.chmod(0o400)
    return _SnapshottedUploadInput(
        path=marker_path,
        remote_path=str(intent["remoteMarkerPath"]),
        sha256=file_sha256(marker_path),
        size_bytes=marker_path.stat().st_size,
    )


def _commit_metadata(commit: Any) -> tuple[str | None, str | None]:
    commit_oid = getattr(commit, "commit_id", None) or getattr(commit, "oid", None)
    title = getattr(commit, "title", None)
    if title is None:
        message = getattr(commit, "message", None)
        if isinstance(message, str):
            title = message.splitlines()[0]
    return commit_oid, title


def _remote_commit_has_expected_parent(
    *,
    api: Any,
    repo_id: str,
    commit_oid: str,
    expected_parent: str | None,
) -> bool:
    history = api.list_repo_commits(
        repo_id=repo_id,
        repo_type="model",
        revision=commit_oid,
    )
    history_ids = [_commit_metadata(commit)[0] for commit in history]
    if not history_ids or history_ids[0] != commit_oid:
        return False
    if expected_parent is None:
        return len(history_ids) == 1
    return len(history_ids) >= 2 and history_ids[1] == expected_parent


def _verify_remote_upload_commit(
    *,
    api: Any,
    hub_module: Any,
    token: str,
    repo_id: str,
    private: bool,
    attempt: Mapping[str, Any],
    commit_oid: str,
    expected_files: Sequence[_SnapshottedUploadInput],
    prefix: str,
) -> Any:
    if re.fullmatch(r"[0-9a-f]{40}", commit_oid) is None:
        raise RuntimeError("Recovered upload commit OID is invalid")
    commits = api.list_repo_commits(repo_id=repo_id, repo_type="model")
    matching_metadata = []
    expected_parent = attempt.get("parentRevision")
    for commit in commits:
        observed_oid, title = _commit_metadata(commit)
        if observed_oid == commit_oid:
            matching_metadata.append(title)
    if matching_metadata != [attempt.get("commitMessage")] or not (
        _remote_commit_has_expected_parent(
            api=api,
            repo_id=repo_id,
            commit_oid=commit_oid,
            expected_parent=expected_parent,
        )
    ):
        raise RuntimeError("Remote upload commit lineage failed verification")

    download = getattr(hub_module, "hf_hub_download", None)
    if not callable(download):
        raise RuntimeError("huggingface_hub lacks immutable download support")
    expected_by_path = {item.remote_path: item for item in expected_files}

    def verify_revision(revision: str, *, label: str) -> None:
        remote_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )
        observed_prefix_paths = {
            path for path in remote_files if path.startswith(prefix)
        }
        if observed_prefix_paths != set(expected_by_path):
            raise RuntimeError(f"Remote upload {label} path set failed verification")
        for remote_path, expected in expected_by_path.items():
            downloaded = Path(
                download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="model",
                    revision=revision,
                    token=token,
                )
            )
            resolved = downloaded.resolve(strict=True)
            if (
                not resolved.is_file()
                or resolved.stat().st_size != expected.size_bytes
                or file_sha256(resolved) != expected.sha256
            ):
                raise RuntimeError(
                    f"Remote upload {label} content failed verification: {remote_path}"
                )

    verify_revision(commit_oid, label="transaction commit")
    final_info = api.repo_info(repo_id=repo_id, repo_type="model")
    final_revision = getattr(final_info, "sha", None)
    if (
        re.fullmatch(r"[0-9a-f]{40}", str(final_revision or "")) is None
        or bool(final_info.private) != private
    ):
        raise RuntimeError("Recovered upload repository head or visibility is invalid")
    if final_revision != commit_oid:
        verify_revision(str(final_revision), label="current head")
    return final_info


def _discover_recoverable_upload_commit(
    *,
    api: Any,
    attempt: Mapping[str, Any],
) -> str:
    candidates: list[str] = []
    for commit in api.list_repo_commits(
        repo_id=attempt["repository"],
        repo_type="model",
    ):
        commit_oid, title = _commit_metadata(commit)
        if (
            re.fullmatch(r"[0-9a-f]{40}", str(commit_oid or "")) is not None
            and title == attempt.get("commitMessage")
            and _remote_commit_has_expected_parent(
                api=api,
                repo_id=str(attempt["repository"]),
                commit_oid=str(commit_oid),
                expected_parent=attempt.get("parentRevision"),
            )
        ):
            candidates.append(str(commit_oid))
    if len(candidates) != 1:
        raise RuntimeError("Remote upload transaction commit is missing or ambiguous")
    return candidates[0]


def _cleanup_upload_transaction_records(paths: Sequence[Path]) -> None:
    changed = False
    for path in paths:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Upload transaction cleanup path is unsafe: {path}")
            path.unlink()
            changed = True
    if changed:
        _fsync_directory(paths[0].parent, label="upload transaction cleanup")


def upload_run(
    *,
    run_root: Path,
    agents: Sequence[str],
    run_id: str,
    private: bool,
    include_gguf: bool,
    token_file: Path,
    allow_diagnostic_upload: bool = False,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import huggingface_hub as hub_module

        CommitOperationAdd = hub_module.CommitOperationAdd
        HfApi = hub_module.HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload") from exc
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", run_id) is None:
        raise RuntimeError("Upload run ID is unsafe")
    run_manifest = _verified_run_manifest(run_root)
    image_source_integrity = current_source_integrity(
        Path(__file__).resolve().parents[3]
    )
    image_source_fields = source_integrity_fields(image_source_integrity)
    if any(
        run_manifest.get(field) != image_source_fields[field]
        for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
    ):
        raise RuntimeError("Upload source does not match the prepared run attestation")
    if run_manifest.get("runID") != run_id:
        raise RuntimeError("Upload run ID does not match the prepared run")
    manifest_agents = run_manifest.get("agents")
    if [item.get("agent") for item in manifest_agents] != list(agents):
        raise RuntimeError("Upload agents do not match the prepared run")
    summary = _verified_completed_summary(run_root, agents)
    publication = _upload_publication_contract(
        summary,
        allow_diagnostic_upload=allow_diagnostic_upload,
    )
    promotion_eligible = bool(publication["promotionEligible"])
    remote_namespace = str(publication["remoteNamespace"])
    publication_root = f"{remote_namespace}/{run_id}"
    if include_gguf and summary.get("ggufStatus") != "verified":
        raise RuntimeError("Upload requested GGUF files that were not verified")
    repo_id = str(run_manifest.get("adapterRepoID") or "").strip()
    runtime_manifest_path = (
        run_root / "generated" / "fine_tuning" / "adapter_runtime_manifest.json"
    )
    if (
        not repo_id
        or file_sha256(runtime_manifest_path)
        != run_manifest.get("adapterRuntimeManifestFileSHA256")
        or read_object(runtime_manifest_path).get("adapterRepoID") != repo_id
    ):
        raise RuntimeError("Prepared adapter repository destination drifted")
    if read_object(run_root / "training_environment.json") != run_manifest.get(
        "trainingEnvironment"
    ):
        raise RuntimeError("Prepared training environment record drifted")

    observed_runtime_binding_smoke = (
        _verified_runtime_binding_smoke_summary_evidence(run_root, agents)
    )
    if any(
        summary.get(field) != observed_runtime_binding_smoke.get(field)
        for field in RUNTIME_BINDING_SMOKE_SUMMARY_FIELDS
    ):
        raise RuntimeError("Upload runtime-binding smoke evidence drifted")
    from tools.fine_tuning.unsloth.runtime_binding_smoke_gate import REPORT_FILENAME

    smoke_report_relative = f"training/{REPORT_FILENAME}"
    if summary.get("runtimeBindingSmokeReport") != str(
        run_root / smoke_report_relative
    ):
        raise RuntimeError("Upload runtime-binding smoke report path drifted")
    smoke_report = read_object(run_root / smoke_report_relative)
    upload_contracts: list[_UploadInputContract] = [
        _UploadInputContract(
            relative_path=smoke_report_relative,
            remote_path=f"{publication_root}/manifests/runtime_binding_smoke.json",
            expected_json=smoke_report,
        )
    ]
    preference_training = bool(summary["preferenceTraining"])
    for agent in agents:
        summary_agent = summary["agents"][agent]
        observed_sft = verify_sft(run_root, agent)
        if observed_sft != summary_agent.get("sft"):
            raise RuntimeError(
                f"Upload SFT lineage drifted from the completed summary for {agent}"
            )
        final_phase = _verified_upload_final_phase(
            summary,
            agent,
            (
                verify_preference(run_root, agent)
                if preference_training
                else observed_sft
            ),
        )
        adapter_root = (
            f"models/lora_qwen3_dpo/{agent}"
            if preference_training
            else f"models/lora_qwen3_bootstrap/{agent}"
        )
        adapter_manifest_relative = f"{adapter_root}/adapter_artifact_manifest.json"
        adapter_manifest = read_object(run_root / adapter_manifest_relative)
        adapter_manifest_digest = adapter_manifest.get("adapterSHA256")
        unsigned_adapter_manifest = dict(adapter_manifest)
        unsigned_adapter_manifest.pop("adapterSHA256", None)
        artifact_files = adapter_manifest.get("files")
        if (
            not isinstance(artifact_files, list)
            or not artifact_files
            or re.fullmatch(r"[0-9a-f]{64}", str(adapter_manifest_digest or ""))
            is None
            or canonical_sha256(unsigned_adapter_manifest) != adapter_manifest_digest
            or final_phase.get("adapterSHA256") != adapter_manifest_digest
        ):
            raise RuntimeError(f"Adapter upload manifest is invalid for {agent}")
        artifact_names: set[str] = set()
        for item in artifact_files:
            name = item.get("path") if isinstance(item, Mapping) else None
            size = item.get("sizeBytes") if isinstance(item, Mapping) else None
            digest = item.get("sha256") if isinstance(item, Mapping) else None
            relative_name = PurePosixPath(name) if isinstance(name, str) else None
            if (
                not isinstance(item, Mapping)
                or set(item) != {"path", "sizeBytes", "sha256"}
                or relative_name is None
                or relative_name.is_absolute()
                or len(relative_name.parts) != 1
                or relative_name.as_posix() != name
                or name in artifact_names
                or type(size) is not int
                or size < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
            ):
                raise RuntimeError(f"Adapter upload manifest is invalid for {agent}")
            artifact_names.add(name)
            upload_contracts.append(
                _UploadInputContract(
                    relative_path=f"{adapter_root}/{name}",
                    remote_path=f"{publication_root}/adapters/{agent}/{name}",
                    expected_sha256=str(digest),
                    expected_size=size,
                )
            )
        upload_contracts.append(
            _UploadInputContract(
                relative_path=adapter_manifest_relative,
                remote_path=(
                    f"{publication_root}/adapters/{agent}/adapter_artifact_manifest.json"
                ),
                expected_json=adapter_manifest,
            )
        )
        finalized_relative = (
            f"training/{agent}/dpo/finalized_variant_manifest.json"
            if preference_training
            else f"training/{agent}/finalized_variant_manifest.json"
        )
        finalized = _verify_manifest_integrity(run_root / finalized_relative)
        if (
            finalized.get("variantManifestSHA256")
            != final_phase.get("finalizedVariantManifestSHA256")
        ):
            raise RuntimeError(f"Upload finalized lineage drifted for {agent}")
        upload_contracts.append(
            _UploadInputContract(
                relative_path=finalized_relative,
                remote_path=f"{publication_root}/manifests/{agent}/variant_manifest.json",
                expected_json=finalized,
            )
        )
        phase_reports = [("sft", observed_sft)]
        if preference_training:
            phase_reports.append(("preference", final_phase))
        for phase_name, phase_evidence in phase_reports:
            report_relative = (
                f"training/{agent}/training_report.json"
                if phase_name == "sft"
                else f"training/{agent}/dpo/dpo_report.json"
            )
            report_path = run_root / report_relative
            report_digest = phase_evidence.get("trainingReportFileSHA256")
            if (
                phase_evidence.get("report") != str(report_path)
                or re.fullmatch(r"[0-9a-f]{64}", str(report_digest or "")) is None
                or file_sha256(report_path) != report_digest
            ):
                raise RuntimeError(
                    f"Upload {phase_name} training report drifted for {agent}"
                )
            upload_contracts.append(
                _UploadInputContract(
                    relative_path=report_relative,
                    remote_path=(
                        f"{publication_root}/manifests/{agent}/"
                        f"{phase_name}_training_report.json"
                    ),
                    expected_sha256=str(report_digest),
                )
            )
        evaluation = summary_agent.get("evaluation") if preference_training else None
        if isinstance(evaluation, Mapping):
            candidate_digest = evaluation.get("candidateOutputsFileSHA256")
            report_digest = evaluation.get("evaluationReportFileSHA256")
            if any(
                re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is None
                for value in (candidate_digest, report_digest)
            ):
                raise RuntimeError(f"Upload evaluation lineage drifted for {agent}")
            upload_contracts.extend(
                (
                    _UploadInputContract(
                        relative_path=f"evaluation/{agent}/candidate_outputs.jsonl",
                        remote_path=(
                            f"{publication_root}/evaluation/{agent}/candidate_outputs.jsonl"
                        ),
                        expected_sha256=str(candidate_digest),
                    ),
                    _UploadInputContract(
                        relative_path=f"evaluation/{agent}/evaluation_report.json",
                        remote_path=(
                            f"{publication_root}/evaluation/{agent}/evaluation_report.json"
                        ),
                        expected_sha256=str(report_digest),
                    ),
                    _UploadInputContract(
                        relative_path=(
                            f"evaluation/{agent}/evaluation_run_manifest.json"
                        ),
                        remote_path=(
                            f"{publication_root}/evaluation/{agent}/evaluation_run_manifest.json"
                        ),
                        expected_json=evaluation,
                    )
                )
            )
        if include_gguf:
            agent_summary = summary["agents"][agent]
            gguf_digest = agent_summary.get("adapterGGUFSHA256")
            gguf_size = agent_summary.get("adapterGGUFSizeBytes")
            if (
                agent_summary.get("adapterGGUFExists") is not True
                or re.fullmatch(r"[0-9a-f]{64}", str(gguf_digest or "")) is None
                or type(gguf_size) is not int
                or gguf_size <= 0
            ):
                raise RuntimeError(f"Upload requires a verified GGUF for {agent}")
            upload_contracts.append(
                _UploadInputContract(
                    relative_path=(
                        f"models/lora_qwen3_gguf/lumen-{agent}-lora.gguf"
                    ),
                    remote_path=f"{publication_root}/gguf/lumen-{agent}-lora.gguf",
                    expected_sha256=str(gguf_digest),
                    expected_size=gguf_size,
                )
            )
            conversion_receipt_relative = (
                "models/lora_qwen3_gguf_receipts/"
                f"{_gguf_conversion_receipt_name(agent)}"
            )
            conversion_receipt = read_object(
                run_root / conversion_receipt_relative
            )
            if (
                agent_summary.get("adapterGGUFConversionReceipt")
                != str(run_root / conversion_receipt_relative)
                or agent_summary.get("adapterGGUFConversionReceiptSHA256")
                != conversion_receipt.get("conversionReceiptSHA256")
                or agent_summary.get("adapterGGUFConversionQualification")
                != GGUF_CONVERSION_QUALIFICATION
                or agent_summary.get("adapterGGUFTensorEquivalenceStatus")
                != GGUF_TENSOR_EQUIVALENCE_STATUS
                or agent_summary.get("adapterGGUFRuntimeModelBindingSHA256")
                != conversion_receipt.get("runtimeModelBindingSHA256")
                or agent_summary.get("adapterGGUFRuntimeTokenizerBindingSHA256")
                != conversion_receipt.get("runtimeTokenizerBindingSHA256")
            ):
                raise RuntimeError(
                    f"Upload GGUF conversion receipt drifted for {agent}"
                )
            upload_contracts.append(
                _UploadInputContract(
                    relative_path=conversion_receipt_relative,
                    remote_path=(
                        f"{publication_root}/gguf/"
                        f"{_gguf_conversion_receipt_name(agent)}"
                    ),
                    expected_json=conversion_receipt,
                )
            )
    upload_contracts.extend(
        (
            _UploadInputContract(
                relative_path="aio_run_manifest.json",
                remote_path=f"{publication_root}/aio_run_manifest.json",
                expected_json=run_manifest,
            ),
            _UploadInputContract(
                relative_path="aio_summary.json",
                remote_path=f"{publication_root}/aio_summary.json",
                expected_json=summary,
            ),
            _UploadInputContract(
                relative_path="training_environment.json",
                remote_path=f"{publication_root}/training_environment.json",
                expected_json=run_manifest["trainingEnvironment"],
            ),
        )
    )
    receipt_path = (
        receipt_path.resolve()
        if receipt_path is not None
        else run_root / "upload_receipts.json"
    )
    if receipt_path.parent.is_symlink() or not receipt_path.parent.is_dir():
        raise RuntimeError("Upload receipt parent must be a regular directory")
    receipt_parent_stat = receipt_path.parent.stat(follow_symlinks=False)
    if (
        receipt_parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(receipt_parent_stat.st_mode) & 0o077
    ):
        raise RuntimeError("Upload receipt parent must be private and process-owned")
    if receipt_path.is_symlink() or (
        receipt_path.exists() and not receipt_path.is_file()
    ):
        raise RuntimeError(f"Upload receipt path is unsafe: {receipt_path}")
    intent_path = receipt_path.parent / UPLOAD_INTENT_FILENAME
    attempt_path = receipt_path.parent / UPLOAD_ATTEMPT_FILENAME
    commit_path = receipt_path.parent / UPLOAD_COMMIT_FILENAME
    transaction_paths = (intent_path, attempt_path, commit_path)

    with tempfile.TemporaryDirectory(
        prefix="lumen-upload-snapshot-",
        dir="/tmp",
    ) as snapshot_name:
        snapshot_root = Path(snapshot_name)
        snapshot_root.chmod(0o700)
        snapshotted_files = _snapshot_verified_upload_inputs(
            run_root,
            upload_contracts,
            snapshot_root,
        )
        intent = _upload_intent_payload(
            repo_id=repo_id,
            private=private,
            run_id=run_id,
            publication_root=publication_root,
            publication=publication,
            include_gguf=include_gguf,
            summary_status=str(summary["status"]),
            snapshotted_files=snapshotted_files,
            image_source_fields=image_source_fields,
        )
        marker = _upload_marker_snapshot(snapshot_root, intent)
        remote_files = [*snapshotted_files, marker]
        remote_paths = [item.remote_path for item in remote_files]
        prefix = str(intent["remotePrefix"])
        if not receipt_path.exists():
            _write_once_upload_record(
                intent_path,
                intent,
                schema=UPLOAD_INTENT_SCHEMA_VERSION,
                digest_field="uploadIntentSHA256",
            )

        token_handle, token_stat = _open_regular_readonly(
            token_file,
            label="Upload token",
        )
        try:
            token_payload = _read_descriptor_bytes(token_handle)
            _require_stable_descriptor(
                token_handle,
                token_stat,
                label="Upload token",
            )
            _require_path_matches_descriptor(
                token_file,
                token_stat,
                label="Upload token",
            )
        finally:
            token_handle.close()
        try:
            token = token_payload.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError("Upload token file is not valid UTF-8") from exc
        if not token or "\n" in token or "\r" in token:
            raise RuntimeError("Upload token file is empty or malformed")
        api = HfApi(token=token)
        identity = api.whoami()
        if not isinstance(identity, Mapping) or not identity.get("name"):
            raise RuntimeError("Hugging Face authentication preflight failed")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if bool(info.private) != private:
            raise RuntimeError(
                "Remote repository visibility does not match the requested policy"
            )
        existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        prefix_exists = any(path.startswith(prefix) for path in existing_files)

        if receipt_path.exists():
            existing_receipt = _verified_self_hashed_upload_record(
                receipt_path,
                schema=UPLOAD_SCHEMA_VERSION,
                digest_field="uploadSHA256",
            )
            parent_revision = existing_receipt.get("parentRevision")
            attempt = _upload_attempt_payload(
                intent,
                parent_revision=(
                    str(parent_revision) if parent_revision is not None else None
                ),
            )
            commit_oid = str(existing_receipt.get("commitOID") or "")
            required_receipt_fields: dict[str, Any] = {
                "repository": repo_id,
                "private": private,
                "runID": run_id,
                **publication,
                "remotePrefix": prefix,
                "ggufIncluded": include_gguf,
                "summaryStatus": summary["status"],
                "uploadedFileCount": len(remote_files),
                "uploadedPaths": remote_paths,
                "commitOID": commit_oid,
                "parentRevision": parent_revision,
                "uploadIntentSHA256": intent["uploadIntentSHA256"],
                "uploadAttemptSHA256": attempt["uploadAttemptSHA256"],
                **image_source_fields,
            }
            if any(
                existing_receipt.get(field) != expected
                for field, expected in required_receipt_fields.items()
            ) or re.fullmatch(
                r"[0-9a-f]{40}", str(existing_receipt.get("headRevision") or "")
            ) is None:
                raise RuntimeError("Existing upload receipt drifted from this run")
            if not prefix_exists:
                raise RuntimeError("Existing upload receipt has no remote run prefix")
            _verify_remote_upload_commit(
                api=api,
                hub_module=hub_module,
                token=token,
                repo_id=repo_id,
                private=private,
                attempt=attempt,
                commit_oid=commit_oid,
                expected_files=remote_files,
                prefix=prefix,
            )
            _cleanup_upload_transaction_records(transaction_paths)
            return existing_receipt

        if attempt_path.exists() or attempt_path.is_symlink():
            attempt = _verified_self_hashed_upload_record(
                attempt_path,
                schema=UPLOAD_ATTEMPT_SCHEMA_VERSION,
                digest_field="uploadAttemptSHA256",
            )
            expected_attempt = _upload_attempt_payload(
                intent,
                parent_revision=attempt.get("parentRevision"),
            )
            if attempt != expected_attempt:
                raise RuntimeError("Upload attempt drifted from its durable intent")
        else:
            if prefix_exists:
                raise RuntimeError(
                    "Remote run prefix exists without a durable local upload attempt"
                )
            parent_revision = getattr(info, "sha", None)
            if parent_revision is not None:
                parent_revision = str(parent_revision)
            attempt = _upload_attempt_payload(
                intent,
                parent_revision=parent_revision,
            )
            _write_once_upload_record(
                attempt_path,
                attempt,
                schema=UPLOAD_ATTEMPT_SCHEMA_VERSION,
                digest_field="uploadAttemptSHA256",
            )

        parent_revision = attempt.get("parentRevision")
        if prefix_exists:
            if commit_path.exists() or commit_path.is_symlink():
                commit_record = _verified_self_hashed_upload_record(
                    commit_path,
                    schema=UPLOAD_COMMIT_SCHEMA_VERSION,
                    digest_field="uploadCommitSHA256",
                )
                expected_commit = _upload_commit_payload(
                    attempt,
                    commit_oid=str(commit_record.get("commitOID") or ""),
                )
                if commit_record != expected_commit:
                    raise RuntimeError("Upload commit record drifted from its attempt")
                commit_oid = str(commit_record["commitOID"])
            else:
                commit_oid = _discover_recoverable_upload_commit(
                    api=api,
                    attempt=attempt,
                )
                commit_record = _upload_commit_payload(
                    attempt,
                    commit_oid=commit_oid,
                )
                _write_once_upload_record(
                    commit_path,
                    commit_record,
                    schema=UPLOAD_COMMIT_SCHEMA_VERSION,
                    digest_field="uploadCommitSHA256",
                )
            final_info = _verify_remote_upload_commit(
                api=api,
                hub_module=hub_module,
                token=token,
                repo_id=repo_id,
                private=private,
                attempt=attempt,
                commit_oid=commit_oid,
                expected_files=remote_files,
                prefix=prefix,
            )
            remote_verification = "recovered_exact_remote_tree"
        else:
            current_parent = getattr(info, "sha", None)
            if current_parent is not None:
                current_parent = str(current_parent)
            if current_parent != parent_revision:
                raise RuntimeError(
                    "Remote repository head changed after the upload attempt was staged"
                )
            commit = api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=item.remote_path,
                        path_or_fileobj=str(item.path),
                    )
                    for item in remote_files
                ],
                commit_message=str(attempt["commitMessage"]),
                parent_commit=parent_revision,
            )
            commit_oid = str(getattr(commit, "oid", None) or "")
            commit_record = _upload_commit_payload(
                attempt,
                commit_oid=commit_oid,
            )
            _write_once_upload_record(
                commit_path,
                commit_record,
                schema=UPLOAD_COMMIT_SCHEMA_VERSION,
                digest_field="uploadCommitSHA256",
            )
            final_info = api.repo_info(repo_id=repo_id, repo_type="model")
            if (
                getattr(final_info, "sha", None) != commit_oid
                or bool(final_info.private) != private
            ):
                raise RuntimeError(
                    "Remote upload head or visibility failed post-commit verification"
                )
            remote_verification = "atomic_create_commit_head"
        final_revision = getattr(final_info, "sha", None)
    result: dict[str, Any] = {
        "schema": UPLOAD_SCHEMA_VERSION,
        "repository": repo_id,
        "private": bool(final_info.private),
        "headRevision": final_revision,
        "parentRevision": parent_revision,
        "runID": run_id,
        **publication,
        "remotePrefix": prefix,
        "ggufIncluded": include_gguf,
        "summaryStatus": summary["status"],
        "uploadedFileCount": len(remote_files),
        "uploadedPaths": remote_paths,
        "commitOID": commit_oid,
        "uploadIntentSHA256": intent["uploadIntentSHA256"],
        "uploadAttemptSHA256": attempt["uploadAttemptSHA256"],
        "remoteVerification": remote_verification,
        **image_source_fields,
    }
    result["uploadSHA256"] = canonical_sha256(result)
    write_object(receipt_path, result)
    receipt_path.chmod(0o600)
    _fsync_directory(receipt_path.parent, label="upload receipt")
    _cleanup_upload_transaction_records(transaction_paths)
    return result


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-source", type=Path, required=True)
    parser.add_argument("--agents", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--container-digest", required=True)


def _execution_plan_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evaluation-scope",
        choices=("full", "smoke", "none"),
        required=True,
    )
    parser.add_argument("--evaluation-max-examples", type=int)
    parser.add_argument(
        "--gguf-requested",
        action=argparse.BooleanOptionalAction,
        required=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed Ubuntu training pipeline helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static-preflight")
    _common_parser(static)
    _execution_plan_parser(static)
    static.add_argument("--run-root", type=Path, required=True)
    static.add_argument("--allowed-run-parent", type=Path, required=True)
    static.add_argument("--run-id")
    static.add_argument("--precreated-bind-root", action="store_true")
    runtime = subparsers.add_parser("runtime-preflight")
    _common_parser(runtime)
    prepare = subparsers.add_parser("prepare")
    _common_parser(prepare)
    _execution_plan_parser(prepare)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--precreated-bind-root", action="store_true")
    recover_prepare = subparsers.add_parser("recover-incomplete-preparation")
    _common_parser(recover_prepare)
    _execution_plan_parser(recover_prepare)
    recover_prepare.add_argument("--run-root", type=Path, required=True)
    recover_prepare.add_argument("--allowed-run-parent", type=Path, required=True)
    recover_prepare.add_argument("--precreated-bind-root", action="store_true")
    validate = subparsers.add_parser("validate-prepared-runtime")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--agents", required=True)
    validate.add_argument("--variant", required=True)
    validate.add_argument("--container-digest", required=True)
    _execution_plan_parser(validate)
    tokenizer_preflight = subparsers.add_parser("global-tokenizer-preflight")
    tokenizer_preflight.add_argument("--run-root", type=Path, required=True)
    tokenizer_preflight.add_argument("--agents", required=True)
    owned = subparsers.add_parser("verify-owned-run")
    owned.add_argument("--run-root", type=Path, required=True)
    owned.add_argument("--variant", required=True)
    reset_owned = subparsers.add_parser("reset-owned-run-root")
    reset_owned.add_argument("--run-root", type=Path, required=True)
    reset_owned.add_argument("--variant", required=True)
    initialize_root = subparsers.add_parser("initialize-bind-root")
    initialize_root.add_argument("--run-root", type=Path, required=True)
    initialize_root.add_argument("--allowed-run-parent", type=Path, required=True)
    initialize_root.add_argument("--create-if-missing", action="store_true")
    verify_root = subparsers.add_parser("verify-bind-root")
    verify_root.add_argument("--run-root", type=Path, required=True)
    verify_root.add_argument("--allowed-run-parent", type=Path, required=True)
    verify_root.add_argument("--expected-identity", required=True)
    verify_root.add_argument("--mounted-bind", action="store_true")
    verify = subparsers.add_parser("verify-phase")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--agent", choices=AGENTS, required=True)
    verify.add_argument("--phase", choices=("sft", "preference"), required=True)
    verify_evaluation_parser = subparsers.add_parser("verify-evaluation")
    verify_evaluation_parser.add_argument("--run-root", type=Path, required=True)
    verify_evaluation_parser.add_argument("--agent", choices=AGENTS, required=True)
    classify_evaluation_parser = subparsers.add_parser(
        "classify-completed-evaluation"
    )
    classify_evaluation_parser.add_argument(
        "--run-root", type=Path, required=True
    )
    classify_evaluation_parser.add_argument(
        "--agent", choices=AGENTS, required=True
    )
    verify_gguf_parser = subparsers.add_parser("verify-gguf")
    verify_gguf_parser.add_argument("--run-root", type=Path, required=True)
    verify_gguf_parser.add_argument("--agent", choices=AGENTS, required=True)
    verify_gguf_file_parser = subparsers.add_parser("verify-gguf-file")
    verify_gguf_file_parser.add_argument("--run-root", type=Path, required=True)
    verify_gguf_file_parser.add_argument("--path", type=Path, required=True)
    write_gguf_receipt_parser = subparsers.add_parser(
        "write-gguf-conversion-receipt"
    )
    write_gguf_receipt_parser.add_argument("--run-root", type=Path, required=True)
    write_gguf_receipt_parser.add_argument(
        "--agent", choices=AGENTS, required=True
    )
    write_gguf_receipt_parser.add_argument(
        "--staging-path", type=Path, required=True
    )
    install_gguf_file_parser = subparsers.add_parser("install-gguf-file")
    install_gguf_file_parser.add_argument("--run-root", type=Path, required=True)
    install_gguf_file_parser.add_argument("--agent", choices=AGENTS, required=True)
    install_gguf_file_parser.add_argument(
        "--staging-path", type=Path, required=True
    )
    clean = subparsers.add_parser("clean-phase")
    clean.add_argument("--run-root", type=Path, required=True)
    clean.add_argument("--agent", choices=AGENTS, required=True)
    clean.add_argument("--phase", choices=("sft", "preference"), required=True)
    final_config = subparsers.add_parser("write-final-config")
    final_config.add_argument("--run-root", type=Path, required=True)
    final_config.add_argument("--agent", choices=AGENTS, required=True)
    summary = subparsers.add_parser("write-summary")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--agents", required=True)
    summary.add_argument("--variant", required=True)
    summary.add_argument("--preference", action="store_true")
    summary.add_argument("--require-gguf", action="store_true")
    summary.add_argument("--require-evaluation", action="store_true")
    postcondition = subparsers.add_parser("verify-container-postcondition")
    postcondition.add_argument("--root", type=Path, required=True)
    postcondition.add_argument("--run-root", type=Path, required=True)
    postcondition.add_argument("--agents", required=True)
    postcondition.add_argument("--variant", required=True)
    postcondition.add_argument("--container-digest", required=True)
    postcondition.add_argument("--prepare-only", action="store_true")
    _execution_plan_parser(postcondition)
    upload = subparsers.add_parser("upload")
    upload.add_argument("--run-root", type=Path, required=True)
    upload.add_argument("--agents", required=True)
    upload.add_argument("--run-id", required=True)
    upload.add_argument("--public", action="store_true")
    upload.add_argument("--include-gguf", action="store_true")
    upload.add_argument("--allow-diagnostic-upload", action="store_true")
    upload.add_argument("--token-file", type=Path, required=True)
    upload.add_argument("--receipt-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(args, "agents"):
        agents = parse_agents(args.agents)
    resolved_run_root: Path | None = None
    if hasattr(args, "run_root") and args.command != "static-preflight":
        raw_run_root = args.run_root.expanduser()
        if raw_run_root.is_symlink():
            raise RuntimeError(f"Run root must not be a symlink: {raw_run_root}")
        resolved_run_root = raw_run_root.resolve()
    if args.command == "static-preflight":
        result = static_preflight(
            root=args.root.resolve(),
            dataset_source=args.dataset_source.resolve(),
            agents=agents,
            variant=args.variant,
            seed=args.seed,
            base_model_override=args.base_model,
            container_digest=args.container_digest,
            run_root=args.run_root,
            allowed_parent=args.allowed_run_parent,
            expected_run_id=args.run_id,
            evaluation_scope=args.evaluation_scope,
            evaluation_max_examples=args.evaluation_max_examples,
            gguf_requested=args.gguf_requested,
            precreated_bind_root=args.precreated_bind_root,
        )
    elif args.command == "runtime-preflight":
        result = runtime_preflight(
            root=args.root.resolve(),
            dataset_source=args.dataset_source.resolve(),
            agent=agents[0],
            variant=args.variant,
            seed=args.seed,
            base_model_override=args.base_model,
            container_digest=args.container_digest,
        )
    elif args.command == "prepare":
        result = prepare_run(
            root=args.root.resolve(),
            dataset_source=args.dataset_source.resolve(),
            run_root=resolved_run_root,
            agents=agents,
            variant=args.variant,
            seed=args.seed,
            base_model_override=args.base_model,
            container_digest=args.container_digest,
            evaluation_scope=args.evaluation_scope,
            evaluation_max_examples=args.evaluation_max_examples,
            gguf_requested=args.gguf_requested,
            precreated_bind_root=args.precreated_bind_root,
        )
    elif args.command == "recover-incomplete-preparation":
        result = recover_incomplete_preparation(
            root=args.root.resolve(),
            dataset_source=args.dataset_source.resolve(),
            run_root=resolved_run_root,
            allowed_parent=args.allowed_run_parent,
            agents=agents,
            variant=args.variant,
            seed=args.seed,
            base_model_override=args.base_model,
            container_digest=args.container_digest,
            evaluation_scope=args.evaluation_scope,
            evaluation_max_examples=args.evaluation_max_examples,
            gguf_requested=args.gguf_requested,
            precreated_bind_root=args.precreated_bind_root,
        )
    elif args.command == "validate-prepared-runtime":
        result = validate_prepared_runtime(
            root=args.root.resolve(),
            run_root=resolved_run_root,
            agents=agents,
            variant=args.variant,
            container_digest=args.container_digest,
            evaluation_scope=args.evaluation_scope,
            evaluation_max_examples=args.evaluation_max_examples,
            gguf_requested=args.gguf_requested,
        )
    elif args.command == "global-tokenizer-preflight":
        result = global_tokenizer_preflight(
            run_root=resolved_run_root,
            agents=agents,
        )
    elif args.command == "verify-owned-run":
        result = verify_owned_run(
            resolved_run_root,
            variant=args.variant,
        )
    elif args.command == "reset-owned-run-root":
        result = reset_owned_run_root(
            resolved_run_root,
            variant=args.variant,
        )
    elif args.command == "initialize-bind-root":
        result = initialize_bind_root(
            resolved_run_root,
            allowed_parent=args.allowed_run_parent,
            create_if_missing=args.create_if_missing,
        )
    elif args.command == "verify-bind-root":
        result = verify_bind_root(
            resolved_run_root,
            allowed_parent=args.allowed_run_parent,
            expected_identity=args.expected_identity,
            mounted_bind=args.mounted_bind,
        )
    elif args.command == "verify-phase":
        result = (
            verify_sft(resolved_run_root, args.agent)
            if args.phase == "sft"
            else verify_preference(resolved_run_root, args.agent)
        )
    elif args.command == "verify-evaluation":
        result = verify_evaluation(resolved_run_root, args.agent)
    elif args.command == "classify-completed-evaluation":
        result = classify_completed_evaluation(
            resolved_run_root,
            args.agent,
        )
    elif args.command == "verify-gguf":
        result = verify_gguf(resolved_run_root, args.agent)
    elif args.command == "verify-gguf-file":
        result = verify_gguf_file(
            resolved_run_root,
            args.path.expanduser().resolve(),
        )
    elif args.command == "write-gguf-conversion-receipt":
        staging_path = args.staging_path.expanduser()
        if not staging_path.is_absolute():
            raise RuntimeError("Staged GGUF path must be absolute")
        result = write_gguf_conversion_receipt(
            resolved_run_root,
            args.agent,
            staging_path,
        )
    elif args.command == "install-gguf-file":
        staging_path = args.staging_path.expanduser()
        if not staging_path.is_absolute():
            raise RuntimeError("Staged GGUF path must be absolute")
        result = install_gguf_file(
            resolved_run_root,
            args.agent,
            staging_path,
        )
    elif args.command == "clean-phase":
        clean_phase(resolved_run_root, args.agent, args.phase)
        result = {"status": "cleaned", "agent": args.agent, "phase": args.phase}
    elif args.command == "write-final-config":
        result = write_final_config(resolved_run_root, args.agent)
    elif args.command == "write-summary":
        result = write_summary(
            run_root=resolved_run_root,
            agents=agents,
            variant=args.variant,
            preference=args.preference,
            require_gguf=args.require_gguf,
            require_evaluation=args.require_evaluation,
        )
    elif args.command == "verify-container-postcondition":
        if args.prepare_only:
            input_closure = _acquire_prepared_input_closure(resolved_run_root)
            try:
                prepared = validate_prepared_runtime(
                    root=args.root.resolve(),
                    run_root=resolved_run_root,
                    agents=agents,
                    variant=args.variant,
                    container_digest=args.container_digest,
                    evaluation_scope=args.evaluation_scope,
                    evaluation_max_examples=args.evaluation_max_examples,
                    gguf_requested=args.gguf_requested,
                    observe_runtime=False,
                )
                global_preflight = _verified_prepared_global_tokenizer_preflight(
                    run_root=resolved_run_root,
                    agents=agents,
                )
                runtime_binding_smoke = (
                    _verified_runtime_binding_smoke_summary_evidence(
                        resolved_run_root,
                        agents,
                    )
                )
                input_closure.verify_unchanged()
                input_closure_sha256 = input_closure.inventory_sha256
                input_closure_entry_count = len(input_closure.inventory)
                input_mount_identity_sha256 = (
                    input_closure.mount_identity_sha256
                )
            finally:
                input_closure.close()
            result = {
                "status": "prepared_postcondition_verified",
                "prepareInputMountStatus": "exact_readonly_mount_verified",
                "prepareInputMountIdentitySHA256": input_mount_identity_sha256,
                "prepareInputClosureSHA256": input_closure_sha256,
                "prepareInputClosureEntryCount": input_closure_entry_count,
                "trainingEnvironmentSHA256": prepared[
                    "trainingEnvironmentSHA256"
                ],
                "observedAccelerator": prepared["observedAccelerator"],
                "globalPreflightSHA256": global_preflight[
                    "globalPreflightSHA256"
                ],
                "runtimeBindingSmokeGateSHA256": runtime_binding_smoke[
                    "runtimeBindingSmokeGateSHA256"
                ],
                "tokenizerClosureSHA256": global_preflight[
                    "tokenizerClosure"
                ]["tokenizerClosureSHA256"],
            }
        else:
            prepared = validate_prepared_runtime(
                root=args.root.resolve(),
                run_root=resolved_run_root,
                agents=agents,
                variant=args.variant,
                container_digest=args.container_digest,
                evaluation_scope=args.evaluation_scope,
                evaluation_max_examples=args.evaluation_max_examples,
                gguf_requested=args.gguf_requested,
                observe_runtime=False,
            )
            completed = _verified_completed_summary(resolved_run_root, agents)
            result = {
                "status": "completed_postcondition_verified",
                "trainingEnvironmentSHA256": prepared[
                    "trainingEnvironmentSHA256"
                ],
                "summarySHA256": completed["summarySHA256"],
                "summaryStatus": completed["status"],
                "evaluationStatus": completed["evaluationStatus"],
                "ggufStatus": completed["ggufStatus"],
                "ggufConversionStatus": completed["ggufConversionStatus"],
                "ggufTensorEquivalenceStatus": completed[
                    "ggufTensorEquivalenceStatus"
                ],
            }
    elif args.command == "upload":
        result = upload_run(
            run_root=resolved_run_root,
            agents=agents,
            run_id=args.run_id,
            private=not args.public,
            include_gguf=args.include_gguf,
            token_file=args.token_file,
            allow_diagnostic_upload=args.allow_diagnostic_upload,
            receipt_path=args.receipt_path,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise RuntimeError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ubuntu pipeline error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
