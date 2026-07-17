from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from tools.fine_tuning.unsloth.training_lineage import (
    DEFAULT_LLAMA_CPP_REVISION,
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
GGUF_FIXED_HEADER_SIZE = 24
GGUF_SUPPORTED_VERSIONS = frozenset({2, 3})
GGUF_READER_RELATIVE_PATH = Path("gguf-py/gguf/scripts/gguf_dump.py")
GGUF_READER_TIMEOUT_SECONDS = 120
SUMMARY_SCHEMA_VERSION = "lumen.ubuntu-training-summary/3.0.0"
UPLOAD_SCHEMA_VERSION = "lumen.ubuntu-training-upload/2.0.0"
EXECUTION_PLAN_SCHEMA_VERSION = "lumen.ubuntu-training-execution-plan/1.0.0"
RUN_SCHEMA_VERSION = "lumen.ubuntu-training-run/3.0.0"
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


def _verified_pinned_gguf_reader_script(
    run_root: Path,
) -> _VerifiedGGUFReaderScript:
    checkout = run_root / "llama.cpp"
    reader_script = checkout / GGUF_READER_RELATIVE_PATH
    if checkout.is_symlink() or not checkout.is_dir():
        raise RuntimeError(
            f"Missing regular pinned llama.cpp checkout for GGUF verification: {checkout}"
        )
    if reader_script.is_symlink() or not reader_script.is_file():
        raise RuntimeError(
            f"Missing regular pinned llama.cpp GGUF reader: {reader_script}"
        )
    if checkout.resolve() not in reader_script.resolve().parents:
        raise RuntimeError(f"Pinned GGUF reader escapes its checkout: {reader_script}")
    head = _git_output(checkout, "rev-parse", "HEAD")
    if head != DEFAULT_LLAMA_CPP_REVISION:
        raise RuntimeError("llama.cpp GGUF reader revision drifted")
    if _git_output(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        raise RuntimeError("llama.cpp GGUF reader checkout is dirty")
    relative_reader = GGUF_READER_RELATIVE_PATH.as_posix()
    expected_blob = _git_output(
        checkout,
        "rev-parse",
        f"HEAD:{relative_reader}",
    )
    if re.fullmatch(r"[0-9a-f]{40}", expected_blob) is None:
        raise RuntimeError("llama.cpp GGUF reader has an invalid pinned blob identity")
    return _VerifiedGGUFReaderScript(
        path=reader_script,
        git_blob_sha1=expected_blob,
    )


def _verify_gguf_with_reader(
    path: Path,
    *,
    artifact_handle: BinaryIO,
    reader_script: Path | _VerifiedGGUFReaderScript,
    tensor_count: int,
    metadata_kv_count: int,
) -> None:
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
        _verify_gguf_with_reader(
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


def validate_variant(
    source_root: Path,
    *,
    agent: str,
    variant: str,
    seed: int,
    base_model_override: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    agent_root = source_root / agent
    variant_root = agent_root / "experiments" / variant
    for filename in DATASET_FILES:
        if not (variant_root / filename).is_file():
            raise RuntimeError(f"Missing controlled dataset: {variant_root / filename}")
    manifest_path = variant_root / "variant_manifest.json"
    config_path = agent_root / "unsloth_config.json"
    manifest = read_object(manifest_path)
    config = read_object(config_path)
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
    unexpected = set(config) - set(controlled) - NON_CONTROLLED_CONFIG_FIELDS
    if (
        manifest.get("trainingConfigSHA256") != canonical_sha256(dict(controlled))
        or any(config.get(key) != value for key, value in controlled.items())
        or unexpected
    ):
        detail = f" unexpected fields: {', '.join(sorted(unexpected))}" if unexpected else ""
        raise RuntimeError(f"Generated config drifted from the controlled variant:{detail} {config_path}")
    if config.get("agent") != agent:
        raise RuntimeError(f"Generated config agent mismatch: {config_path}")
    if manifest.get("seed") != seed:
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
) -> dict[str, Any]:
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
    if not dataset_source.is_dir():
        raise RuntimeError(f"Dataset source does not exist: {dataset_source}")
    runtime_manifest = read_object(dataset_source / "adapter_runtime_manifest.json")
    read_object(
        root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
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
        checked.append(
            {
                "agent": agent,
                "variantManifestSHA256": manifest["variantManifestSHA256"],
                "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
            }
        )
    return {
        "schema": "lumen.ubuntu-training-static-preflight/1.0.0",
        "status": "static_ready",
        "trainingReady": False,
        "unchecked": ["python_environment", "cuda_runtime", "accelerator", "network"],
        "variant": variant,
        "agents": checked,
        "runRoot": str(run_root.resolve()),
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
    controlled = manifest.get("controlledTrainingConfig")
    if not isinstance(controlled, Mapping):
        raise RuntimeError("Variant manifest lacks controlled training config")
    effective = {key: config.get(key) for key in controlled}
    if effective != dict(controlled):
        raise RuntimeError("Effective training config drifted from the controlled variant")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise RuntimeError("Variant manifest lacks dataset lineage")
    return {
        "schema": "lumen.training-variant-attestation/1.0.0",
        "variant": manifest["variant"],
        "variantManifestSHA256": manifest["variantManifestSHA256"],
        "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
        "laneHashes": {
            name: contract["sha256"]
            for name, contract in sorted(datasets.items())
            if isinstance(contract, Mapping) and isinstance(contract.get("sha256"), str)
        },
        "effectiveTrainingConfigSHA256": canonical_sha256(effective),
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
        "trainingEnvironmentLockSHA256": manifest[
            "trainingEnvironmentLockSHA256"
        ],
        "trainingEnvironmentSHA256": config["trainingEnvironmentSHA256"],
        "trainingCodeSHA256": config["trainingCodeSHA256"],
        "trainingDependencyLockSHA256": config[
            "trainingDependencyLockSHA256"
        ],
        "requirementsSHA256": config["requirementsSHA256"],
        "runtimeImageBindingStatus": config[
            "trainingRuntimeImageBindingStatus"
        ],
        "runtimeImageBindingVerified": config[
            "trainingRuntimeImageBindingVerified"
        ],
        **{field: config[field] for field in RUNTIME_SOURCE_FIELDS},
        **{field: config[field] for field in UBUNTU_SOURCE_INTEGRITY_FIELDS},
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
) -> dict[str, Any]:
    if run_root.exists():
        raise RuntimeError(f"Run root already exists: {run_root}")
    source_config, _, _ = validate_variant(
        dataset_source,
        agent=agents[0],
        variant=variant,
        seed=seed,
        base_model_override=base_model_override,
    )
    source_integrity = current_source_integrity(root)
    runtime_lineage, runtime_environment = _runtime_lineage(
        root=root,
        source_config=source_config,
        container_digest=container_digest,
        source_integrity=source_integrity,
    )
    prepared_execution_plan = execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=evaluation_max_examples,
        gguf_requested=gguf_requested,
    )
    run_root.mkdir(parents=True)
    snapshot_root = run_root / "generated" / "fine_tuning"
    shutil.copytree(dataset_source, snapshot_root)
    behavior_manifest_source = (
        root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    )
    if not behavior_manifest_source.is_file():
        raise RuntimeError(
            f"Missing frozen behavior manifest: {behavior_manifest_source}"
        )
    behavior_manifest_snapshot = (
        run_root
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    behavior_manifest_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(behavior_manifest_source, behavior_manifest_snapshot)
    for directory in (
        run_root / "configs",
        run_root / "logs",
        run_root / "training",
        run_root / "models" / "lora_qwen3_bootstrap",
        run_root / "models" / "lora_qwen3_dpo",
        run_root / "models" / "lora_qwen3_gguf",
        run_root / "evaluation",
    ):
        directory.mkdir(parents=True, exist_ok=True)

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
        ):
            if config.get(field) != source_config.get(field):
                raise RuntimeError(f"Shared training lineage differs for {agent}: {field}")
        base_model = (
            base_model_override
            or str(base_by_agent.get(agent) or "")
            or str(config.get("base_model_name") or "")
        )
        adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
        preference_adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
        training_dir = run_root / "training" / agent
        gguf_path = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        config["base_model_name"] = base_model
        config["baseModelID"] = base_model
        config["trainingContainerImageDigest"] = container_digest
        config["trainingContainerImageDigestSource"] = "operator_declared"
        config["trainingRuntimeImageBindingStatus"] = "manual_validation_required"
        config["trainingRuntimeImageBindingVerified"] = False
        config["runExecutionPlan"] = prepared_execution_plan
        config.update(runtime_source)
        config.update(integrity_fields)
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
                "preferenceTrainer": config.get("preference_trainer", "dpo"),
                "preferenceAdapterDir": str(preference_adapter_dir),
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
        **runtime_source,
        **integrity_fields,
        "agents": prepared,
    }
    run_manifest["runManifestSHA256"] = canonical_sha256(run_manifest)
    write_object(run_root / "aio_run_manifest.json", run_manifest)
    write_object(run_root / "training_environment.json", runtime_environment)
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
    _verified_execution_plan(manifest.get("executionPlan"))
    if (
        manifest.get("schema") != RUN_SCHEMA_VERSION
        or manifest.get("adapterFirst") is not True
        or manifest.get("trainBaseModelWeights") is not False
        or manifest.get("runID") != run_root.name
        or manifest.get("runRoot") != str(run_root)
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
    }


def _reject_managed_symlinks(run_root: Path) -> None:
    for name in (
        "configs",
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
        "preferenceTrainer": preference_trainer,
        "preferenceAdapterDir": str(paths["dpo_output_dir"]),
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
) -> dict[str, Any]:
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
    seed = manifest.get("seed")
    if type(seed) is not int:
        raise RuntimeError("Prepared run manifest has an invalid seed")
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
        verify_embedded_source_integrity(prepared_config)
        _, pending_manifest, variant_root = validate_variant(
            snapshot_root,
            agent=agent,
            variant=variant,
            seed=seed,
            base_model_override=str(prepared_config.get("base_model_name") or ""),
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
            preference_trainer=str(
                prepared_config.get("preference_trainer", "dpo")
            ),
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
    }


def _verify_gguf_inventory(
    run_root: Path,
    agents: Sequence[str],
    *,
    require_all: bool,
) -> dict[str, Path]:
    gguf_dir = run_root / "models" / "lora_qwen3_gguf"
    if gguf_dir.is_symlink() or not gguf_dir.is_dir():
        raise RuntimeError(f"Missing regular GGUF artifact directory: {gguf_dir}")
    expected = {
        f"lumen-{agent}-lora.gguf": gguf_dir / f"lumen-{agent}-lora.gguf"
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
    observed = {entry.name for entry in entries}
    if require_all and observed != set(expected):
        missing = sorted(set(expected) - observed)
        raise RuntimeError(
            "GGUF artifact directory is missing required entries: "
            + ", ".join(missing)
        )
    return {name: path for name, path in expected.items() if name in observed}


def verify_gguf_file(run_root: Path, path: Path) -> dict[str, Any]:
    prepared_run = _verified_run_manifest(run_root)
    prepared_entries = prepared_run.get("agents")
    if not isinstance(prepared_entries, list) or any(
        not isinstance(item, Mapping) for item in prepared_entries
    ):
        raise RuntimeError("Prepared run lacks exact agent ownership")
    prepared_agents = tuple(str(item.get("agent") or "") for item in prepared_entries)
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
        or path.name
        not in {f"lumen-{agent}-lora.gguf" for agent in prepared_agents}
    ):
        raise RuntimeError(f"GGUF artifact is not owned by the prepared run: {path}")
    reader_script = _verified_pinned_gguf_reader_script(run_root)
    return verify_gguf_artifact(path, reader_script=reader_script)


def verify_gguf(run_root: Path, agent: str) -> dict[str, Any]:
    prepared_run = _verified_run_manifest(run_root)
    prepared_agent_entries = prepared_run.get("agents")
    if not isinstance(prepared_agent_entries, list) or any(
        not isinstance(item, Mapping) for item in prepared_agent_entries
    ):
        raise RuntimeError("Prepared run lacks exact agent ownership")
    prepared_agents = tuple(str(item["agent"]) for item in prepared_agent_entries)
    if agent not in prepared_agents:
        raise RuntimeError(f"Prepared run does not own agent {agent}")
    summary = _verified_completed_summary(run_root, prepared_agents)
    if summary.get("status") != "complete":
        raise RuntimeError(
            "Existing GGUF reuse requires a complete canonical training summary"
        )
    agent_summary = summary["agents"].get(agent)
    if not isinstance(agent_summary, Mapping):
        raise RuntimeError(f"Existing summary lacks agent {agent}")
    path = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
    expected_digest = agent_summary.get("adapterGGUFSHA256")
    expected_size = agent_summary.get("adapterGGUFSizeBytes")
    reader_script = _verified_pinned_gguf_reader_script(run_root)
    gguf = verify_gguf_artifact(path, reader_script=reader_script)
    if (
        type(expected_size) is not int
        or expected_size <= 0
        or gguf["adapterGGUFSizeBytes"] != expected_size
        or re.fullmatch(r"[0-9a-f]{64}", str(expected_digest or "")) is None
        or gguf["adapterGGUFSHA256"] != expected_digest
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


def _verify_training_report(
    path: Path,
    *,
    phase: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
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
    return report


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


def verify_sft(run_root: Path, agent: str) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.adapter_artifact import verify_adapter_artifact

    config = read_object(run_root / "configs" / f"{agent}.json")
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
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
    )
    _verify_training_report(
        report_path,
        phase="SFT",
        expected={
            "schema": "lumen.train_sft.manifest/1.0.0",
            "agent": agent,
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
        },
    )
    return {
        "phase": "sft",
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "report": str(report_path),
    }


def verify_preference(run_root: Path, agent: str) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.adapter_artifact import verify_adapter_artifact

    config = read_object(run_root / "configs" / f"{agent}.json")
    sft = verify_sft(run_root, agent)
    adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
    finalized_path = run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    report_path = run_root / "training" / agent / "dpo" / "dpo_report.json"
    finalized = _verify_manifest_integrity(finalized_path)
    trainer = str(config.get("preference_trainer", "dpo")).lower()
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
    )
    _verify_training_report(
        report_path,
        phase=trainer.upper(),
        expected={
            "agent": agent,
            "trainer": "ORPOTrainer" if trainer == "orpo" else "DPOTrainer",
            "training_phase": "sft_dpo",
            "seed": config.get("seed"),
            "variantManifestSHA256": config.get("variantManifestSHA256"),
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
        },
    )
    return {
        "phase": trainer,
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "parentSFTAdapterSHA256": sft["adapterSHA256"],
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "report": str(report_path),
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
    config = dict(base_config)
    finalized_path = (
        run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    )
    trainer = str(config.get("preference_trainer", "dpo")).lower()
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
    targets: list[Path] = [
        run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf",
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

    prepared_run = _verified_run_manifest(run_root)
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
    if (
        type(generated_case_count) is not int
        or generated_case_count <= 0
        or generated_case_count > len(evaluation_records)
        or type(evaluation_run.get("fullCaseCount")) is not int
        or evaluation_run.get("fullCaseCount") != len(evaluation_records)
        or complete_evaluation is not (generated_case_count == len(evaluation_records))
    ):
        raise RuntimeError(f"Evaluation case-count lineage failed verification: {run_path}")
    if complete_evaluation:
        selected_records = list(evaluation_records)
    else:
        try:
            selected_records = evaluate_adapter.select_evaluation_records(
                evaluation_records,
                max_examples=generated_case_count,
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
    structured_eligible = agent in evaluate_adapter.JSON_OUTPUT_AGENTS
    expected_generation_keys = {
        "doSample",
        "numBeams",
        "repetitionPenalty",
        "thinkingEnabled",
        "maxNewTokens",
        "maxSequenceLength",
        "seed",
        "structuredOutputContractEligible",
        "structuredOutputContractVersion",
        "structuredOutputContractSHA256",
        "strictJSONRetryEligible",
        "strictJSONMaxAttempts",
        "strictJSONRetryContractVersion",
        "strictJSONRetryContractSHA256",
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
        and (
            generation.get("structuredOutputContractEligible")
            is structured_eligible
            and generation.get("structuredOutputContractVersion")
            == evaluate_adapter.STRUCTURED_OUTPUT_CONTRACT_VERSION
            and generation.get("structuredOutputContractSHA256")
            == evaluate_adapter._structured_output_contract_sha256(
                agent,
                tool_contracts=tool_contracts,
            )
            and generation.get("strictJSONRetryEligible") is structured_eligible
            and type(generation.get("strictJSONMaxAttempts")) is int
            and generation.get("strictJSONMaxAttempts")
            == (
                evaluate_adapter.STRICT_JSON_MAX_ATTEMPTS
                if structured_eligible
                else 1
            )
            and generation.get("strictJSONRetryContractVersion")
            == evaluate_adapter.STRICT_JSON_RETRY_CONTRACT_VERSION
            and generation.get("strictJSONRetryContractSHA256")
            == hashlib.sha256(
                evaluate_adapter.STRICT_JSON_RETRY_INSTRUCTION.encode("utf-8")
            ).hexdigest()
            and generation.get("doSample") is False
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
    if evaluation_status not in {"quality_gate_passed", "smoke_complete"}:
        raise RuntimeError(f"Evaluation did not pass or complete a smoke run: {run_path}")
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


def _derived_summary_state(
    *,
    plan: Mapping[str, Any],
    evaluation_statuses: Sequence[str],
    agent_count: int,
    gguf_count: int,
) -> dict[str, Any]:
    if agent_count <= 0:
        raise RuntimeError("Summary state requires at least one prepared agent")
    verified_plan = _verified_execution_plan(plan)
    evaluation_scope = str(verified_plan["evaluationScope"])
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
    elif verified_plan["ggufRequested"] is False and gguf_count == 0:
        gguf_status = "skipped_by_operator"
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
        "evaluationStatus": evaluation_status,
        "evaluationScope": evaluation_scope,
        "ggufStatus": gguf_status,
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
    gguf_inventory = _verify_gguf_inventory(
        run_root,
        agents,
        require_all=require_gguf,
    )
    reader_script = (
        _verified_pinned_gguf_reader_script(run_root)
        if require_gguf or gguf_inventory
        else None
    )
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA_VERSION,
        "status": "pending_verification",
        "evaluationStatus": "pending_verification",
        "evaluationScope": "pending_verification",
        "ggufStatus": "pending_verification",
        "qualification": "pending_verification",
        "promotionEligible": False,
        "executionPlanSHA256": prepared_execution_plan["executionPlanSHA256"],
        "variant": variant,
        "runRoot": str(run_root),
        "preferenceTraining": preference,
        **{
            field: run_manifest[field]
            for field in UBUNTU_SOURCE_INTEGRITY_FIELDS
        },
        "agents": {},
    }
    for agent in agents:
        sft = verify_sft(run_root, agent)
        final_phase = verify_preference(run_root, agent) if preference else sft
        gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        gguf_metadata = (
            verify_gguf_artifact(gguf, reader_script=reader_script)
            if gguf.name in gguf_inventory and reader_script is not None
            else None
        )
        gguf_exists = gguf_metadata is not None
        if require_gguf and not gguf_exists:
            raise RuntimeError(f"Missing required GGUF adapter: {gguf}")
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
            "evaluationStatus",
            "evaluationScope",
            "ggufStatus",
            "qualification",
            "promotionEligible",
            "executionPlanSHA256",
            "variant",
            "runRoot",
            "preferenceTraining",
            *UBUNTU_SOURCE_INTEGRITY_FIELDS,
            "agents",
            "summarySHA256",
        }
        or summary.get("schema") != SUMMARY_SCHEMA_VERSION
        or summary.get("runRoot") != str(run_root)
        or summary.get("variant") != run_manifest.get("variant")
        or summary.get("preferenceTraining") is not True
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
        }
        or summary.get("evaluationStatus")
        not in {"quality_gate_passed", "smoke_complete", "not_run"}
        or summary.get("evaluationScope") not in {"full", "smoke", "none"}
        or summary.get("ggufStatus")
        not in {"verified", "skipped_by_operator"}
        or summary.get("qualification")
        not in {"quality_gate_passed", "diagnostic_only"}
        or type(summary.get("promotionEligible")) is not bool
        or summary.get("executionPlanSHA256")
        != prepared_execution_plan["executionPlanSHA256"]
        or not isinstance(summary_agents, Mapping)
        or set(summary_agents) != set(agents)
    ):
        raise RuntimeError("Completed Ubuntu training summary failed verification")
    gguf_inventory = _verify_gguf_inventory(
        run_root,
        agents,
        require_all=False,
    )
    reader_script = (
        _verified_pinned_gguf_reader_script(run_root)
        if gguf_inventory
        else None
    )
    evaluation_statuses: list[str] = []
    for agent in agents:
        item = summary_agents.get(agent)
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "sft",
                "finalPhase",
                "adapterGGUF",
                "adapterGGUFExists",
                "adapterGGUFSHA256",
                "adapterGGUFSizeBytes",
                "evaluationReport",
                "evaluationReportExists",
                "evaluation",
            }
        ):
            raise RuntimeError(f"Completed summary lacks agent {agent}")
        sft = verify_sft(run_root, agent)
        final_phase = verify_preference(run_root, agent)
        if item.get("sft") != sft or item.get("finalPhase") != final_phase:
            raise RuntimeError(f"Completed summary adapter lineage drifted for {agent}")
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
            evaluation_statuses.append(str(verified_evaluation.get("status")))
        elif evaluation_report_exists:
            raise RuntimeError(f"Completed summary evaluation flag drifted for {agent}")
        gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        gguf_metadata = (
            verify_gguf_artifact(gguf, reader_script=reader_script)
            if gguf.name in gguf_inventory and reader_script is not None
            else None
        )
        gguf_exists = gguf_metadata is not None
        if (
            item.get("adapterGGUF") != str(gguf)
            or item.get("adapterGGUFExists") is not gguf_exists
        ):
            raise RuntimeError(f"Completed summary GGUF flag drifted for {agent}")
        if gguf_exists:
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
            ):
                raise RuntimeError(f"Completed summary GGUF drifted for {agent}")
        elif (
            gguf.exists()
            or gguf.is_symlink()
            or item.get("adapterGGUFSizeBytes") != 0
            or item.get("adapterGGUFSHA256") is not None
        ):
            raise RuntimeError(f"Unbound or unsafe GGUF exists for {agent}")
    expected_state = _derived_summary_state(
        plan=prepared_execution_plan,
        evaluation_statuses=evaluation_statuses,
        agent_count=len(agents),
        gguf_count=len(gguf_inventory),
    )
    if any(summary.get(field) != value for field, value in expected_state.items()):
        raise RuntimeError("Completed summary state does not match its verified evidence")
    return summary


def _upload_publication_contract(
    summary: Mapping[str, Any],
    *,
    allow_diagnostic_upload: bool,
) -> dict[str, Any]:
    if type(allow_diagnostic_upload) is not bool:
        raise RuntimeError("Diagnostic upload override must be boolean")
    promotion_eligible = summary.get("promotionEligible") is True
    qualification = str(summary.get("qualification") or "")
    if promotion_eligible:
        if (
            qualification != "quality_gate_passed"
            or summary.get("evaluationStatus") != "quality_gate_passed"
            or summary.get("evaluationScope") != "full"
            or summary.get("status") not in {"complete", "complete_without_gguf"}
        ):
            raise RuntimeError("Upload summary has inconsistent qualification state")
        remote_namespace = "runs"
    else:
        if (
            qualification != "diagnostic_only"
            or summary.get("evaluationStatus")
            not in {"smoke_complete", "not_run"}
            or summary.get("evaluationScope") not in {"smoke", "none"}
            or summary.get("status")
            not in {"smoke_complete", "training_complete_without_full_evaluation"}
        ):
            raise RuntimeError("Upload summary has inconsistent diagnostic state")
        if not allow_diagnostic_upload:
            raise RuntimeError(
                "Diagnostic upload requires --allow-diagnostic-upload"
            )
        remote_namespace = "diagnostic-runs"
    return {
        "remoteNamespace": remote_namespace,
        "qualification": qualification,
        "promotionEligible": promotion_eligible,
        "diagnosticUploadOverrideApplied": not promotion_eligible,
        "evaluationStatus": summary["evaluationStatus"],
        "evaluationScope": summary["evaluationScope"],
        "ggufStatus": summary["ggufStatus"],
    }


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
        from huggingface_hub import CommitOperationAdd, HfApi
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

    local_files: list[tuple[Path, str]] = []
    for agent in agents:
        preference = verify_preference(run_root, agent)
        adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
        adapter_manifest = read_object(adapter_dir / "adapter_artifact_manifest.json")
        artifact_files = adapter_manifest.get("files")
        if not isinstance(artifact_files, list):
            raise RuntimeError(f"Adapter upload manifest is invalid for {agent}")
        for item in artifact_files:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                raise RuntimeError(f"Adapter upload manifest is invalid for {agent}")
            local_files.append(
                (
                    adapter_dir / item["path"],
                    f"{publication_root}/adapters/{agent}/{item['path']}",
                )
            )
        local_files.append(
            (
                adapter_dir / "adapter_artifact_manifest.json",
                f"{publication_root}/adapters/{agent}/adapter_artifact_manifest.json",
            )
        )
        local_files.append(
            (
                run_root
                / "training"
                / agent
                / "dpo"
                / "finalized_variant_manifest.json",
                f"{publication_root}/manifests/{agent}/variant_manifest.json",
            )
        )
        evaluation = summary["agents"][agent].get("evaluation")
        if isinstance(evaluation, Mapping):
            for filename in (
                "candidate_outputs.jsonl",
                "evaluation_report.json",
                "evaluation_run_manifest.json",
            ):
                local_files.append(
                    (
                        run_root / "evaluation" / agent / filename,
                        f"{publication_root}/evaluation/{agent}/{filename}",
                    )
                )
        if include_gguf:
            if summary["agents"][agent].get("adapterGGUFExists") is not True:
                raise RuntimeError(f"Upload requires a verified GGUF for {agent}")
            local_files.append(
                (
                    run_root
                    / "models"
                    / "lora_qwen3_gguf"
                    / f"lumen-{agent}-lora.gguf",
                    f"{publication_root}/gguf/lumen-{agent}-lora.gguf",
                )
            )
        if preference.get("adapterSHA256") != adapter_manifest.get("adapterSHA256"):
            raise RuntimeError(f"Upload adapter lineage drifted for {agent}")
    for filename in (
        "aio_run_manifest.json",
        "aio_summary.json",
        "training_environment.json",
    ):
        local_files.append((run_root / filename, f"{publication_root}/{filename}"))
    remote_paths = [remote for _, remote in local_files]
    if len(set(remote_paths)) != len(remote_paths):
        raise RuntimeError("Upload file contract contains duplicate remote paths")
    for local_path, _ in local_files:
        if local_path.is_symlink() or not local_path.is_file():
            raise RuntimeError(f"Upload input is not a regular verified file: {local_path}")
    receipt_path = (
        receipt_path.resolve()
        if receipt_path is not None
        else run_root / "upload_receipts.json"
    )
    if receipt_path.parent.is_symlink() or not receipt_path.parent.is_dir():
        raise RuntimeError("Upload receipt parent must be a regular directory")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError(f"Upload receipt path already exists: {receipt_path}")

    if token_file.is_symlink() or not token_file.is_file():
        raise RuntimeError("Upload token path is not a regular mounted secret")
    token = token_file.read_text(encoding="utf-8").strip()
    if not token or "\n" in token or "\r" in token:
        raise RuntimeError("Upload token file is empty or malformed")
    api = HfApi(token=token)
    identity = api.whoami()
    if not isinstance(identity, Mapping) or not identity.get("name"):
        raise RuntimeError("Hugging Face authentication preflight failed")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if bool(info.private) != private:
        raise RuntimeError("Remote repository visibility does not match the requested policy")
    existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    prefix = f"{publication_root}/"
    if any(path.startswith(prefix) for path in existing_files):
        raise RuntimeError(f"Remote run prefix already exists: {prefix}")
    parent_revision = getattr(info, "sha", None)
    if parent_revision is not None and re.fullmatch(
        r"[0-9a-f]{40}", str(parent_revision)
    ) is None:
        raise RuntimeError("Remote parent revision is not immutable")
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for local, remote in local_files
        ],
        commit_message=(
            f"Upload verified Lumen training run {run_id}"
            if promotion_eligible
            else f"Upload diagnostic-only Lumen training run {run_id}"
        ),
        parent_commit=parent_revision,
    )
    commit_oid = getattr(commit, "oid", None)
    if re.fullmatch(r"[0-9a-f]{40}", str(commit_oid or "")) is None:
        raise RuntimeError("Hugging Face upload did not return an immutable commit OID")
    final_info = api.repo_info(repo_id=repo_id, repo_type="model")
    final_revision = getattr(final_info, "sha", None)
    if final_revision != commit_oid or bool(final_info.private) != private:
        raise RuntimeError("Remote upload head or visibility failed post-commit verification")
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
        "uploadedFileCount": len(local_files),
        "uploadedPaths": remote_paths,
        "commitOID": commit_oid,
        **image_source_fields,
    }
    result["uploadSHA256"] = canonical_sha256(result)
    write_object(receipt_path, result)
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
    static.add_argument("--run-root", type=Path, required=True)
    static.add_argument("--allowed-run-parent", type=Path, required=True)
    static.add_argument("--run-id")
    runtime = subparsers.add_parser("runtime-preflight")
    _common_parser(runtime)
    prepare = subparsers.add_parser("prepare")
    _common_parser(prepare)
    _execution_plan_parser(prepare)
    prepare.add_argument("--run-root", type=Path, required=True)
    validate = subparsers.add_parser("validate-prepared-runtime")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--agents", required=True)
    validate.add_argument("--variant", required=True)
    validate.add_argument("--container-digest", required=True)
    _execution_plan_parser(validate)
    owned = subparsers.add_parser("verify-owned-run")
    owned.add_argument("--run-root", type=Path, required=True)
    owned.add_argument("--variant", required=True)
    verify = subparsers.add_parser("verify-phase")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--agent", choices=AGENTS, required=True)
    verify.add_argument("--phase", choices=("sft", "preference"), required=True)
    verify_gguf_parser = subparsers.add_parser("verify-gguf")
    verify_gguf_parser.add_argument("--run-root", type=Path, required=True)
    verify_gguf_parser.add_argument("--agent", choices=AGENTS, required=True)
    verify_gguf_file_parser = subparsers.add_parser("verify-gguf-file")
    verify_gguf_file_parser.add_argument("--run-root", type=Path, required=True)
    verify_gguf_file_parser.add_argument("--path", type=Path, required=True)
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
    elif args.command == "verify-owned-run":
        result = verify_owned_run(
            resolved_run_root,
            variant=args.variant,
        )
    elif args.command == "verify-phase":
        result = (
            verify_sft(resolved_run_root, args.agent)
            if args.phase == "sft"
            else verify_preference(resolved_run_root, args.agent)
        )
    elif args.command == "verify-gguf":
        result = verify_gguf(resolved_run_root, args.agent)
    elif args.command == "verify-gguf-file":
        result = verify_gguf_file(
            resolved_run_root,
            args.path.expanduser().resolve(),
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
