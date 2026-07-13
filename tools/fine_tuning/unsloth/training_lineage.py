from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


TRAINING_CODE_MANIFEST_SCHEMA_VERSION = "lumen.training-code-manifest/1.0.0"
TRAINING_CODE_BUNDLE_SCHEMA_VERSION = "lumen.training-code-bundle/1.0.0"
TRAINING_DEPENDENCY_LOCK_SCHEMA_VERSION = (
    "lumen.adapter-training-dependency-lock/1.0.0"
)

DEFAULT_PYTHON_VERSION = "3.10"
DEFAULT_CUDA_VERSION = "12.8"
DEFAULT_UNSLOTH_REVISION = "935474c20aabc2aadb1da17338959c7c6f9bdafe"
DEFAULT_LLAMA_CPP_REVISION = "34558825a27f4d74dcfd7a91bfde4464baa2a30a"
DEFAULT_PACKAGE_VERSIONS: dict[str, str] = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "datasets": "4.3.0",
    "gradio": "6.20.0",
    "hf_transfer": "0.1.9",
    "huggingface_hub": "1.23.0",
    "peft": "0.19.1",
    "protobuf": "7.35.1",
    "sentencepiece": "0.2.2",
    "spaces": "0.51.0",
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "torchvision": "0.23.0",
    "trackio": "0.30.3",
    "transformers": "4.57.6",
    "trl": "0.24.0",
    "unsloth_zoo": "2026.7.2",
}

_REQUIREMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PHASES = ("sft", "dpo", "orpo")


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


def build_training_code_manifest(
    *,
    phase: str,
    files: Mapping[str, Path],
) -> dict[str, Any]:
    if phase not in _PHASES:
        raise ValueError(f"Unsupported training phase: {phase}")
    if not files:
        raise ValueError("Training-code manifest must contain at least one file")

    entries: list[dict[str, Any]] = []
    for logical_path, source_path in sorted(files.items()):
        logical = _safe_logical_path(logical_path)
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
    if (
        manifest.get("schemaVersion") != TRAINING_CODE_MANIFEST_SCHEMA_VERSION
        or phase not in _PHASES
        or not isinstance(files, list)
        or not files
    ):
        raise ValueError("Invalid training-code manifest contract")

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

    if normalized != sorted(normalized, key=lambda item: item["path"]):
        raise ValueError("Training-code manifest files must be sorted")
    payload = {
        "schemaVersion": TRAINING_CODE_MANIFEST_SCHEMA_VERSION,
        "phase": phase,
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
    common = {
        "adapter_artifact.py": root / "tools/fine_tuning/unsloth/adapter_artifact.py",
        "app.py": root / "tools/hf_zerogpu/space_template/app.py",
        "lumen_manifest_crawler/dataset/adapter_evaluation.py": root
        / "tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/adapter_evaluation.py",
        "lumen_manifest_crawler/dataset/adapter_export.py": root
        / "tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/adapter_export.py",
        "lumen_manifest_crawler/dataset/fine_tuning.py": root
        / "tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/fine_tuning.py",
        "lumen_manifest_crawler/dataset/public_adapter_eval_registry.py": root
        / "tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/public_adapter_eval_registry.py",
        "requirements.txt": root / "tools/hf_zerogpu/space_template/requirements.txt",
        "training_lineage.py": root / "tools/fine_tuning/unsloth/training_lineage.py",
    }
    sft = {
        **common,
        "lumen_train_sft.py": root / "tools/fine_tuning/unsloth/train_sft.py",
    }
    preference = {
        **common,
        "lumen_train_dpo.py": root / "tools/fine_tuning/unsloth/train_dpo.py",
        # train_dpo imports controlled environment and seed helpers from train_sft.
        "lumen_train_sft.py": root / "tools/fine_tuning/unsloth/train_sft.py",
    }
    return build_training_code_bundle(
        {
            "sft": build_training_code_manifest(phase="sft", files=sft),
            "dpo": build_training_code_manifest(phase="dpo", files=preference),
            "orpo": build_training_code_manifest(phase="orpo", files=preference),
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
    if installed_versions is not None and dict(installed_versions) != dict(packages):
        raise ValueError("Installed controlled package versions drifted from the lock")
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


def installed_controlled_package_versions(lock: Mapping[str, Any]) -> dict[str, str]:
    packages = lock.get("packageVersions")
    if not isinstance(packages, Mapping):
        raise ValueError("trainingDependencyLock.packageVersions must be an object")
    return {
        name: importlib_metadata.version(name)
        for name in sorted(packages)
    }


def validate_runtime_source(*, kind: Any, revision: Any) -> tuple[str, str]:
    if kind not in {"git", "huggingface_space"}:
        raise ValueError("runtimeSourceKind must be git or huggingface_space")
    if not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("runtimeSourceRevision must be a full lowercase commit SHA")
    return kind, revision
