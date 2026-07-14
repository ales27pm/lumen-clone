from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


TRAINING_CODE_MANIFEST_SCHEMA_VERSION = "lumen.training-code-manifest/2.0.0"
TRAINING_CODE_BUNDLE_SCHEMA_VERSION = "lumen.training-code-bundle/2.0.0"
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
        if status != RUNTIME_SOURCE_BINDING_LOCAL:
            raise ValueError(
                "Local Git runtime source binding status must be "
                "local_checkout_observed"
            )
        if method != RUNTIME_SOURCE_BINDING_LOCAL_METHOD:
            raise ValueError(
                "Local Git runtime source binding method must be "
                "git_head_plus_training_code_manifest"
            )

    return {
        field: value.get(field)
        for field in RUNTIME_SOURCE_AUDIT_FIELDS
    }
