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
from pathlib import Path, PurePosixPath
from typing import Any


SOURCE_INTEGRITY_SCHEMA = "lumen.ubuntu-source-integrity/1.0.0"
WORKING_TREE_SCHEMA = "lumen.git-working-tree/1.0.0"
ORCHESTRATION_MANIFEST_SCHEMA = "lumen.ubuntu-orchestration-code/1.0.0"
SOURCE_INTEGRITY_ENV = "LUMEN_UBUNTU_SOURCE_ATTESTATION_PATH"
DEFAULT_IMAGE_ATTESTATION_PATH = Path("/opt/lumen/ubuntu-source-integrity.json")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GIT_INDEX_ENTRY = re.compile(
    rb"(?P<mode>[0-7]{6}) (?P<object>[0-9a-f]{40,64}) (?P<stage>[0-3])\t(?P<path>.+)",
    re.DOTALL,
)

# This is intentionally a superset of the code imported by the Ubuntu pipeline.
# It binds the launchers, container recipe, uploader, trainers, evaluator, GGUF
# helper, crawler scoring package, and the ZeroGPU dependency/runtime sources.
ORCHESTRATION_EXACT_PATHS = frozenset(
    {
        "scripts/ubuntu_train_lumen_full_pipeline.sh",
        "scripts/ubuntu_train_lumen_adapters_aio.sh",
    }
)
ORCHESTRATION_PREFIXES = (
    "tools/fine_tuning/unsloth/",
    "tools/lumen_manifest_crawler/lumen_manifest_crawler/",
    "tools/hf_zerogpu/space_template/",
)
REQUIRED_ORCHESTRATION_PATHS = frozenset(
    {
        *ORCHESTRATION_EXACT_PATHS,
        "tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128",
        "tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128.dockerignore",
        "tools/fine_tuning/unsloth/evaluate_adapter.py",
        "tools/fine_tuning/unsloth/export_gguf.py",
        "tools/fine_tuning/unsloth/train_dpo.py",
        "tools/fine_tuning/unsloth/train_sft.py",
        "tools/fine_tuning/unsloth/training_lineage.py",
        "tools/fine_tuning/unsloth/ubuntu_pipeline.py",
        "tools/fine_tuning/unsloth/ubuntu_postcondition.py",
        "tools/fine_tuning/unsloth/ubuntu_source_integrity.py",
        "tools/fine_tuning/unsloth/ubuntu_uploader.py",
        "tools/hf_zerogpu/space_template/app.py",
        "tools/hf_zerogpu/space_template/requirements.txt",
    }
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> tuple[int, str]:
    if path.is_symlink():
        raise RuntimeError(f"Ubuntu orchestration source must not be a symlink: {path}")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to open Ubuntu orchestration source: {path}") from exc
    try:
        file_stat = os.fstat(descriptor)
        stability = (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_mode,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"Ubuntu orchestration source is not a regular file: {path}")
        digest = hashlib.sha256()
        offset = 0
        while True:
            chunk = os.pread(descriptor, 1 << 20, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
        current = os.fstat(descriptor)
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != stability:
            raise RuntimeError(f"Ubuntu orchestration source changed while hashing: {path}")
        return file_stat.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _git_output(root: Path, arguments: Sequence[str], *, binary: bool = False) -> Any:
    environment = dict(os.environ)
    blocked = {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    }
    for key in tuple(environment):
        if key in blocked or key.startswith("GIT_CONFIG_KEY_") or key.startswith(
            "GIT_CONFIG_VALUE_"
        ):
            environment.pop(key, None)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ],
            cwd=root,
            env=environment,
            stderr=subprocess.PIPE,
            text=not binary,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = f": {str(exc.stderr).strip()}"
        raise RuntimeError(f"Unable to inspect the repository source{detail}") from exc


def _git_head(root: Path) -> str:
    revision = str(_git_output(root, ("rev-parse", "--verify", "HEAD"))).strip()
    if _COMMIT.fullmatch(revision) is None:
        raise RuntimeError("Repository HEAD is not a full immutable commit SHA")
    return revision


def _tracked_entries(root: Path) -> list[dict[str, str]]:
    raw = bytes(_git_output(root, ("ls-files", "--stage", "-z"), binary=True))
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        match = _GIT_INDEX_ENTRY.fullmatch(item)
        if match is None or match.group("stage") != b"0":
            raise RuntimeError("Repository index contains an invalid or unmerged entry")
        try:
            path = match.group("path").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("Repository paths must be valid UTF-8") from exc
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or path in seen:
            raise RuntimeError(f"Repository index contains an unsafe path: {path}")
        seen.add(path)
        entries.append(
            {
                "path": path,
                "mode": match.group("mode").decode("ascii"),
                "gitObjectID": match.group("object").decode("ascii"),
            }
        )
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise RuntimeError("Repository index is empty")
    return entries


def _reject_hidden_index_state(root: Path) -> None:
    verbose = bytes(_git_output(root, ("ls-files", "-v", "-z"), binary=True))
    tagged = bytes(_git_output(root, ("ls-files", "-t", "-z"), binary=True))
    for item in verbose.split(b"\0"):
        if not item:
            continue
        if len(item) < 3 or item[1:2] != b" ":
            raise RuntimeError("Repository index visibility flags could not be inspected")
        if item[:1].islower():
            raise RuntimeError(
                "Ubuntu pipeline rejects assume-unchanged repository entries"
            )
    for item in tagged.split(b"\0"):
        if not item:
            continue
        if len(item) < 3 or item[1:2] != b" ":
            raise RuntimeError("Repository index visibility flags could not be inspected")
        if item[:1] == b"S":
            raise RuntimeError(
                "Ubuntu pipeline rejects skip-worktree repository entries"
            )


def require_clean_repository(root: Path) -> tuple[str, list[dict[str, str]]]:
    root = root.resolve()
    observed_root = Path(
        str(_git_output(root, ("rev-parse", "--show-toplevel"))).strip()
    ).resolve()
    if observed_root != root:
        raise RuntimeError(
            f"Ubuntu source root must be the exact Git worktree root: {root}"
        )
    _reject_hidden_index_state(root)
    revision = _git_head(root)
    status = bytes(
        _git_output(
            root,
            (
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ),
            binary=True,
        )
    )
    if status:
        raise RuntimeError(
            "Ubuntu pipeline requires a clean checkout with no staged, unstaged, "
            "or untracked files"
        )
    submodules = str(_git_output(root, ("submodule", "status", "--recursive")))
    invalid_submodules = [
        line for line in submodules.splitlines() if line and line[0] != " "
    ]
    if invalid_submodules:
        raise RuntimeError(
            "Ubuntu pipeline requires every submodule to be initialized at its exact "
            "recorded commit"
        )
    entries = _tracked_entries(root)
    return revision, entries


def _is_orchestration_path(path: str) -> bool:
    return path in ORCHESTRATION_EXACT_PATHS or path.startswith(
        ORCHESTRATION_PREFIXES
    )


def _discover_orchestration_paths(root: Path) -> set[str]:
    discovered = set(ORCHESTRATION_EXACT_PATHS)
    for prefix in ORCHESTRATION_PREFIXES:
        directory = root / prefix
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"Missing Ubuntu orchestration source directory: {directory}")
        for path in directory.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            discovered.add(relative)
    return discovered


def build_orchestration_manifest(
    root: Path,
    *,
    tracked_paths: set[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    paths = _discover_orchestration_paths(root)
    missing = sorted(REQUIRED_ORCHESTRATION_PATHS - paths)
    if missing:
        raise RuntimeError(
            "Ubuntu orchestration closure is incomplete: " + ", ".join(missing)
        )
    if tracked_paths is not None:
        untracked = sorted(paths - tracked_paths)
        if untracked:
            raise RuntimeError(
                "Ubuntu orchestration closure contains untracked or ignored files: "
                + ", ".join(untracked)
            )
        omitted = sorted(
            path for path in tracked_paths if _is_orchestration_path(path) and path not in paths
        )
        if omitted:
            raise RuntimeError(
                "Tracked Ubuntu orchestration files are missing from the checkout: "
                + ", ".join(omitted)
            )
    files: list[dict[str, Any]] = []
    for relative in sorted(paths):
        size, digest = _file_sha256(root / relative)
        files.append({"path": relative, "size": size, "sha256": digest})
    return {"schemaVersion": ORCHESTRATION_MANIFEST_SCHEMA, "files": files}


def verify_orchestration_against_index(
    root: Path,
    entries: Sequence[Mapping[str, str]],
    orchestration: Mapping[str, Any],
) -> None:
    by_path = {entry["path"]: entry for entry in entries}
    files = orchestration.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Ubuntu orchestration manifest files are unavailable")
    for item in files:
        relative = str(item["path"])
        index_entry = by_path.get(relative)
        if index_entry is None:
            raise RuntimeError(f"Ubuntu orchestration source is not tracked: {relative}")
        path = root / relative
        file_stat = path.stat(follow_symlinks=False)
        actual_mode = "100755" if file_stat.st_mode & 0o111 else "100644"
        if index_entry["mode"] != actual_mode:
            raise RuntimeError(
                f"Ubuntu orchestration source mode differs from the Git index: {relative}"
            )
        observed_object = str(
            _git_output(
                root,
                ("hash-object", "--no-filters", "--", relative),
            )
        ).strip()
        if observed_object != index_entry["gitObjectID"]:
            raise RuntimeError(
                f"Ubuntu orchestration source bytes differ from the staged Git blob: {relative}"
            )


def attest_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    revision, entries = require_clean_repository(root)
    tracked_paths = {entry["path"] for entry in entries}
    orchestration = build_orchestration_manifest(root, tracked_paths=tracked_paths)
    verify_orchestration_against_index(root, entries, orchestration)
    working_tree = {
        "schemaVersion": WORKING_TREE_SCHEMA,
        "baseCommit": revision,
        "entries": entries,
    }
    record: dict[str, Any] = {
        "schema": SOURCE_INTEGRITY_SCHEMA,
        "baseCommit": revision,
        "workingTreeDigest": canonical_sha256(working_tree),
        "dirtyState": False,
        "ubuntuOrchestrationCodeSHA256": canonical_sha256(orchestration),
        "orchestrationManifest": orchestration,
    }
    record["sourceIntegritySHA256"] = canonical_sha256(record)
    final_revision, final_entries = require_clean_repository(root)
    final_orchestration = build_orchestration_manifest(
        root,
        tracked_paths={entry["path"] for entry in final_entries},
    )
    verify_orchestration_against_index(root, final_entries, final_orchestration)
    if (
        final_revision != revision
        or final_entries != entries
        or final_orchestration != orchestration
    ):
        raise RuntimeError("Repository source changed while it was being attested")
    return record


def _require_record_shape(record: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "baseCommit",
        "workingTreeDigest",
        "dirtyState",
        "ubuntuOrchestrationCodeSHA256",
        "orchestrationManifest",
        "sourceIntegritySHA256",
    }
    if set(record) != expected_keys:
        raise RuntimeError("Ubuntu source-integrity record has an unexpected schema")
    orchestration = record.get("orchestrationManifest")
    if (
        not isinstance(orchestration, Mapping)
        or set(orchestration) != {"schemaVersion", "files"}
        or orchestration.get("schemaVersion") != ORCHESTRATION_MANIFEST_SCHEMA
        or not isinstance(orchestration.get("files"), list)
        or not orchestration["files"]
    ):
        raise RuntimeError("Ubuntu orchestration manifest has an unexpected schema")
    paths: list[str] = []
    for item in orchestration["files"]:
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise RuntimeError("Ubuntu orchestration manifest contains an invalid file")
        path = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        pure = PurePosixPath(path) if isinstance(path, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or not _is_orchestration_path(path)
            or type(size) is not int
            or size < 0
            or _SHA256.fullmatch(str(digest or "")) is None
        ):
            raise RuntimeError("Ubuntu orchestration manifest contains an unsafe file")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError("Ubuntu orchestration manifest paths must be sorted and unique")
    if not REQUIRED_ORCHESTRATION_PATHS.issubset(paths):
        raise RuntimeError("Ubuntu orchestration manifest omits required execution sources")
    orchestration_digest = canonical_sha256(orchestration)
    unsigned = dict(record)
    declared = unsigned.pop("sourceIntegritySHA256", None)
    if (
        record.get("schema") != SOURCE_INTEGRITY_SCHEMA
        or _COMMIT.fullmatch(str(record.get("baseCommit") or "")) is None
        or _SHA256.fullmatch(str(record.get("workingTreeDigest") or "")) is None
        or record.get("dirtyState") is not False
        or _SHA256.fullmatch(
            str(record.get("ubuntuOrchestrationCodeSHA256") or "")
        )
        is None
        or record.get("ubuntuOrchestrationCodeSHA256") != orchestration_digest
        or _SHA256.fullmatch(str(declared or "")) is None
        or canonical_sha256(unsigned) != declared
    ):
        raise RuntimeError("Ubuntu source-integrity record failed verification")
    return dict(record)


def validate_attestation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return _require_record_shape(record)


def build_image_attestation(
    root: Path,
    *,
    base_commit: str,
    working_tree_digest: str,
    expected_orchestration_digest: str,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(base_commit) is None:
        raise RuntimeError("Image source base commit is invalid")
    if _SHA256.fullmatch(working_tree_digest) is None:
        raise RuntimeError("Image working-tree digest is invalid")
    if _SHA256.fullmatch(expected_orchestration_digest) is None:
        raise RuntimeError("Expected image orchestration digest is invalid")
    orchestration = build_orchestration_manifest(root.resolve())
    actual = canonical_sha256(orchestration)
    if actual != expected_orchestration_digest:
        raise RuntimeError("Image-baked Ubuntu orchestration closure drifted")
    record: dict[str, Any] = {
        "schema": SOURCE_INTEGRITY_SCHEMA,
        "baseCommit": base_commit,
        "workingTreeDigest": working_tree_digest,
        "dirtyState": False,
        "ubuntuOrchestrationCodeSHA256": actual,
        "orchestrationManifest": orchestration,
    }
    record["sourceIntegritySHA256"] = canonical_sha256(record)
    return record


def verify_snapshot_attestation(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    verified = _require_record_shape(record)
    orchestration = build_orchestration_manifest(root.resolve())
    if (
        orchestration != verified["orchestrationManifest"]
        or canonical_sha256(orchestration)
        != verified["ubuntuOrchestrationCodeSHA256"]
    ):
        raise RuntimeError("Image-baked Ubuntu source no longer matches its attestation")
    return verified


def load_verified_attestation(
    root: Path,
    record_path: Path | None = None,
) -> dict[str, Any]:
    path = record_path or Path(
        os.environ.get(SOURCE_INTEGRITY_ENV, str(DEFAULT_IMAGE_ATTESTATION_PATH))
    )
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Missing regular Ubuntu source-integrity record: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read Ubuntu source-integrity record: {path}") from exc
    if not isinstance(record, dict):
        raise RuntimeError("Ubuntu source-integrity record must be a JSON object")
    return verify_snapshot_attestation(root, record)


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Short write while creating source-integrity record")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attest Ubuntu pipeline source closure")
    subparsers = parser.add_subparsers(dest="command", required=True)
    host = subparsers.add_parser("attest-host")
    host.add_argument("--root", type=Path, required=True)
    image = subparsers.add_parser("build-image-record")
    image.add_argument("--root", type=Path, required=True)
    image.add_argument("--base-commit", required=True)
    image.add_argument("--working-tree-digest", required=True)
    image.add_argument("--orchestration-digest", required=True)
    image.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-image")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--record", type=Path, required=True)
    verify.add_argument("--base-commit", required=True)
    verify.add_argument("--working-tree-digest", required=True)
    verify.add_argument("--orchestration-digest", required=True)
    verify.add_argument("--source-integrity-digest", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "attest-host":
        record = attest_repository(args.root)
    elif args.command == "build-image-record":
        record = build_image_attestation(
            args.root,
            base_commit=args.base_commit,
            working_tree_digest=args.working_tree_digest,
            expected_orchestration_digest=args.orchestration_digest,
        )
        _write_record(args.output, record)
    else:
        record = load_verified_attestation(args.root, args.record)
        expected = {
            "baseCommit": args.base_commit,
            "workingTreeDigest": args.working_tree_digest,
            "ubuntuOrchestrationCodeSHA256": args.orchestration_digest,
            "sourceIntegritySHA256": args.source_integrity_digest,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Image source attestation does not match the clean host source")
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ubuntu source-integrity error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
