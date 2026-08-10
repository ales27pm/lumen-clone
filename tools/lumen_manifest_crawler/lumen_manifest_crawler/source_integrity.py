from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


SOURCE_INTEGRITY_EXCLUDED_PREFIXES = (
    "generated/",
    "lumen_manifest_crawler/generated/",
    "tools/lumen_manifest_crawler/generated/",
)
SOURCE_INTEGRITY_EXCLUDED_PATHS = {
    "ios/Lumen/AgentBehaviorManifest.json",
    # `uv run` can create this package-local lock as a validation byproduct.
    # It is not the repository's controlled dependency lock.
    "tools/lumen_manifest_crawler/uv.lock",
}


def repository_working_tree_provenance(root: Path) -> tuple[str, bool]:
    """Return the canonical digest and dirty state for repository source inputs.

    The digest covers the current bytes and executable bit of every tracked or
    non-ignored untracked file except generated output trees and the synced app
    manifest. The implementation is dependency-free so the Xcode build can use
    exactly the same contract as the manifest crawler when it stamps Info.plist.
    """

    resolved_root = root.resolve()
    tracked = _git_path_list(resolved_root, ["ls-files", "-z"])
    untracked = _git_path_list(
        resolved_root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    changed = _git_path_list(
        resolved_root,
        ["diff", "--name-only", "-z", "HEAD", "--"],
    )

    candidates = sorted(
        path
        for path in set(tracked + untracked)
        if not source_integrity_path_is_excluded(path)
    )
    entries: list[dict[str, object]] = []
    for relative_path in candidates:
        path = resolved_root / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            entries.append({"path": relative_path, "state": "deleted"})
            continue

        if stat.S_ISLNK(metadata.st_mode):
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
            kind = "symlink"
        elif stat.S_ISREG(metadata.st_mode):
            payload = path.read_bytes()
            kind = "file"
        else:
            # A gitlink or other non-regular entry remains visible in the
            # canonical path set without following content outside this repo.
            payload = b""
            kind = "other"
        entries.append(
            {
                "path": relative_path,
                "kind": kind,
                "executable": bool(metadata.st_mode & stat.S_IXUSR),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    encoded = json.dumps(
        {
            "schema": "lumen.source-working-tree/1.0.0",
            "entries": entries,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    relevant_changes = {
        path
        for path in changed + untracked
        if not source_integrity_path_is_excluded(path)
    }
    return hashlib.sha256(encoded).hexdigest(), bool(relevant_changes)


def _git_path_list(root: Path, arguments: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in completed.stdout.split(b"\0")
        if value
    ]


def source_integrity_path_is_excluded(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized in SOURCE_INTEGRITY_EXCLUDED_PATHS or normalized.startswith(
        SOURCE_INTEGRITY_EXCLUDED_PREFIXES
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Lumen's canonical source working-tree attestation."
    )
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        digest, dirty = repository_working_tree_provenance(args.root)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"source working-tree attestation failed: {error}") from error
    print(f"{digest} {'true' if dirty else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
