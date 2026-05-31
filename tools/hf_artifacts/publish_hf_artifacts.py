#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HF artifact manifest must be a JSON object")
    return payload


def iter_artifacts(manifest: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    base = manifest.get("baseModel")
    if isinstance(base, dict):
        yield "baseModel", base
    embedding = manifest.get("embedding")
    if isinstance(embedding, dict):
        yield "embedding", embedding
    adapters = manifest.get("adapters")
    if isinstance(adapters, dict):
        for name, artifact in sorted(adapters.items()):
            if isinstance(artifact, dict):
                yield f"adapters.{name}", artifact
    release_bakes = manifest.get("releaseBakes")
    if isinstance(release_bakes, dict):
        for name, artifact in sorted(release_bakes.items()):
            if isinstance(artifact, dict):
                yield f"releaseBakes.{name}", artifact


def resolve_local_path(root: Path, artifact: dict[str, Any]) -> Path:
    local_path = artifact.get("localPath")
    if not isinstance(local_path, str) or not local_path.strip():
        raise ValueError(f"Artifact {artifact.get('id') or artifact} is missing localPath")
    path = Path(local_path).expanduser()
    return path if path.is_absolute() else root / path


def upload_file(repo_id: str, local_path: Path, path_in_repo: str, *, repo_type: str, token: str | None, dry_run: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
        "upload",
        repo_id,
        str(local_path),
        path_in_repo,
        "--repo-type",
        repo_type,
    ]
    if token:
        command.extend(["--token", token])
    print("$", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def build_resolved_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(manifest))
    for key, artifact in iter_artifacts(resolved):
        local = resolve_local_path(root, artifact)
        if not local.exists():
            if artifact.get("required", True):
                raise FileNotFoundError(f"Missing required artifact {key}: {local}")
            artifact["missing"] = True
            continue
        artifact["sizeBytes"] = local.stat().st_size
        artifact["sha256"] = sha256_file(local)
        artifact["downloadURL"] = f"https://huggingface.co/{artifact['repoId']}/resolve/main/{artifact['fileName']}?download=true"
    return resolved


def write_resolved_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Lumen model/adaptor artifacts to Hugging Face from a manifest.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root used to resolve relative localPath entries.")
    parser.add_argument("--manifest", type=Path, default=Path("tools/hf_artifacts/lumen_hf_artifact_manifest.template.json"), help="Artifact manifest JSON.")
    parser.add_argument("--resolved-manifest", type=Path, default=Path("generated/hf_artifacts/lumen_hf_artifact_manifest.resolved.json"), help="Output manifest with sha256, sizes, and download URLs.")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="Hugging Face repository type.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing the Hugging Face token.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print uploads without calling Hugging Face.")
    parser.add_argument("--skip-upload", action="store_true", help="Only write resolved manifest; do not upload artifacts.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    resolved_path = args.resolved_manifest if args.resolved_manifest.is_absolute() else root / args.resolved_manifest
    manifest = read_manifest(manifest_path)
    resolved = build_resolved_manifest(root, manifest)
    write_resolved_manifest(resolved_path, resolved)
    print(f"Wrote resolved manifest: {resolved_path}")

    if args.skip_upload:
        return 0

    token = os.environ.get(args.token_env)
    for key, artifact in iter_artifacts(resolved):
        if artifact.get("missing"):
            print(f"Skipping missing optional artifact: {key}")
            continue
        local = resolve_local_path(root, artifact)
        upload_file(
            str(artifact["repoId"]),
            local,
            str(artifact["fileName"]),
            repo_type=args.repo_type,
            token=token,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
