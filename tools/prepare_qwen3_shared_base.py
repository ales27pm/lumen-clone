#!/usr/bin/env python3
"""Prepare the Qwen3 shared base GGUF for Lumen's adapter runtime.

The adapter-first runtime expects this exact artifact:

  repo: ales27pm/lumen-qwen3-bootstrap-gguf
  file: lumen-qwen3-fast-shared-q4_k_m.gguf

This file is NOT a role-baked model and must NOT live in the adapter repo.
Use this utility to either:

1. download a compatible public Qwen3-1.7B Q4_K_M GGUF and rename it; or
2. export the base from Unsloth's Qwen3 1.7B base package; then
3. validate the GGUF magic/size; and
4. optionally upload it to Hugging Face.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

EXPECTED_FILE_NAME = "lumen-qwen3-fast-shared-q4_k_m.gguf"
DEFAULT_OUTPUT = Path("models/base_qwen3_fast") / EXPECTED_FILE_NAME
DEFAULT_SOURCE_REPO = "rippertnt/Qwen3-1.7B-Q4_K_M-GGUF"
DEFAULT_TARGET_REPO = "ales27pm/lumen-qwen3-bootstrap-gguf"
MIN_BYTES = 1_000_000_000


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(root: Path, path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else root / p


def validate_gguf(path: Path, *, min_bytes: int = MIN_BYTES) -> None:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise SystemExit(f"File is too small: {size} bytes; expected at least {min_bytes} bytes")
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"Not a GGUF file: {path} magic={magic!r}")
    print(f"OK: {path} ({size / 1024 / 1024 / 1024:.2f} GiB)")


def find_largest_gguf(directory: Path) -> Path:
    ggufs = sorted(directory.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if not ggufs:
        raise SystemExit(f"No GGUF files found in {directory}")
    return ggufs[0]


def download_public_source(root: Path, output: Path, source_repo: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as exc:
        raise SystemExit("Missing huggingface_hub. Install with: pip install -U huggingface_hub") from exc

    files = [f for f in list_repo_files(source_repo, repo_type="model") if f.lower().endswith(".gguf")]
    if not files:
        raise SystemExit(f"No GGUF files found in source repo: {source_repo}")
    q4 = [f for f in files if "q4_k_m" in f.lower()]
    selected = q4[0] if q4 else files[0]
    print(f"Downloading {source_repo}/{selected}")
    downloaded = Path(hf_hub_download(repo_id=source_repo, filename=selected, repo_type="model"))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, output)
    validate_gguf(output)
    return output


def export_unsloth_base(root: Path, output: Path, unsloth_model: str, max_seq_length: int) -> Path:
    script = root / "generated" / "scripts" / "export_qwen3_shared_base_unsloth.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    staging = root / "models" / "base_qwen3_fast" / "_unsloth_export"
    script.write_text(
        f'''
from pathlib import Path
from unsloth import FastLanguageModel

out = Path({str(staging)!r})
out.mkdir(parents=True, exist_ok=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={unsloth_model!r},
    max_seq_length={max_seq_length},
    dtype=None,
    load_in_4bit=True,
)
model.save_pretrained_gguf(str(out), tokenizer, quantization_method="q4_k_m")
print(out)
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(script)], cwd=root, check=True)
    source = find_largest_gguf(staging)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    validate_gguf(output)
    return output


def upload(output: Path, target_repo: str, private: bool) -> None:
    create = ["hf", "repo", "create", target_repo, "--type", "model", "--yes"]
    if private:
        create.append("--private")
    subprocess.run(create, check=False)
    subprocess.run(
        [
            "hf",
            "upload",
            target_repo,
            str(output),
            EXPECTED_FILE_NAME,
            "--repo-type",
            "model",
            "--commit-message",
            "Upload Lumen Qwen3 shared base GGUF",
        ],
        check=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Lumen's Qwen3 shared base GGUF.")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=["download", "unsloth", "validate"], default="download")
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--unsloth-model", default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = resolve(Path.cwd(), args.root).resolve()
    output = resolve(root, args.output)

    if args.method == "download":
        download_public_source(root, output, args.source_repo)
    elif args.method == "unsloth":
        export_unsloth_base(root, output, args.unsloth_model, args.max_seq_length)
    else:
        validate_gguf(output)

    if args.upload:
        upload(output, args.target_repo, args.private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
