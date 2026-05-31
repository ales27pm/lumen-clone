#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _arg_value(argv: Sequence[str], name: str, default: str) -> str:
    try:
        index = list(argv).index(name)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = Path(_arg_value(args, "--root", ".")).resolve()
    output = Path(_arg_value(args, "--output", "generated/agent_manifest"))
    loop_output = Path(_arg_value(args, "--loop-output", "generated/agent_improvement_loop"))

    visual_runner = root / "tools" / "run_visual_improve_loop_v2.py"
    augment_runner = root / "tools" / "augment_loop_state_embedding.py"

    visual = subprocess.run([sys.executable, str(visual_runner), *args], cwd=root, check=False)
    if visual.returncode != 0:
        return visual.returncode

    output_path = output if output.is_absolute() else root / output
    loop_output_path = loop_output if loop_output.is_absolute() else root / loop_output
    augment = subprocess.run(
        [
            sys.executable,
            str(augment_runner),
            "--loop-state",
            str(loop_output_path / "loop_state.json"),
            "--embedding-dir",
            str(output_path / "embedding"),
            "--print-summary",
        ],
        cwd=root,
        check=False,
    )
    return augment.returncode


if __name__ == "__main__":
    raise SystemExit(main())
