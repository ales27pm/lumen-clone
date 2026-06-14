#!/usr/bin/env python3
"""Fail if listed pipeline files use subprocess shell execution."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKED_FILES = (
    "tools/hf_artifacts/publish_hf_artifacts.py",
    "tools/lumen_manifest_crawler/lumen_manifest_crawler/improvement_loop.py",
    "tools/lumen_terminal_improve_loop.py",
    "tools/prepare_qwen3_shared_base.py",
    "tools/run_visual_improve_loop.py",
    "tools/run_visual_improve_loop_v2.py",
    "tools/run_visual_improve_loop_with_embedding_state.py",
    "tools/serve_visual_improve_loop.py",
)
SUBPROCESS_CALLS = {"run", "Popen", "call", "check_call", "check_output"}


def is_subprocess_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in SUBPROCESS_CALLS
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )


def shell_true_arg(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is True
    return False


def main() -> int:
    failures: list[str] = []
    for relative in CHECKED_FILES:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and is_subprocess_call(node) and shell_true_arg(node):
                failures.append(f"{relative}:{node.lineno}: subprocess shell=True is not allowed")
    if failures:
        print("\n".join(failures))
        return 1
    print("PASS: no subprocess shell=True usage in checked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
