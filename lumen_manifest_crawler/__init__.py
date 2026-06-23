"""Repo-root shim for running the crawler without installing it first."""

from __future__ import annotations

from pathlib import Path

_TOOLS_PACKAGE = Path(__file__).resolve().parent.parent / "tools" / "lumen_manifest_crawler" / "lumen_manifest_crawler"
if _TOOLS_PACKAGE.exists():
    __path__.append(str(_TOOLS_PACKAGE))  # type: ignore[name-defined]

__version__ = "0.1.0"
