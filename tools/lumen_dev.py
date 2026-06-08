#!/usr/bin/env python3
"""Thin entry point for the Lumen developer framework CLI."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CRAWLER = ROOT / "tools" / "lumen_manifest_crawler"
if str(CRAWLER) not in sys.path:
    sys.path.insert(0, str(CRAWLER))

from lumen_manifest_crawler.cli import framework_app  # noqa: E402


if __name__ == "__main__":
    framework_app()
