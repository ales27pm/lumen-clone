from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    _ROOT = Path(__file__).resolve().parent.parent
    _VENV_PYTHON = _ROOT / "tools" / "lumen_manifest_crawler" / ".venv" / "bin" / "python3"
    if _VENV_PYTHON.exists():
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), "-m", "lumen_manifest_crawler", *sys.argv[1:]])
    raise SystemExit(
        "lumen_manifest_crawler requires Python 3.11+. "
        "Run tools/lumen_manifest_crawler/.venv/bin/python3 -m lumen_manifest_crawler or install Python 3.11+."
    )

from lumen_manifest_crawler.cli import app

if __name__ == "__main__":
    app()
