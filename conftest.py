import sys

import pytest


def pytest_configure(config):
    if sys.version_info < (3, 11):
        pytest.exit(
            "Lumen Python tests require Python 3.11+. "
            "Run with a 3.11+ interpreter, for example: "
            "uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler "
            "--with pytest --with pydantic --with typer --with rich "
            "python -m pytest",
            returncode=4,
        )
