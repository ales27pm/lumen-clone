import sys

import pytest


def pytest_configure(config):
    if sys.version_info < (3, 11):
        pytest.exit(
            "Lumen manifest crawler tests require Python 3.11+. "
            "Run with a 3.11+ interpreter, for example: "
            "uv run --python 3.12 --with-editable . --with pytest "
            "python -m pytest",
            returncode=4,
        )
