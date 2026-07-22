from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from audit_to_adapter_contract import CONTRACT, write_contract_json  # noqa: E402


def test_write_contract_json_creates_deterministic_contract(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "audit-to-adapter-contract.json"

    write_contract_json(output)

    expected = (
        json.dumps(
            CONTRACT.as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert output.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize(
    "entrypoint",
    (
        "validate_audit_to_adapter_pipeline.py",
        "validate_audit_to_adapter_pipeline_deep.py",
    ),
)
def test_validator_entrypoint_imports_contract_writer(entrypoint: str) -> None:
    result = subprocess.run(
        [sys.executable, str(PIPELINE_DIR / entrypoint), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--write-contract-json" in result.stdout
