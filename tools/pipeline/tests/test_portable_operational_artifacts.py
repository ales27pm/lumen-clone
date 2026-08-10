from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import inspect_audit_to_adapter_inputs as inspector  # noqa: E402
import run_audit_to_adapter_pipeline as audit_runner  # noqa: E402
import run_e2e_workflow_pipeline as e2e_runner  # noqa: E402
from portable_artifact_paths import (  # noqa: E402
    PortableArtifactError,
    path_replacements,
    sanitize_payload,
)


def test_content_refs_are_full_digest_and_replacement_is_collision_safe(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    internal = root / "runtime-audits" / "same name.json"
    internal.parent.mkdir()
    internal.write_bytes(b"inside")
    external_parent = tmp_path / "External Secret"
    external_parent.mkdir()
    external = external_parent / "same name.json"
    external.write_bytes(b"outside")

    replacements = path_replacements(
        root,
        [internal, external],
        external_prefix="runtime-audit",
    )
    payload = sanitize_payload(
        {
            "internal": str(internal),
            "external": str(external),
            "warning": f"{external}: failed",
            "uri": f"file://{external}",
        },
        replacements=replacements,
    )

    expected_ref = "runtime-audit-sha256-" + hashlib.sha256(b"outside").hexdigest()
    assert payload == {
        "internal": "runtime-audits/same name.json",
        "external": expected_ref,
        "warning": f"{expected_ref}: failed",
        "uri": expected_ref,
    }
    assert replacements[str(internal)] != replacements[str(external)]

    with pytest.raises(PortableArtifactError, match="unclassified"):
        sanitize_payload(
            {"sibling": f"{external}.extra"},
            replacements=replacements,
        )


def test_unknown_absolute_path_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PortableArtifactError, match="unclassified"):
        sanitize_payload(
            {"message": "/Users/example/Client Secret/private.json"},
            replacements={},
        )


def test_inspection_output_uses_content_ref_for_external_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "Private Audit Name.json"
    external.write_text(json.dumps({"unknown": True}), encoding="utf-8")
    output = root / "generated" / "inspection.json"

    result = inspector.main(
        [
            "--root",
            str(root),
            "--runtime-audit",
            str(external),
            "--write",
            str(output),
            "--json",
        ]
    )

    assert result == 0
    raw = output.read_text(encoding="utf-8")
    payload = json.loads(raw)
    expected_ref = "runtime-audit-sha256-" + hashlib.sha256(
        external.read_bytes()
    ).hexdigest()
    assert payload["schema"].endswith("/1.1.0")
    assert payload["files"][0]["source"] == expected_ref
    assert expected_ref in payload["warnings"][0]
    assert str(tmp_path) not in raw
    assert external.name not in raw


def test_e2e_state_is_portable_even_when_explicitly_written_under_generated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}")
    state = root / "generated" / "agent_improvement_loop" / "state.json"
    args = SimpleNamespace(
        dry_run=False,
        state_file=state,
        python=sys.executable,
        train_python=sys.executable,
        runtime_audit=[str(audit)],
    )
    result = e2e_runner.StageResult(
        id="inspect",
        description="inspect",
        command=[
            sys.executable,
            "tools/pipeline/inspect_audit_to_adapter_inputs.py",
            "--root",
            str(root),
            "--runtime-audit",
            str(audit),
        ],
        returncode=0,
        elapsed_s=1.0,
        started_at="2026-08-10T00:00:00+00:00",
        finished_at="2026-08-10T00:00:01+00:00",
        log_path=".local/lumen/e2e_workflow/logs/01-inspect.log",
    )

    e2e_runner.write_state(root, args, [result], ok=True)

    raw = state.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["schema"].endswith("/1.1.0")
    assert payload["root"] == "."
    assert payload["runtime_audit_refs"] == [
        "runtime-audit-sha256-" + hashlib.sha256(b"{}").hexdigest()
    ]
    assert str(tmp_path) not in raw
    assert sys.executable not in raw


def test_runner_defaults_keep_raw_state_and_logs_in_ignored_local_tree() -> None:
    e2e = e2e_runner.parse_args([])
    audit = audit_runner.parse_args([])

    assert e2e.state_file == Path(".local/lumen/e2e_workflow/state.json")
    assert e2e.logs_dir == Path(".local/lumen/e2e_workflow/logs")
    assert audit.state_file == Path(
        ".local/lumen/audit_to_adapter_pipeline/state.json"
    )


def test_audit_pipeline_state_sanitizes_commands(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    state = root / "generated" / "agent_improvement_loop" / "pipeline.json"
    args = SimpleNamespace(
        dry_run=False,
        state_file=state,
        python=sys.executable,
        train_python=sys.executable,
        runtime_audit=[],
        mode="validate",
        agents="cortex",
    )
    result = audit_runner.StageResult(
        stage="validate",
        command=[sys.executable, "tool.py", "--root", str(root)],
        returncode=0,
        elapsed_s=0.1,
        started_at="2026-08-10T00:00:00+00:00",
        finished_at="2026-08-10T00:00:01+00:00",
    )

    audit_runner.write_state(root, args, [result])

    raw = state.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["schema"].endswith("/1.1.0")
    assert payload["stages"][0]["command"][-1] == "."
    assert str(tmp_path) not in raw
    assert sys.executable not in raw
