"""Tests for manifest validation rules."""

# pylint: disable=missing-function-docstring,line-too-long

from lumen_manifest_crawler.manifest import AgentBehaviorManifest, IntentManifest, ToolArgumentManifest, ToolManifest
from lumen_manifest_crawler.validators import validate_manifest


def test_duplicate_tool_id_failure():
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="web.search"), ToolManifest(id="web.search")])
    report = validate_manifest(manifest)
    assert not report.passed
    assert any(f.code == "duplicate_tool_id" for f in report.failures)


def test_unknown_intent_tool_failure():
    manifest = AgentBehaviorManifest(intents=[IntentManifest(id="search", allowedToolIDs=["web.search"])])
    report = validate_manifest(manifest)
    assert not report.passed
    assert any(f.code == "unknown_intent_tool" for f in report.failures)


def test_unsupported_argument_type_failure():
    manifest = AgentBehaviorManifest(
        tools=[ToolManifest(id="x.run", arguments=[ToolArgumentManifest(name="payload", type="binary", required=True)])]
    )
    report = validate_manifest(manifest)
    assert not report.passed
    assert any(f.code == "unsupported_argument_type" for f in report.failures)


def test_runtime_repair_record_requires_provenance_and_repair_action():
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="web.search")])
    dataset = {
        "runtime_audit_repairs": [
            {
                "id": "runtime-repair-1",
                "schemaVersion": "2.0.0",
                "split": "train",
                "sourceFamily": "runtime_audit_repairs",
                "agentRole": "rem",
                "taskType": "runtime_manifest_drift_repair",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "{}"},
                    {"role": "assistant", "content": "{\"failureType\":\"runtime_audit_clean\",\"repair\":{\"action\":\"document_runtime_pass_and_expand_coverage\"}}"},
                ],
                "metadata": {"source": "lumen_in_app_dataset_package", "sourceFile": "runtime-audits/latest-testflight-export.json"},
            }
        ]
    }
    report = validate_manifest(manifest, dataset)
    assert not any(
        f.code in {"runtime_repair_missing_source_family", "runtime_repair_missing_provenance", "runtime_repair_missing_action", "runtime_repair_missing_failure_type"}
        for f in report.failures
    )


def test_runtime_repair_record_fails_without_repair_action():
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="web.search")])
    dataset = {
        "runtime_audit_repairs": [
            {
                "id": "runtime-repair-2",
                "schemaVersion": "2.0.0",
                "split": "train",
                "agentRole": "rem",
                "taskType": "runtime_manifest_drift_repair",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "{}"},
                    {"role": "assistant", "content": "{\"repair\":{}}"},
                ],
                "metadata": {"source": "lumen_in_app_dataset_package", "sourceFile": "runtime-audits/latest-testflight-export.json"},
            }
        ]
    }
    report = validate_manifest(manifest, dataset)
    assert any(f.code == "runtime_repair_missing_source_family" for f in report.failures)
    assert any(f.code == "runtime_repair_missing_action" for f in report.failures)
    assert any(f.code == "runtime_repair_missing_failure_type" for f in report.failures)
