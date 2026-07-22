"""Tests for manifest validation rules."""

# pylint: disable=missing-function-docstring,line-too-long

from lumen_manifest_crawler.manifest import (
    AgentBehaviorManifest,
    FleetManifest,
    IntentManifest,
    ModelSlotManifest,
    ToolArgumentManifest,
    ToolManifest,
)
from lumen_manifest_crawler.validators import validate_manifest


def test_duplicate_tool_id_failure():
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="web.search"), ToolManifest(id="web.search")])
    report = validate_manifest(manifest)
    assert not report.passed
    assert any(f.code == "duplicate_tool_id" for f in report.failures)


def test_model_slot_without_source_grounded_responsibilities_fails_validation():
    manifest = AgentBehaviorManifest(
        fleet=FleetManifest(
            slots=[ModelSlotManifest(id="embedding", role="embedding")]
        )
    )

    report = validate_manifest(manifest)

    assert any(
        failure.code == "model_slot_missing_responsibilities"
        and failure.path == "fleet.slots.embedding.responsibilities"
        for failure in report.failures
    )


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


def test_enum_argument_type_is_supported():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="trigger.create",
                arguments=[
                    ToolArgumentManifest(
                        name="schedule",
                        type="enum",
                        required=True,
                        allowedValues=["absolute", "interval", "relative"],
                    )
                ],
            )
        ]
    )
    report = validate_manifest(manifest)
    assert not any(f.code == "unsupported_argument_type" for f in report.failures)


def test_enum_argument_type_is_supported_when_executor_lists_only_json_primitives():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="trigger.create",
                arguments=[
                    ToolArgumentManifest(
                        name="schedule",
                        type="enum",
                        required=True,
                        allowedValues=["absolute", "interval", "relative"],
                    )
                ],
            )
        ]
    )
    manifest.agentProtocols.executorOutput["supportedJSONTypes"] = [
        "array",
        "bool",
        "null",
        "number",
        "object",
        "string",
    ]

    report = validate_manifest(manifest)

    assert not any(f.code == "unsupported_argument_type" for f in report.failures)


def test_inferred_tool_argument_contract_is_hard_failure():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="outlook.folders.list",
                arguments=[
                    ToolArgumentManifest(
                        name="includeHidden",
                        type="string",
                        required=False,
                        description="Inferred from ToolDefinition description Args contract: optional includeHidden true/false",
                    )
                ],
            )
        ]
    )
    report = validate_manifest(manifest)
    assert any(f.code == "inferred_tool_argument_contract" for f in report.failures)


def test_literal_boolean_value_cannot_be_argument_name():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="outlook.folders.list",
                arguments=[
                    ToolArgumentManifest(name="includeHidden", type="bool", required=False),
                    ToolArgumentManifest(name="false", type="string", required=False),
                ],
            )
        ]
    )
    report = validate_manifest(manifest)
    assert any(f.code == "literal_value_argument_name" for f in report.failures)


def test_tool_capability_contract_rejects_bad_confirmation_and_permission_kind():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="camera.capture",
                requiresApproval=True,
                permissionKind="cameraRoll",
                confirmationMode="none",
            )
        ]
    )
    report = validate_manifest(manifest)
    assert any(f.code == "unsupported_permission_kind" for f in report.failures)
    assert any(f.code == "confirmation_mode_approval_mismatch" for f in report.failures)


def test_codebase_home_rejects_generated_manifest_self_ingestion():
    manifest = AgentBehaviorManifest()
    dataset = {
        "codebase_home_corpus": [
            {"id": "generated-manifest", "path": "generated/agent_manifest/AgentBehaviorManifest.json"},
            {"id": "nested-generated", "path": "tools/lumen_manifest_crawler/generated/example.json"},
            {"id": "ios-copy", "path": "ios/Lumen/AgentBehaviorManifest.json"},
        ],
        "codebase_home_sft": [
            {
                "id": "sft-generated-manifest",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "where"},
                    {"role": "assistant", "content": "{\"path\":\"generated/agent_manifest/runtime_grounding_bundle.json\"}"},
                ],
            }
        ],
    }
    report = validate_manifest(manifest, dataset)
    failures = [f for f in report.failures if f.code == "generated_output_in_codebase_home"]
    assert len(failures) == 4


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
