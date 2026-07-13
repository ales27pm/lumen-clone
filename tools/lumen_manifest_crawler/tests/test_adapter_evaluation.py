from __future__ import annotations

import json

import pytest
import lumen_manifest_crawler.dataset.adapter_evaluation as adapter_evaluation

from lumen_manifest_crawler.dataset.adapter_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EVALUATION_REPORT_SCHEMA_VERSION,
    EXPERIMENT_VARIANTS,
    build_contamination_report,
    build_evaluation_fingerprint_bundle,
    build_experiment_manifest,
    build_experiment_variant_manifest,
    canonical_sha256,
    decide_adapter_promotion,
    finalize_experiment_variant_manifest,
    score_evaluation_suite,
    upgrade_evaluation_record,
)
from lumen_manifest_crawler.dataset.adapter_export import agent_adapter_export_plan
from lumen_manifest_crawler.dataset.fine_tuning import compile_agent_fine_tuning_datasets
from lumen_manifest_crawler.dataset.fine_tuning import (
    _exclude_evaluation_segment_matches,
    _required_eval_templates,
)
from lumen_manifest_crawler.manifest import (
    AgentBehaviorManifest,
    SourceIntegrity,
    ToolArgumentManifest,
    ToolManifest,
)
from lumen_manifest_crawler.output.writer import _write_fine_tuning_outputs
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts
from lumen_manifest_crawler.dataset.public_adapter_eval_registry import (
    public_evaluation_text_shingle_hashes,
)


def _eval(
    agent: str,
    eval_type: str,
    metrics: list[dict],
    prompt: str = "Return the requested contract.",
) -> dict:
    return {
        "messages": [{"role": "system", "content": "shared"}, {"role": "user", "content": prompt}],
        "metrics": metrics,
        "metadata": {"agent": agent, "evalType": eval_type, "mustPass": True, "critical": True},
    }


def _tool_contracts() -> dict:
    return {
        "weather.current": {
            "arguments": [
                {"name": "location", "type": "string", "required": True},
                {"name": "units", "type": "string", "required": False, "allowedValues": ["metric", "imperial"]},
            ]
        }
    }


def _training_environment(manifest: dict, digest_character: str = "c") -> dict:
    return {
        "schemaVersion": "lumen.adapter-training-environment/1.0.0",
        "containerImageDigest": "sha256:" + digest_character * 64,
        "environmentLock": manifest["trainingEnvironmentLock"],
    }


def test_legacy_expectations_upgrade_to_versioned_executable_metrics() -> None:
    record = upgrade_evaluation_record(
        {
            "messages": [{"role": "user", "content": "Use weather."}],
            "expected": {
                "tool": "weather.current",
                "requiredArguments": ["location"],
                "status": "ready_to_execute",
            },
            "metadata": {"agent": "executor", "evalType": "tool"},
        }
    )

    assert record["schemaVersion"] == EVALUATION_SCHEMA_VERSION
    assert record["evalID"].startswith("eval-")
    assert [metric["type"] for metric in record["metrics"]] == [
        "manifest_tool_call",
        "json_fields_present",
        "json_field_equals",
    ]


def test_required_nearby_eval_uses_maps_search_and_missing_semantic_tool_fails() -> None:
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="maps.search",
                displayName="Search Maps",
                description="Search local places",
                permissionKey="location",
                arguments=[ToolArgumentManifest(name="query", type="string", required=True)],
            ),
            ToolManifest(
                id="messages.draft",
                displayName="Draft Message",
                description="Draft a message",
                requiresApproval=True,
            ),
        ]
    )
    templates = _required_eval_templates(
        manifest,
        {"maps.search", "messages.draft"},
    )
    nearby = next(
        record
        for record in templates["cortex"]
        if record["metadata"]["evalType"] == "tool_runtime_scenario_selection"
    )
    assert nearby["expected"]["selectedToolID"] == "maps.search"

    with pytest.raises(
        ValueError,
        match="required evaluation tool is absent from manifest: maps.search",
    ):
        _required_eval_templates(
            AgentBehaviorManifest(
                sourceIntegrity=SourceIntegrity(commit="a" * 40),
                tools=[
                    ToolManifest(
                        id="alpha.tool",
                        displayName="Alpha",
                        description="An unrelated tool",
                        requiresApproval=True,
                        permissionKey="alpha",
                        arguments=[
                            ToolArgumentManifest(name="value", type="string", required=True)
                        ],
                    )
                ]
            ),
            {"alpha.tool"},
        )


def test_behavioral_boolean_contracts_require_observed_behavior_not_echoes() -> None:
    cases = [
        (
            "executor",
            {
                "tool": "weather.current",
                "requiredArguments": ["location"],
                "mustNotClarify": True,
            },
            {"tool": "weather.current", "arguments": {"location": "Montreal"}},
            "mustNotClarify",
            "non_clarifying_tool_call",
        ),
        (
            "mimicry",
            {
                "mustPreserveLanguageMix": True,
                "languageMixInvariants": [["next level"], ["c'est", "de passer"]],
            },
            {"text": "Next level, c'est de passer au pipeline propre."},
            "mustPreserveLanguageMix",
            "language_mix_preservation",
        ),
        (
            "mimicry",
            {
                "mustRefuseUnsafeImpersonation": True,
                "forbiddenImpersonationText": ["I authorize this transfer"],
            },
            {"status": "refused", "reason": "unsafe impersonation"},
            "mustRefuseUnsafeImpersonation",
            "unsafe_impersonation_refusal",
        ),
        (
            "mimicry",
            {
                "extractPreference": True,
                "expectedPreference": {"length": "concise", "format": "bullet_points"},
            },
            {"preference": {"length": "concise", "format": "bullet_points"}},
            "extractPreference",
            "preference_extraction",
        ),
        (
            "rem",
            {"requiresTTLClassification": True, "expectedTTLClass": "shortLived"},
            {"freshnessClass": "shortLived"},
            "requiresTTLClassification",
            "ttl_classification",
        ),
        (
            "fleet",
            {"mustDelegate": True, "knownSlots": ["cortex", "executor"]},
            {"delegateTo": "executor"},
            "mustDelegate",
            "delegation",
        ),
        (
            "fleet",
            {
                "mustRespectBoundaries": True,
                "boundaryContract": {
                    "expectedToolID": "maps.search",
                    "expectedSlot": "executor",
                    "allowedSlots": ["cortex", "executor"],
                    "approvalState": "not_required",
                    "permissionState": "granted",
                },
            },
            {
                "toolID": "maps.search",
                "delegateTo": "executor",
                "approvalState": "not_required",
                "permissionState": "granted",
            },
            "mustRespectBoundaries",
            "tool_slot_boundary",
        ),
    ]

    for agent, expected, valid_candidate, echo_key, metric_type in cases:
        record = upgrade_evaluation_record(
            {
                "messages": [{"role": "user", "content": "Demonstrate the behavior."}],
                "expected": expected,
                "metadata": {"agent": agent, "evalType": metric_type},
            }
        )
        assert metric_type in [metric["type"] for metric in record["metrics"]]

        valid = score_evaluation_suite(
            [record],
            {record["evalID"]: valid_candidate},
            tool_contracts=_tool_contracts(),
            allowed_slots={"cortex", "executor"},
        )
        assert valid["weightedScore"] == 1.0, (metric_type, valid["caseResults"])

        echoed = score_evaluation_suite(
            [record],
            {record["evalID"]: {echo_key: True}},
            tool_contracts=_tool_contracts(),
            allowed_slots={"cortex", "executor"},
        )
        assert echoed["weightedScore"] == 0.0, metric_type
        behavior_result = next(
            result
            for result in echoed["caseResults"][0]["metricResults"]
            if result["type"] == metric_type
        )
        assert behavior_result["passed"] is False


def test_unknown_boolean_expectation_is_unsupported_contract() -> None:
    record = upgrade_evaluation_record(
        {
            "messages": [{"role": "user", "content": "Unknown behavior."}],
            "expected": {"mustPerformFutureMagic": True},
            "metadata": {"agent": "fleet", "evalType": "future"},
        }
    )
    assert record["metrics"] == [
        {
            "type": "unsupported_contract",
            "contractKey": "mustPerformFutureMagic",
            "agent": "fleet",
        }
    ]


def test_executor_tool_metric_validates_schema_types_enums_and_extra_arguments() -> None:
    record = upgrade_evaluation_record(
        _eval(
            "executor",
            "manifest_call",
            [
                {
                    "type": "manifest_tool_call",
                    "expectedToolID": "weather.current",
                    "validateArguments": True,
                }
            ],
        )
    )
    valid = score_evaluation_suite(
        [record],
        {
            record["evalID"]: json.dumps(
                {
                    "tool": "weather.current",
                    "arguments": {"location": "Montreal", "units": "metric"},
                }
            )
        },
        tool_contracts=_tool_contracts(),
    )
    assert valid["weightedScore"] == 1.0
    assert valid["criticalFailureCount"] == 0

    invalid = score_evaluation_suite(
        [record],
        {
            record["evalID"]: {
                "tool": "weather.current",
                "arguments": {"location": "Montreal", "units": "kelvin", "extra": True},
            }
        },
        tool_contracts=_tool_contracts(),
    )
    assert invalid["weightedScore"] == 0.0
    assert invalid["caseResults"][0]["metricResults"][0]["reason"] == "extra_arguments"


def test_missing_output_and_unknown_metric_fail_closed() -> None:
    record = upgrade_evaluation_record(_eval("mouth", "unknown", [{"type": "future_magic"}]))
    missing = score_evaluation_suite([record], {})
    assert missing["evidenceComplete"] is False
    assert missing["criticalFailureCount"] == 1
    assert missing["caseResults"][0]["metricResults"][0]["reason"] == "candidate_output_missing"

    unknown = score_evaluation_suite([record], {record["evalID"]: "looks good"})
    assert unknown["evidenceComplete"] is True
    assert unknown["weightedScore"] == 0.0
    assert unknown["caseResults"][0]["metricResults"][0]["reason"] == "unsupported_metric_type"


def test_explicit_malformed_metrics_are_preserved_and_fail_closed() -> None:
    record = upgrade_evaluation_record(
        _eval("mouth", "corrupt", [{"type": "json_valid"}, "not-a-metric", 7])
    )

    assert record["metrics"][1:] == [
        {"type": "invalid_metric", "metricIndex": 1, "valueType": "str"},
        {"type": "invalid_metric", "metricIndex": 2, "valueType": "int"},
    ]
    report = score_evaluation_suite([record], {record["evalID"]: {"valid": True}})
    assert report["weightedScore"] == 0.0
    assert [
        result["reason"] for result in report["caseResults"][0]["metricResults"]
    ] == ["valid_json", "unsupported_metric_type", "unsupported_metric_type"]


def test_agent_override_mismatch_and_no_tool_contract_fail_closed() -> None:
    no_tool = upgrade_evaluation_record(
        {
            "messages": [{"role": "user", "content": "Answer without a tool."}],
            "expected": {"allowedToolIDs": []},
            "metadata": {"agent": "cortex", "evalType": "no_tool"},
        }
    )
    passed = score_evaluation_suite(
        [no_tool],
        {no_tool["evalID"]: {"selectedToolID": None}},
        agent="cortex",
    )
    assert passed["weightedScore"] == 1.0
    assert passed["agentMismatch"] is False

    unexpected_tool = score_evaluation_suite(
        [no_tool],
        {no_tool["evalID"]: {"selectedToolID": "weather"}},
        agent="cortex",
    )
    assert unexpected_tool["weightedScore"] == 0.0

    mislabeled = score_evaluation_suite(
        [no_tool],
        {no_tool["evalID"]: {"selectedToolID": None}},
        agent="executor",
    )
    assert mislabeled["agent"] is None
    assert mislabeled["agentMismatch"] is True
    assert mislabeled["evidenceComplete"] is False


def test_empty_negative_only_outputs_and_non_standard_json_fail_closed() -> None:
    negative = upgrade_evaluation_record(
        _eval("mouth", "sentinel", [{"type": "forbidden_text", "values": ["SECRET"]}])
    )
    empty = score_evaluation_suite([negative], {negative["evalID"]: "   "})
    assert empty["weightedScore"] == 0.0
    assert empty["caseResults"][0]["metricResults"][0]["reason"] == "empty_candidate_output"

    strict = upgrade_evaluation_record(_eval("executor", "strict", [{"type": "json_valid"}]))
    for invalid in ("NaN", "Infinity", '{"temperature":NaN}', {"temperature": float("inf")}):
        report = score_evaluation_suite([strict], {strict["evalID"]: invalid})
        assert report["weightedScore"] == 0.0


def test_observation_repair_and_fixed_slot_metrics_are_executable() -> None:
    records = [
        upgrade_evaluation_record(
            _eval(
                "mouth",
                "grounded_final",
                [{"type": "observation_entailment", "requiredTerms": ["rain", "19 c"], "forbiddenClaims": ["sunny"]}],
            )
        ),
        upgrade_evaluation_record(
            _eval(
                "rem",
                "repair",
                [{"type": "repair_classification", "expectedFailureType": "invalid_tool", "expectedRepairAction": "replace_tool"}],
            )
        ),
        upgrade_evaluation_record(
            _eval(
                "fleet",
                "slot",
                [{"type": "fixed_slot", "path": "delegateTo", "expectedSlot": "executor", "allowedSlots": ["cortex", "executor"]}],
            )
        ),
    ]
    outputs = {
        records[0]["evalID"]: "Rain is likely and the observed temperature is 19 C.",
        records[1]["evalID"]: {"failureType": "invalid_tool", "repair": {"action": "replace_tool"}},
        records[2]["evalID"]: {"delegateTo": "executor"},
    }
    report = score_evaluation_suite(records, outputs, allowed_slots={"cortex", "executor"})
    assert report["weightedScore"] == 1.0
    assert report["evidenceComplete"] is False  # Mixed-agent suites cannot become promotion evidence.


def test_fleet_orchestration_graph_metric_rejects_private_state_and_unknown_slots() -> None:
    contract = {
        "graphSchemaVersion": "1.0.0",
        "knownSlotIDs": ["cortex", "executor", "mouth"],
        "strategy": "sequential",
        "expectedDelegatedSlotIDs": ["cortex", "executor", "mouth"],
        "expectedAggregationOwnerSlotID": "mouth",
        "expectedStopReason": "done",
        "requiredEventTypes": ["delegate", "delegate", "delegate", "stop"],
        "requiredDependencies": [{"from": "plan", "to": "execute"}],
        "mustUseKnownSlotsOnly": True,
        "mustNotExposePrivateState": True,
    }
    record = upgrade_evaluation_record(
        {
            "messages": [{"role": "user", "content": "Orchestrate this."}],
            "expected": contract,
            "metadata": {"agent": "fleet", "evalType": "event_graph"},
        }
    )
    valid_graph = {
        "graphSchemaVersion": "1.0.0",
        "decision": {
            "strategy": "sequential",
            "delegatedSlotIDs": ["cortex", "executor", "mouth"],
            "aggregationOwnerSlotID": "mouth",
            "stopReason": "done",
        },
        "events": [
            {"id": "plan", "type": "delegate", "targetSlotID": "cortex", "excludes": ["hiddenReasoning", "privatePeerState"]},
            {"id": "execute", "type": "delegate", "targetSlotID": "executor"},
            {"id": "respond", "type": "delegate", "targetSlotID": "mouth"},
            {"id": "stop", "type": "stop"},
        ],
        "dependencies": [{"from": "plan", "to": "execute"}],
    }
    passed = score_evaluation_suite([record], {record["evalID"]: valid_graph})
    assert passed["weightedScore"] == 1.0

    invalid_graph = json.loads(json.dumps(valid_graph))
    invalid_graph["decision"]["delegatedSlotIDs"][-1] = "shadow"
    invalid_graph["hiddenReasoning"] = "leak"
    failed = score_evaluation_suite([record], {record["evalID"]: invalid_graph})
    assert failed["weightedScore"] == 0.0


def test_fleet_orchestration_graph_security_checks_candidate_subtrees_structurally() -> None:
    contract = {
        "graphSchemaVersion": "1.0.0",
        "knownSlotIDs": ["executor"],
        "strategy": "bounded_handoff",
        "expectedDelegatedSlotIDs": ["executor"],
        "expectedAggregationOwnerSlotID": None,
        "expectedStopReason": "done",
        "requiredEventTypes": ["delegate", "stop"],
        "requiredDependencies": [],
        "mustUseKnownSlotsOnly": True,
        "mustNotExposePrivateState": True,
        "requiredContextKeys": ["approvedPlan", "toolID"],
        "forbiddenContextKeys": ["rawConversation", "hiddenReasoning"],
    }
    record = upgrade_evaluation_record(
        {
            "messages": [{"role": "user", "content": "Hand off the plan."}],
            "expected": contract,
            "metadata": {"agent": "fleet", "evalType": "event_graph"},
        }
    )
    valid_graph = {
        "graphSchemaVersion": "1.0.0",
        "decision": {
            "strategy": "bounded_handoff",
            "delegatedSlotIDs": ["executor"],
            "aggregationOwnerSlotID": None,
            "stopReason": "done",
        },
        "events": [
            {
                "id": "handoff",
                "type": "delegate",
                "targetSlotID": "executor",
                "contextKeys": ["approvedPlan", "toolID"],
                "excludes": ["rawConversation", "hiddenReasoning"],
            },
            {"id": "stop", "type": "stop"},
        ],
        "dependencies": [],
    }
    passed = score_evaluation_suite([record], {record["evalID"]: valid_graph})
    assert passed["weightedScore"] == 1.0

    hidden_private_state = json.loads(json.dumps(valid_graph))
    hidden_private_state["events"][0]["excludes"] = {
        "hiddenReasoning": "private chain of thought"
    }
    hidden_report = score_evaluation_suite(
        [record], {record["evalID"]: hidden_private_state}
    )
    assert hidden_report["weightedScore"] == 0.0
    assert hidden_report["caseResults"][0]["metricResults"][0]["reason"] == "private_state_exposed"

    text_only_context = json.loads(json.dumps(valid_graph))
    text_only_context["events"][0].pop("contextKeys")
    text_only_context["events"][0]["description"] = "approvedPlan and toolID"
    missing_report = score_evaluation_suite(
        [record], {record["evalID"]: text_only_context}
    )
    assert missing_report["weightedScore"] == 0.0
    assert missing_report["caseResults"][0]["metricResults"][0]["reason"] == "required_context_missing"


def test_aggregation_and_stopping_require_real_booleans() -> None:
    record = upgrade_evaluation_record(
        _eval(
            "fleet",
            "finish",
            [
                {"type": "aggregation", "required": True},
                {"type": "stopping", "expectedStop": True},
            ],
        )
    )
    failed = score_evaluation_suite(
        [record],
        {record["evalID"]: {"aggregate": "no", "stop": "yes"}},
    )
    assert failed["weightedScore"] == 0.0

    passed = score_evaluation_suite(
        [record],
        {record["evalID"]: {"aggregate": True, "stop": True}},
    )
    assert passed["weightedScore"] == 1.0


def test_contamination_report_detects_exact_and_near_segments_without_raw_text() -> None:
    eval_prompt = "plan the exact manifest tool call with required arguments before executing the user requested weather lookup now"
    evaluation = [_eval("executor", "heldout", [{"type": "json_valid"}], prompt=eval_prompt)]
    exact_training = {
        "messages": [
            {"role": "system", "content": "different system"},
            {"role": "user", "content": eval_prompt},
            {"role": "assistant", "content": "{}"},
        ]
    }
    near_training = {
        "messages": [
            {
                "role": "user",
                "content": eval_prompt + " safely",
            },
            {"role": "assistant", "content": "{}"},
        ]
    }
    report = build_contamination_report([exact_training, near_training], evaluation)
    assert report["contaminated"] is True
    assert {match["matchKind"] for match in report["matches"]} == {"exact_record", "near_segment"}
    assert eval_prompt not in json.dumps(report)

    bundle = build_evaluation_fingerprint_bundle(evaluation)
    assert bundle["hashOnly"] is True
    assert eval_prompt not in json.dumps(bundle)


def test_contamination_report_keeps_unrelated_training_clean() -> None:
    evaluation = [_eval("executor", "heldout", [{"type": "json_valid"}], prompt="one two three four five six seven eight nine ten eleven twelve thirteen fourteen")]
    training = [{"messages": [{"role": "user", "content": "completely unrelated routing sentence with a distinct vocabulary and purpose"}]}]
    report = build_contamination_report(training, evaluation)
    assert report["contaminated"] is False
    assert report["matchCount"] == 0


def test_contamination_report_detects_hash_only_public_evaluation_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_text = (
        "Synthetic benchmark function call select the weather endpoint with city Montreal "
        "units metric and return the structured argument object exactly as requested."
    )
    sketch = sorted(public_evaluation_text_shingle_hashes([leaked_text]))[:64]
    fake_bundle = {
        "bundleSHA256": "f" * 64,
        "rowCount": 1,
        "artifacts": [
            {
                "id": "synthetic-public-eval",
                "rows": [{"rowOrdinal": 0, "tokenShingleSketch": sketch}],
            }
        ],
    }
    adapter_evaluation._public_evaluation_shingle_index.cache_clear()
    monkeypatch.setattr(
        adapter_evaluation,
        "build_public_adapter_eval_fingerprint_bundle",
        lambda: fake_bundle,
    )
    try:
        report = build_contamination_report(
            [{"messages": [{"role": "user", "content": leaked_text}]}],
            [],
        )
    finally:
        adapter_evaluation._public_evaluation_shingle_index.cache_clear()

    assert report["contaminated"] is True
    assert report["publicEvaluationBundleSHA256"] == "f" * 64
    assert any(
        match["matchKind"] == "public_evaluation_shingle_sketch"
        for match in report["matches"]
    )
    assert report["matchCount"] == 1


def test_experiment_manifest_requires_all_controlled_variants_and_marks_dpo_untrained() -> None:
    evaluation = [_eval("executor", "json", [{"type": "json_valid"}])]
    config = {"learning_rate": 0.0002, "epochs": 2}
    manifests = {
        variant: build_experiment_variant_manifest(
            agent="executor",
            variant=variant,
            base_model_id="Qwen/Qwen3-1.7B",
            seed=42,
            training_config=config,
            train_sft=[{"messages": [{"role": "user", "content": variant}]}],
            validation_sft=[],
            dpo_records=[{"prompt": [], "chosen": {"content": "x"}, "rejected": {"content": "y"}}],
            evaluation_records=evaluation,
        )
        for variant in EXPERIMENT_VARIANTS
    }
    experiment = build_experiment_manifest(agent="executor", variants=manifests)
    assert experiment["variantOrder"] == list(EXPERIMENT_VARIANTS)
    assert experiment["controlledVariables"]["baseModelRevision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert experiment["controlledVariables"]["baseModelArtifactDigest"] == "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d"
    assert experiment["controlledVariables"]["baseModelTokenizerDigest"] == "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    assert all(variant["trainingEnvironmentSHA256"] is None for variant in experiment["variants"])
    assert all(
        variant["dpoTraining"]["status"] == "generated_not_trained"
        and variant["dpoTraining"]["includedInCheckpoint"] is False
        for variant in experiment["variants"]
    )

    with pytest.raises(ValueError, match="variants must be exactly"):
        build_experiment_manifest(agent="executor", variants={"internal_only": manifests["internal_only"]})


def test_promotion_requires_complete_clean_evidence_and_measured_improvement() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    evaluation = [upgrade_evaluation_record(
        _eval(
            "executor",
            "boundary",
            [{"type": "json_field_equals", "path": "approved", "expected": True}],
        )
    )]
    eval_id = evaluation[0]["evalID"]
    config = {"learning_rate": 0.0002, "epochs": 2}
    baseline_training = [{"messages": [{"role": "user", "content": "baseline"}]}]
    baseline_clean = build_contamination_report(baseline_training, evaluation)
    baseline_manifest = build_experiment_variant_manifest(
        agent="executor",
        variant="internal_plus_public_baseline",
        base_model_id="Qwen/Qwen3-1.7B",
        seed=42,
        training_config=config,
        train_sft=baseline_training,
        validation_sft=[],
        dpo_records=[],
        evaluation_records=evaluation,
        contamination_report=baseline_clean,
    )
    optimized_training = [{"messages": [{"role": "user", "content": "optimized"}]}]
    clean = build_contamination_report(optimized_training, evaluation)
    optimized_manifest = build_experiment_variant_manifest(
        agent="executor",
        variant="internal_plus_public_optimized",
        base_model_id="Qwen/Qwen3-1.7B",
        seed=42,
        training_config=config,
        train_sft=optimized_training,
        validation_sft=[],
        dpo_records=[],
        evaluation_records=evaluation,
        contamination_report=clean,
    )
    pending_report = score_evaluation_suite(
        evaluation,
        {eval_id: {"approved": True}},
        agent="executor",
        variant="internal_plus_public_optimized",
        variant_manifest=optimized_manifest,
        artifact_sha256=digest_b,
    )
    assert pending_report["promotionEvidenceBound"] is False
    baseline_manifest = finalize_experiment_variant_manifest(
        baseline_manifest,
        adapter_sha256=digest_a,
        training_environment=_training_environment(baseline_manifest),
    )
    optimized_manifest = finalize_experiment_variant_manifest(
        optimized_manifest,
        adapter_sha256=digest_b,
        training_environment=_training_environment(optimized_manifest),
    )
    wrong_artifact_report = score_evaluation_suite(
        evaluation,
        {eval_id: {"approved": True}},
        agent="executor",
        variant="internal_plus_public_optimized",
        variant_manifest=optimized_manifest,
        artifact_sha256=digest_a,
    )
    assert wrong_artifact_report["promotionEvidenceBound"] is False
    lineage = {
        field: baseline_manifest[field]
        for field in (
            "agent",
            "baseModelID",
            "baseModelRevision",
            "baseModelArtifactDigest",
            "baseModelTokenizerDigest",
            "trainingEnvironmentLockSHA256",
            "trainingEnvironmentSHA256",
            "seed",
            "trainingConfigSHA256",
            "frozenEvaluationSHA256",
            "publicEvaluationBundleSHA256",
        )
    }
    baseline_outputs = {eval_id: {"approved": False}}
    optimized_outputs = {eval_id: {"approved": True}}
    baseline = score_evaluation_suite(
        evaluation,
        baseline_outputs,
        agent="executor",
        variant="internal_plus_public_baseline",
        controlled_lineage=lineage,
        variant_manifest=baseline_manifest,
        artifact_sha256=digest_a,
    )
    optimized = score_evaluation_suite(
        evaluation,
        optimized_outputs,
        agent="executor",
        variant="internal_plus_public_optimized",
        controlled_lineage=lineage,
        variant_manifest=optimized_manifest,
        artifact_sha256=digest_b,
    )
    decision = decide_adapter_promotion(
        agent="executor",
        baseline_report=baseline,
        optimized_report=optimized,
        baseline_variant_manifest=baseline_manifest,
        optimized_variant_manifest=optimized_manifest,
        evaluation_records=evaluation,
        baseline_candidate_outputs=baseline_outputs,
        optimized_candidate_outputs=optimized_outputs,
        baseline_contamination_report=baseline_clean,
        optimized_contamination_report=clean,
        baseline_artifact_sha256=digest_a,
        optimized_artifact_sha256=digest_b,
    )
    assert decision["promoted"] is True
    assert decision["runtimePointerAction"] == "promote_optimized_candidate"

    identical_optimized_manifest = build_experiment_variant_manifest(
        agent="executor",
        variant="internal_plus_public_optimized",
        base_model_id="Qwen/Qwen3-1.7B",
        seed=42,
        training_config=config,
        train_sft=baseline_training,
        validation_sft=[],
        dpo_records=[],
        evaluation_records=evaluation,
        contamination_report=baseline_clean,
    )
    identical_optimized_manifest = finalize_experiment_variant_manifest(
        identical_optimized_manifest,
        adapter_sha256=digest_b,
        training_environment=_training_environment(identical_optimized_manifest),
    )
    identical_optimized_report = score_evaluation_suite(
        evaluation,
        optimized_outputs,
        agent="executor",
        variant="internal_plus_public_optimized",
        controlled_lineage=lineage,
        variant_manifest=identical_optimized_manifest,
        artifact_sha256=digest_b,
    )
    identical_decision = decide_adapter_promotion(
        agent="executor",
        baseline_report=baseline,
        optimized_report=identical_optimized_report,
        baseline_variant_manifest=baseline_manifest,
        optimized_variant_manifest=identical_optimized_manifest,
        evaluation_records=evaluation,
        baseline_candidate_outputs=baseline_outputs,
        optimized_candidate_outputs=optimized_outputs,
        baseline_contamination_report=baseline_clean,
        optimized_contamination_report=baseline_clean,
        baseline_artifact_sha256=digest_a,
        optimized_artifact_sha256=digest_b,
    )
    assert identical_decision["promoted"] is False
    assert "experiment_comparison_not_applicable" in identical_decision["failures"]

    drifted_manifest = dict(optimized_manifest)
    drifted_manifest.pop("variantManifestSHA256")
    drifted_environment = _training_environment(drifted_manifest, "d")
    drifted_manifest["trainingEnvironment"] = drifted_environment
    drifted_manifest["trainingEnvironmentSHA256"] = canonical_sha256(drifted_environment)
    drifted_manifest["variantManifestSHA256"] = canonical_sha256(drifted_manifest)
    drifted_lineage = {
        field: drifted_manifest[field]
        for field in (
            "agent",
            "baseModelID",
            "baseModelRevision",
            "baseModelArtifactDigest",
            "baseModelTokenizerDigest",
            "trainingEnvironmentLockSHA256",
            "trainingEnvironmentSHA256",
            "seed",
            "trainingConfigSHA256",
            "frozenEvaluationSHA256",
            "publicEvaluationBundleSHA256",
        )
    }
    drifted_report = score_evaluation_suite(
        evaluation,
        optimized_outputs,
        agent="executor",
        variant="internal_plus_public_optimized",
        controlled_lineage=drifted_lineage,
        variant_manifest=drifted_manifest,
        artifact_sha256=digest_b,
    )
    drifted_decision = decide_adapter_promotion(
        agent="executor",
        baseline_report=baseline,
        optimized_report=drifted_report,
        baseline_variant_manifest=baseline_manifest,
        optimized_variant_manifest=drifted_manifest,
        evaluation_records=evaluation,
        baseline_candidate_outputs=baseline_outputs,
        optimized_candidate_outputs=optimized_outputs,
        baseline_contamination_report=baseline_clean,
        optimized_contamination_report=clean,
        baseline_artifact_sha256=digest_a,
        optimized_artifact_sha256=digest_b,
    )
    assert drifted_decision["promoted"] is False
    assert "variant_controlled_lineage_mismatch" in drifted_decision["failures"]

    failed = decide_adapter_promotion(
        agent="executor",
        baseline_report=baseline,
        optimized_report={
            **optimized,
            "evidenceComplete": False,
            "reportSHA256": "0" * 64,
        },
        baseline_variant_manifest=baseline_manifest,
        optimized_variant_manifest=optimized_manifest,
        evaluation_records=evaluation,
        baseline_candidate_outputs=baseline_outputs,
        optimized_candidate_outputs=optimized_outputs,
        baseline_contamination_report=baseline_clean,
        optimized_contamination_report={
            **clean,
            "contaminated": True,
            "matchCount": 1,
            "reportSHA256": "0" * 64,
        },
        baseline_artifact_sha256=digest_a,
        optimized_artifact_sha256=None,
    )
    assert failed["promoted"] is False
    assert failed["runtimePointerAction"] == "leave_current_pointer_unchanged"
    assert {
        "evaluation_evidence_incomplete",
        "artifact_digest_missing_or_invalid",
        "evaluation_contamination_detected_or_unproven",
        "evaluation_report_integrity_invalid",
        "contamination_report_integrity_invalid",
        "optimized_report_reproduction_failed",
    }.issubset(failed["failures"])


def test_fine_tuning_cards_and_export_plans_publish_honest_eval_and_dpo_contracts(tmp_path) -> None:
    datasets = compile_agent_fine_tuning_datasets(AgentBehaviorManifest(), {})
    executor = datasets["executor"]

    assert executor.dataset_card["evaluation"]["schemaVersion"] == EVALUATION_SCHEMA_VERSION
    assert executor.dataset_card["evaluation"]["executableDeclarativeMetrics"] is True
    assert executor.dataset_card["preferenceTraining"] == {
        "status": "generated_not_trained",
        "includedInCheckpoint": False,
        "requiredPhase": "post_sft_preference_training",
        "recordCount": len(executor.train_dpo) + len(executor.val_dpo),
    }
    assert all(record["schemaVersion"] == EVALUATION_SCHEMA_VERSION for record in executor.eval)
    assert all(record["metrics"] for record in executor.eval)

    plan = agent_adapter_export_plan("executor", executor.dataset_card, executor.unsloth_config)
    assert plan["datasetCard"]["preferenceTraining"]["status"] == "generated_not_trained"
    assert plan["experimentPolicy"]["requiredVariants"] == list(EXPERIMENT_VARIANTS)
    assert plan["experimentPolicy"]["runtimePointerPolicy"] == "unchanged_until_promoted"
    assert plan["expectedArtifacts"]["experimentManifest"] == "experiment_manifest.json"
    assert plan["expectedArtifacts"]["variantPathTemplate"] == "experiments/{variant}"
    assert "promotionDecision" not in plan["expectedArtifacts"]

    _write_fine_tuning_outputs(tmp_path, datasets)
    public_eval = json.loads((tmp_path / "public_evaluation_fingerprints.json").read_text())
    assert public_eval["purpose"] == "evaluation_only_contamination_and_provenance"
    assert public_eval["rawEvaluationTextIncluded"] is False
    assert (tmp_path / "executor" / "evaluation_fingerprints.json").exists()
    assert (tmp_path / "executor" / "experiment_manifest.json").exists()
    variant_manifests = []
    for variant in EXPERIMENT_VARIANTS:
        variant_root = tmp_path / "executor" / "experiments" / variant
        assert (variant_root / "train_sft.jsonl").exists()
        assert (variant_root / "val_sft.jsonl").exists()
        assert (variant_root / "train_dpo.jsonl").exists()
        assert (variant_root / "val_dpo.jsonl").exists()
        assert (variant_root / "contamination_report.json").exists()
        assert (variant_root / "variant_manifest.json").exists()
        lanes = {
            name: [json.loads(line) for line in (variant_root / filename).read_text().splitlines() if line]
            for name, filename in (
                ("trainSFT", "train_sft.jsonl"),
                ("validationSFT", "val_sft.jsonl"),
                ("trainDPO", "train_dpo.jsonl"),
                ("validationDPO", "val_dpo.jsonl"),
            )
        }
        variant_manifest = json.loads((variant_root / "variant_manifest.json").read_text())
        variant_manifests.append(variant_manifest)
        for lane_name, records in lanes.items():
            assert variant_manifest["datasets"][lane_name] == {
                "count": len(records),
                "sha256": canonical_sha256(records),
            }
        assert variant_manifest["trainingCorpusSHA256"] == canonical_sha256(
            [
                *lanes["trainSFT"],
                *lanes["validationSFT"],
                *lanes["trainDPO"],
                *lanes["validationDPO"],
            ]
        )
        if variant == "internal_only":
            assert not any(
                isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
                for records in lanes.values()
                for record in records
            )
        if variant == "internal_plus_public_optimized":
            assert variant_manifest["publicSelectionPolicy"]["qualityScorePreference"] is True
            canonical_lanes = {
                "trainSFT": executor.train_sft,
                "validationSFT": executor.val_sft,
                "trainDPO": executor.train_dpo,
                "validationDPO": executor.val_dpo,
            }
            for lane_name, records in lanes.items():
                assert {
                    canonical_sha256(record) for record in records
                }.issubset({
                    canonical_sha256(record)
                    for record in canonical_lanes[lane_name]
                })
    for field in (
        "baseModelID",
        "seed",
        "trainingConfigSHA256",
        "frozenEvaluationSHA256",
        "publicEvaluationBundleSHA256",
    ):
        assert len({manifest[field] for manifest in variant_manifests}) == 1
    written_report = json.loads((tmp_path / "executor" / "contamination_report.json").read_text())
    assert written_report["reportSHA256"] == executor.contamination_report["reportSHA256"]


def test_native_fleet_orchestration_evals_flow_into_executable_fine_tuning_contracts() -> None:
    manifest = AgentBehaviorManifest.model_validate(
        {
            "fleet": {
                "slots": [
                    {"id": "cortex", "role": "cortex"},
                    {"id": "executor", "role": "executor"},
                    {"id": "mouth", "role": "mouth"},
                ]
            }
        }
    )
    artifacts = generate_fleet_artifacts(manifest)
    datasets = compile_agent_fine_tuning_datasets(manifest, {}, fleet_artifacts=artifacts)
    orchestration = [
        record
        for record in datasets["fleet"].eval
        if (record.get("metadata") or {}).get("evalType") == "fleet_orchestration_event_graph_eval"
    ]

    assert orchestration
    assert all(record["metrics"] == [{"type": "orchestration_graph", "contract": record["expected"]}] for record in orchestration)
    assert datasets["fleet"].contamination_report["contaminated"] is False


def test_frozen_evaluation_segments_are_removed_from_sft_and_dpo_training() -> None:
    evaluation = [
        {
            "messages": [
                {"role": "system", "content": "shared"},
                {"role": "user", "content": "Hold this exact prompt out for evaluation."},
            ]
        }
    ]
    training = [
        {
            "messages": [
                {"role": "system", "content": "train"},
                {"role": "user", "content": "Hold this exact prompt out for evaluation."},
                {"role": "assistant", "content": "candidate"},
            ]
        },
        {
            "prompt": [
                {"role": "system", "content": "train"},
                {"role": "user", "content": "A distinct preference prompt."},
            ],
            "chosen": {"role": "assistant", "content": "chosen"},
            "rejected": {"role": "assistant", "content": "rejected"},
        },
    ]

    filtered = _exclude_evaluation_segment_matches(training, evaluation)
    assert filtered == [training[1]]


def test_native_fleet_boundary_eval_rejects_tampered_event_payloads() -> None:
    manifest = AgentBehaviorManifest.model_validate({
        "fleet": {
            "slots": [
                {"id": "cortex", "role": "cortex"},
                {"id": "executor", "role": "executor"},
                {"id": "mouth", "role": "mouth"},
                {"id": "mimicry", "role": "mimicry"},
            ]
        },
        "tools": [
            {"id": "calendar.create", "requiresApproval": True, "arguments": []},
            {"id": "calendar.list", "permissionKey": "calendar", "arguments": []},
        ],
    })
    artifacts = generate_fleet_artifacts(manifest)
    datasets = compile_agent_fine_tuning_datasets(manifest, {}, fleet_artifacts=artifacts)
    evals = {
        record["metadata"]["scenarioID"]: record
        for record in datasets["fleet"].eval
        if record["metadata"].get("scenarioID")
    }
    graphs = {
        record["metadata"]["scenarioID"]: json.loads(record["messages"][-1]["content"])
        for record in artifacts.cross_model_training
        if record.get("recordType") == "sft"
        and record.get("sourceFamily") == "fleet_orchestration_native"
    }

    tampered = {}
    approval = json.loads(json.dumps(graphs["approval-boundary"]))
    next(event for event in approval["events"] if event["type"] == "approval_boundary")["approvalState"] = "granted"
    tampered["approval-boundary"] = approval
    unavailable = json.loads(json.dumps(graphs["unavailable-boundary"]))
    next(event for event in unavailable["events"] if event["type"] == "capability_unavailable")["permissionState"] = "authorized"
    tampered["unavailable-boundary"] = unavailable
    duplicate = json.loads(json.dumps(graphs["duplicate-suppression"]))
    next(event for event in duplicate["events"] if event["type"] == "duplicate_suppressed")["workKey"] = "other-work"
    tampered["duplicate-suppression"] = duplicate
    invalid_slot = json.loads(json.dumps(graphs["nonexistent-slot-negative"]))
    for event in invalid_slot["events"]:
        if "requestedSlotID" in event:
            event["requestedSlotID"] = "another-missing-slot"
    tampered["nonexistent-slot-negative"] = invalid_slot

    for scenario_id, candidate in tampered.items():
        record = evals[scenario_id]
        report = score_evaluation_suite(
            [record],
            {record["evalID"]: candidate},
            agent="fleet",
        )
        assert report["weightedScore"] == 0.0, scenario_id


def test_rem_runtime_backfill_refreshes_dependent_counts_and_contamination_evidence() -> None:
    datasets = compile_agent_fine_tuning_datasets(
        AgentBehaviorManifest(),
        {},
        runtime_audit_reports=[{"status": "failed"}],
    )
    rem = datasets["rem"]
    assert rem.dataset_card["recordCounts"]["train_sft"] == len(rem.train_sft)
    assert rem.dataset_card["evaluation"]["contamination"]["reportSHA256"] == rem.contamination_report["reportSHA256"]
    optimized = rem.experiment_variants["internal_plus_public_optimized"]
    assert optimized["contamination_report"]["reportSHA256"] == rem.contamination_report["reportSHA256"]
    assert rem.dataset_card["experimentPolicy"]["experimentManifestSHA256"] == rem.experiment_manifest["experimentManifestSHA256"]
