import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from lumen_manifest_crawler import fleet_artifacts as fleet_artifact_module
from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset import fine_tuning as fine_tuning_module
from lumen_manifest_crawler.dataset.chat_template_contract import (
    generic_strict_json_retry_instruction,
)
from lumen_manifest_crawler.dataset.adapter_evaluation import (
    _score_orchestration_graph,
    canonical_sha256,
    upgrade_evaluation_record,
)
from lumen_manifest_crawler.dataset.fine_tuning import (
    _fleet_artifact_training_records,
    compile_agent_fine_tuning_datasets,
)
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts
from lumen_manifest_crawler.manifest import ModelSlotManifest, ToolManifest
from lumen_manifest_crawler.output.writer import _write_fleet_artifacts
from lumen_manifest_crawler.runtime_prompt_contract import (
    FLEET_SYSTEM_PROMPT_CONTRACT_SCHEMA_VERSION,
    RUNTIME_PROMPT_COMPOSER_POLICY_SHA256,
    prompt_sha256,
)
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets


@pytest.fixture(scope="module")
def production_fleet_compilation():
    """Compile the real source corpus once for production-routing assertions."""

    root = Path(__file__).resolve().parents[3]
    manifest = generate_manifest(root)
    datasets = generate_all_datasets(manifest, root=root)
    artifacts = generate_fleet_artifacts(manifest)
    compiled = compile_agent_fine_tuning_datasets(
        manifest,
        datasets,
        fleet_artifacts=artifacts,
    )
    return manifest, datasets, artifacts, compiled


@pytest.fixture(scope="module")
def production_compilation_without_fleet_artifacts(
    production_fleet_compilation,
):
    manifest, datasets, _, _ = production_fleet_compilation
    return compile_agent_fine_tuning_datasets(manifest, datasets)


def test_fleet_artifacts_include_source_code_map_and_whole_system_records():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)

    assert artifacts.system_prompts
    assert artifacts.cross_model_training
    assert "## System Identity" in artifacts.markdown
    assert "## Source Integrity" in artifacts.markdown

    first_prompt = next(iter(artifacts.system_prompts.values()))
    assert first_prompt["promptContractSchemaVersion"] == (
        FLEET_SYSTEM_PROMPT_CONTRACT_SCHEMA_VERSION
    )
    assert first_prompt["fleetContractVersion"] == manifest.fleet.contractVersion
    assert first_prompt["composerPolicySHA256"] == (
        RUNTIME_PROMPT_COMPOSER_POLICY_SHA256
    )
    assert first_prompt["systemPromptSHA256"] == prompt_sha256(
        first_prompt["systemPrompt"]
    )
    payload = first_prompt["contextPayload"]
    assert "sourceCodeMap" in payload
    assert payload["sourceCodeMap"]["baseCommit"] == manifest.sourceIntegrity.baseCommit
    assert payload["sourceCodeMap"]["workingTreeDigest"] == manifest.sourceIntegrity.workingTreeDigest
    assert payload["sourceCodeMap"]["dirtyState"] == manifest.sourceIntegrity.dirtyState
    assert payload["sourceCodeMap"]["fileCount"] == len(manifest.sourceIntegrity.files)
    assert payload["sourceCodeMap"]["boundary"]
    assert "source_code_map" in first_prompt

    task_types = {record.get("taskType") for record in artifacts.cross_model_training}
    assert "fleet_whole_system_identity" in task_types
    assert "source_code_self_knowledge" in task_types
    assert "source_tool_registry_knowledge" in task_types
    assert "source_routing_knowledge" in task_types


def test_embedding_self_knowledge_is_specific_and_has_no_assigned_tools():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    embedding = next(slot for slot in manifest.fleet.slots if slot.id == "embedding")
    prompt = artifacts.system_prompts["embedding"]

    assert embedding.responsibilities == [
        "semantic memory embedding",
        "embedding vector generation",
    ]
    assert prompt["contextPayload"]["responsibilities"] == sorted(
        embedding.responsibilities
    )
    assert prompt["contextPayload"]["availableTools"] == []
    assert "Follow the role contract extracted from the Swift source." not in prompt[
        "systemPrompt"
    ]
    topology = prompt["contextPayload"]["topology"]
    assert topology["purpose"] == (
        "Generate semantic vector representations for memory indexing and retrieval."
    )
    assert topology["outputSignature"] == (
        "Embedding vector only; no user-facing text, tool call, or hidden reasoning."
    )


def test_embedding_slot_is_structurally_excluded_from_tool_assignment():
    embedding = ModelSlotManifest(
        id="embedding",
        role="embedding",
        responsibilities=[
            "semantic memory embedding",
            "embedding vector generation",
        ],
    )
    executor = ModelSlotManifest(
        id="executor",
        role="tool_executor",
        responsibilities=["strict JSON generation"],
    )
    memory_tool = ToolManifest(
        id="memory.recall",
        displayName="Recall Memory",
        description="Recall relevant memory by exact query.",
    )

    assigned = fleet_artifact_module._best_slot_for_tool(
        memory_tool,
        [embedding, executor],
    )

    assert assigned.id == "executor"


def test_embedding_only_fleet_fails_closed_for_tool_assignment():
    embedding = ModelSlotManifest(
        id="embedding",
        role="embedding",
        responsibilities=["semantic memory embedding"],
    )
    memory_tool = ToolManifest(
        id="memory.recall",
        displayName="Recall Memory",
        description="Recall relevant memory by exact query.",
    )

    with pytest.raises(
        ValueError,
        match="Fleet tool assignment requires a non-embedding slot",
    ):
        fleet_artifact_module._best_slot_for_tool(memory_tool, [embedding])


def test_fleet_records_teach_peer_source_awareness_and_private_boundaries():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    task_types = {record.get("taskType") for record in artifacts.cross_model_training}

    if len(manifest.fleet.slots) > 1:
        assert "fleet_peer_source_knowledge" in task_types
        assert "fleet_private_state_boundary" in task_types

    serialized = "\n".join(str(record) for record in artifacts.cross_model_training)
    assert "single logical agent" in serialized or "one logical agent" in serialized
    assert "must not claim direct access" in serialized or "cannot inspect" in serialized


def test_production_cross_model_training_is_fleet_owned(
    production_fleet_compilation,
):
    _, _, _, compiled = production_fleet_compilation
    expected_sft_task_types = {
        "fleet_delegation",
        "fleet_peer_source_knowledge",
        "source_code_self_knowledge",
    }
    expected_dpo_task_types = {
        "fleet_delegation_preference",
        "fleet_private_state_boundary",
    }

    fleet_sft_task_types = {
        record["metadata"]["taskType"]
        for record in [*compiled["fleet"].train_sft, *compiled["fleet"].val_sft]
        if record["metadata"].get("sourceFamily") == "cross_model_training"
    }
    fleet_dpo_task_types = {
        record["metadata"]["taskType"]
        for record in [*compiled["fleet"].train_dpo, *compiled["fleet"].val_dpo]
        if record["metadata"].get("sourceFamily") == "cross_model_training"
    }

    assert expected_sft_task_types <= fleet_sft_task_types
    assert expected_dpo_task_types <= fleet_dpo_task_types

    for role_locked_agent in ("cortex", "executor"):
        role_locked_records = [
            *compiled[role_locked_agent].train_sft,
            *compiled[role_locked_agent].val_sft,
            *compiled[role_locked_agent].train_dpo,
            *compiled[role_locked_agent].val_dpo,
        ]
        assert not [
            record
            for record in role_locked_records
            if record["metadata"].get("sourceFamily") == "cross_model_training"
        ]


def test_compiler_preserves_real_source_slots_without_prefixing_native_orchestration():
    manifest = generate_manifest(Path(__file__).resolve().parents[3])
    artifacts = generate_fleet_artifacts(manifest)
    records = _fleet_artifact_training_records(artifacts)
    native_records = [
        record
        for record in records
        if record.get("sourceFamily") == "fleet_orchestration_native"
    ]
    user_prompts = [
        str(message.get("content") or "")
        for record in records
        for message in record.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "user"
    ]

    assert native_records
    assert not [
        message
        for record in native_records
        for message in record.get("messages") or []
        if isinstance(message, dict)
        and message.get("role") == "user"
        and str(message.get("content") or "").startswith("For the")
    ]
    assert any(prompt.startswith("For the `embedding` source slot:") for prompt in user_prompts)
    assert not any(prompt.startswith("For the `the` source slot:") for prompt in user_prompts)


def test_cortex_private_state_preferences_use_matched_json_envelopes():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    records = [
        record
        for record in artifacts.cross_model_training
        if record.get("taskType") == "fleet_private_state_boundary"
        and record.get("agentRole") == "orchestrator"
    ]
    required_fields = {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "reasoningSummary",
    }

    assert records
    for record in records:
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        assert set(chosen) == set(rejected) == required_fields
        assert chosen["selectedToolID"] is None
        assert chosen["requiresApproval"] is False
        assert "unavailable" in chosen["reasoningSummary"]
        assert "fabricated_internal_state" in rejected["reasoningSummary"]
        assert "Return exactly one valid JSON object and nothing else." in record["prompt"][0]["content"]


def test_native_fleet_orchestration_covers_event_graph_boundaries_and_eval_contracts():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    native_training = [
        record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
    ]
    sft_records = [record for record in native_training if record.get("recordType") == "sft"]
    expected_classes = {
        "no-delegation",
        "sequential-dependencies",
        "parallel-dependencies",
        "context-handoff",
        "duplicate-suppression",
        "aggregation-owner",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }

    assert Counter(
        record["metadata"]["behaviorClass"] for record in sft_records
    ) == Counter({behavior_class: 10 for behavior_class in expected_classes})
    assert Counter(
        (record["recordType"], record["metadata"]["requiredSplit"])
        for record in native_training
    ) == Counter(
        {
            ("sft", "train"): 81,
            ("sft", "validation"): 9,
            ("dpo", "train"): 81,
            ("dpo", "validation"): 4,
        }
    )
    assert Counter(
        record["metadata"]["behaviorClass"]
        for record in artifacts.orchestration_evals
    ) == Counter({behavior_class: 1 for behavior_class in expected_classes})
    assert {
        record["metadata"]["scenarioID"] for record in sft_records
    }.isdisjoint(
        record["metadata"]["scenarioID"]
        for record in artifacts.orchestration_evals
    )
    assert all(record["taskType"] == "fleet_orchestration_event_graph" for record in sft_records)
    assert all(record["taskType"] == "fleet_orchestration_event_graph_eval" for record in artifacts.orchestration_evals)
    assert all(
        record["metadata"]["requiredSplit"]
        == (
            "validation"
            if record["metadata"]["trainingMatrixVariant"] in {
                "normalized-intake",
                "policy-audited",
                "normalization-policy-audited",
            }
            else "train"
        )
        for record in native_training
    )
    for record in native_training:
        graph = json.loads(
            record["messages"][-1]["content"]
            if record["recordType"] == "sft"
            else record["chosen"]["content"]
        )
        assert record["metadata"]["trainingTopologySHA256"] == canonical_sha256(
            fleet_artifact_module._orchestration_topology_contract(graph)
        )


def test_compiled_fleet_native_matrices_are_optimizer_visible_per_behavior(
    production_fleet_compilation,
) -> None:
    _, _, _, compiled = production_fleet_compilation
    fleet = compiled["fleet"]

    expected_sft = {
        "no-delegation",
        "sequential-dependencies",
        "parallel-dependencies",
        "context-handoff",
        "duplicate-suppression",
        "aggregation-owner",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }
    core_dpo = expected_sft
    validation_dpo = {
        "duplicate-suppression",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }
    expected_dpo = expected_sft

    def native_by_behavior(records):
        grouped = defaultdict(list)
        for record in records:
            metadata = record.get("metadata") or {}
            if metadata.get("sourceFamily") == "fleet_orchestration_native":
                grouped[metadata["behaviorClass"]].append(record)
        return grouped

    train_sft = native_by_behavior(fleet.train_sft)
    train_dpo = native_by_behavior(fleet.train_dpo)
    assert set(train_sft) == expected_sft
    assert set(train_dpo) == expected_dpo
    assert sum(map(len, train_sft.values())) == 81
    assert sum(map(len, train_dpo.values())) == 81
    assert all(len(records) == 9 for records in train_sft.values())
    assert all(len(records) == 9 for records in train_dpo.values())
    for records in (
        fleet.train_dpo,
        *(
            variant["train_dpo"]
            for variant in fleet.experiment_variants.values()
        ),
    ):
        native_count = sum(
            1
            for record in records
            if record["metadata"].get("sourceFamily")
            == "fleet_orchestration_native"
        )
        assert native_count == 81
        assert (
            native_count
            * fine_tuning_module.FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR
            >= len(records)
            * fine_tuning_module.FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MIN_BASIS_POINTS
        )
        assert (
            native_count
            * fine_tuning_module.FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR
            <= len(records)
            * fine_tuning_module.FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MAX_BASIS_POINTS
        )
    assert all(
        record["metadata"]["sftOptimizerVisible"] is True
        for records in train_sft.values()
        for record in records
    )
    for behavior, records in train_dpo.items():
        for record in records:
            metadata = record["metadata"]
            assert metadata["preferenceSourceScenarioID"]
            assert metadata["sftAnchorScenarioID"]
            if metadata["trainingMatrixVariant"] == "behavior-conditioned":
                expected_visibility = True
                assert metadata["sftOptimizerVisible"] is expected_visibility
                assert metadata["sftAnchorBindingMode"] == "exact_scenario"
                assert metadata["sftAnchorScenarioID"] == (
                    metadata["preferenceSourceScenarioID"]
                )
            elif metadata["trainingMatrixVariant"] == "core":
                assert metadata["sftOptimizerVisible"] is True
                assert metadata["sftAnchorBindingMode"] == "exact_scenario"
                assert metadata["sftAnchorScenarioID"] == (
                    metadata["preferenceSourceScenarioID"]
                )
    val_sft = native_by_behavior(fleet.val_sft)
    val_dpo = native_by_behavior(fleet.val_dpo)
    assert set(val_sft) == expected_sft
    assert set(val_dpo) == validation_dpo
    assert sum(map(len, val_sft.values())) == 9
    assert sum(map(len, val_dpo.values())) == 4
    assert all(len(records) == 1 for records in val_sft.values())
    assert all(len(records) == 1 for records in val_dpo.values())
    assert fleet.val_sft
    assert fleet.val_dpo

    expected_sft_train_variants = {"core", "behavior-conditioned"}
    expected_dpo_train_variants = {"core", "behavior-conditioned"}
    for behavior, records in train_sft.items():
        assert {
            record["metadata"]["trainingMatrixVariant"]
            for record in records
        } == expected_sft_train_variants
    for behavior, records in train_dpo.items():
        expected_variants = (
            expected_dpo_train_variants
            if behavior in core_dpo
            else {"behavior-conditioned"}
        )
        assert {
            record["metadata"]["trainingMatrixVariant"]
            for record in records
        } == expected_variants

    for grouped in (train_sft, train_dpo):
        for behavior, records in grouped.items():
            expected_topology_count = 1
            topology_hashes = set()
            for record in records:
                graph = json.loads(
                    record["messages"][-1]["content"]
                    if "messages" in record
                    else record["chosen"]["content"]
                )
                topology_hash = canonical_sha256(
                    fleet_artifact_module._orchestration_topology_contract(graph)
                )
                assert topology_hash == record["metadata"][
                    "trainingTopologySHA256"
                ]
                topology_hashes.add(topology_hash)
            assert len(topology_hashes) == expected_topology_count
            behavior_conditioned = [
                record
                for record in records
                if record["metadata"]["trainingMatrixVariant"]
                == "behavior-conditioned"
            ]
            expected_conditioned_count = 8
            assert len(behavior_conditioned) == expected_conditioned_count
            assert {
                record["metadata"]["behaviorConditionedInstanceIndex"]
                for record in behavior_conditioned
            } == set(range(1, expected_conditioned_count + 1))
            assert {
                record["metadata"]["atomicPreferenceMutation"]
                for record in behavior_conditioned
            } == set(
                fleet_artifact_module.ORCHESTRATION_ATOMIC_MUTATION_KINDS[
                    :expected_conditioned_count
                ]
            )
            assert all(
                record["metadata"]["topologyCoverageMode"]
                == "trained_policy_topology_unseen_frozen_instance"
                for record in behavior_conditioned
            )
    for grouped in (val_sft, val_dpo):
        for records in grouped.values():
            assert {
                record["metadata"]["trainingMatrixVariant"]
                for record in records
            } == {"normalization-policy-audited"}


def test_compiled_fleet_has_three_positive_policy_anchors_per_core_behavior(
    production_fleet_compilation,
) -> None:
    _, _, _, compiled = production_fleet_compilation
    fleet = compiled["fleet"]
    expected_behaviors = set(
        fine_tuning_module.FLEET_NATIVE_ORCHESTRATION_SFT_BEHAVIORS
    )
    expected_anchor_kinds = {
        "graph_contract",
        "semantic_role_binding",
        "topology_identity",
    }

    for records in (
        fleet.train_sft,
        *(
            variant["train_sft"]
            for variant in fleet.experiment_variants.values()
        ),
    ):
        anchors = [
            record
            for record in records
            if record["metadata"].get("taskType")
            == fine_tuning_module.FLEET_POLICY_VOCABULARY_SFT_TASK_TYPE
        ]
        assert len(anchors) == len(expected_behaviors) * len(
            expected_anchor_kinds
        )
        core_graphs = {
            record["metadata"]["behaviorClass"]: json.loads(
                record["messages"][-1]["content"]
            )
            for record in records
            if record["metadata"].get("sourceFamily")
            == "fleet_orchestration_native"
            and record["metadata"].get("trainingMatrixVariant") == "core"
        }
        assert set(core_graphs) == expected_behaviors
        assert Counter(
            anchor["metadata"]["behaviorClass"] for anchor in anchors
        ) == Counter({behavior: 3 for behavior in expected_behaviors})
        assert {
            anchor["metadata"]["policyVocabularyAnchorKind"]
            for anchor in anchors
        } == expected_anchor_kinds
        assert {
            behavior: {
                anchor["metadata"]["policyVocabularyAnchorSurfaceIndex"]
                for anchor in anchors
                if anchor["metadata"]["behaviorClass"] == behavior
            }
            for behavior in expected_behaviors
        } == {behavior: set(range(3)) for behavior in expected_behaviors}

        for anchor in anchors:
            metadata = anchor["metadata"]
            behavior = metadata["behaviorClass"]
            graph = core_graphs[behavior]
            target = json.loads(anchor["messages"][-1]["content"])
            event_order_by_id = {
                event["id"]: index
                for index, event in enumerate(graph["events"], start=1)
            }
            dependency_orders = [
                {
                    "fromOrder": event_order_by_id[
                        dependency["fromEventID"]
                    ],
                    "toOrder": event_order_by_id[dependency["toEventID"]],
                }
                for dependency in graph["dependencies"]
            ]
            graph_contract = {
                "behaviorClass": behavior,
                "graphTopLevelKeys": list(graph),
                "decisionKeys": list(graph["decision"]),
                "eventID": {
                    "namespaceKey": "scenarioID",
                    "separator": "::event::",
                    "orderEncoding": "two_digit_one_based",
                },
                "eventSchemas": [
                    {
                        "type": event["type"],
                        "payloadKeys": [
                            key
                            for key in event
                            if key not in {"id", "type"}
                        ],
                    }
                    for event in graph["events"]
                ],
                "dependencyOrders": dependency_orders,
                "decisionContract": graph["decision"],
            }
            terminal_rejection = (
                fleet_artifact_module._terminal_decision_rejection(graph)
            )
            topology_identity = {
                "behaviorClass": behavior,
                "scenarioIdentitySource": "supplied_scenario_id",
                "eventIdentity": {
                    "namespaceKey": "scenarioID",
                    "separator": "::event::",
                    "orderEncoding": "two_digit_one_based",
                },
                "eventTypeSequence": [
                    event["type"] for event in graph["events"]
                ],
                "dependencyOrders": [
                    {
                        "fromOrder": dependency["fromOrder"],
                        "kind": "requires",
                        "toOrder": dependency["toOrder"],
                    }
                    for dependency in dependency_orders
                ],
                "terminalEventOrder": len(graph["events"]),
            }
            payload_roles = {
                "approvalRequestID": "copy_userApprovalRequestIdentifier",
                "requestID": "request_not_scenario",
                "targetSlotID": "delegation_target",
                "sourceSlotID": "result_source",
                "workKey": "persistent_work_key",
                "contextKeys": "slot_scoped_context",
                "requestedSlotID": "requested_slot",
                "reason": "exact_stop_reason",
            }
            semantic_role_binding = {
                "behaviorClass": behavior,
                "scenarioIdentitySource": "supplied_scenario_id",
                "eventBindings": [
                    {
                        "eventOrder": event_order,
                        "canonicalType": event["type"],
                        "requiredPayloadBindings": [
                            {
                                "key": key,
                                "semanticRole": payload_roles[key],
                            }
                            for key in event
                            if key in payload_roles
                        ],
                    }
                    for event_order, event in enumerate(
                        graph["events"],
                        start=1,
                    )
                ],
                "requiredInteriorEventTypes": [
                    event["type"]
                    for event in graph["events"]
                    if event["type"] not in {"request_received", "stop"}
                ],
                "decisionBinding": {
                    "strategy": graph["decision"]["strategy"],
                    "delegatedSlotIDs": graph["decision"]["delegatedSlotIDs"],
                    "aggregationOwnerSlotID": graph["decision"][
                        "aggregationOwnerSlotID"
                    ],
                    "stopReason": graph["decision"]["stopReason"],
                },
            }
            expected_targets = {
                "graph_contract": graph_contract,
                "semantic_role_binding": semantic_role_binding,
                "topology_identity": topology_identity,
            }

            def string_values(value):
                if isinstance(value, dict):
                    return {
                        item
                        for nested in value.values()
                        for item in string_values(nested)
                    }
                if isinstance(value, list):
                    return {
                        item
                        for nested in value
                        for item in string_values(nested)
                    }
                return {value} if isinstance(value, str) else set()

            rejected_values = {
                fleet_artifact_module._natural_noncanonical_event_type_alias(
                    graph,
                    event,
                )
                for event in graph["events"]
            } | {
                terminal_rejection["decision"]["strategy"],
                terminal_rejection["decision"]["stopReason"],
            }
            assert string_values(target).isdisjoint(rejected_values)

            assert metadata["requiredSplit"] == "train"
            assert metadata["policyVocabularyAnchor"] is True
            assert metadata["derivedCoreGraphCount"] == 1
            assert metadata["derivedBehaviorClasses"] == [behavior]
            assert metadata["policyVocabularySHA256"] == canonical_sha256(
                target
            )
            assert target == expected_targets[
                metadata["policyVocabularyAnchorKind"]
            ]
            assert fine_tuning_module._fleet_source_role(anchor) == (
                fine_tuning_module.FLEET_SOURCE_ROLE_BEHAVIORAL_PRIMARY
            )


def test_compiled_fleet_native_coverage_guard_rejects_replica_loss_and_mutation_collapse(
    production_fleet_compilation,
) -> None:
    _, _, _, compiled = production_fleet_compilation
    fleet = compiled["fleet"]
    target = next(
        record
        for record in fleet.train_sft
        if record["metadata"].get("sourceFamily")
        == "fleet_orchestration_native"
        and record["metadata"].get("behaviorClass") == "no-delegation"
        and record["metadata"].get("trainingMatrixVariant")
        == "behavior-conditioned"
        and record["metadata"].get("behaviorConditionedInstanceIndex") == 7
    )

    with pytest.raises(ValueError, match="variant counts are incomplete"):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=[record for record in fleet.train_sft if record is not target],
            val_sft=fleet.val_sft,
            train_dpo=fleet.train_dpo,
            val_dpo=fleet.val_dpo,
        )

    collapsed = []
    for record in fleet.train_sft:
        if record is target:
            record = {**record, "metadata": dict(record["metadata"])}
            record["metadata"]["atomicPreferenceMutation"] = (
                "terminal_decision_contract"
            )
        collapsed.append(record)
    with pytest.raises(ValueError, match="atomic mutation coverage is incomplete"):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=collapsed,
            val_sft=fleet.val_sft,
            train_dpo=fleet.train_dpo,
            val_dpo=fleet.val_dpo,
        )

    missing_target = {
        **target,
        "messages": [],
        "metadata": dict(target["metadata"]),
    }
    with pytest.raises(ValueError, match="lacks one assistant target"):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=[
                missing_target if record is target else record
                for record in fleet.train_sft
            ],
            val_sft=fleet.val_sft,
            train_dpo=fleet.train_dpo,
            val_dpo=fleet.val_dpo,
        )

    dpo_target = next(
        record
        for record in fleet.train_dpo
        if record["metadata"].get("sourceFamily")
        == "fleet_orchestration_native"
        and record["metadata"].get("behaviorClass") == "no-delegation"
        and record["metadata"].get("trainingMatrixVariant")
        == "behavior-conditioned"
        and record["metadata"].get("behaviorConditionedInstanceIndex") == 8
    )
    collapsed_preference = {
        **dpo_target,
        "rejected": dict(dpo_target["chosen"]),
        "metadata": dict(dpo_target["metadata"]),
    }
    with pytest.raises(ValueError, match="preference mutation is invalid"):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=fleet.train_sft,
            val_sft=fleet.val_sft,
            train_dpo=[
                collapsed_preference if record is dpo_target else record
                for record in fleet.train_dpo
            ],
            val_dpo=fleet.val_dpo,
        )

    core_target = next(
        record
        for record in fleet.train_dpo
        if record["metadata"].get("sourceFamily")
        == "fleet_orchestration_native"
        and record["metadata"].get("behaviorClass") == "aggregation-owner"
        and record["metadata"].get("trainingMatrixVariant") == "core"
    )
    collapsed_core_preference = {
        **core_target,
        "rejected": dict(core_target["chosen"]),
        "metadata": dict(core_target["metadata"]),
    }
    with pytest.raises(
        ValueError,
        match=(
            "invalid rejected graph schema|"
            "core (?:schema|failure-family) contrast is invalid"
        ),
    ):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=fleet.train_sft,
            val_sft=fleet.val_sft,
            train_dpo=[
                collapsed_core_preference if record is core_target else record
                for record in fleet.train_dpo
            ],
            val_dpo=fleet.val_dpo,
        )

    wrong_anchor = copy.deepcopy(dpo_target)
    wrong_anchor["metadata"]["sftAnchorScenarioID"] = (
        fleet_artifact_module._orchestration_training_scenario_id(
            behavior="aggregation-owner",
            variant="core",
            replica_index=None,
        )
    )
    with pytest.raises(ValueError, match="invalid SFT anchor identity"):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=fleet.train_sft,
            val_sft=fleet.val_sft,
            train_dpo=[
                wrong_anchor if record is dpo_target else record
                for record in fleet.train_dpo
            ],
            val_dpo=fleet.val_dpo,
        )

    event_id_target = next(
        record
        for record in fleet.train_dpo
        if record["metadata"].get("sourceFamily")
        == "fleet_orchestration_native"
        and record["metadata"].get("behaviorClass") == "no-delegation"
        and record["metadata"].get("atomicPreferenceMutation")
        == "event_id_grammar"
    )
    rebound_fact = copy.deepcopy(event_id_target)
    rebound_prompt = rebound_fact["prompt"][1]["content"]
    request_identifier = fine_tuning_module._fleet_native_prompt_request_identifier(
        rebound_prompt
    )
    assert request_identifier is not None
    rebound_fact["prompt"][1]["content"] = rebound_prompt.replace(
        f'"requestIdentifier": "{request_identifier}"',
        '"requestIdentifier": "id-aaaaaa"',
        1,
    )
    assert rebound_fact["prompt"][1]["content"] != rebound_prompt
    with pytest.raises(ValueError, match="preference mutation is invalid"):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=fleet.train_sft,
            val_sft=fleet.val_sft,
            train_dpo=[
                rebound_fact if record is event_id_target else record
                for record in fleet.train_dpo
            ],
            val_dpo=fleet.val_dpo,
        )

    duplicate_identity = copy.deepcopy(target)
    duplicate_identity["metadata"]["scenarioID"] = next(
        record["metadata"]["scenarioID"]
        for record in fleet.train_sft
        if record is not target
        and record["metadata"].get("sourceFamily")
        == "fleet_orchestration_native"
    )
    with pytest.raises(ValueError, match="scenario identity collision"):
        fine_tuning_module._assert_fleet_native_orchestration_training_coverage(
            train_sft=[
                duplicate_identity if record is target else record
                for record in fleet.train_sft
            ],
            val_sft=fleet.val_sft,
            train_dpo=fleet.train_dpo,
            val_dpo=fleet.val_dpo,
        )


def test_native_fleet_orchestration_is_fleet_owned_and_manifest_grounded():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    known_slots = {slot.id for slot in manifest.fleet.slots}
    native_records = [
        record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
    ]

    assert native_records
    for record in [*native_records, *artifacts.orchestration_evals]:
        assert record["agentRole"] == "fleet"
        assert record["sourceFamily"] == "fleet_orchestration_native"
        assert record["metadata"]["sourceClass"] == "lumen_native_manifest_derived"
        assert record["metadata"]["derivedFrom"] == "AgentBehaviorManifest"
        assert record["metadata"]["sourceIntegrity"] == manifest.sourceIntegrity.lineage_dict()
        assert "publicCorpus" not in record
        assert "publicCorpus" not in record["metadata"]

    for record in native_records:
        outputs = []
        if record["recordType"] == "sft":
            outputs.append(record["messages"][-1]["content"])
        else:
            outputs.append(record["chosen"]["content"])
        for output in outputs:
            graph = json.loads(output)
            assert set(graph["knownSlotIDs"]) == known_slots
            assert set(graph["decision"]["delegatedSlotIDs"]).issubset(known_slots)
            assert all("role" not in event for event in graph["events"])
            for event in graph["events"]:
                for key in ("targetSlotID", "sourceSlotID"):
                    if key in event:
                        assert event[key] in known_slots


def test_native_fleet_orchestration_graphs_have_valid_dependencies_and_explicit_stops():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    sft_records = [
        record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native" and record.get("recordType") == "sft"
    ]

    for record in sft_records:
        graph = json.loads(record["messages"][-1]["content"])
        event_ids = {event["id"] for event in graph["events"]}
        assert len(event_ids) == len(graph["events"])
        assert graph["events"][-1]["type"] == "stop"
        assert graph["events"][-1]["reason"] == graph["decision"]["stopReason"]
        assert graph["decision"]["stopReason"]
        for dependency in graph["dependencies"]:
            assert dependency["fromEventID"] in event_ids
            assert dependency["toEventID"] in event_ids
            assert dependency["kind"] == "requires"

    eval_by_scenario = {
        record["metadata"]["behaviorClass"]: record["expected"]
        for record in artifacts.orchestration_evals
    }
    assert eval_by_scenario["no-delegation"]["mustNotDelegate"] is True
    assert eval_by_scenario["sequential-dependencies"]["mustRespectDependencyOrder"] is True
    assert eval_by_scenario["parallel-dependencies"]["mustWaitForAllDependenciesBeforeAggregation"] is True
    assert eval_by_scenario["context-handoff"]["forbiddenContextKeys"] == [
        "conversationTranscript",
        "peerRuntimeSnapshot",
        "internalReasoningTrace",
    ]
    assert eval_by_scenario["duplicate-suppression"]["maximumDelegationsPerWorkKey"] == 1
    assert eval_by_scenario["aggregation-owner"]["mustHaveExactlyOneAggregationOwner"] is True
    assert eval_by_scenario["approval-boundary"]["mustNotExecuteBeforeApproval"] is True
    assert eval_by_scenario["unavailable-boundary"]["mustNotDelegateUnavailableCapability"] is True
    assert eval_by_scenario["nonexistent-slot-negative"]["maximumDelegationCount"] == 0


def test_native_fleet_orchestration_preferences_reject_boundary_violations():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    known_slots = {slot.id for slot in manifest.fleet.slots}
    dpo_by_class = defaultdict(list)
    for record in artifacts.cross_model_training:
        if (
            record.get("sourceFamily") == "fleet_orchestration_native"
            and record.get("recordType") == "dpo"
        ):
            dpo_by_class[record["metadata"]["behaviorClass"]].append(record)

    validation_dpo_classes = {
        "duplicate-suppression",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }
    expected_classes = {
        "no-delegation",
        "sequential-dependencies",
        "parallel-dependencies",
        "context-handoff",
        "duplicate-suppression",
        "aggregation-owner",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }
    assert set(dpo_by_class) == expected_classes
    assert sum(map(len, dpo_by_class.values())) == 85
    assert all(
        len(records) == (
            10 if behavior_class in validation_dpo_classes else 9
        )
        for behavior_class, records in dpo_by_class.items()
    )
    for behavior_class, records in dpo_by_class.items():
        expected_variants = (
            {
                "core",
                "behavior-conditioned",
                "normalization-policy-audited",
            }
            if behavior_class in validation_dpo_classes
            else {"core", "behavior-conditioned"}
        )
        assert {
            record["metadata"]["trainingMatrixVariant"] for record in records
        } == expected_variants
        for record in records:
            chosen = json.loads(record["chosen"]["content"])
            rejected = json.loads(record["rejected"]["content"])
            assert canonical_sha256(chosen) != canonical_sha256(rejected)
            if record["metadata"]["trainingMatrixVariant"] == "behavior-conditioned":
                assert chosen["scenarioID"] == rejected["scenarioID"]
                continue
            if record["metadata"]["trainingMatrixVariant"] == "core":
                assert record["metadata"]["preferenceContrastMode"] == (
                    fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_CONTRAST_MODE
                )
                assert record["metadata"]["coreFailureFamilyMutation"] == (
                    fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_FAMILY_MUTATIONS[
                        behavior_class
                    ]
                )
                continue
            assert chosen["scenarioID"] == rejected["scenarioID"]
            chosen_delegations = [
                event
                for event in chosen["events"]
                if event["type"] == "delegate"
            ]
            rejected_delegations = [
                event
                for event in rejected["events"]
                if event["type"] == "delegate"
            ]
            if behavior_class == "duplicate-suppression":
                assert len(chosen_delegations) == 1
                assert len(rejected_delegations) == (
                    3
                    if record["metadata"].get("preferenceContrastMode")
                    == "compound_schema_repetition"
                    else 2
                )
            else:
                assert not chosen_delegations
                assert len(rejected_delegations) == (
                    2
                    if record["metadata"].get("preferenceContrastMode")
                    == "compound_schema_repetition"
                    else 1
                )
    core_by_class = {
        behavior_class: next(
            record
            for record in records
            if record["metadata"]["trainingMatrixVariant"] == "core"
        )
        for behavior_class, records in dpo_by_class.items()
    }
    assert set(core_by_class) == expected_classes
    duplicate_rejected = json.loads(core_by_class["duplicate-suppression"]["rejected"]["content"])
    duplicate_targets = [event["targetSlotID"] for event in duplicate_rejected["events"] if event["type"] == "delegate"]
    assert duplicate_targets == ["executor"]

    approval_chosen = json.loads(core_by_class["approval-boundary"]["chosen"]["content"])
    unavailable_chosen = json.loads(core_by_class["unavailable-boundary"]["chosen"]["content"])
    assert not approval_chosen["decision"]["delegatedSlotIDs"]
    assert not unavailable_chosen["decision"]["delegatedSlotIDs"]

    assert all(
        set(json.loads(record["chosen"]["content"])["knownSlotIDs"])
        == known_slots
        for record in core_by_class.values()
    )


def test_behavior_conditioned_preferences_are_atomic_and_scorer_invalid():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    training_scenarios = fleet_artifact_module._orchestration_training_scenarios(
        manifest
    )
    scenarios_by_id = {
        preference["id"]: preference
        for scenario in training_scenarios
        if isinstance(scenario.get("rejectedGraph"), dict)
        for preference in [
            fleet_artifact_module._orchestration_preference_scenario(scenario)
        ]
    }
    atomic_records = [
        record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
        and record.get("recordType") == "dpo"
        and record.get("metadata", {}).get("trainingMatrixVariant")
        == "behavior-conditioned"
    ]
    expected_mutations = {
        "terminal_decision_contract": (
            None,
            "strategy_mismatch",
        ),
        "event_type_vocabulary": (
            None,
            "event_type_schema_unknown",
        ),
        "event_completeness_contract": (
            "$.events",
            "event_sequence_mismatch",
        ),
        "event_order": (
            "$.events",
            "event_sequence_mismatch",
        ),
        "event_id_grammar": (
            None,
            "event_id_grammar_mismatch",
        ),
        "dependency_endpoint_reference": (
            None,
            "dependency_set_mismatch",
        ),
        "event_payload_schema": (
            None,
            "event_schema_invalid",
        ),
        "delegated_slot_contract": (
            None,
            "unknown_slot_used",
        ),
    }

    assert len(atomic_records) == 72
    assert Counter(
        record["metadata"]["behaviorClass"] for record in atomic_records
    ) == Counter(
        {
            behavior_class: 8
            for behavior_class in {
                "no-delegation",
                "sequential-dependencies",
                "parallel-dependencies",
                "context-handoff",
                "duplicate-suppression",
                "aggregation-owner",
                "approval-boundary",
                "unavailable-boundary",
                "nonexistent-slot-negative",
            }
        }
    )
    assert Counter(
        record["metadata"]["atomicPreferenceMutation"]
        for record in atomic_records
    ) == Counter(
        {
            "terminal_decision_contract": 9,
            "event_type_vocabulary": 9,
            "event_completeness_contract": 9,
            "event_order": 9,
            "event_id_grammar": 9,
            "dependency_endpoint_reference": 9,
            "event_payload_schema": 9,
            "delegated_slot_contract": 9,
        }
    )
    mutations_by_behavior = defaultdict(set)
    completeness_modes = set()
    event_type_aliases = []
    payload_key_aliases = []
    payload_omission_count = 0
    event_id_modes = set()
    for record in atomic_records:
        mutations_by_behavior[record["metadata"]["behaviorClass"]].add(
            record["metadata"]["atomicPreferenceMutation"]
        )
    assert all(
        mutations == set(expected_mutations)
        for mutations in mutations_by_behavior.values()
    )

    for record in atomic_records:
        metadata = record["metadata"]
        behavior_class = metadata["behaviorClass"]
        expected_kind = metadata["atomicPreferenceMutation"]
        expected_path, rejected_reason = expected_mutations[expected_kind]
        scenario = scenarios_by_id[metadata["scenarioID"]]
        event_id_fact = (
            fleet_artifact_module._orchestration_event_id_negative_fact(
                scenario["canonicalDerivation"]["facts"]
            )
        )
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        differences = fleet_artifact_module._orchestration_scalar_leaf_differences(
            chosen,
            rejected,
        )
        fleet_artifact_module._validate_atomic_orchestration_rejection(
            chosen,
            rejected,
            mutation_kind=expected_kind,
            event_id_fact=event_id_fact,
        )

        assert metadata["atomicPreferenceMutation"] == expected_kind
        if expected_kind == "terminal_decision_contract":
            terminal_index = len(chosen["events"]) - 1
            assert set(differences) == {
                "$.decision.stopReason",
                "$.decision.strategy",
                f"$.events[{terminal_index}].reason",
            }
        elif expected_kind == "event_completeness_contract":
            assert differences
        elif expected_kind == "event_order":
            assert differences
            assert all(path.startswith("$.events[") for path in differences)
        elif expected_kind == "event_id_grammar":
            assert differences
        elif expected_kind == "event_type_vocabulary":
            event_index, _, _ = (
                fleet_artifact_module._event_type_vocabulary_rejection_target(
                    chosen
                )
            )
            assert differences == [f"$.events[{event_index}].type"]
        elif expected_kind == "event_payload_schema":
            event_index, _, _ = (
                fleet_artifact_module._event_payload_schema_rejection_target(
                    chosen
                )
            )
            assert differences == [f"$.events[{event_index}]"]
        elif expected_kind == "delegated_slot_contract":
            assert tuple(differences) in {
                ("$.decision.delegatedSlotIDs",),
                ("$.decision.delegatedSlotIDs[0]",),
            }
        elif expected_kind == "dependency_endpoint_reference":
            assert differences == [
                f"$.dependencies[{len(chosen['dependencies']) - 1}].toEventID"
            ]
        else:
            assert differences == [expected_path]
        if expected_kind == "terminal_decision_contract":
            assert rejected["decision"]["strategy"] != chosen["decision"][
                "strategy"
            ]
            assert rejected["decision"]["strategy"] in (
                fleet_artifact_module._CANONICAL_ORCHESTRATION_STRATEGIES
            )
            assert rejected["decision"]["stopReason"] != chosen["decision"][
                "stopReason"
            ]
            assert rejected["events"][-1]["reason"] == rejected["decision"][
                "stopReason"
            ]
            assert rejected["decision"]["aggregationOwnerSlotID"] == chosen[
                "decision"
            ]["aggregationOwnerSlotID"]
            observed_alias = (
                fleet_artifact_module._OBSERVED_STOP_REASON_ALIAS_BY_STRATEGY.get(
                    chosen["decision"]["strategy"]
                )
            )
            if observed_alias is not None:
                assert (
                    chosen["decision"]["stopReason"],
                    rejected["decision"]["stopReason"],
                ) == observed_alias
        elif expected_kind == "event_type_vocabulary":
            event_index, canonical_event_type, expected_alias = (
                fleet_artifact_module._event_type_vocabulary_rejection_target(
                    chosen
                )
            )
            alias = rejected["events"][event_index]["type"]
            assert chosen["events"][event_index]["type"] == canonical_event_type
            assert alias == expected_alias
            assert alias != canonical_event_type
            event_type_aliases.append((canonical_event_type, alias))
        elif expected_kind == "event_completeness_contract":
            expected_rejection = (
                fleet_artifact_module._atomic_event_completeness_rejection(
                    json.loads(json.dumps(chosen, ensure_ascii=False))
                )
            )
            assert rejected == expected_rejection
            if not any(event["type"] == "stop" for event in rejected["events"]):
                completeness_modes.add("terminal_stop")
                assert rejected["decision"] == chosen["decision"]
            else:
                completeness_modes.add("interior_event")
            if behavior_class == "unavailable-boundary":
                assert not any(
                    event["type"] == "request_received"
                    for event in rejected["events"]
                )
                assert rejected["events"][0]["type"] == (
                    "permission_state_checked"
                )
            elif behavior_class == "nonexistent-slot-negative":
                assert not any(
                    event["type"] == "invalid_slot_rejected"
                    for event in rejected["events"]
                )
                assert any(
                    event["type"] == "rejection_recorded"
                    for event in rejected["events"]
                )
        elif expected_kind == "event_order":
            first, second = fleet_artifact_module._atomic_event_order_indices(
                chosen["events"]
            )
            assert [event["id"] for event in rejected["events"]] == [
                event["id"] for event in chosen["events"]
            ]
            assert rejected["dependencies"] == chosen["dependencies"]
            assert rejected["events"][first]["type"] == (
                chosen["events"][second]["type"]
            )
            assert rejected["events"][second]["type"] == (
                chosen["events"][first]["type"]
            )
        elif expected_kind == "event_id_grammar":
            mode = fleet_artifact_module._event_id_grammar_contrast_mode(
                chosen
            )
            event_id_modes.add(mode)
            assert rejected == (
                fleet_artifact_module._event_id_grammar_rejection(
                    chosen,
                    event_id_fact=event_id_fact,
                )
            )
            if behavior_class in {
                "approval-boundary",
                "unavailable-boundary",
            }:
                assert mode == "fact_namespace_all_events"
                assert all(
                    event["id"].startswith(event_id_fact + "::event::")
                    for event in rejected["events"]
                )
            if mode not in {"duplicate_identity", "missing_identity"}:
                assert fleet_artifact_module._orchestration_topology_contract(
                    rejected
                ) == fleet_artifact_module._orchestration_topology_contract(
                    chosen
                )
        elif expected_kind == "dependency_endpoint_reference":
            assert rejected["dependencies"][-1]["toEventID"] != (
                chosen["dependencies"][-1]["toEventID"]
            )
            assert chosen["dependencies"][-1]["toEventID"] == (
                chosen["events"][-1]["id"]
            )
            assert rejected["dependencies"][-1]["toEventID"] == (
                chosen["events"][0]["id"]
            )
        elif expected_kind == "event_payload_schema":
            event_index, required_key, alias_key = (
                fleet_artifact_module._event_payload_schema_rejection_target(
                    chosen
                )
            )
            chosen_event = chosen["events"][event_index]
            rejected_event = rejected["events"][event_index]
            assert required_key in chosen_event
            assert required_key not in rejected_event
            if alias_key is None:
                payload_omission_count += 1
                assert set(rejected_event) < set(chosen_event)
            else:
                payload_key_aliases.append((required_key, alias_key))
                assert alias_key not in chosen_event
                assert rejected_event[alias_key] == chosen_event[required_key]
                assert set(chosen_event) - set(rejected_event) == {
                    required_key
                }
                assert set(rejected_event) - set(chosen_event) == {alias_key}
        else:
            assert "invented_shadow_slot" in rejected["decision"][
                "delegatedSlotIDs"
            ]

        if expected_kind in {
            "event_completeness_contract",
            "event_order",
            "event_id_grammar",
        }:
            event_ids = [event["id"] for event in rejected["events"]]
            positions = {
                event_id: index for index, event_id in enumerate(event_ids)
            }
            if expected_kind != "event_id_grammar":
                expected_ids = [
                    f"{rejected['scenarioID']}::event::{index:02d}"
                    for index in range(1, len(event_ids) + 1)
                ]
                assert event_ids == expected_ids
            if expected_kind != "event_id_grammar" or (
                fleet_artifact_module._event_id_grammar_contrast_mode(chosen)
                not in {"duplicate_identity", "missing_identity"}
            ):
                assert len(event_ids) == len(set(event_ids))
                assert all(
                    dependency["kind"] == "requires"
                    and dependency["fromEventID"] in positions
                    and dependency["toEventID"] in positions
                    and positions[dependency["fromEventID"]]
                    < positions[dependency["toEventID"]]
                    for dependency in rejected["dependencies"]
                )

        graph = scenario["graph"]
        decision = graph["decision"]
        metric = {
            "type": "orchestration_graph",
            "contract": {
                "metricVersion": "1.0.0",
                "graphSchemaVersion": graph["graphSchemaVersion"],
                "scenarioID": scenario["id"],
                "strategy": decision["strategy"],
                "knownSlotIDs": graph["knownSlotIDs"],
                "expectedDelegatedSlotIDs": decision["delegatedSlotIDs"],
                "expectedAggregationOwnerSlotID": decision[
                    "aggregationOwnerSlotID"
                ],
                "expectedStopReason": decision["stopReason"],
                "requiredEventTypes": [
                    event["type"] for event in graph["events"]
                ],
                "requiredDependencies": graph["dependencies"],
                "requiresCanonicalDerivation": True,
                "canonicalDerivation": scenario["canonicalDerivation"],
                "mustUseKnownSlotsOnly": True,
                "mustNotExposePrivateState": True,
                **scenario["evalConstraints"],
                "expectedCandidateHashSchemaVersion": (
                    "lumen.eval-candidate-hash/1.0.0"
                ),
                "expectedCandidateSHA256": canonical_sha256(graph),
                "expectedCandidateTopologyHashSchemaVersion": (
                    "lumen.eval-candidate-topology-hash/1.0.0"
                ),
                "expectedCandidateTopologySHA256": canonical_sha256(
                    fleet_artifact_module._orchestration_topology_contract(graph)
                ),
            },
        }
        chosen_result = _score_orchestration_graph(metric, chosen)
        rejected_result = _score_orchestration_graph(metric, rejected)

        if expected_kind == "event_id_grammar" and (
            fleet_artifact_module._event_id_grammar_contrast_mode(chosen)
            in {"duplicate_identity", "missing_identity"}
        ):
            rejected_reason = "event_id_invalid_or_duplicate"

        assert chosen_result["passed"] is True, behavior_class
        assert chosen_result["reason"] == "orchestration_graph_valid"
        assert rejected_result["passed"] is False, behavior_class
        assert rejected_result["reason"] == rejected_reason

    assert completeness_modes == {"interior_event", "terminal_stop"}
    assert len(event_type_aliases) == 9
    assert set(
        fleet_artifact_module._OBSERVED_EVENT_TYPE_ALIAS_BY_STRATEGY.values()
    ) <= set(event_type_aliases)
    assert Counter(payload_key_aliases) == Counter(
        {
            ("targetSlotID", "targetSlot"): 3,
            ("sourceSlotID", "sourceSlot"): 1,
            ("branchID", "branchIDs"): 1,
        }
    )
    assert payload_omission_count == 4
    assert "holdout" not in json.dumps(
        {
            "eventTypeAliases": (
                fleet_artifact_module._OBSERVED_EVENT_TYPE_ALIAS_BY_STRATEGY
            ),
            "payloadKeyAliases": (
                fleet_artifact_module._OBSERVED_EVENT_PAYLOAD_KEY_ALIASES
            ),
            "stopReasonAliases": (
                fleet_artifact_module._OBSERVED_STOP_REASON_ALIAS_BY_STRATEGY
            ),
            "omissionTargets": (
                fleet_artifact_module._REQUIRED_EVENT_OMISSION_TARGET_BY_STRATEGY
            ),
        },
        sort_keys=True,
    )
    assert event_id_modes == set(
        fleet_artifact_module._EVENT_ID_GRAMMAR_CONTRAST_BY_STRATEGY.values()
    )

    typed_guard_record = atomic_records[0]
    typed_guard_chosen = json.loads(typed_guard_record["chosen"]["content"])
    typed_guard_rejected = json.loads(typed_guard_record["rejected"]["content"])
    typed_guard_rejected["graphSchemaVersion"] = "tampered"
    with pytest.raises(
        ValueError,
        match="changes more than its typed contract dimension",
    ):
        fleet_artifact_module._validate_atomic_orchestration_rejection(
            typed_guard_chosen,
            typed_guard_rejected,
            mutation_kind=typed_guard_record["metadata"][
                "atomicPreferenceMutation"
            ],
        )


def test_native_fleet_orchestration_generation_is_deterministic():
    manifest = generate_manifest(Path(".").resolve())

    first = generate_fleet_artifacts(manifest)
    second = generate_fleet_artifacts(manifest)

    assert first.cross_model_training == second.cross_model_training
    assert first.orchestration_evals == second.orchestration_evals


def test_native_fleet_graph_prompts_match_initial_generation_runtime_contract():
    manifest = generate_manifest(Path(".").resolve())
    native = [
        record
        for record in generate_fleet_artifacts(manifest).cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
    ]
    retry_suffix = "\n\n" + generic_strict_json_retry_instruction(
        "invalid_json"
    )
    native_sft = [
        record
        for record in native
        if record["recordType"] == "sft"
    ]
    native_dpo = [
        record
        for record in native
        if record["recordType"] == "dpo"
    ]

    assert len(native_sft) == 90
    assert all(
        record["metadata"]["generationPromptMode"] == "initial_generation"
        and "retryFailureCode" not in record["metadata"]
        and not record["messages"][-2]["content"].endswith(retry_suffix)
        for record in native_sft
    )
    assert len(native_dpo) == 85
    assert all(
        record["metadata"]["generationPromptMode"] == "initial_generation"
        and "retryFailureCode" not in record["metadata"]
        and not record["prompt"][-1]["content"].endswith(retry_suffix)
        for record in native_dpo
    )
    assert all(
        [message["role"] for message in record["prompt"]] == ["system", "user"]
        for record in native_dpo
    )


def test_native_fleet_core_dpo_rejections_cover_observed_failure_families():
    manifest = generate_manifest(Path(".").resolve())
    preference_scenarios = {
        preference["id"]: preference
        for source in fleet_artifact_module._orchestration_training_scenarios(
            manifest
        )
        if source["trainingMatrixVariant"] == "core"
        and isinstance(source.get("rejectedGraph"), dict)
        for preference in [
            fleet_artifact_module._orchestration_preference_scenario(source)
        ]
    }
    records = [
        record
        for record in generate_fleet_artifacts(manifest).cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
        and record["recordType"] == "dpo"
        and record["metadata"]["requiredSplit"] == "train"
        and record["metadata"]["trainingMatrixVariant"] == "core"
    ]

    assert len(records) == 9
    assert {
        record["metadata"]["behaviorClass"] for record in records
    } == {
        "no-delegation",
        "sequential-dependencies",
        "parallel-dependencies",
        "context-handoff",
        "duplicate-suppression",
        "aggregation-owner",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }
    assert fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_FAMILY_MUTATIONS == {
        "parallel-dependencies": "top_level_dependencies_omission",
        "no-delegation": "event_type_vocabulary",
        "approval-boundary": "approval_request_event_vocabulary",
        "unavailable-boundary": "scenario_identity_role",
        "context-handoff": "decision_strategy_role",
        "aggregation-owner": "terminal_stop_reason",
        "nonexistent-slot-negative": "terminal_stop_reason",
        "duplicate-suppression": "event_payload_schema",
        "sequential-dependencies": "result_source_slot_role",
    }

    for record in records:
        metadata = record["metadata"]
        behavior = metadata["behaviorClass"]
        preference = preference_scenarios[metadata["scenarioID"]]
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        assert metadata["preferenceContrastMode"] == (
            fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_CONTRAST_MODE
        )
        assert metadata["coreFailureFamilyMutation"] == (
            fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_FAMILY_MUTATIONS[
                behavior
            ]
        )
        assert "compoundPreferenceDimensions" not in metadata
        assert chosen == preference["graph"]
        assert rejected == preference["rejectedGraph"]
        event_id_fact = (
            fleet_artifact_module._orchestration_event_id_negative_fact(
                preference["canonicalDerivation"]["facts"]
            )
        )
        assert rejected == fleet_artifact_module._core_failure_family_rejection(
            chosen,
            behavior=behavior,
            event_id_fact=event_id_fact,
        )

        if behavior == "parallel-dependencies":
            assert set(rejected) == set(chosen) - {"dependencies"}
        elif behavior == "approval-boundary":
            changed = [
                (chosen_event, rejected_event)
                for chosen_event, rejected_event in zip(
                    chosen["events"],
                    rejected["events"],
                    strict=True,
                )
                if chosen_event != rejected_event
            ]
            assert len(changed) == 1
            assert changed[0][0]["type"] == "request_user_approval"
            assert changed[0][1]["type"] == "user_approval_request"
        elif behavior == "unavailable-boundary":
            permission_check = preference["canonicalDerivation"]["facts"][
                "permissionCheckIdentifier"
            ]
            assert rejected["scenarioID"] == permission_check
            assert all(
                event["id"].startswith(permission_check + "::event::")
                for event in rejected["events"]
            )
        elif behavior == "context-handoff":
            assert rejected["decision"]["strategy"] == "context_handoff"
        elif behavior == "aggregation-owner":
            assert rejected["decision"]["stopReason"] == (
                "single_render_owner_complete"
            )
            assert rejected["events"][-1]["reason"] == (
                "single_render_owner_complete"
            )
        elif behavior == "sequential-dependencies":
            chosen_result = next(
                event
                for event in chosen["events"]
                if event["type"] == "result_received"
            )
            rejected_result = next(
                event
                for event in rejected["events"]
                if event["type"] == "result_received"
            )
            assert chosen_result["sourceSlotID"] == "executor"
            assert rejected_result["sourceSlotID"] == "rem"
        else:
            assert rejected["scenarioID"] == chosen["scenarioID"]

        expected_reason = {
            "approval-boundary": "event_type_schema_unknown",
            "unavailable-boundary": "scenario_id_mismatch",
            "aggregation-owner": "stop_reason_mismatch",
            "sequential-dependencies": "exact_candidate_hash_mismatch",
        }.get(behavior)
        if expected_reason is not None:
            decision = chosen["decision"]
            metric = {
                "type": "orchestration_graph",
                "contract": {
                    "metricVersion": "1.0.0",
                    "graphSchemaVersion": chosen["graphSchemaVersion"],
                    "scenarioID": preference["id"],
                    "strategy": decision["strategy"],
                    "knownSlotIDs": chosen["knownSlotIDs"],
                    "expectedDelegatedSlotIDs": decision[
                        "delegatedSlotIDs"
                    ],
                    "expectedAggregationOwnerSlotID": decision[
                        "aggregationOwnerSlotID"
                    ],
                    "expectedStopReason": decision["stopReason"],
                    "requiredEventTypes": [
                        event["type"] for event in chosen["events"]
                    ],
                    "requiredDependencies": chosen["dependencies"],
                    "requiresCanonicalDerivation": True,
                    "canonicalDerivation": preference[
                        "canonicalDerivation"
                    ],
                    "mustUseKnownSlotsOnly": True,
                    "mustNotExposePrivateState": True,
                    **preference["evalConstraints"],
                    "expectedCandidateHashSchemaVersion": (
                        "lumen.eval-candidate-hash/1.0.0"
                    ),
                    "expectedCandidateSHA256": canonical_sha256(chosen),
                    "expectedCandidateTopologyHashSchemaVersion": (
                        "lumen.eval-candidate-topology-hash/1.0.0"
                    ),
                    "expectedCandidateTopologySHA256": canonical_sha256(
                        fleet_artifact_module._orchestration_topology_contract(
                            chosen
                        )
                    ),
                },
            }
            rejected_result = _score_orchestration_graph(metric, rejected)
            assert rejected_result["passed"] is False
            assert rejected_result["reason"] == expected_reason


def test_core_native_metadata_binds_failure_family_to_exact_behavior():
    base = {
        "behaviorClass": "unavailable-boundary",
        "scenarioID": "id-abcdef",
        "trainingMatrixVariant": "core",
        "trainingTopologySHA256": "a" * 64,
        "sftOptimizerVisible": True,
        "generationPromptMode": "initial_generation",
        "preferenceType": "manifest_grounded_orchestration",
        "preferenceSourceScenarioID": "id-abcdef",
        "sftAnchorScenarioID": "id-abcdef",
        "sftAnchorBindingMode": "exact_scenario",
        "preferenceContrastMode": (
            fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_CONTRAST_MODE
        ),
        "coreFailureFamilyMutation": (
            fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_FAMILY_MUTATIONS[
                "unavailable-boundary"
            ]
        ),
    }
    resolved = fine_tuning_module._fleet_native_matrix_metadata(base)
    assert resolved["preferenceContrastMode"] == (
        fleet_artifact_module.ORCHESTRATION_CORE_FAILURE_CONTRAST_MODE
    )
    assert resolved["coreFailureFamilyMutation"] == "scenario_identity_role"

    with pytest.raises(ValueError, match="invalid core failure-family metadata"):
        fine_tuning_module._fleet_native_matrix_metadata(
            {**base, "behaviorClass": "parallel-dependencies"}
        )

    with pytest.raises(ValueError, match="invalid core failure-family metadata"):
        fine_tuning_module._fleet_native_matrix_metadata(
            {
                **base,
                "compoundPreferenceDimensions": ["unrelated_compound_error"],
            }
        )


def test_conditioned_approval_training_covers_missing_external_state():
    manifest = generate_manifest(Path(".").resolve())
    scenarios = [
        scenario
        for scenario in fleet_artifact_module._orchestration_training_scenarios(
            manifest
        )
        if scenario["behaviorClass"] == "approval-boundary"
        and scenario["trainingMatrixVariant"] == "behavior-conditioned"
    ]

    assert Counter(
        scenario["canonicalDerivation"]["facts"]["approvalState"]
        for scenario in scenarios
    ) == Counter({"missing": 4, "required": 4})
    for scenario in scenarios:
        policy = next(
            event
            for event in scenario["graph"]["events"]
            if event["type"] == "approval_policy_evaluated"
        )
        boundary = next(
            event
            for event in scenario["graph"]["events"]
            if event["type"] == "approval_boundary"
        )
        assert policy["approvalState"] == scenario["canonicalDerivation"][
            "facts"
        ]["approvalState"]
        assert boundary["approvalState"] == "required"


def test_conditioned_training_completion_envelopes_cover_frozen_graphs():
    manifest = generate_manifest(Path(".").resolve())
    training_by_behavior = defaultdict(list)
    for scenario in fleet_artifact_module._orchestration_training_scenarios(
        manifest
    ):
        if scenario["trainingMatrixVariant"] == "behavior-conditioned":
            training_by_behavior[scenario["behaviorClass"]].append(
                len(
                    fleet_artifact_module._serialize_orchestration_graph(
                        scenario["graph"]
                    )
                )
            )
    for frozen in fleet_artifact_module._orchestration_eval_scenarios(manifest):
        frozen_length = len(
            fleet_artifact_module._serialize_orchestration_graph(frozen["graph"])
        )
        assert max(training_by_behavior[frozen["behaviorClass"]]) >= (
            frozen_length + 64
        )


def test_conditioned_training_varies_a_nonrequest_payload_fact_per_behavior():
    manifest = generate_manifest(Path(".").resolve())
    identities_by_behavior = defaultdict(set)
    semantic_roles_by_behavior = defaultdict(set)

    def collect(value, identities, semantic_roles):
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "requestIdentifier":
                    collect(child, identities, semantic_roles)
        elif isinstance(value, list):
            for child in value:
                collect(child, identities, semantic_roles)
        elif (
            isinstance(value, str)
            and fleet_artifact_module._is_orchestration_training_fact_id(
                value
            )
        ):
            role = (
                fleet_artifact_module._semantic_orchestration_training_fact_role(
                    value
                )
            )
            assert role is not None
            if role != "request":
                identities.add(value)
                semantic_roles.add(role)

    for scenario in fleet_artifact_module._orchestration_training_scenarios(
        manifest
    ):
        if scenario["trainingMatrixVariant"] != "behavior-conditioned":
            continue
        behavior = scenario["behaviorClass"]
        collect(
            scenario["canonicalDerivation"]["facts"],
            identities_by_behavior[behavior],
            semantic_roles_by_behavior[behavior],
        )

    assert set(identities_by_behavior) == set(
        fleet_artifact_module._ORCHESTRATION_VARIED_FACT_KINDS_BY_BEHAVIOR
    )
    # Every conditioned replica contributes at least one distinct non-request
    # identity. Semantic prefixes now carry the visible role; punctuation and
    # length are no longer a meaningful uniqueness proxy.
    assert all(
        len(identities) >= 8
        for identities in identities_by_behavior.values()
    )
    required_semantic_roles = {
        "no-delegation": {
            "trusted-context-snapshot",
            "trusted-evidence",
        },
        "sequential-dependencies": {"executor-observation"},
        "parallel-dependencies": {
            "parallel-executor-branch",
            "parallel-mimicry-branch",
            "parallel-join",
        },
        "context-handoff": {"approved-action", "executor-result"},
        "duplicate-suppression": {
            "candidate-branch-a",
            "candidate-branch-b",
        },
        "aggregation-owner": {
            "aggregation-executor-result",
            "aggregation-mimicry-result",
            "validated-response",
        },
        "approval-boundary": {
            "approval-policy-snapshot",
            "user-approval-request",
        },
        "unavailable-boundary": {"permission-check"},
        "nonexistent-slot-negative": {
            "requested-unlisted-slot",
            "slot-directory-snapshot",
            "rejection-record",
        },
    }
    assert set(semantic_roles_by_behavior) == set(required_semantic_roles)
    for behavior, required_roles in required_semantic_roles.items():
        assert required_roles <= semantic_roles_by_behavior[behavior]
    assert len(
        {
            role
            for role in semantic_roles_by_behavior["duplicate-suppression"]
            if role.startswith("shared-work-")
        }
    ) == 8


def test_semantic_context_roles_reject_reserved_and_colliding_slugs():
    with pytest.raises(ValueError, match="source is unsafe"):
        fleet_artifact_module._semantic_training_fact_slug("scenario")

    conditions = fleet_artifact_module._orchestration_policy_conditions(
        behavior="sequential-dependencies",
        training_variant=(
            fleet_artifact_module.ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
        ),
    )
    with pytest.raises(ValueError, match="fact slug collision"):
        fleet_artifact_module._training_feature_facts(
            base_facts={
                "peerContext": {
                    "cortex": ["shared.context"],
                    "executor": ["shared_context"],
                    "mouth": ["renderContext"],
                }
            },
            behavior="sequential-dependencies",
            variant=(
                fleet_artifact_module.ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
            ),
            replica_index=0,
            conditions=conditions,
            identity_registry={},
        )


def test_conditioned_semantic_context_survives_sft_and_dpo_rebinding():
    manifest = generate_manifest(Path(".").resolve())

    def slug(value: str) -> str:
        separated = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
        return re.sub(r"[^a-z0-9]+", "-", separated.lower()).strip("-")

    conditioned = [
        scenario
        for scenario in fleet_artifact_module._orchestration_training_scenarios(
            manifest
        )
        if scenario["trainingMatrixVariant"] == "behavior-conditioned"
    ]
    assert len(conditioned) == 72
    assert all(
        fleet_artifact_module._is_semantic_orchestration_training_scenario_id(
            scenario["id"]
        )
        for scenario in conditioned
    )

    for scenario in conditioned:
        behavior = scenario["behaviorClass"]
        replica_index = scenario["behaviorConditionedInstanceIndex"] - 1
        facts = scenario["canonicalDerivation"]["facts"]
        preference = fleet_artifact_module._orchestration_preference_scenario(
            scenario
        )
        preference_facts = preference["canonicalDerivation"]["facts"]

        def assert_role_rebound(sft_value: str, dpo_value: str, role: str) -> None:
            assert (
                fleet_artifact_module._semantic_orchestration_training_fact_role(
                    sft_value
                )
                == role
            )
            assert (
                fleet_artifact_module._semantic_orchestration_training_fact_role(
                    dpo_value
                )
                == role
            )
            assert sft_value != dpo_value

        assert (
            fleet_artifact_module._is_semantic_orchestration_training_scenario_id(
                preference["id"]
            )
        )
        assert scenario["id"] != preference["id"]

        def assert_all_fact_roles_rebound(sft_value, dpo_value) -> None:
            if isinstance(sft_value, dict):
                assert isinstance(dpo_value, dict)
                assert set(sft_value) == set(dpo_value)
                for key in sft_value:
                    assert_all_fact_roles_rebound(
                        sft_value[key],
                        dpo_value[key],
                    )
                return
            if isinstance(sft_value, list):
                assert isinstance(dpo_value, list)
                assert len(sft_value) == len(dpo_value)
                for left, right in zip(sft_value, dpo_value, strict=True):
                    assert_all_fact_roles_rebound(left, right)
                return
            if not isinstance(sft_value, str) or not (
                fleet_artifact_module._is_orchestration_training_fact_id(
                    sft_value
                )
            ):
                return
            role = (
                fleet_artifact_module._semantic_orchestration_training_fact_role(
                    sft_value
                )
            )
            assert role is not None
            assert_role_rebound(sft_value, dpo_value, role)

        assert_all_fact_roles_rebound(facts, preference_facts)
        if behavior != "context-handoff":
            assert_role_rebound(
                facts["requestIdentifier"],
                preference_facts["requestIdentifier"],
                "request",
            )

        if behavior in {"sequential-dependencies", "parallel-dependencies"}:
            variants = (
                fleet_artifact_module._SEQUENTIAL_CONTEXT_VARIANTS
                if behavior == "sequential-dependencies"
                else fleet_artifact_module._PARALLEL_CONTEXT_VARIANTS
            )
            slots = (
                ("cortex", "executor", "mouth")
                if behavior == "sequential-dependencies"
                else ("cortex", "executor", "mimicry", "mouth")
            )
            for slot_id, source_values in zip(
                slots,
                variants[replica_index],
                strict=True,
            ):
                expected_roles = [slug(value) for value in source_values]
                assert [
                    fleet_artifact_module._semantic_orchestration_training_fact_role(
                        value
                    )
                    for value in facts["peerContext"][slot_id]
                ] == expected_roles
                for sft_value, dpo_value, role in zip(
                    facts["peerContext"][slot_id],
                    preference_facts["peerContext"][slot_id],
                    expected_roles,
                    strict=True,
                ):
                    assert_role_rebound(sft_value, dpo_value, role)
                assert fleet_artifact_module._delegation_to(
                    scenario["graph"],
                    slot_id,
                )["contextKeys"] == facts["peerContext"][slot_id]
                assert fleet_artifact_module._delegation_to(
                    preference["graph"],
                    slot_id,
                )["contextKeys"] == preference_facts["peerContext"][slot_id]
        elif behavior == "context-handoff":
            allowed, forbidden = fleet_artifact_module._HANDOFF_CONTEXT_VARIANTS[
                replica_index
            ]
            for fact_key, source_values, role_prefix in (
                ("allowedExecutorContext", allowed, "allowed-executor-context"),
                (
                    "forbiddenExecutorContext",
                    forbidden,
                    "forbidden-executor-context",
                ),
            ):
                expected_roles = [
                    f"{role_prefix}-{slug(value)}" for value in source_values
                ]
                for sft_value, dpo_value, role in zip(
                    facts[fact_key],
                    preference_facts[fact_key],
                    expected_roles,
                    strict=True,
                ):
                    assert_role_rebound(sft_value, dpo_value, role)
        elif behavior == "aggregation-owner":
            expected_roles = [
                f"render-context-{slug(value)}"
                for value in fleet_artifact_module._AGGREGATION_CONTEXT_VARIANTS[
                    replica_index
                ]
            ]
            for sft_value, dpo_value, role in zip(
                facts["renderContext"],
                preference_facts["renderContext"],
                expected_roles,
                strict=True,
            ):
                assert_role_rebound(sft_value, dpo_value, role)
        elif behavior == "duplicate-suppression":
            expected_role = (
                "shared-work-"
                + slug(
                    fleet_artifact_module._DUPLICATE_WORK_KEYS[replica_index]
                )
            )
            assert_role_rebound(
                facts["sharedWorkKey"],
                preference_facts["sharedWorkKey"],
                expected_role,
            )


def test_compact_training_identity_encoding_is_bounded_and_deterministic():
    compact = fleet_artifact_module._compact_orchestration_training_digest

    assert compact("0" * 64, width=6) == "a" * 6
    encoded = compact("f" * 64, width=8)
    assert re.fullmatch(r"[a-z]{8}", encoded)
    assert encoded == compact("f" * 64, width=8)
    with pytest.raises(ValueError, match="compact identity input is invalid"):
        compact("not-a-sha256", width=8)
    with pytest.raises(ValueError, match="compact identity input is invalid"):
        compact("0" * 64, width=0)


def test_training_generation_rejects_compact_identity_collisions(monkeypatch):
    manifest = generate_manifest(Path(".").resolve())
    scenarios = fleet_artifact_module._orchestration_training_scenarios(
        manifest
    )
    original_formatter = (
        fleet_artifact_module._format_orchestration_training_identity
    )

    def collide_scenarios(
        *, identity_class: str, digest: str, surface_index: int
    ) -> str:
        if identity_class == "scenario":
            return "id-aaaaaa"
        return original_formatter(
            identity_class=identity_class,
            digest=digest,
            surface_index=surface_index,
        )

    monkeypatch.setattr(
        fleet_artifact_module,
        "_format_orchestration_training_identity",
        collide_scenarios,
    )
    with pytest.raises(ValueError, match="scenario identity collision"):
        fleet_artifact_module._orchestration_training_scenarios(manifest)

    def collide_facts(
        *, identity_class: str, digest: str, surface_index: int
    ) -> str:
        if identity_class == "fact":
            return "id-aaaaaa"
        return original_formatter(
            identity_class=identity_class,
            digest=digest,
            surface_index=surface_index,
        )

    monkeypatch.setattr(
        fleet_artifact_module,
        "_format_orchestration_training_identity",
        collide_facts,
    )
    with pytest.raises(ValueError, match="fact identity collision"):
        fleet_artifact_module._orchestration_training_scenarios(manifest)

    preference_source = next(
        scenario
        for scenario in scenarios
        if isinstance(scenario.get("rejectedGraph"), dict)
        and scenario.get("trainingMatrixVariant")
        == "normalization-policy-audited"
    )

    def collide_preference_facts(
        *, identity_class: str, digest: str, surface_index: int
    ) -> str:
        if identity_class == "fact":
            return "id-aaaaaa"
        return original_formatter(
            identity_class=identity_class,
            digest=digest,
            surface_index=surface_index,
        )

    monkeypatch.setattr(
        fleet_artifact_module,
        "_format_orchestration_training_identity",
        collide_preference_facts,
    )
    with pytest.raises(ValueError, match="fact identity collision"):
        fleet_artifact_module._orchestration_preference_scenario(
            preference_source
        )


def test_native_training_identities_mix_opaque_and_role_bearing_shapes():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    native = [
        record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
    ]
    scenarios = fleet_artifact_module._orchestration_training_scenarios(
        manifest
    )
    preference_scenarios = [
        fleet_artifact_module._orchestration_preference_scenario(scenario)
        for scenario in scenarios
        if isinstance(scenario.get("rejectedGraph"), dict)
    ]

    all_scenario_ids = {
        scenario["id"] for scenario in [*scenarios, *preference_scenarios]
    }
    assert len(all_scenario_ids) == len(scenarios) + len(
        preference_scenarios
    )
    assert all(
        fleet_artifact_module._is_orchestration_training_scenario_id(
            scenario_id
        )
        for scenario_id in all_scenario_ids
    )
    semantic_scenario_ids = {
        scenario_id
        for scenario_id in all_scenario_ids
        if fleet_artifact_module._is_semantic_orchestration_training_scenario_id(
            scenario_id
        )
    }
    opaque_scenario_ids = all_scenario_ids - semantic_scenario_ids
    assert semantic_scenario_ids
    assert opaque_scenario_ids
    assert all(
        not fleet_artifact_module._is_semantic_orchestration_training_fact_id(
            scenario_id
        )
        for scenario_id in semantic_scenario_ids
    )
    identity_prefix = (
        fleet_artifact_module.ORCHESTRATION_TRAINING_IDENTITY_PREFIX
    )
    scenario_payloads = [
        scenario_id.removeprefix(identity_prefix)
        for scenario_id in opaque_scenario_ids
    ]
    assert any("-" in payload for payload in scenario_payloads)
    assert any("_" in payload for payload in scenario_payloads)
    assert max(map(len, scenario_payloads)) >= 30
    assert not any(
        scenario_id.startswith("train-") for scenario_id in all_scenario_ids
    )

    fact_ids: set[str] = set()

    def collect_fact_ids(value):
        if isinstance(value, dict):
            for child in value.values():
                collect_fact_ids(child)
        elif isinstance(value, list):
            for child in value:
                collect_fact_ids(child)
        elif isinstance(
            value, str
        ) and fleet_artifact_module._is_orchestration_training_fact_id(
            value
        ):
            fact_ids.add(value)

    for scenario in [*scenarios, *preference_scenarios]:
        scenario_id = scenario["id"]
        derivation = scenario["canonicalDerivation"]
        prompt = fleet_artifact_module._orchestration_training_prompt(
            scenario,
            derivation,
        )
        assert derivation["scenarioID"] == scenario_id
        assert prompt.count(scenario_id) == 1
        assert "Use matrix instance" not in prompt
        assert "Apply the enabled canonical behavior-class conditions" not in prompt
        assert scenario_id not in json.dumps(
            derivation["facts"],
            ensure_ascii=False,
            sort_keys=True,
        )
        assert all(
            event["id"]
            == f"{scenario_id}::event::{index:02d}"
            for index, event in enumerate(scenario["graph"]["events"], start=1)
        )
        collect_fact_ids(derivation["facts"])

    assert fact_ids
    semantic_fact_ids = {
        fact_id
        for fact_id in fact_ids
        if fleet_artifact_module._is_semantic_orchestration_training_fact_id(
            fact_id
        )
    }
    opaque_fact_ids = fact_ids - semantic_fact_ids
    assert semantic_fact_ids
    assert opaque_fact_ids
    assert all(
        fleet_artifact_module._semantic_orchestration_training_fact_role(fact_id)
        not in {None, "scenario"}
        for fact_id in semantic_fact_ids
    )
    fact_payloads = [
        fact_id.removeprefix(identity_prefix) for fact_id in opaque_fact_ids
    ]
    assert any("-" in payload for payload in fact_payloads)
    assert any("_" in payload for payload in fact_payloads)
    assert max(map(len, fact_payloads)) >= 30
    assert not any(fact_id.startswith("train-") for fact_id in fact_ids)
    assert all(
        scenario_id not in fact_id and fact_id not in scenario_id
        for scenario_id in all_scenario_ids
        for fact_id in fact_ids
    )
    assert all(
        not fleet_artifact_module._is_semantic_orchestration_training_scenario_id(
            fact_id
        )
        for fact_id in semantic_fact_ids
    )

    def visible_shape(value: str) -> str:
        return re.sub(r"[a-z]", "x", value.removeprefix(identity_prefix))

    assert {
        visible_shape(scenario_id) for scenario_id in opaque_scenario_ids
    } & {visible_shape(fact_id) for fact_id in opaque_fact_ids}

    for scenario in scenarios:
        variant = scenario["trainingMatrixVariant"]
        replica_index = (
            scenario.get("behaviorConditionedInstanceIndex", 0) - 1
            if variant == "behavior-conditioned"
            else None
        )
        expected_semantic = (
            variant == "behavior-conditioned"
            and replica_index
            in fleet_artifact_module._ORCHESTRATION_SEMANTIC_IDENTITY_REPLICA_INDICES
        )
        assert (
            fleet_artifact_module._is_semantic_orchestration_training_scenario_id(
                scenario["id"]
            )
            is expected_semantic
        )

    for record in native:
        content_fields = (
            [("sft", record["messages"][-1]["content"])]
            if record["recordType"] == "sft"
            else [
                ("chosen", record["chosen"]["content"]),
                ("rejected", record["rejected"]["content"]),
            ]
        )
        for output_kind, content in content_fields:
            graph = json.loads(content)
            expected_keys = [
                "graphSchemaVersion",
                "scenarioID",
                "knownSlotIDs",
                "events",
                "dependencies",
                "decision",
            ]
            if (
                output_kind == "rejected"
                and record["metadata"].get("coreFailureFamilyMutation")
                == "top_level_dependencies_omission"
            ):
                expected_keys.remove("dependencies")
            assert list(graph) == expected_keys
            assert all(
                list(event)[:2] == ["id", "type"]
                for event in graph["events"]
            )
            assert all(
                list(dependency)
                == ["fromEventID", "kind", "toEventID"]
                for dependency in graph.get("dependencies", [])
            )
            expected_decision_keys = [
                "strategy",
                "delegatedSlotIDs",
                "aggregationOwnerSlotID",
                "stopReason",
            ]
            if (
                output_kind == "rejected"
                and record["metadata"].get("coreFailureFamilyMutation")
                == "decision_aggregation_owner_omission"
            ):
                expected_decision_keys.remove("aggregationOwnerSlotID")
            assert list(graph["decision"]) == expected_decision_keys

    opaque_wrapper = next(
        scenario
        for scenario in preference_scenarios
        if scenario["trainingMatrixVariant"]
        == "normalization-policy-audited"
    )
    assert (
        fleet_artifact_module._derive_orchestration_graph_from_contract(
            opaque_wrapper["canonicalDerivation"]
        )
        == opaque_wrapper["graph"]
    )
    tampered = json.loads(
        json.dumps(opaque_wrapper["canonicalDerivation"], ensure_ascii=False)
    )
    tampered["scenarioID"] = "scenario-0000000000000000"
    with pytest.raises(ValueError, match="scenario identity is invalid"):
        fleet_artifact_module._derive_orchestration_graph_from_contract(
            tampered
        )


def test_context_handoff_event_id_negative_covers_dotted_separator():
    manifest = generate_manifest(Path(".").resolve())
    source = next(
        scenario
        for scenario in fleet_artifact_module._orchestration_training_scenarios(
            manifest
        )
        if scenario["behaviorClass"] == "context-handoff"
        and scenario["trainingMatrixVariant"] == "behavior-conditioned"
        and scenario["behaviorConditionedInstanceIndex"] == 5
    )
    assert source["atomicPreferenceMutation"] == "event_id_grammar"
    preference = fleet_artifact_module._orchestration_preference_scenario(
        source
    )
    facts = preference["canonicalDerivation"]["facts"]
    chosen = preference["graph"]
    rejected = preference["rejectedGraph"]
    request = chosen["events"][0]

    assert fleet_artifact_module._is_orchestration_training_fact_id(
        facts["approvedActionIdentifier"]
    )
    assert "requestIdentifier" not in facts
    assert "requestID" not in request
    assert request["actionID"] == facts["approvedActionIdentifier"]
    assert fleet_artifact_module._event_id_grammar_contrast_mode(chosen) == (
        "dotted_event_separator"
    )
    assert [event["id"] for event in rejected["events"]] == [
        f"{chosen['scenarioID']}::event.{index:02d}"
        for index in range(1, len(chosen["events"]) + 1)
    ]
    fleet_artifact_module._validate_atomic_orchestration_rejection(
        chosen,
        rejected,
        mutation_kind="event_id_grammar",
        event_id_fact=facts["approvedActionIdentifier"],
    )
    assert fleet_artifact_module._orchestration_topology_contract(
        rejected
    ) == fleet_artifact_module._orchestration_topology_contract(chosen)
    assert (
        fleet_artifact_module._derive_orchestration_graph_from_contract(
            preference["canonicalDerivation"]
        )
        == chosen
    )
    decision = chosen["decision"]
    metric = {
        "type": "orchestration_graph",
        "contract": {
            "metricVersion": "1.0.0",
            "graphSchemaVersion": chosen["graphSchemaVersion"],
            "scenarioID": preference["id"],
            "strategy": decision["strategy"],
            "knownSlotIDs": chosen["knownSlotIDs"],
            "expectedDelegatedSlotIDs": decision["delegatedSlotIDs"],
            "expectedAggregationOwnerSlotID": decision[
                "aggregationOwnerSlotID"
            ],
            "expectedStopReason": decision["stopReason"],
            "requiredEventTypes": [
                event["type"] for event in chosen["events"]
            ],
            "requiredDependencies": chosen["dependencies"],
            "requiresCanonicalDerivation": True,
            "canonicalDerivation": preference["canonicalDerivation"],
            "mustUseKnownSlotsOnly": True,
            "mustNotExposePrivateState": True,
            **preference["evalConstraints"],
            "expectedCandidateHashSchemaVersion": (
                "lumen.eval-candidate-hash/1.0.0"
            ),
            "expectedCandidateSHA256": canonical_sha256(chosen),
            "expectedCandidateTopologyHashSchemaVersion": (
                "lumen.eval-candidate-topology-hash/1.0.0"
            ),
            "expectedCandidateTopologySHA256": canonical_sha256(
                fleet_artifact_module._orchestration_topology_contract(chosen)
            ),
        },
    }
    assert _score_orchestration_graph(metric, chosen) == {
        "type": "orchestration_graph",
        "passed": True,
        "reason": "orchestration_graph_valid",
    }
    assert _score_orchestration_graph(metric, rejected) == {
        "type": "orchestration_graph",
        "passed": False,
        "reason": "event_id_grammar_mismatch",
    }


def test_native_fleet_dpo_instances_are_disjoint_from_sft_anchors():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    native = [
        record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
    ]
    sft = [record for record in native if record["recordType"] == "sft"]
    dpo = [record for record in native if record["recordType"] == "dpo"]
    sft_by_scenario = {
        record["metadata"]["scenarioID"]: record for record in sft
    }
    sft_prompt_hashes = {
        canonical_sha256(record["messages"][:-1]) for record in sft
    }
    dpo_prompt_hashes = {
        canonical_sha256(record["prompt"]) for record in dpo
    }
    sft_target_hashes = {
        canonical_sha256(json.loads(record["messages"][-1]["content"]))
        for record in sft
    }
    dpo_chosen_hashes = {
        canonical_sha256(json.loads(record["chosen"]["content"]))
        for record in dpo
    }

    assert dpo
    assert all(record["metadata"]["sftOptimizerVisible"] is True for record in sft)
    assert Counter(
        record["metadata"]["sftOptimizerVisible"] for record in dpo
    ) == Counter({True: 85})
    assert sft_prompt_hashes.isdisjoint(dpo_prompt_hashes)
    assert sft_target_hashes.isdisjoint(dpo_chosen_hashes)
    for record in dpo:
        serialized_record = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
        )
        assert "--dpo" not in serialized_record
        assert "--preference" not in serialized_record
        assert (
            "Resolve this independent Fleet request instance"
            not in serialized_record
        )
        metadata = record["metadata"]
        anchor = sft_by_scenario[metadata["sftAnchorScenarioID"]]
        assert metadata["behaviorClass"] == anchor["metadata"]["behaviorClass"]
        assert metadata["trainingTopologySHA256"] == anchor["metadata"][
            "trainingTopologySHA256"
        ]
        assert metadata["preferenceSourceScenarioID"]
        assert metadata["sftOptimizerVisible"] is True
        assert metadata["sftAnchorBindingMode"] == "exact_scenario"
        assert metadata["preferenceSourceScenarioID"] == metadata[
            "sftAnchorScenarioID"
        ]
        chosen = json.loads(record["chosen"]["content"])
        assert chosen["scenarioID"] == metadata["scenarioID"]

        def string_values(value):
            if isinstance(value, dict):
                return [
                    item
                    for child in value.values()
                    for item in string_values(child)
                ]
            if isinstance(value, list):
                return [
                    item
                    for child in value
                    for item in string_values(child)
                ]
            return [value] if isinstance(value, str) else []

        assert not any(
            value.startswith("holdout")
            for value in string_values(
                {
                    "prompt": record["prompt"],
                    "chosen": chosen,
                    "rejected": json.loads(record["rejected"]["content"]),
                }
            )
        )

    for source in fleet_artifact_module._orchestration_training_scenarios(
        manifest
    ):
        if not isinstance(source.get("rejectedGraph"), dict):
            continue
        preference = fleet_artifact_module._orchestration_preference_scenario(
            source
        )
        replacements = fleet_artifact_module._preference_graph_replacements(
            source["graph"],
            preference["graph"],
        )
        prompt = fleet_artifact_module._orchestration_training_prompt(
            preference,
            preference["canonicalDerivation"],
        )
        source_prompt = fleet_artifact_module._orchestration_training_prompt(
            source,
            source["canonicalDerivation"],
        )
        assert f"scenario `{preference['id']}`;" in prompt
        for original, rebound in replacements.items():
            if original == rebound:
                continue
            if original not in source_prompt:
                continue
            assert rebound in prompt, (
                source["id"],
                original,
                rebound,
            )


def test_native_fleet_holdouts_are_hash_bound_and_semantically_disjoint():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    training_scenarios = fleet_artifact_module._orchestration_training_scenarios(
        manifest
    )
    eval_scenarios = fleet_artifact_module._orchestration_eval_scenarios(manifest)
    training_by_class = defaultdict(list)
    eval_by_class = defaultdict(list)
    for scenario in training_scenarios:
        training_by_class[scenario["behaviorClass"]].append(scenario)
    for scenario in eval_scenarios:
        eval_by_class[scenario["behaviorClass"]].append(scenario)

    assert set(training_by_class) == set(eval_by_class)
    assert all(len(scenarios) == 10 for scenarios in training_by_class.values())
    assert all(len(scenarios) == 1 for scenarios in eval_by_class.values())

    training_exact_hashes = {
        canonical_sha256(scenario["graph"])
        for scenario in training_scenarios
    }
    training_topology_hashes = {
        canonical_sha256(
            fleet_artifact_module._orchestration_topology_contract(
                scenario["graph"]
            )
        )
        for scenario in training_scenarios
    }
    eval_exact_hashes = {
        canonical_sha256(scenario["graph"])
        for scenario in eval_scenarios
    }
    eval_topology_hashes = {
        canonical_sha256(
            fleet_artifact_module._orchestration_topology_contract(
                scenario["graph"]
            )
        )
        for scenario in eval_scenarios
    }
    assert len(training_exact_hashes) == 90
    assert len(eval_exact_hashes) == 9
    assert len(eval_topology_hashes) == 9
    assert training_exact_hashes.isdisjoint(eval_exact_hashes)
    assert eval_topology_hashes <= training_topology_hashes
    for scenarios in training_by_class.values():
        assert len(
            {
                canonical_sha256(
                    fleet_artifact_module._orchestration_topology_contract(
                        scenario["graph"]
                    )
                )
                for scenario in scenarios
            }
        ) == 2
        assert all(
            fleet_artifact_module._orchestration_prompt_reconstructs_graph(
                fleet_artifact_module._orchestration_training_prompt(scenario),
                scenario["graph"],
            )
            for scenario in scenarios
        )
        for scenario in scenarios:
            derivation = scenario["canonicalDerivation"]
            prompt = fleet_artifact_module._orchestration_training_prompt(
                scenario,
                derivation,
            )
            assert (
                fleet_artifact_module._derive_orchestration_graph_from_contract(
                    derivation
                )
                == scenario["graph"]
            )
            assert "policyProfile" not in derivation
            assert set(derivation["policyConditions"]) == set(
                fleet_artifact_module._ORCHESTRATION_POLICY_CONDITION_KEYS
            )
            assert '"events"' not in prompt
            assert '"dependencies"' not in prompt
            assert '"decision"' not in prompt
            assert prompt.count(
                fleet_artifact_module.ORCHESTRATION_OUTPUT_INTERFACE
            ) == 1
            for required_key in (
                "`decision`",
                "`dependencies`",
                "`events`",
                "`graphSchemaVersion`",
                "`knownSlotIDs`",
                "`scenarioID`",
                "`aggregationOwnerSlotID`",
                "`delegatedSlotIDs`",
                "`stopReason`",
                "`strategy`",
                "`fromEventID`",
                "`kind`",
                "`toEventID`",
            ):
                assert required_key in prompt
            assert "never emit the literal `<scenarioID>` placeholder" in prompt
            assert "no more than 12 events or 16 dependencies" in prompt
            assert "closed canonical enum token" in prompt
            assert "never skip or collapse that stage" in prompt
            assert "Input fact IDs are payload values only" in prompt
            assert "`peerContext` object is input-only" in prompt
            assert not any(
                f'"{event["type"]}"' in prompt
                for event in scenario["graph"]["events"]
            )

    training_enabled_conditions = {
        key
        for scenario in training_scenarios
        for key, enabled in scenario["canonicalDerivation"][
            "policyConditions"
        ].items()
        if enabled
    }
    holdout_enabled_conditions = {
        key
        for scenario in eval_scenarios
        for key, enabled in fleet_artifact_module._orchestration_derivation_contract(
            scenario
        )["policyConditions"].items()
        if enabled
    }
    assert holdout_enabled_conditions <= training_enabled_conditions
    for condition in holdout_enabled_conditions:
        supporting_training = [
            scenario
            for scenario in training_scenarios
            if scenario["canonicalDerivation"]["policyConditions"][condition]
        ]
        assert len(supporting_training) >= 2, condition
        assert len(
            {
                canonical_sha256(
                    fleet_artifact_module._orchestration_topology_contract(
                        scenario["graph"]
                    )
                )
                for scenario in supporting_training
            }
        ) >= 2, condition
    assert not any(
        "policyProfile"
        in fleet_artifact_module._orchestration_derivation_contract(scenario)
        for scenario in [*training_scenarios, *eval_scenarios]
    )

    def values_for_key(value, key):
        values = set()
        if isinstance(value, dict):
            if key in value:
                child = value[key]
                if isinstance(child, list):
                    values.update(item for item in child if isinstance(item, str))
                elif isinstance(child, str):
                    values.add(child)
            for child in value.values():
                values.update(values_for_key(child, key))
        elif isinstance(value, list):
            for child in value:
                values.update(values_for_key(child, key))
        return values

    training_graphs = [scenario["graph"] for scenario in training_scenarios]
    eval_graphs = [scenario["graph"] for scenario in eval_scenarios]
    for key in ("id", "workKey", "contextKeys", "requestedSlotID"):
        training_values = values_for_key(training_graphs, key)
        eval_values = values_for_key(eval_graphs, key)
        assert training_values.isdisjoint(eval_values), key

    for behavior_class, heldout in eval_by_class.items():
        heldout_topology_hash = canonical_sha256(
            fleet_artifact_module._orchestration_topology_contract(
                heldout[0]["graph"]
            )
        )
        topology_matches = [
            scenario
            for scenario in training_by_class[behavior_class]
            if canonical_sha256(
                fleet_artifact_module._orchestration_topology_contract(
                    scenario["graph"]
                )
            )
            == heldout_topology_hash
        ]
        heldout_event_types = [
            event["type"] for event in heldout[0]["graph"]["events"]
        ]
        assert len(topology_matches) == 9
        assert Counter(
            match["trainingMatrixVariant"] for match in topology_matches
        ) == Counter({"core": 1, "behavior-conditioned": 8})
        assert all(
            heldout_event_types
            == [event["type"] for event in match["graph"]["events"]]
            for match in topology_matches
        )
        assert all(
            heldout_event_types
            != [event["type"] for event in scenario["graph"]["events"]]
            for scenario in training_by_class[behavior_class]
            if scenario["trainingMatrixVariant"]
            not in {"core", "behavior-conditioned"}
        )
        canonical_training = [
            scenario
            for scenario in training_by_class[behavior_class]
            if scenario["trainingMatrixVariant"]
            in {"core", "behavior-conditioned"}
        ]
        assert len(canonical_training) == 9
        assert len(
            {canonical_sha256(scenario["graph"]) for scenario in canonical_training}
        ) == 9
        assert len(
            {
                canonical_sha256(scenario["canonicalDerivation"]["facts"])
                for scenario in canonical_training
            }
        ) == 9
        assert canonical_sha256(
            fleet_artifact_module._orchestration_derivation_contract(heldout[0])[
                "facts"
            ]
        ) not in {
            canonical_sha256(scenario["canonicalDerivation"]["facts"])
            for scenario in canonical_training
        }
        conditioned = [
            scenario
            for scenario in canonical_training
            if scenario["trainingMatrixVariant"] == "behavior-conditioned"
        ]
        heldout_fact_keys = set(
            fleet_artifact_module._orchestration_derivation_contract(
                heldout[0]
            )["facts"]
        )
        assert all(
            "behaviorInstanceFacts"
            not in scenario["canonicalDerivation"]["facts"]
            for scenario in conditioned
        )
        assert all(
            set(scenario["canonicalDerivation"]["facts"])
            == heldout_fact_keys
            for scenario in conditioned
        )
        assert len(
            {
                canonical_sha256(
                    fleet_artifact_module._orchestration_training_prompt(
                        scenario,
                        scenario["canonicalDerivation"],
                    )
                )
                for scenario in conditioned
            }
        ) == 8

    eval_records_by_id = {
        record["metadata"]["scenarioID"]: record
        for record in artifacts.orchestration_evals
    }
    assert set(eval_records_by_id) == {
        scenario["id"] for scenario in eval_scenarios
    }
    for scenario in eval_scenarios:
        record = eval_records_by_id[scenario["id"]]
        metadata = record["metadata"]
        prompt = record["messages"][0]["content"]
        assert metadata["expectedCandidateHashSchemaVersion"] == (
            "lumen.eval-candidate-hash/1.0.0"
        )
        assert metadata["expectedCandidateTopologyHashSchemaVersion"] == (
            "lumen.eval-candidate-topology-hash/1.0.0"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", metadata["expectedCandidateSHA256"])
        assert re.fullmatch(
            r"[0-9a-f]{64}", metadata["expectedCandidateTopologySHA256"]
        )
        assert metadata["expectedCandidateSHA256"] == canonical_sha256(
            scenario["graph"]
        )
        assert metadata["expectedCandidateTopologySHA256"] == canonical_sha256(
            fleet_artifact_module._orchestration_topology_contract(
                scenario["graph"]
            )
        )
        assert fleet_artifact_module._orchestration_prompt_reconstructs_graph(
            prompt,
            scenario["graph"],
        )
        assert prompt.count(
            fleet_artifact_module.ORCHESTRATION_OUTPUT_INTERFACE
        ) == 1
        derivation = record["expected"]["canonicalDerivation"]
        assert record["expected"]["requiresCanonicalDerivation"] is True
        assert fleet_artifact_module._derive_orchestration_graph_from_contract(
            derivation
        ) == scenario["graph"]
        assert canonical_sha256(scenario["graph"]) not in prompt
        assert json.dumps(
            scenario["graph"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) not in prompt
        assert scenario["graph"]["decision"]["stopReason"] not in prompt
        assert not any(
            event["id"] in prompt for event in scenario["graph"]["events"]
        )
        upgraded = upgrade_evaluation_record(
            {
                **record,
                "metadata": {
                    **record["metadata"],
                    "agent": "fleet",
                    "evalType": record["taskType"],
                },
            }
        )
        orchestration_metric = next(
            metric
            for metric in upgraded["metrics"]
            if metric.get("type") == "orchestration_graph"
        )
        assert _score_orchestration_graph(
            orchestration_metric,
            scenario["graph"],
        )["passed"] is True
        mutated_metric = json.loads(json.dumps(orchestration_metric))
        mutated_facts = mutated_metric["contract"]["canonicalDerivation"][
            "facts"
        ]
        first_fact = next(iter(mutated_facts))
        mutated_facts[first_fact] = "tampered-derivation-fact"
        assert _score_orchestration_graph(
            mutated_metric,
            scenario["graph"],
        )["reason"] in {
            "canonical_derivation_contract_invalid",
            "canonical_derivation_hash_mismatch",
        }
        assert "graph" not in record
        assert record["expected"] != scenario["graph"]

    boundary_classes = {"approval-boundary", "unavailable-boundary"}
    for behavior_class in boundary_classes:
        training_tool_ids = values_for_key(
            [scenario["graph"] for scenario in training_by_class[behavior_class]],
            "toolID",
        )
        eval_tool_ids = values_for_key(eval_by_class[behavior_class][0]["graph"], "toolID")
        assert training_tool_ids.isdisjoint(eval_tool_ids)


def test_native_fleet_holdout_guard_rejects_exact_and_invalid_topology_coverage():
    manifest = generate_manifest(Path(".").resolve())
    training = fleet_artifact_module._orchestration_training_scenarios(manifest)
    heldout = fleet_artifact_module._orchestration_eval_scenarios(manifest)[0]

    exact_collision = json.loads(json.dumps(training[0]))
    exact_collision["id"] = "renamed-exact-collision"
    with pytest.raises(ValueError, match="exact graph collides"):
        fleet_artifact_module._assert_orchestration_holdouts_disjoint(
            training_scenarios=training,
            eval_scenarios=[exact_collision],
        )

    controlled_matches = [
        scenario
        for scenario in training
        if scenario["behaviorClass"] == heldout["behaviorClass"]
        and scenario["trainingMatrixVariant"] == "behavior-conditioned"
    ]
    assert len(controlled_matches) == 8
    missing_coverage = [
        scenario for scenario in training if scenario not in controlled_matches
    ]
    with pytest.raises(
        ValueError,
        match=(
            "lacks controlled trained-topology coverage: .*observed=0 required=8"
        ),
    ):
        fleet_artifact_module._assert_orchestration_holdouts_disjoint(
            training_scenarios=missing_coverage,
            eval_scenarios=[heldout],
        )

    duplicate_match = json.loads(json.dumps(controlled_matches[0]))
    duplicate_match["id"] = "duplicate-controlled-topology"
    duplicate_match["graph"]["scenarioID"] = "duplicate-controlled-topology"
    id_map = {
        event["id"]: f"duplicate-{event['id']}"
        for event in duplicate_match["graph"]["events"]
    }
    for event in duplicate_match["graph"]["events"]:
        event["id"] = id_map[event["id"]]
    for dependency in duplicate_match["graph"]["dependencies"]:
        dependency["fromEventID"] = id_map[dependency["fromEventID"]]
        dependency["toEventID"] = id_map[dependency["toEventID"]]
    with pytest.raises(
        ValueError,
        match=(
            "lacks controlled trained-topology coverage: .*observed=9 required=8"
        ),
    ):
        fleet_artifact_module._assert_orchestration_holdouts_disjoint(
            training_scenarios=[*training, duplicate_match],
            eval_scenarios=[heldout],
        )

    repeated_match_training = [
        scenario
        for scenario in training
        if scenario not in controlled_matches
    ] + [controlled_matches[0]] * 8
    with pytest.raises(
        ValueError,
        match="trained-topology replicas are not distinct and complete",
    ):
        fleet_artifact_module._assert_orchestration_holdouts_disjoint(
            training_scenarios=repeated_match_training,
            eval_scenarios=[heldout],
        )


def test_native_orchestration_training_routes_only_to_fleet_adapter(
    production_fleet_compilation,
    production_compilation_without_fleet_artifacts,
):
    manifest, _, _, compiled = production_fleet_compilation
    fleet_orchestration_eval_classes = {
        record["metadata"]["behaviorClass"]
        for record in compiled["fleet"].eval
        if record.get("metadata", {}).get("evalType")
        == "fleet_orchestration_event_graph_eval"
    }
    assert len(compiled["fleet"].eval) == 15
    assert fleet_orchestration_eval_classes == {
        "no-delegation",
        "sequential-dependencies",
        "parallel-dependencies",
        "context-handoff",
        "duplicate-suppression",
        "aggregation-owner",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }
    complete_failures = validate_agent_fine_tuning_datasets(
        manifest,
        compiled,
    )
    assert "fleet_orchestration_eval_coverage_missing" not in {
        failure.code for failure in complete_failures
    }

    truncated = production_compilation_without_fleet_artifacts
    truncated_failures = validate_agent_fine_tuning_datasets(
        manifest,
        truncated,
    )
    assert "fleet_orchestration_eval_coverage_missing" in {
        failure.code for failure in truncated_failures
    }

    for agent, dataset in compiled.items():
        train_orchestration_sft = [
            record
            for record in dataset.train_sft
            if record["metadata"].get("sourceFamily") == "fleet_orchestration_native"
        ]
        val_orchestration_sft = [
            record
            for record in dataset.val_sft
            if record["metadata"].get("sourceFamily") == "fleet_orchestration_native"
        ]
        train_orchestration_dpo = [
            record
            for record in dataset.train_dpo
            if record["metadata"].get("taskType")
            == "fleet_orchestration_event_graph_preference"
        ]
        val_orchestration_dpo = [
            record
            for record in dataset.val_dpo
            if record["metadata"].get("taskType")
            == "fleet_orchestration_event_graph_preference"
        ]
        if agent == "fleet":
            assert len(train_orchestration_sft) == 81
            assert len(val_orchestration_sft) == 9
            assert len(train_orchestration_dpo) == 81
            assert len(val_orchestration_dpo) == 4
            assert len(train_orchestration_sft + val_orchestration_sft) == 90
            assert len(train_orchestration_dpo + val_orchestration_dpo) == 85
        else:
            assert not train_orchestration_sft
            assert not val_orchestration_sft
            assert not train_orchestration_dpo
            assert not val_orchestration_dpo


def test_fleet_writer_persists_and_mirrors_cross_model_artifacts(tmp_path: Path):
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)

    default_output = tmp_path / "default" / "agent_manifest"
    default_output.mkdir(parents=True)
    _write_fleet_artifacts(default_output, artifacts, None)

    eval_path = default_output / "cross_model_training" / "orchestration_evals.jsonl"
    written = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()]
    assert written == artifacts.orchestration_evals

    mirrored_output = tmp_path / "mirrored" / "agent_manifest"
    external_output = tmp_path / "mirrored" / "cross_model_training"
    mirrored_output.mkdir(parents=True)
    _write_fleet_artifacts(mirrored_output, artifacts, external_output)

    nested_output = mirrored_output / "cross_model_training"
    expected_files = {
        "cross_model_training.jsonl",
        "cross_model_training_index.csv",
        "dpo_train_cross.jsonl",
        "dpo_val_cross.jsonl",
        "orchestration_evals.jsonl",
        "train_sft_cross.jsonl",
        "val_sft_cross.jsonl",
    }
    assert {path.name for path in nested_output.iterdir()} == expected_files
    assert {path.name for path in external_output.iterdir()} == expected_files
    for filename in expected_files:
        assert (nested_output / filename).read_bytes() == (
            external_output / filename
        ).read_bytes()

    same_path_output = tmp_path / "same-path" / "agent_manifest"
    same_path_output.mkdir(parents=True)
    same_path_cross_model = same_path_output / "cross_model_training"
    _write_fleet_artifacts(
        same_path_output,
        artifacts,
        same_path_cross_model,
    )
    assert {path.name for path in same_path_cross_model.iterdir()} == expected_files
