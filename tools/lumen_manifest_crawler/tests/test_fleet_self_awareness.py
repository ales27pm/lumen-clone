import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from lumen_manifest_crawler import fleet_artifacts as fleet_artifact_module
from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
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
    ) == Counter({behavior_class: 5 for behavior_class in expected_classes})
    assert Counter(
        (record["recordType"], record["metadata"]["requiredSplit"])
        for record in native_training
    ) == Counter(
        {
            ("sft", "train"): 36,
            ("sft", "validation"): 9,
            ("dpo", "train"): 21,
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
            if record["metadata"]["trainingMatrixVariant"]
            == "normalization-policy-audited"
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
    full_matrix_dpo = {
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
    assert sum(map(len, train_sft.values())) == 36
    assert sum(map(len, train_dpo.values())) == 21
    assert all(len(records) == 4 for records in train_sft.values())
    assert all(
        len(records) == (4 if behavior in full_matrix_dpo else 1)
        for behavior, records in train_dpo.items()
    )
    val_sft = native_by_behavior(fleet.val_sft)
    val_dpo = native_by_behavior(fleet.val_dpo)
    assert set(val_sft) == expected_sft
    assert set(val_dpo) == full_matrix_dpo
    assert sum(map(len, val_sft.values())) == 9
    assert sum(map(len, val_dpo.values())) == 4
    assert all(len(records) == 1 for records in val_sft.values())
    assert all(len(records) == 1 for records in val_dpo.values())
    assert fleet.val_sft
    assert fleet.val_dpo

    expected_train_variants = {
        "core",
        "behavior-conditioned",
        "normalized-intake",
        "policy-audited",
    }
    for behavior, records in train_sft.items():
        assert {
            record["metadata"]["trainingMatrixVariant"]
            for record in records
        } == expected_train_variants
    for behavior, records in train_dpo.items():
        expected_variants = (
            expected_train_variants
            if behavior in full_matrix_dpo
            else {"behavior-conditioned"}
        )
        assert {
            record["metadata"]["trainingMatrixVariant"]
            for record in records
        } == expected_variants

    for grouped in (train_sft, train_dpo):
        for behavior, records in grouped.items():
            expected_topology_count = (
                4
                if grouped is train_sft or behavior in full_matrix_dpo
                else 1
            )
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
    for grouped in (val_sft, val_dpo):
        for records in grouped.values():
            assert {
                record["metadata"]["trainingMatrixVariant"]
                for record in records
            } == {"normalization-policy-audited"}


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

    full_matrix_classes = {
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
    assert sum(map(len, dpo_by_class.values())) == 25
    assert all(
        len(records) == (5 if behavior_class in full_matrix_classes else 1)
        for behavior_class, records in dpo_by_class.items()
    )
    for behavior_class, records in dpo_by_class.items():
        expected_variants = (
            {
                "core",
                "behavior-conditioned",
                "normalized-intake",
                "policy-audited",
                "normalization-policy-audited",
            }
            if behavior_class in full_matrix_classes
            else {"behavior-conditioned"}
        )
        assert {
            record["metadata"]["trainingMatrixVariant"] for record in records
        } == expected_variants
        for record in records:
            chosen = json.loads(record["chosen"]["content"])
            rejected = json.loads(record["rejected"]["content"])
            assert chosen["scenarioID"] == rejected["scenarioID"]
            assert canonical_sha256(chosen) != canonical_sha256(rejected)
            if record["metadata"]["trainingMatrixVariant"] == "behavior-conditioned":
                continue
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
                assert len(rejected_delegations) == 2
            else:
                assert not chosen_delegations
                assert len(rejected_delegations) == 1
    core_by_class = {
        behavior_class: next(
            record
            for record in records
            if record["metadata"]["trainingMatrixVariant"] == "core"
        )
        for behavior_class, records in dpo_by_class.items()
        if behavior_class in full_matrix_classes
    }
    duplicate_rejected = json.loads(core_by_class["duplicate-suppression"]["rejected"]["content"])
    duplicate_targets = [event["targetSlotID"] for event in duplicate_rejected["events"] if event["type"] == "delegate"]
    assert duplicate_targets == ["executor", "executor"]

    approval_chosen = json.loads(core_by_class["approval-boundary"]["chosen"]["content"])
    unavailable_chosen = json.loads(core_by_class["unavailable-boundary"]["chosen"]["content"])
    assert not approval_chosen["decision"]["delegatedSlotIDs"]
    assert not unavailable_chosen["decision"]["delegatedSlotIDs"]

    invalid_rejected = json.loads(core_by_class["nonexistent-slot-negative"]["rejected"]["content"])
    invalid_target = invalid_rejected["decision"]["delegatedSlotIDs"][0]
    assert invalid_target not in known_slots


def test_behavior_conditioned_preferences_are_atomic_and_scorer_invalid():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    scenarios_by_id = {
        scenario["id"]: scenario
        for scenario in fleet_artifact_module._orchestration_training_scenarios(
            manifest
        )
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
        "no-delegation": (
            "aggregation_owner_type",
            "$.decision.aggregationOwnerSlotID",
            "aggregation_owner_mismatch",
        ),
        "sequential-dependencies": (
            "event_type_vocabulary",
            "$.events[1].type",
            "event_type_schema_unknown",
        ),
        "parallel-dependencies": (
            "decision_strategy_literal",
            "$.decision.strategy",
            "strategy_mismatch",
        ),
        "context-handoff": (
            "decision_stop_reason_literal",
            "$.decision.stopReason",
            "stop_reason_mismatch",
        ),
        "duplicate-suppression": (
            "aggregation_owner_type",
            "$.decision.aggregationOwnerSlotID",
            "aggregation_owner_mismatch",
        ),
        "aggregation-owner": (
            "event_type_vocabulary",
            "$.events[1].type",
            "event_type_schema_unknown",
        ),
        "approval-boundary": (
            "decision_strategy_literal",
            "$.decision.strategy",
            "strategy_mismatch",
        ),
        "unavailable-boundary": (
            "decision_stop_reason_literal",
            "$.decision.stopReason",
            "stop_reason_mismatch",
        ),
        "nonexistent-slot-negative": (
            "aggregation_owner_type",
            "$.decision.aggregationOwnerSlotID",
            "aggregation_owner_mismatch",
        ),
    }

    assert len(atomic_records) == 9
    assert {
        record["metadata"]["behaviorClass"] for record in atomic_records
    } == set(expected_mutations)
    assert Counter(
        record["metadata"]["atomicPreferenceMutation"]
        for record in atomic_records
    ) == Counter(
        {
            "aggregation_owner_type": 3,
            "event_type_vocabulary": 2,
            "decision_strategy_literal": 2,
            "decision_stop_reason_literal": 2,
        }
    )

    for record in atomic_records:
        metadata = record["metadata"]
        behavior_class = metadata["behaviorClass"]
        expected_kind, expected_path, rejected_reason = expected_mutations[
            behavior_class
        ]
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        differences = fleet_artifact_module._orchestration_scalar_leaf_differences(
            chosen,
            rejected,
        )

        assert metadata["atomicPreferenceMutation"] == expected_kind
        assert differences == [expected_path]
        if expected_kind == "aggregation_owner_type":
            assert chosen["decision"]["aggregationOwnerSlotID"] is None
            assert rejected["decision"]["aggregationOwnerSlotID"] == "null"
        elif expected_kind == "event_type_vocabulary":
            assert rejected["events"][1]["type"] == "invented_event_alias"
        elif expected_kind == "decision_strategy_literal":
            assert rejected["decision"]["strategy"] == "invented_strategy"
        else:
            assert rejected["decision"]["stopReason"] == "invented_stop_reason"

        scenario = scenarios_by_id[metadata["scenarioID"]]
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

        assert chosen_result["passed"] is True, behavior_class
        assert chosen_result["reason"] == "orchestration_graph_valid"
        assert rejected_result["passed"] is False, behavior_class
        assert rejected_result["reason"] == rejected_reason


def test_native_fleet_orchestration_generation_is_deterministic():
    manifest = generate_manifest(Path(".").resolve())

    first = generate_fleet_artifacts(manifest)
    second = generate_fleet_artifacts(manifest)

    assert first.cross_model_training == second.cross_model_training
    assert first.orchestration_evals == second.orchestration_evals


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
    assert all(len(scenarios) == 5 for scenarios in training_by_class.values())
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
    assert len(training_exact_hashes) == 45
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
        ) == 5
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
        assert len(topology_matches) == 1
        assert topology_matches[0]["trainingMatrixVariant"] == (
            "behavior-conditioned"
        )
        assert heldout_event_types == [
            event["type"] for event in topology_matches[0]["graph"]["events"]
        ]
        assert all(
            heldout_event_types
            != [event["type"] for event in scenario["graph"]["events"]]
            for scenario in training_by_class[behavior_class]
            if scenario["trainingMatrixVariant"] != "behavior-conditioned"
        )

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

    controlled_match = next(
        scenario
        for scenario in training
        if scenario["behaviorClass"] == heldout["behaviorClass"]
        and scenario["trainingMatrixVariant"] == "behavior-conditioned"
    )
    missing_coverage = [
        scenario for scenario in training if scenario is not controlled_match
    ]
    with pytest.raises(
        ValueError,
        match=(
            "lacks controlled trained-topology coverage: .*observed=0 required=1"
        ),
    ):
        fleet_artifact_module._assert_orchestration_holdouts_disjoint(
            training_scenarios=missing_coverage,
            eval_scenarios=[heldout],
        )

    duplicate_match = json.loads(json.dumps(controlled_match))
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
            "lacks controlled trained-topology coverage: .*observed=2 required=1"
        ),
    ):
        fleet_artifact_module._assert_orchestration_holdouts_disjoint(
            training_scenarios=[*training, duplicate_match],
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
            assert len(train_orchestration_sft) == 36
            assert len(val_orchestration_sft) == 9
            assert len(train_orchestration_dpo) == 21
            assert len(val_orchestration_dpo) == 4
            assert len(train_orchestration_sft + val_orchestration_sft) == 45
            assert len(train_orchestration_dpo + val_orchestration_dpo) == 25
        else:
            assert not train_orchestration_sft
            assert not val_orchestration_sft
            assert not train_orchestration_dpo
            assert not val_orchestration_dpo


def test_fleet_writer_persists_orchestration_evals(tmp_path: Path):
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)

    _write_fleet_artifacts(tmp_path, artifacts, None)

    eval_path = tmp_path / "cross_model_training" / "orchestration_evals.jsonl"
    written = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()]
    assert written == artifacts.orchestration_evals
