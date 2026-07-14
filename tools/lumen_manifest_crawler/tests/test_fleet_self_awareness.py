import json
from pathlib import Path

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset.fine_tuning import compile_agent_fine_tuning_datasets
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts
from lumen_manifest_crawler.output.writer import _write_fleet_artifacts


def test_fleet_artifacts_include_source_code_map_and_whole_system_records():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)

    assert artifacts.system_prompts
    assert artifacts.cross_model_training
    assert "## System Identity" in artifacts.markdown
    assert "## Source Integrity" in artifacts.markdown

    first_prompt = next(iter(artifacts.system_prompts.values()))
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


def test_native_fleet_orchestration_covers_event_graph_boundaries_and_eval_contracts():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)
    native_training = [
        record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native"
    ]
    sft_records = [record for record in native_training if record.get("recordType") == "sft"]
    scenario_ids = {record["metadata"]["scenarioID"] for record in sft_records}
    expected_scenarios = {
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

    assert scenario_ids == expected_scenarios
    assert {record["metadata"]["scenarioID"] for record in artifacts.orchestration_evals} == expected_scenarios
    assert all(record["taskType"] == "fleet_orchestration_event_graph" for record in sft_records)
    assert all(record["taskType"] == "fleet_orchestration_event_graph_eval" for record in artifacts.orchestration_evals)


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
        assert graph["events"][-1] == {
            "id": "stop",
            "type": "stop",
            "reason": graph["decision"]["stopReason"],
        }
        assert graph["decision"]["stopReason"]
        for dependency in graph["dependencies"]:
            assert dependency["fromEventID"] in event_ids
            assert dependency["toEventID"] in event_ids
            assert dependency["kind"] == "requires"

    eval_by_scenario = {
        record["metadata"]["scenarioID"]: record["expected"]
        for record in artifacts.orchestration_evals
    }
    assert eval_by_scenario["no-delegation"]["mustNotDelegate"] is True
    assert eval_by_scenario["sequential-dependencies"]["mustRespectDependencyOrder"] is True
    assert eval_by_scenario["parallel-dependencies"]["mustWaitForAllDependenciesBeforeAggregation"] is True
    assert eval_by_scenario["context-handoff"]["forbiddenContextKeys"] == [
        "rawConversation",
        "privatePeerState",
        "hiddenReasoning",
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
    dpo_by_scenario = {
        record["metadata"]["scenarioID"]: record
        for record in artifacts.cross_model_training
        if record.get("sourceFamily") == "fleet_orchestration_native" and record.get("recordType") == "dpo"
    }

    assert set(dpo_by_scenario) == {
        "duplicate-suppression",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }
    duplicate_rejected = json.loads(dpo_by_scenario["duplicate-suppression"]["rejected"]["content"])
    duplicate_targets = [event["targetSlotID"] for event in duplicate_rejected["events"] if event["type"] == "delegate"]
    assert duplicate_targets == ["executor", "executor"]

    approval_chosen = json.loads(dpo_by_scenario["approval-boundary"]["chosen"]["content"])
    unavailable_chosen = json.loads(dpo_by_scenario["unavailable-boundary"]["chosen"]["content"])
    assert not approval_chosen["decision"]["delegatedSlotIDs"]
    assert not unavailable_chosen["decision"]["delegatedSlotIDs"]

    invalid_rejected = json.loads(dpo_by_scenario["nonexistent-slot-negative"]["rejected"]["content"])
    invalid_target = invalid_rejected["decision"]["delegatedSlotIDs"][0]
    assert invalid_target not in known_slots


def test_native_fleet_orchestration_generation_is_deterministic():
    manifest = generate_manifest(Path(".").resolve())

    first = generate_fleet_artifacts(manifest)
    second = generate_fleet_artifacts(manifest)

    assert first.cross_model_training == second.cross_model_training
    assert first.orchestration_evals == second.orchestration_evals


def test_native_orchestration_training_routes_only_to_fleet_adapter():
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)

    compiled = compile_agent_fine_tuning_datasets(manifest, {}, fleet_artifacts=artifacts)
    for agent, dataset in compiled.items():
        sft = [*dataset.train_sft, *dataset.val_sft]
        dpo = [*dataset.train_dpo, *dataset.val_dpo]
        orchestration_sft = [
            record
            for record in sft
            if record["metadata"].get("sourceFamily") == "fleet_orchestration_native"
        ]
        orchestration_dpo = [
            record
            for record in dpo
            if record["metadata"].get("taskType") == "fleet_orchestration_event_graph_preference"
        ]
        if agent == "fleet":
            assert len(orchestration_sft) == 9
            assert len(orchestration_dpo) == 4
        else:
            assert not orchestration_sft
            assert not orchestration_dpo


def test_fleet_writer_persists_orchestration_evals(tmp_path: Path):
    manifest = generate_manifest(Path(".").resolve())
    artifacts = generate_fleet_artifacts(manifest)

    _write_fleet_artifacts(tmp_path, artifacts, None)

    eval_path = tmp_path / "cross_model_training" / "orchestration_evals.jsonl"
    written = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()]
    assert written == artifacts.orchestration_evals
