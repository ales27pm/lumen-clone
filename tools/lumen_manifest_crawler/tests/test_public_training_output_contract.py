"""Public training validation cannot be bypassed by metadata or earlier turns."""
import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest

from lumen_manifest_crawler.dataset.fine_tuning import AgentFineTuningDataset
from lumen_manifest_crawler.fleet_artifacts import generate_orchestration_evals
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolManifest, ToolArgumentManifest
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets

AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
ROUTE_SYSTEM = "Task mode: Cortex route mode. Return the route as JSON."
VALID_ROUTE = json.dumps({
    "intent": "probe",
    "selectedToolID": "fixture.numeric",
    "requiresApproval": False,
    "nextModel": "executor",
    "reasoningSummary": "Use the declared fixture.",
})
VALID_ACTION = '{"action":{"tool":"fixture.numeric","args":{"value":1}}}'
NONFINITE_ACTION = '{"action":{"tool":"fixture.numeric","args":{"value":NaN}}}'
MANIFEST = AgentBehaviorManifest(tools=[
    ToolManifest(
        id="fixture.numeric",
        arguments=[ToolArgumentManifest(name="value", type="number", required=True)],
    ),
])


def sft(agent, assistant, user):
    return {
        "messages": [
            {
                "role": "system",
                "content": ROUTE_SYSTEM if agent == "cortex" else "Return a native action or final.",
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "agent": agent,
            "toolIDs": ["fixture.numeric"],
            "sourceFamily": "executor_tool_calls" if agent == "executor" else "synthetic_route",
            "taskType": "tool_call_generation" if agent == "executor" else "route",
        },
    }


def card(records):
    source = Counter()
    task = Counter()
    for row in records:
        metadata = row.get("metadata") if isinstance(row, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        source[str(metadata.get("sourceFamily") or "unknown")] += 1
        task[str(metadata.get("taskType") or "unknown")] += 1
    return {"sourceFamilyCounts": dict(source), "taskTypeCounts": dict(task)}


def base():
    """Provide all six datasets so unrelated coverage failures cannot hide bypasses."""
    datasets = {}
    for agent in AGENTS:
        config = {
            "agent": agent,
            "base_model_name": "synthetic-no-model",
            "max_seq_length": 4096,
            "load_in_4bit": False,
            "lora_r": 1,
            "lora_alpha": 1,
            "learning_rate": 0.001,
            "dpo_rpo_alpha": 1.0 if agent == "fleet" else None,
            "dataset_dir": "unused-fixture",
            "output_dir": "unused-fixture",
        }
        datasets[agent] = AgentFineTuningDataset(
            agent=agent,
            train_sft=[],
            val_sft=[],
            train_dpo=[],
            val_dpo=[],
            eval=[],
            dataset_card=card([]),
            unsloth_config=config,
            contamination_report={},
            experiment_variants={},
            experiment_manifest={},
        )
    for agent, text in [("executor", VALID_ACTION), ("cortex", VALID_ROUTE)]:
        train = [sft(agent, text, "Use fixture one.")]
        val = [sft(agent, text.replace(":1}", ":2}"), "Use fixture two.")]
        datasets[agent] = replace(
            datasets[agent], train_sft=train, val_sft=val,
            dataset_card=card(train + val),
        )
    mouth_eval = {
        "expected": {},
        "messages": [
            {"role": "system", "content": "Answer completely."},
            {"role": "user", "content": "Check the sentinel fixture."},
        ],
        "metadata": {"evalType": "sentinel_suppression"},
        "metrics": [{"type": "complete_final_text"}],
    }
    datasets["mouth"] = replace(datasets["mouth"], eval=[mouth_eval])
    # Only scenario IDs satisfy the unrelated Fleet coverage guard.
    fleet_evals = [
        {
            "expected": {},
            "messages": [
                {"role": "system", "content": "Synthetic boundary."},
                {"role": "user", "content": "Synthetic request."},
            ],
            "metadata": {
                "evalType": "fleet_orchestration_event_graph_eval",
                "scenarioID": record["metadata"]["scenarioID"],
            },
        }
        for record in generate_orchestration_evals(MANIFEST)
    ]
    fleet_card = {
        **card([]),
        "constraints": {
            "fleetOrchestrationEvaluationRequired": False,
            "fleetValidationSamplingPolicy": {
                "schemaVersion": "lumen.fleet-validation-sampling/1.0.0",
                "status": "bounded_stratified_observation",
                "lane": "sft_validation",
                "maximumRecords": 1,
                "selectedRecordCount": 0,
                "candidateBeforePublicSelectionRecordCount": 0,
                "samplingInputRecordCount": 0,
                "rejectedByPublicSelectionCount": 0,
                "rejectedByValidationSamplingCount": 0,
                "explicitValidationAssignmentsPreserved": True,
                "allSamplingInputStrataPreserved": True,
                "optimizerLossShareCapsApplied": False,
                "selectionStrategy": "least_represented_stratum_then_salted_stable_hash_v1",
                "selectedCohortSHA256": hashlib.sha256(b"[]").hexdigest(),
                "samplingInputStratumCount": 0,
                "selectedStratumCount": 0,
            },
        },
    }
    datasets["fleet"] = replace(
        datasets["fleet"], eval=fleet_evals, dataset_card=fleet_card,
    )
    return datasets


def failures_for(datasets, manifest=MANIFEST):
    return validate_agent_fine_tuning_datasets(manifest, datasets)


def record_at(datasets, agent, split):
    return getattr(datasets[agent], split)[0]


def test_synthetic_public_baseline_has_no_failures():
    assert failures_for(base()) == []


@pytest.mark.parametrize("split", ["train_sft", "val_sft"])
@pytest.mark.parametrize("tool_ids", ["missing", None, "fixture.numeric", {}, True, [], ["fixture.numeric"]])
@pytest.mark.parametrize("output", [NONFINITE_ACTION, "not JSON"])
def test_executor_output_contract_does_not_depend_on_tool_id_metadata(split, tool_ids, output):
    datasets = base()
    row = record_at(datasets, "executor", split)
    row["messages"][2]["content"] = output
    if tool_ids == "missing":
        row["metadata"].pop("toolIDs")
    else:
        row["metadata"]["toolIDs"] = tool_ids
    failures = failures_for(datasets)
    assert any(f.code == "executor_non_json_output" for f in failures), failures


@pytest.mark.parametrize("missing", [False, True])
def test_executor_output_is_checked_even_when_sft_metadata_is_missing(missing):
    datasets = base()
    row = record_at(datasets, "executor", "train_sft")
    row["messages"][2]["content"] = NONFINITE_ACTION
    if missing:
        row.pop("metadata")
    else:
        row["metadata"] = None
    ds = datasets["executor"]
    datasets["executor"] = replace(ds, dataset_card=card(ds.train_sft + ds.val_sft))
    failures = failures_for(datasets)
    assert any(f.code == "executor_non_json_output" for f in failures), failures
    assert any(f.code == "missing_sft_metadata" for f in failures), failures


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("split", ["train_sft", "val_sft"])
@pytest.mark.parametrize("output", ["not JSON", '{"probe":NaN}', " ", None])
def test_every_assistant_turn_is_checked_through_public_validator(agent, split, output):
    datasets = base()
    row = record_at(datasets, agent, split)
    row["messages"].extend([
        {"role": "user", "content": "Continue the synthetic request."},
        {"role": "assistant", "content": output},
    ])
    failures = failures_for(datasets)
    expected = "empty_assistant_output" if output is None or output == " " else f"{agent}_non_json_output"
    assert any(f.code == expected for f in failures), failures


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("split", ["train_sft", "val_sft"])
def test_later_assistant_turn_cannot_leak_a_forbidden_sentinel(agent, split):
    datasets = base()
    manifest = MANIFEST.model_copy(deep=True)
    sentinel = "SYNTHETIC_TEST_CANARY"
    manifest.sentinels.forbiddenInUserOutput.append(sentinel)
    text = json.dumps({"final": sentinel}) if agent == "executor" else VALID_ROUTE.replace("Use the declared fixture.", sentinel)
    record_at(datasets, agent, split)["messages"].extend([
        {"role": "user", "content": "Continue the synthetic request."},
        {"role": "assistant", "content": text},
    ])
    failures = failures_for(datasets, manifest)
    assert any(f.code == "sentinel_leak" for f in failures), failures


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("split", ["train_sft", "val_sft"])
def test_valid_multiturn_sft_remains_supported(agent, split):
    datasets = base()
    text = VALID_ACTION if agent == "executor" else VALID_ROUTE
    record_at(datasets, agent, split)["messages"].extend([
        {"role": "user", "content": "Continue the synthetic request."},
        {"role": "assistant", "content": text},
    ])
    assert failures_for(datasets) == []


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("metadata", ["missing", None, "invalid metadata"])
@pytest.mark.parametrize("output", ["not JSON", '{"probe":NaN}'])
def test_chosen_dpo_contract_is_independent_of_optional_metadata(agent, metadata, output):
    datasets = base()
    row = {"prompt": [{"role": "system", "content": ROUTE_SYSTEM if agent == "cortex" else "Return a native output."}, {"role": "user", "content": "Use the synthetic fixture."}], "chosen": {"role": "assistant", "content": output}, "rejected": {"role": "assistant", "content": "invalid alternative"}}
    if metadata != "missing":
        row["metadata"] = metadata
    datasets[agent].train_dpo.append(row)
    failures = failures_for(datasets)
    expected = "executor_non_json_output" if agent == "executor" else "cortex_dpo_non_json_output"
    assert any(f.code == expected for f in failures), failures


@pytest.mark.parametrize("agent", ["executor", "cortex"])
def test_invalid_rejected_dpo_does_not_poison_a_valid_chosen_output(agent):
    datasets = base()
    chosen = VALID_ACTION if agent == "executor" else VALID_ROUTE
    datasets[agent].train_dpo.append({"prompt": [{"role": "system", "content": ROUTE_SYSTEM if agent == "cortex" else "Return a native output."}, {"role": "user", "content": "Use the synthetic fixture."}], "chosen": {"role": "assistant", "content": chosen}, "rejected": {"role": "assistant", "content": NONFINITE_ACTION}})
    assert failures_for(datasets) == []
