"""JSON-compatible training records fail with typed diagnostics at the public API."""
import copy
import json
from dataclasses import replace

import pytest

from test_public_training_output_contract import (
    AGENTS,
    MANIFEST,
    ROUTE_SYSTEM,
    VALID_ACTION,
    VALID_ROUTE,
    base,
    card,
)
from lumen_manifest_crawler.manifest import ValidationReport
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets

def sft(agent):
    text = (
        VALID_ACTION if agent == "executor"
        else VALID_ROUTE if agent == "cortex"
        else "A complete synthetic response."
    )
    source = {
        "executor": "executor_tool_calls",
        "mouth": "mouth_responses",
        "mimicry": "mimicry_style",
        "rem": "rem_reflection",
    }.get(agent, "synthetic_route")
    return {
        "messages": [
            {"role": "system", "content": ROUTE_SYSTEM if agent == "cortex" else "Return the synthetic response."},
            {"role": "user", "content": "Synthetic request."},
            {"role": "assistant", "content": text},
        ],
        "metadata": {
            "agent": agent,
            "sourceFamily": source,
            "taskType": "synthetic",
            "toolIDs": ["fixture.numeric"],
        },
    }

def dpo(agent):
    row = sft(agent)
    return {
        "prompt": row["messages"][:2],
        "chosen": row["messages"][2],
        "rejected": {"role": "assistant", "content": "A different complete response."},
        "metadata": copy.deepcopy(row["metadata"]),
    }

def fixtures(agent):
    datasets = base()
    ds = datasets[agent]
    train = copy.deepcopy(ds.train_sft) or [sft(agent)]
    datasets[agent] = replace(
        ds,
        train_sft=train,
        train_dpo=[dpo(agent)],
        dataset_card={**ds.dataset_card, **card(train + ds.val_sft)},
    )
    return datasets


def set_collection(datasets, agent, field, value):
    datasets[agent] = replace(datasets[agent], **{field: copy.deepcopy(value)})


def set_record(datasets, agent, field, value):
    getattr(datasets[agent], field)[0] = copy.deepcopy(value)


def set_sft_field(datasets, agent, field, value):
    datasets[agent].train_sft[0][field] = copy.deepcopy(value)


def set_message_field(datasets, agent, index, field, value):
    datasets[agent].train_sft[0]["messages"][index][field] = copy.deepcopy(value)


def set_dpo_field(datasets, agent, field, value):
    datasets[agent].train_dpo[0][field] = copy.deepcopy(value)


def set_dpo_message_field(datasets, agent, part, field, value):
    datasets[agent].train_dpo[0][part][field] = copy.deepcopy(value)


def variant(datasets, agent, lane, value):
    ds = datasets[agent]
    payload = {
        "train_sft": copy.deepcopy(ds.train_sft),
        "val_sft": copy.deepcopy(ds.val_sft),
        "train_dpo": copy.deepcopy(ds.train_dpo),
        "val_dpo": [],
        "variant_manifest": {
            "validationSamplingPolicy": copy.deepcopy(
                ds.dataset_card.get("constraints", {}).get("fleetValidationSamplingPolicy", {})
            ),
        },
        "contamination_report": {},
    }
    payload[lane] = copy.deepcopy(value)
    datasets[agent] = replace(ds, experiment_variants={"synthetic_variant": payload})


def rejected(datasets):
    failures = validate_agent_fine_tuning_datasets(MANIFEST, datasets)
    assert failures
    ValidationReport(passed=False, failures=failures).model_dump_json().encode("utf-8")
    return failures


@pytest.mark.parametrize("agent", AGENTS)
def test_six_agent_synthetic_controls(agent):
    assert validate_agent_fine_tuning_datasets(MANIFEST, fixtures(agent)) == []


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("lane", ["train_sft", "val_sft", "train_dpo", "val_dpo", "eval"])
@pytest.mark.parametrize("value", [None, {}, "invalid"])
def test_nonlist_collections_return_diagnostics(agent, lane, value):
    datasets = fixtures(agent)
    set_collection(datasets, agent, lane, value)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("lane", ["train_sft", "train_dpo", "eval"])
@pytest.mark.parametrize("value", [None, [], True, "invalid"])
def test_nondict_records_return_diagnostics(agent, lane, value):
    datasets = fixtures(agent)
    if not getattr(datasets[agent], lane):
        datasets[agent] = replace(datasets[agent], **{lane: [value]})
    else:
        set_record(datasets, agent, lane, value)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("role", [None, [], {}, True, "unknown"])
def test_sft_roles_must_be_supported_strings(agent, role):
    datasets = fixtures(agent)
    set_message_field(datasets, agent, 1, "role", role)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("value", [None, [], {}, True])
def test_sft_json_user_content_compatibility_is_preserved(agent, value):
    datasets = fixtures(agent)
    set_message_field(datasets, agent, 1, "content", value)
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("value", [None, [], True, "invalid"])
def test_sft_metadata_malformed_returns_diagnostics(agent, value):
    datasets = fixtures(agent)
    set_sft_field(datasets, agent, "metadata", value)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("value", [[], True, "invalid"])
def test_present_dpo_metadata_must_be_an_object(agent, value):
    datasets = fixtures(agent)
    set_dpo_field(datasets, agent, "metadata", value)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
def test_null_optional_dpo_metadata_is_preserved(agent):
    datasets = fixtures(agent)
    set_dpo_field(datasets, agent, "metadata", None)
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("part", ["chosen", "rejected"])
@pytest.mark.parametrize("role", [None, [], {}, "user"])
def test_dpo_completion_role_must_be_assistant(agent, part, role):
    datasets = fixtures(agent)
    set_dpo_message_field(datasets, agent, part, "role", role)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("prompt", [
    [],
    [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
    [{"role": "assistant", "content": "unrequested"}],
    [{"role": "user", "content": "one"}, {"role": "system", "content": "late"}],
    [{"role": "user", "content": "one"}, {"role": "assistant", "content": "last"}],
    [{"role": "tool", "content": "unowned"}],
    [None],
])
def test_dpo_prompt_matches_the_training_consumer_contract(agent, prompt):
    datasets = fixtures(agent)
    set_dpo_field(datasets, agent, "prompt", prompt)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
def test_valid_dpo_multiturn_prompt_remains_supported(agent):
    datasets = fixtures(agent)
    datasets[agent].train_dpo[0]["prompt"].extend([
        {"role": "assistant", "content": "Earlier response."},
        {"role": "user", "content": "Follow up."},
    ])
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("part", ["chosen", "rejected"])
@pytest.mark.parametrize("content", ["", " "])
def test_dpo_completion_content_must_be_nonempty(agent, part, content):
    datasets = fixtures(agent)
    set_dpo_message_field(datasets, agent, part, "content", content)
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("lane", ["train_sft", "val_sft", "train_dpo", "val_dpo"])
@pytest.mark.parametrize("value", [None, [None]])
def test_all_variant_lanes_validate_their_record_boundaries(agent, lane, value):
    datasets = fixtures(agent)
    variant(datasets, agent, lane, value)
    rejected(datasets)


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("lane", ["train_sft", "val_sft", "train_dpo", "val_dpo"])
def test_all_variant_lanes_validate_preferred_outputs(agent, lane):
    datasets = fixtures(agent)
    row = sft(agent) if lane.endswith("sft") else dpo(agent)
    if lane.endswith("sft"):
        row["messages"][-1]["content"] = '{"probe":NaN}'
    else:
        row["chosen"]["content"] = '{"probe":NaN}'
    variant(datasets, agent, lane, [row])
    failures = rejected(datasets)
    assert any(".experiments.synthetic_variant." in failure.path for failure in failures)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "/absolute", "nested/child", "nested\\child"])
def test_variant_names_must_be_safe_single_path_components(agent, name):
    datasets = fixtures(agent)
    variant(datasets, agent, "train_dpo", [dpo(agent)])
    payload = datasets[agent].experiment_variants["synthetic_variant"]
    datasets[agent] = replace(datasets[agent], experiment_variants={name: payload})
    rejected(datasets)


@pytest.mark.parametrize("agent", AGENTS)
def test_valid_variant_payloads_remain_supported(agent):
    datasets = fixtures(agent)
    variant(datasets, agent, "train_dpo", [dpo(agent)])
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


@pytest.mark.parametrize("agent,case,expected", [
    (agent, case, expected)
    for agent in AGENTS
    for case, expected in [
        ("duplicate", "duplicate_sft_messages"),
        ("overlap", "sft_split_overlap"),
        ("conflict", "conflicting_sft_prompt_labels"),
        ("off_role", "off_role_sft_source"),
    ]
    if case != "off_role" or agent not in {"cortex", "fleet"}
])
def test_variant_collections_enforce_identity_splits_labels_and_role(agent, case, expected):
    datasets = fixtures(agent)
    first = copy.deepcopy(datasets[agent].train_sft[0])
    variant(datasets, agent, "train_sft", [first])
    payload = datasets[agent].experiment_variants["synthetic_variant"]
    payload["val_sft"] = []
    if case == "duplicate":
        payload["train_sft"].append(copy.deepcopy(first))
    elif case == "overlap":
        payload["val_sft"] = [copy.deepcopy(first)]
    elif case == "conflict":
        second = copy.deepcopy(first)
        if agent == "executor":
            text = '{"action":{"tool":"fixture.numeric","args":{"value":2}}}'
        elif agent == "cortex":
            text = VALID_ROUTE.replace("Use the declared fixture.", "Use a different reasoning summary.")
        else:
            text = "A different complete synthetic response."
        second["messages"][-1]["content"] = text
        payload["train_sft"].append(second)
    else:
        payload["train_sft"][0]["metadata"]["sourceFamily"] = "undeclared_synthetic_role_source"
    failures = rejected(datasets)
    assert any(
        failure.code == expected and ".experiments.synthetic_variant" in failure.path
        for failure in failures
    )


def test_prior_assistant_context_is_not_canonicalized_as_the_current_target():
    datasets = base()
    first = copy.deepcopy(datasets["executor"].train_sft[0])
    second = copy.deepcopy(first)
    second["messages"][2]["content"] = '{ "action": {"args": {"value": 1}, "tool": "fixture.numeric"}}'
    for row in [first, second]:
        row["messages"].extend([
            {"role": "user", "content": "Same followup, different raw history."},
            {"role": "assistant", "content": VALID_ACTION},
        ])
    datasets["executor"] = replace(datasets["executor"], train_sft=[first], val_sft=[second], dataset_card=card([first, second]))
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


def test_conflicting_nonfinal_assistant_targets_are_detected():
    datasets = base()
    first = copy.deepcopy(datasets["executor"].train_sft[0])
    second = copy.deepcopy(first)
    second["messages"][2]["content"] = VALID_ACTION.replace(":1}", ":2}")
    for index, row in enumerate([first, second]):
        row["messages"].extend([
            {"role": "user", "content": f"Different future input {index}."},
            {"role": "assistant", "content": VALID_ACTION},
        ])
    datasets["executor"] = replace(datasets["executor"], train_sft=[first], val_sft=[second], dataset_card=card([first, second]))
    failures = rejected(datasets)
    assert any(f.code == "conflicting_sft_prompt_labels" for f in failures)


@pytest.mark.parametrize("field", ["user_content", "config", "card", "variant_manifest"])
def test_unserializable_large_json_integer_returns_typed_failure(field):
    datasets = fixtures("executor")
    value = 10 ** 5000
    if field == "user_content":
        datasets["executor"].train_sft[0]["messages"][1]["content"] = value
    elif field == "config":
        datasets["executor"].unsloth_config["lora_r"] = value
    elif field == "card":
        datasets["executor"].dataset_card["synthetic_count"] = value
    else:
        variant(datasets, "executor", "train_dpo", [dpo("executor")])
        datasets["executor"].experiment_variants["synthetic_variant"]["variant_manifest"]["synthetic_count"] = value
    failures = rejected(datasets)
    assert any(failure.code == "invalid_training_json" for failure in failures)


def test_json_integer_with_two_thousand_digits_remains_supported():
    datasets = fixtures("executor")
    datasets["executor"].train_sft[0]["messages"][1]["content"] = 10 ** 2000
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


def test_json_container_beyond_serializer_recursion_limit_returns_diagnostic():
    datasets = fixtures("executor")
    nested = 0
    for _ in range(10000):
        nested = [nested]
    datasets["executor"].train_sft[0]["messages"][1]["content"] = nested
    failures = rejected(datasets)
    assert any(failure.code == "invalid_training_json" for failure in failures)


def test_shared_acyclic_json_containers_remain_supported():
    datasets = fixtures("executor")
    shared = {"synthetic": [1, 2, 3]}
    datasets["executor"].train_sft[0]["messages"][1]["content"] = [shared, shared]
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


@pytest.mark.parametrize("agent", ["executor", "fleet"])
@pytest.mark.parametrize("bad_text", ["\ud800", "\udc80"])
def test_surrogate_inside_json_serializable_tuple_is_rejected(agent, bad_text):
    datasets = fixtures(agent)
    datasets[agent].train_sft[0]["messages"][1]["content"] = ({"nested": (bad_text,)},)
    failures = rejected(datasets)
    assert any(failure.code == "invalid_training_unicode" for failure in failures)


@pytest.mark.parametrize("agent", ["executor", "fleet"])
def test_valid_json_serializable_tuple_remains_supported(agent):
    datasets = fixtures(agent)
    datasets[agent].train_sft[0]["messages"][1]["content"] = ({"nested": ("Astral \U0001f600",)},)
    assert validate_agent_fine_tuning_datasets(MANIFEST, datasets) == []


def test_cyclic_python_container_is_rejected_without_hanging():
    datasets = fixtures("executor")
    cycle = []
    cycle.append(cycle)
    datasets["executor"].train_sft[0]["messages"][1]["content"] = cycle
    failures = rejected(datasets)
    assert any(failure.code == "invalid_training_json" for failure in failures)


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("lane", ["train_dpo", "val_dpo", "variant_train_dpo", "variant_val_dpo"])
@pytest.mark.parametrize("part", ["chosen", "rejected"])
def test_dpo_forbidden_sentinels_rejected_only_in_preferred_output(agent, lane, part):
    datasets = fixtures(agent)
    manifest = MANIFEST.model_copy(deep=True)
    sentinel = "SYNTHETIC_PREFERENCE_CANARY"
    manifest.sentinels.forbiddenInUserOutput.append(sentinel)
    row = dpo(agent)
    if part == "chosen" and agent == "executor":
        text = json.dumps({"final": sentinel})
    elif part == "chosen" and agent == "cortex":
        text = VALID_ROUTE.replace("Use the declared fixture.", sentinel)
    else:
        text = f"A complete synthetic answer with {sentinel}."
    row[part]["content"] = text
    if lane.startswith("variant_"):
        variant(datasets, agent, lane.removeprefix("variant_"), [row])
    else:
        set_collection(datasets, agent, lane, [row])
    failures = validate_agent_fine_tuning_datasets(manifest, datasets)
    if part == "chosen":
        assert any(failure.code == "sentinel_leak" for failure in failures)
    else:
        assert failures == []


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("lane", ["train_sft", "val_sft", "train_dpo", "val_dpo"])
@pytest.mark.parametrize("in_variant", [False, True])
def test_structured_preferred_outputs_cannot_escape_sentinels_in_json(agent, lane, in_variant):
    datasets = fixtures(agent)
    manifest = MANIFEST.model_copy(deep=True)
    sentinel = "SYNTHETIC_ESCAPED_CANARY"
    manifest.sentinels.forbiddenInUserOutput.append(sentinel)
    escaped = "".join(f"\\u{ord(char):04x}" for char in sentinel)
    output = (
        '{"final":"' + escaped + '"}' if agent == "executor"
        else VALID_ROUTE.replace("Use the declared fixture.", escaped)
    )
    decoded = json.loads(output)
    assert decoded["final" if agent == "executor" else "reasoningSummary"] == sentinel
    row = sft(agent) if lane.endswith("sft") else dpo(agent)
    if lane.endswith("sft"):
        row["messages"][1]["content"] = "Distinct escaped-canary request."
        row["messages"][-1]["content"] = output
    else:
        row["chosen"]["content"] = output
    if in_variant:
        variant(datasets, agent, lane, [row])
    else:
        set_collection(datasets, agent, lane, [row])
        if lane.endswith("sft"):
            ds = datasets[agent]
            datasets[agent] = replace(ds, dataset_card={**ds.dataset_card, **card(ds.train_sft + ds.val_sft)})
    failures = validate_agent_fine_tuning_datasets(manifest, datasets)
    assert any(failure.code == "sentinel_leak" for failure in failures)


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("case", ["rejected", "literal_escape_text"])
def test_sentinel_scan_preserves_rejected_negatives_and_single_json_decoding(agent, case):
    datasets = fixtures(agent)
    manifest = MANIFEST.model_copy(deep=True)
    sentinel = "SYNTHETIC_ESCAPED_CANARY"
    manifest.sentinels.forbiddenInUserOutput.append(sentinel)
    escaped = "".join(f"\\u{ord(char):04x}" for char in sentinel)
    row = dpo(agent)
    if case == "rejected":
        row["rejected"]["content"] = '{"final":"' + escaped + '"}'
    elif agent == "executor":
        row["chosen"]["content"] = json.dumps({"final": escaped})
    else:
        payload = json.loads(VALID_ROUTE)
        payload["reasoningSummary"] = escaped
        row["chosen"]["content"] = json.dumps(payload)
    set_collection(datasets, agent, "train_dpo", [row])
    assert validate_agent_fine_tuning_datasets(manifest, datasets) == []
