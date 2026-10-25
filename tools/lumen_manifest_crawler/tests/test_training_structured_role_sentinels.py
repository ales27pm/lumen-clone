"""Sentinel decoding follows each role's actual structured-output contract."""

import json
from dataclasses import replace

import pytest

from lumen_manifest_crawler.dataset.chat_template_contract import STRUCTURED_OUTPUT_INSTRUCTION
from lumen_manifest_crawler.dataset.fine_tuning import _bind_structured_output_instruction
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets
from test_public_training_output_contract import MANIFEST, card
from test_training_record_boundaries import dpo, fixtures, set_collection, sft, variant


SENTINEL = "SYNTHETIC_ROLE_CANARY"
ESCAPED = "".join(f"\\u{ord(char):04x}" for char in SENTINEL)


def prepared(agent, lane, *, chosen=True, bind=True, marker_position="suffix"):
    manifest = MANIFEST.model_copy(deep=True)
    manifest.sentinels.forbiddenInUserOutput.append(SENTINEL)
    datasets = fixtures(agent)
    row = sft(agent) if lane.endswith("sft") else dpo(agent)
    messages_key = "messages" if lane.endswith("sft") else "prompt"
    row[messages_key][1]["content"] = "A distinct synthetic structured-role request."
    if bind:
        _bind_structured_output_instruction(row, messages_key=messages_key)
        if marker_position == "prefix":
            row[messages_key][0]["content"] = STRUCTURED_OUTPUT_INSTRUCTION + "\n\nUse the synthetic fixture."
        elif marker_position == "middle":
            row[messages_key][0]["content"] = "Use the fixture.\n" + STRUCTURED_OUTPUT_INSTRUCTION + "\nEnd of fixture policy."
    output = '{"policy":"' + ESCAPED + '"}'
    if lane.endswith("sft"):
        row["messages"][-1]["content"] = output
    else:
        row["chosen" if chosen else "rejected"]["content"] = output
    return manifest, datasets, row


def install(datasets, agent, lane, row, in_variant):
    if in_variant:
        variant(datasets, agent, lane, [row])
    else:
        set_collection(datasets, agent, lane, [row])
        if lane.endswith("sft"):
            dataset = datasets[agent]
            datasets[agent] = replace(dataset, dataset_card={**dataset.dataset_card, **card(dataset.train_sft + dataset.val_sft)})


@pytest.mark.parametrize("agent", ["fleet", "rem", "mimicry"])
@pytest.mark.parametrize("lane", ["train_sft", "val_sft", "train_dpo", "val_dpo"])
@pytest.mark.parametrize("in_variant", [False, True])
def test_json_roles_reject_escaped_sentinels_in_preferred_outputs(agent, lane, in_variant):
    manifest, datasets, row = prepared(agent, lane, bind=agent == "mimicry")
    output = row["messages"][-1]["content"] if lane.endswith("sft") else row["chosen"]["content"]
    assert json.loads(output)["policy"] == SENTINEL
    install(datasets, agent, lane, row, in_variant)
    failures = validate_agent_fine_tuning_datasets(manifest, datasets)
    assert any(failure.code == "sentinel_leak" for failure in failures)


@pytest.mark.parametrize("agent", ["fleet", "rem", "mimicry"])
@pytest.mark.parametrize("lane", ["train_dpo", "val_dpo"])
@pytest.mark.parametrize("in_variant", [False, True])
def test_rejected_structured_role_negative_examples_remain_allowed(agent, lane, in_variant):
    manifest, datasets, row = prepared(agent, lane, chosen=False, bind=agent == "mimicry")
    install(datasets, agent, lane, row, in_variant)
    assert validate_agent_fine_tuning_datasets(manifest, datasets) == []


@pytest.mark.parametrize("agent", ["fleet", "rem", "mimicry"])
@pytest.mark.parametrize("lane", ["train_sft", "train_dpo"])
def test_structured_roles_do_not_decode_literal_escape_text_twice(agent, lane):
    manifest, datasets, row = prepared(agent, lane, bind=agent == "mimicry")
    output = json.dumps({"policy": ESCAPED})
    if lane.endswith("sft"):
        row["messages"][-1]["content"] = output
    else:
        row["chosen"]["content"] = output
    install(datasets, agent, lane, row, False)
    assert validate_agent_fine_tuning_datasets(manifest, datasets) == []


@pytest.mark.parametrize("lane", ["train_sft", "train_dpo"])
@pytest.mark.parametrize("in_variant", [False, True])
@pytest.mark.parametrize("marker", ["absent", "drifted", "duplicate"])
def test_mimicry_text_without_exact_once_json_instruction_is_not_interpreted(lane, in_variant, marker):
    manifest, datasets, row = prepared("mimicry", lane, bind=False)
    messages_key = "messages" if lane.endswith("sft") else "prompt"
    if marker == "drifted":
        row[messages_key][0]["content"] += "\nResponse format contract: use the preferred display text."
    elif marker == "duplicate":
        row[messages_key][0]["content"] += "\n" + STRUCTURED_OUTPUT_INSTRUCTION * 2
    install(datasets, "mimicry", lane, row, in_variant)
    assert validate_agent_fine_tuning_datasets(manifest, datasets) == []


@pytest.mark.parametrize("lane", ["train_sft", "train_dpo"])
@pytest.mark.parametrize("position", ["prefix", "middle", "suffix"])
def test_exact_mimicry_instruction_contract_does_not_require_suffix_placement(lane, position):
    manifest, datasets, row = prepared("mimicry", lane, marker_position=position)
    install(datasets, "mimicry", lane, row, False)
    assert any(failure.code == "sentinel_leak" for failure in validate_agent_fine_tuning_datasets(manifest, datasets))
