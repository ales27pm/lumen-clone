"""Training producers redact sentinels without repairing invalid JSON targets."""

import copy
import json
from dataclasses import replace

import pytest

from lumen_manifest_crawler.dataset.compiler import DatasetCompilerConfig, _build_manifest_grounding_cards
from lumen_manifest_crawler.dataset.fine_tuning import (
    _normalize_candidate_record,
    _scrub_forbidden_sentinels,
    _to_sft_record,
)
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets
from test_public_training_output_contract import MANIFEST, base, card


SWIFT_SENTINEL = r"\nAction: \(action.displayContent)\nObservation: \(compactScratchpadObservation(result))"


@pytest.mark.parametrize("location", [None, "internal_only", "internal_plus_public_baseline", "internal_plus_public_optimized"])
def test_real_manifest_sentinel_card_produces_safe_cortex_target(location):
    manifest = MANIFEST.model_copy(deep=True)
    manifest.sentinels.forbiddenInUserOutput = ["SYNTHETIC_SENTINEL", SWIFT_SENTINEL]
    source_card = next(
        record for record in _build_manifest_grounding_cards(manifest, DatasetCompilerConfig())
        if record["card"] == "sentinel_policy"
    )
    original = copy.deepcopy(source_card)
    normalized = _normalize_candidate_record(source_card, "manifest_grounding_cards")
    record = _to_sft_record(manifest, normalized, "cortex", {tool.id for tool in manifest.tools})
    assert record is not None
    target = json.loads(record["messages"][-1]["content"])
    assert target["forbiddenInUserOutput"] == ["[REDACTED_SENTINEL]"] * 2
    assert source_card == original
    assert record["messages"][1]["content"] == original["messages"][1]["content"]

    datasets = base()
    dataset = datasets["cortex"]
    if location is None:
        datasets["cortex"] = replace(dataset, train_sft=[record], dataset_card=card([record] + dataset.val_sft))
    else:
        datasets["cortex"] = replace(dataset, experiment_variants={location: {
            "train_sft": [record], "val_sft": [], "train_dpo": [], "val_dpo": [],
            "variant_manifest": {}, "contamination_report": {},
        }})
    assert validate_agent_fine_tuning_datasets(manifest, datasets) == []


def test_scrubber_redacts_once_decoded_json_values_and_keys_preserving_types():
    source = {"SYNTHETIC_SENTINEL": [SWIFT_SENTINEL, {"nested": "SYNTHETIC_SENTINEL"}, 7, False, None]}
    result = _scrub_forbidden_sentinels(json.dumps(source), ["SYNTHETIC_SENTINEL", SWIFT_SENTINEL])
    assert json.loads(result) == {"[REDACTED_SENTINEL]": ["[REDACTED_SENTINEL]", {"nested": "[REDACTED_SENTINEL]"}, 7, False, None]}


@pytest.mark.parametrize("source", [
    '{ "safe" : [true, 1, null], "text": "ordinary" }',
    '{"text":"\\u0053\\u0059\\u004e"}',
])
def test_json_without_decoded_sentinel_is_preserved_byte_for_byte(source):
    assert _scrub_forbidden_sentinels(source, ["SYNTHETIC_SENTINEL"]) == source


def test_literal_escape_text_is_not_decoded_twice():
    literal = "".join(f"\\u{ord(char):04x}" for char in "SYNTHETIC_SENTINEL")
    source = json.dumps({"text": literal})
    assert _scrub_forbidden_sentinels(source, ["SYNTHETIC_SENTINEL"]) == source


@pytest.mark.parametrize("source", [
    '{"text":"SYNTHETIC_SENTINEL",}',
    '{"text":"SYNTHETIC_SENTINEL","text":"duplicate"}',
    '{"value":NaN,"text":"SYNTHETIC_SENTINEL"}',
    '{"value":1e400,"text":"SYNTHETIC_SENTINEL"}',
])
def test_invalid_json_targets_are_preserved_for_strict_validation(source):
    assert _scrub_forbidden_sentinels(source, ["SYNTHETIC_SENTINEL"]) == source


def test_invalid_json_escape_cannot_be_healed_by_sentinel_replacement():
    source = '{"text":"\\q"}'
    assert _scrub_forbidden_sentinels(source, [r"\q"]) == source


def test_redaction_does_not_publish_a_decoded_unpaired_surrogate():
    source = '{"text":"\\ud800","canary":"SYNTHETIC_SENTINEL"}'
    assert _scrub_forbidden_sentinels(source, ["SYNTHETIC_SENTINEL"]) == source


def test_scrubbed_json_key_collision_fails_explicitly():
    source = json.dumps({"SYNTHETIC_SENTINEL": 1, "[REDACTED_SENTINEL]": 2})
    with pytest.raises(ValueError, match="^Sentinel redaction would create duplicate JSON keys$"):
        _scrub_forbidden_sentinels(source, ["SYNTHETIC_SENTINEL"])


def test_plain_text_redaction_remains_supported():
    assert _scrub_forbidden_sentinels("A SYNTHETIC_SENTINEL answer.", ["SYNTHETIC_SENTINEL"]) == "A [REDACTED_SENTINEL] answer."
