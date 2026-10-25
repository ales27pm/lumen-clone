"""Public training validation and UTF-8 publication agree on Unicode scalars."""

import json
from dataclasses import asdict, replace

import pytest

from lumen_manifest_crawler.dataset.fine_tuning import AgentFineTuningDataset
from lumen_manifest_crawler.manifest import ValidationReport
from lumen_manifest_crawler.output.writer import _write_jsonl
from test_public_training_output_contract import (
    ROUTE_SYSTEM,
    VALID_ACTION,
    VALID_ROUTE,
    base,
    card,
    failures_for,
    sft,
)


def json_transport(datasets):
    """Prove the input is representable as ordinary UTF-8 JSON on the wire."""
    wire = json.dumps(
        {agent: asdict(dataset) for agent, dataset in datasets.items()},
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        agent: AgentFineTuningDataset(**payload)
        for agent, payload in json.loads(wire).items()
    }


def output_for(agent, text, *, escaped=False):
    if agent == "executor":
        if escaped:
            return '{"final":"' + text + '"}'
        return json.dumps({"final": text}, ensure_ascii=False)
    route = json.loads(VALID_ROUTE)
    if escaped:
        return VALID_ROUTE.replace("Use the declared fixture.", text)
    route["reasoningSummary"] = text
    return json.dumps(route, ensure_ascii=False)


def preference(agent):
    return {
        "prompt": [
            {"role": "system", "content": ROUTE_SYSTEM if agent == "cortex" else "Return a native output."},
            {"role": "user", "content": "Synthetic Unicode request."},
        ],
        "chosen": {"role": "assistant", "content": VALID_ROUTE if agent == "cortex" else VALID_ACTION},
        "rejected": {"role": "assistant", "content": "synthetic rejected alternative"},
    }


def assert_serializable_rejection(datasets):
    failures = failures_for(json_transport(datasets))
    assert failures, "An unrepresentable Unicode string cannot be a publishable training record"
    report = ValidationReport(passed=False, failures=failures)
    json.loads(report.model_dump_json().encode("utf-8"))
    return failures


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("split", ["train_sft", "val_sft"])
@pytest.mark.parametrize("surrogate", [chr(0xD800), chr(0xDC00)])
def test_public_sft_rejects_unpaired_surrogates_without_encoding_crash(agent, split, surrogate):
    datasets = base()
    getattr(datasets[agent], split)[0]["messages"][2]["content"] = output_for(agent, surrogate)
    assert_serializable_rejection(datasets)


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("lane", ["chosen", "rejected", "prompt"])
@pytest.mark.parametrize("surrogate", [chr(0xD800), chr(0xDC00)])
def test_public_dpo_rejects_unrepresentable_text_in_all_published_fields(agent, lane, surrogate):
    datasets = base()
    row = preference(agent)
    if lane == "chosen":
        row["chosen"]["content"] = output_for(agent, surrogate)
    elif lane == "rejected":
        row["rejected"]["content"] = surrogate
    else:
        row["prompt"][1]["content"] = surrogate
    datasets[agent].train_dpo.append(row)
    assert_serializable_rejection(datasets)


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("escape", [r"\ud800", r"\udc00", r"\udc00\ud800"])
def test_decoded_training_json_cannot_contain_unpaired_surrogate_scalars(agent, escape):
    datasets = base()
    raw = output_for(agent, escape, escaped=True)
    assert raw.isascii()
    # The training text itself is ASCII; its decoded output is not a scalar string.
    decoded = json.loads(raw)
    value = decoded.get("final", decoded.get("reasoningSummary"))
    with pytest.raises(UnicodeEncodeError):
        value.encode("utf-8")
    datasets[agent].train_sft[0]["messages"][2]["content"] = raw
    assert_serializable_rejection(datasets)


def test_duplicate_surrogate_key_rejection_has_utf8_serializable_diagnostics():
    datasets = base()
    datasets["cortex"].train_sft[0]["messages"][2]["content"] = r'{"\ud800":1,"\ud800":2}'
    failures = assert_serializable_rejection(datasets)
    assert any("json" in failure.code or "unicode" in failure.code for failure in failures)


@pytest.mark.parametrize("raw", [r'{"text":"\ud800"}', r'{"text":"\udc80"}'])
def test_fleet_cohort_hashing_does_not_decode_raw_text_into_unencodable_keys(raw):
    datasets = base()
    record = sft("fleet", raw, "Synthetic distinct validation request.")
    fleet = datasets["fleet"]
    datasets["fleet"] = replace(
        fleet,
        val_sft=[record],
        dataset_card={**fleet.dataset_card, **card([record])},
    )
    failures = failures_for(json_transport(datasets))
    # The synthetic cohort policy is now stale. This must be a serializable
    # diagnostic, rather than an exception while hashing the UTF-8 input text.
    assert any(failure.code == "fleet_validation_sampling_contract_invalid" for failure in failures)
    json.loads(ValidationReport(passed=False, failures=failures).model_dump_json().encode("utf-8"))


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("text", ["caf\u00e9", "cafe\u0301", "\U0001f600", r"\ud83d\ude00"])
def test_valid_astral_and_unicode_text_stays_valid_and_publishable(agent, text, tmp_path):
    datasets = base()
    escaped = text == r"\ud83d\ude00"
    raw = output_for(agent, text, escaped=escaped)
    datasets[agent].train_sft[0]["messages"][2]["content"] = raw
    datasets[agent].train_dpo.append({
        **preference(agent),
        "chosen": {"role": "assistant", "content": raw},
    })
    datasets = json_transport(datasets)
    assert failures_for(datasets) == []
    for name, rows in [("sft", datasets[agent].train_sft), ("dpo", datasets[agent].train_dpo)]:
        destination = tmp_path / f"{name}.jsonl"
        _write_jsonl(destination, rows)
        assert [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()] == rows


@pytest.mark.parametrize("agent", ["executor", "cortex"])
def test_utf8_representable_invalid_json_remains_allowed_as_rejected_dpo(agent, tmp_path):
    datasets = base()
    row = preference(agent)
    row["rejected"]["content"] = '{"invalid":NaN}'
    datasets[agent].train_dpo.append(row)
    datasets = json_transport(datasets)
    assert failures_for(datasets) == []
    destination = tmp_path / "rejected-negative.jsonl"
    _write_jsonl(destination, datasets[agent].train_dpo)
    assert json.loads(destination.read_text())["rejected"]["content"] == '{"invalid":NaN}'
