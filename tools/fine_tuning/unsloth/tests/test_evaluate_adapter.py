from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumen_manifest_crawler.dataset import adapter_evaluation
from tools.fine_tuning.unsloth import evaluate_adapter


def _record(eval_id: str, *, agent: str = "executor") -> dict:
    return {
        "schemaVersion": adapter_evaluation.EVALUATION_SCHEMA_VERSION,
        "evalID": eval_id,
        "messages": [
            {"role": "system", "content": "Follow the contract."},
            {"role": "user", "content": "Select the tool."},
        ],
        "metrics": [{"type": "json_valid"}],
        "metadata": {
            "agent": agent,
            "evalType": "unit",
            "mustPass": True,
            "critical": True,
        },
        "weight": 1.0,
    }


class _FakeTensor:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.shape = (len(values), len(values[0]))

    def to(self, _device: str) -> "_FakeTensor":
        return self

    def __getitem__(self, index: int) -> list[int]:
        return self.values[index]


class _FakeModel:
    def __init__(self) -> None:
        self.generation_kwargs: list[dict] = []

    def parameters(self):
        yield SimpleNamespace(device="cuda:0")

    def generate(self, **kwargs):
        self.generation_kwargs.append(kwargs)
        input_ids = kwargs["input_ids"].values[0]
        return _FakeTensor([input_ids + [91, 92, 93]])


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = None

    def __init__(self, completions: list[str]) -> None:
        self.completions = iter(completions)
        self.template_kwargs: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs.append({"messages": messages, **kwargs})
        return {"input_ids": _FakeTensor([[1, 2, 3, 4]])}

    def decode(self, _tokens, **_kwargs):
        return next(self.completions)


def test_load_evaluation_records_upgrades_and_hashes_frozen_suite(
    tmp_path: Path,
) -> None:
    records = [_record("eval-one"), _record("eval-two")]
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded, digest = evaluate_adapter.load_evaluation_records(
        path,
        agent="executor",
        evaluation_module=adapter_evaluation,
    )

    assert [record["evalID"] for record in loaded] == ["eval-one", "eval-two"]
    assert digest == adapter_evaluation.canonical_sha256(loaded)


@pytest.mark.parametrize(
    ("records", "error"),
    [
        ([_record("same"), _record("same")], "duplicates evalID"),
        ([_record("wrong", agent="mouth")], "belongs to agent mouth"),
    ],
)
def test_load_evaluation_records_rejects_ambiguous_identity(
    tmp_path: Path,
    records: list[dict],
    error: str,
) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        evaluate_adapter.load_evaluation_records(
            path,
            agent="executor",
            evaluation_module=adapter_evaluation,
        )


def test_behavior_contract_is_strict_and_canonically_hashed(tmp_path: Path) -> None:
    manifest = {
        "schemaVersion": "1.0.0",
        "tools": [
            {
                "id": "weather",
                "arguments": [
                    {
                        "name": "location",
                        "type": "string",
                        "required": True,
                        "allowedValues": None,
                    }
                ],
            }
        ],
        "fleet": {"slots": [{"id": "cortex"}, {"id": "executor"}]},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    tools, slots, digest = evaluate_adapter.load_behavior_contract(path)

    assert set(tools) == {"weather"}
    assert slots == {"cortex", "executor"}
    assert digest == evaluate_adapter._canonical_sha256(manifest)


def test_scoring_contract_rejects_missing_tool_before_inference() -> None:
    record = _record("eval-one")
    record["metrics"] = [
        {
            "type": "manifest_tool_call",
            "expectedToolID": "weather",
            "validateArguments": True,
        }
    ]

    with pytest.raises(ValueError, match="missing evaluation tool contracts: weather"):
        evaluate_adapter.validate_scoring_contracts(
            [record],
            tool_contracts={},
            allowed_slots={"executor"},
        )


def test_finalized_manifest_must_bind_exact_frozen_evaluation(tmp_path: Path) -> None:
    payload = {
        "agent": "executor",
        "variant": "internal_plus_public_optimized",
        "sourceVariantManifestSHA256": "a" * 64,
        "frozenEvaluationSHA256": "b" * 64,
        "artifact": {"status": "trained", "adapterSHA256": "c" * 64},
    }
    payload["variantManifestSHA256"] = evaluate_adapter._canonical_sha256(payload)
    path = tmp_path / "finalized.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    fake_module = SimpleNamespace(
        canonical_sha256=evaluate_adapter._canonical_sha256,
        _valid_variant_manifest=lambda *_args, **_kwargs: True,
    )

    with pytest.raises(ValueError, match="not bound.*frozen evaluation"):
        evaluate_adapter.load_finalized_manifest(
            path,
            cfg={
                "agent": "executor",
                "variant": "internal_plus_public_optimized",
                "variantManifestSHA256": "a" * 64,
            },
            evaluation_sha256="d" * 64,
            evaluation_module=fake_module,
        )


def test_generation_is_deterministic_thinking_off_and_sequence_bounded() -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer(["result"])

    completion, input_count, generated_count = evaluate_adapter.generate_completion(
        model,
        tokenizer,
        _record("eval-one")["messages"],
        max_seq_length=6,
        max_new_tokens=1024,
        torch_module=SimpleNamespace(inference_mode=nullcontext),
    )

    assert completion == "result"
    assert input_count == 4
    assert generated_count == 3
    assert tokenizer.template_kwargs[0]["enable_thinking"] is False
    kwargs = model.generation_kwargs[0]
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert kwargs["max_new_tokens"] == 2
    assert kwargs["pad_token_id"] == tokenizer.eos_token_id


def test_json_roles_are_parsed_but_text_roles_remain_verbatim() -> None:
    parsed, kind, error = evaluate_adapter.normalize_candidate_output(
        "executor",
        '{"tool":"weather","arguments":{}}',
        evaluation_module=adapter_evaluation,
    )
    text, text_kind, text_error = evaluate_adapter.normalize_candidate_output(
        "mouth",
        '{"tool":"weather"}',
        evaluation_module=adapter_evaluation,
    )

    assert parsed == {"tool": "weather", "arguments": {}}
    assert (kind, error) == ("json_object", None)
    assert text == '{"tool":"weather"}'
    assert (text_kind, text_error) == ("text", None)


def test_malformed_json_output_is_preserved_as_failed_evidence() -> None:
    output, kind, error = evaluate_adapter.normalize_candidate_output(
        "cortex",
        "```json\n{}\n```",
        evaluation_module=adapter_evaluation,
    )

    assert output == "```json\n{}\n```"
    assert kind == "invalid_json"
    assert error == "invalid_json"


def test_evaluate_records_self_hashes_outputs_and_counts_format_failures() -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer(['{"status":"ready"}', "not-json"])
    records = [_record("eval-one"), _record("eval-two")]

    outputs, rows, failures = evaluate_adapter.evaluate_records(
        records,
        agent="executor",
        model=model,
        tokenizer=tokenizer,
        max_seq_length=64,
        max_new_tokens=8,
        evaluation_module=adapter_evaluation,
        torch_module=SimpleNamespace(inference_mode=nullcontext),
    )

    assert outputs == {"eval-one": {"status": "ready"}, "eval-two": "not-json"}
    assert failures == 1
    assert [row["outputKind"] for row in rows] == ["json_object", "invalid_json"]
    for row in rows:
        expected = row.pop("candidateRecordSHA256")
        assert expected == evaluate_adapter._canonical_sha256(row)


def test_candidate_output_loader_rejects_mutation_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    row = {
        "schemaVersion": evaluate_adapter.CANDIDATE_OUTPUT_SCHEMA_VERSION,
        "evalID": "eval-one",
        "agent": "executor",
        "output": {"status": "ready"},
        "outputKind": "json_object",
        "formatError": None,
        "inputTokenCount": 4,
        "generatedTokenCount": 3,
    }
    row["candidateRecordSHA256"] = evaluate_adapter._canonical_sha256(row)
    path = tmp_path / "candidate_outputs.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert evaluate_adapter.load_candidate_outputs(
        path,
        agent="executor",
    ) == {"eval-one": {"status": "ready"}}

    mutated = dict(row)
    mutated["output"] = {"status": "mutated"}
    path.write_text(json.dumps(mutated) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed candidate lineage validation"):
        evaluate_adapter.load_candidate_outputs(path, agent="executor")

    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed candidate lineage validation"):
        evaluate_adapter.load_candidate_outputs(path, agent="executor")
