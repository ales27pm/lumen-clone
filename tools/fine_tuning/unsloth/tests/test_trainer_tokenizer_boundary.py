from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import pytest

from tools.fine_tuning.unsloth import train_dpo, train_sft, ubuntu_pipeline
from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.adapter_evaluation import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_BASE_MODEL_TOKENIZER_FILES,
    canonical_base_model_tokenizer_closure,
)


class _StableShapeTokenizer:
    chat_template = "{% generation %}"
    eos_token_id = 151_645

    def __init__(self, offset: int) -> None:
        self.offset = offset

    def _token_id(self, word: str) -> int:
        return self.offset + sum(word.encode("utf-8"))

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {
            "input_ids": [self._token_id(word) for word in value.split()]
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_assistant_tokens_mask: bool,
        enable_thinking: bool,
    ) -> dict[str, list[int]]:
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        assert return_assistant_tokens_mask is True
        assert enable_thinking is False
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        for message in messages:
            words = message["content"].split()
            token_count = len(words)
            input_ids.extend(self._token_id(word) for word in words)
            assistant_masks.extend(
                [1 if message["role"] == "assistant" else 0] * token_count
            )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "assistant_masks": assistant_masks,
        }


def _render_preference(
    row: dict[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    del tokenizer
    return row


def test_sft_transcript_detects_equal_length_token_id_drift(tmp_path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": "one two"},
                {"role": "assistant", "content": "three four"},
            ]
        }
    ]
    first = train_sft._preflight_sft_token_lengths(
        {"train": (records, tmp_path / "train.jsonl")},
        tokenizer=_StableShapeTokenizer(0),
        max_sequence_length=256,
    )
    second = train_sft._preflight_sft_token_lengths(
        {"train": (records, tmp_path / "train.jsonl")},
        tokenizer=_StableShapeTokenizer(10_000),
        max_sequence_length=256,
    )

    assert first["totalTokens"] == second["totalTokens"]
    assert first["assistantTargetTokens"] == second["assistantTargetTokens"]
    assert (
        first["tokenizationTranscriptSHA256"]
        != second["tokenizationTranscriptSHA256"]
    )


def test_sft_transcript_binds_row_order(tmp_path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": "alpha beta"},
                {"role": "assistant", "content": "gamma delta"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "epsilon zeta"},
                {"role": "assistant", "content": "eta theta"},
            ]
        },
    ]
    first = train_sft._preflight_sft_token_lengths(
        {"train": (records, tmp_path / "train.jsonl")},
        tokenizer=_StableShapeTokenizer(0),
        max_sequence_length=256,
    )
    second = train_sft._preflight_sft_token_lengths(
        {"train": (list(reversed(records)), tmp_path / "train.jsonl")},
        tokenizer=_StableShapeTokenizer(0),
        max_sequence_length=256,
    )

    assert first["totalTokens"] == second["totalTokens"]
    assert (
        first["tokenizationTranscriptSHA256"]
        != second["tokenizationTranscriptSHA256"]
    )


def test_preference_transcript_detects_equal_length_token_id_drift() -> None:
    rows = [
        {
            "prompt": "one two",
            "chosen": "three four",
            "rejected": "five six",
        }
    ]
    first = train_dpo._preflight_preference_token_lengths(
        {"train": rows},
        tokenizer=_StableShapeTokenizer(0),
        render_preference=_render_preference,
        max_prompt_length=128,
        max_sequence_length=256,
    )
    second = train_dpo._preflight_preference_token_lengths(
        {"train": rows},
        tokenizer=_StableShapeTokenizer(10_000),
        render_preference=_render_preference,
        max_prompt_length=128,
        max_sequence_length=256,
    )

    assert first["maximumTotalTokens"] == second["maximumTotalTokens"]
    assert (
        first["tokenizationTranscriptSHA256"]
        != second["tokenizationTranscriptSHA256"]
    )


def test_default_qwen_rejects_self_consistent_altered_tokenizer_closure() -> None:
    files = copy.deepcopy(DEFAULT_BASE_MODEL_TOKENIZER_FILES)
    files[0]["sha256"] = "0" * 64
    files[0]["huggingFaceBlobID"] = "0" * 64
    closure = canonical_base_model_tokenizer_closure(
        base_model_id=DEFAULT_BASE_MODEL_ID,
        base_model_revision=DEFAULT_BASE_MODEL_REVISION,
        files=files,
    )
    tokenizer_json = next(
        item for item in files if item["path"] == "tokenizer.json"
    )

    with pytest.raises(RuntimeError, match="trusted registry"):
        train_sft._validated_base_model_tokenizer_closure(
            {
                "base_model_name": DEFAULT_BASE_MODEL_ID,
                "baseModelID": DEFAULT_BASE_MODEL_ID,
                "baseModelRevision": DEFAULT_BASE_MODEL_REVISION,
                "baseModelTokenizerDigest": tokenizer_json["sha256"],
                "baseModelTokenizerFiles": files,
                "baseModelTokenizerClosureSHA256": train_sft._canonical_sha256(
                    closure
                ),
            }
        )


def test_trainer_rejects_split_base_model_identity() -> None:
    files = copy.deepcopy(DEFAULT_BASE_MODEL_TOKENIZER_FILES)
    closure = canonical_base_model_tokenizer_closure(
        base_model_id=DEFAULT_BASE_MODEL_ID,
        base_model_revision=DEFAULT_BASE_MODEL_REVISION,
        files=files,
    )
    tokenizer_json = next(
        item for item in files if item["path"] == "tokenizer.json"
    )

    with pytest.raises(
        RuntimeError,
        match="baseModelID must exactly match base_model_name",
    ):
        train_sft._validated_base_model_tokenizer_closure(
            {
                "base_model_name": DEFAULT_BASE_MODEL_ID,
                "baseModelID": "example/different-model",
                "baseModelRevision": DEFAULT_BASE_MODEL_REVISION,
                "baseModelTokenizerDigest": tokenizer_json["sha256"],
                "baseModelTokenizerFiles": files,
                "baseModelTokenizerClosureSHA256": (
                    train_sft._canonical_sha256(closure)
                ),
            }
        )


def test_trainer_has_no_shared_hub_tokenizer_runtime_reentry() -> None:
    source = inspect.getsource(train_sft)
    assert "_verify_hugging_face_tokenizer_file" not in source
    assert "hf_hub_download" not in source


def test_production_trainer_rejects_injected_global_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "run" / "configs" / "cortex.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_global_tokenizer_preflight",
        lambda **_: {
            "tokenizerClosure": {
                "schemaVersion": (
                    "lumen.global-tokenizer-snapshot/injected-test-double"
                )
            }
        },
    )

    with pytest.raises(RuntimeError, match="rejects injected"):
        train_sft._verify_prepared_global_tokenizer_preflight(
            {"agent": "cortex"},
            cfg_path=cfg_path,
            phase="sft",
            bound_preflight={},
        )


def test_global_comparison_precedes_expensive_trainer_construction() -> None:
    sft_source = inspect.getsource(train_sft.main)
    dpo_source = inspect.getsource(train_dpo.main)

    assert sft_source.index("_verify_prepared_global_tokenizer_preflight(") < (
        sft_source.index("FastLanguageModel.get_peft_model(")
    )
    dpo_global_boundary = dpo_source.index(
        "_verify_prepared_global_tokenizer_preflight("
    )
    dpo_policy_load = dpo_source.index("_load_sft_policy(")
    assert dpo_global_boundary < dpo_policy_load
    dpo_global_call = dpo_source[dpo_global_boundary:dpo_policy_load]
    assert 'phase="preference"' in dpo_global_call
    assert "phase=preference_trainer" not in dpo_global_call


def _qualified_upload_summary() -> dict[str, Any]:
    runtime_model_binding_sha256 = "8" * 64
    runtime_tokenizer_binding_sha256 = "9" * 64

    def phase_runtime_evidence(*, report_digest: str) -> dict[str, str]:
        return {
            "trainingReportFileSHA256": report_digest * 64,
            "runtimeModelBindingSHA256": runtime_model_binding_sha256,
            "runtimeTokenizerBindingSHA256": (
                runtime_tokenizer_binding_sha256
            ),
            "peftBaseModelIdentitySHA256": "a" * 64,
            "adapterTokenizerBindingSHA256": "b" * 64,
            "baseModelTokenizerSnapshotVerificationSHA256": "c" * 64,
            "baseModelRuntimeSnapshotVerificationSHA256": "d" * 64,
        }

    return {
        "promotionEligible": True,
        "qualification": "quality_gate_passed",
        "preferenceTraining": True,
        "trainingScope": "sft_preference",
        "evaluationStatus": "quality_gate_passed",
        "evaluationScope": "full",
        "status": "complete_without_gguf",
        "ggufStatus": "skipped_by_operator",
        "ggufConversionStatus": "skipped_by_operator",
        "ggufTensorEquivalenceStatus": "not_applicable",
        "executionPlanSHA256": "1" * 64,
        "baseModelID": DEFAULT_BASE_MODEL_ID,
        "baseModelRevision": DEFAULT_BASE_MODEL_REVISION,
        "baseModelTokenizerDigest": next(
            item["sha256"]
            for item in DEFAULT_BASE_MODEL_TOKENIZER_FILES
            if item["path"] == "tokenizer.json"
        ),
        "baseModelTokenizerFiles": copy.deepcopy(
            DEFAULT_BASE_MODEL_TOKENIZER_FILES
        ),
        "baseModelTokenizerClosureSHA256": "2" * 64,
        "baseModelTokenizerSnapshotPath": "/tmp/tokenizer-snapshot",
        "baseModelTokenizerSnapshotVerification": {
            "snapshotVerificationSHA256": "3" * 64,
        },
        "baseModelGenerationConfigFile": {
            "path": "generation_config.json",
            "sizeBytes": 1,
            "sha256": "4" * 64,
            "huggingFaceBlobID": "5" * 40,
        },
        "baseModelRuntimeSnapshotPath": "/tmp/runtime-snapshot",
        "baseModelRuntimeSnapshotVerification": {
            "snapshotVerificationSHA256": "6" * 64,
        },
        "runtimeBindingSmokeReport": "/tmp/runtime-binding-smoke.json",
        "runtimeBindingSmokeReportFileSHA256": "e" * 64,
        "runtimeBindingSmokeGateSHA256": "f" * 64,
        "runtimeBindingSmokeContractEvidence": [],
        "runtimeBindingSmokeBindingsByAgent": {
            "cortex": {
                "runtimeModelBindingSHA256": runtime_model_binding_sha256,
                "runtimeTokenizerBindingSHA256": (
                    runtime_tokenizer_binding_sha256
                ),
            },
        },
        "agents": {
            "cortex": {
                "sft": phase_runtime_evidence(report_digest="1"),
                "finalPhase": phase_runtime_evidence(report_digest="2"),
            },
        },
        "runManifestSHA256": "7" * 64,
    }


def test_upload_publication_binds_tokenizer_json_digest() -> None:
    summary = _qualified_upload_summary()

    publication = ubuntu_pipeline._upload_publication_contract(
        summary,
        allow_diagnostic_upload=False,
    )

    assert publication["baseModelTokenizerDigest"] == summary[
        "baseModelTokenizerDigest"
    ]
    assert publication["preferenceTraining"] is True
    assert publication["trainingScope"] == "sft_preference"
    assert publication["phaseRuntimeEvidenceByAgent"]["cortex"] == {
        "sft": summary["agents"]["cortex"]["sft"],
        "preference": summary["agents"]["cortex"]["finalPhase"],
    }


def test_upload_publication_rejects_split_tokenizer_digest() -> None:
    summary = _qualified_upload_summary()
    summary["baseModelTokenizerDigest"] = "0" * 64

    with pytest.raises(RuntimeError, match="not bound to tokenizer.json"):
        ubuntu_pipeline._upload_publication_contract(
            summary,
            allow_diagnostic_upload=False,
        )
