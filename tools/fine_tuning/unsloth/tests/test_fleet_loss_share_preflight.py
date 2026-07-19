from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tools.fine_tuning.unsloth import train_dpo, train_sft, ubuntu_pipeline


BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
BASE_MODEL_REVISION = "a" * 40
TOKENIZER_SHA256 = "b" * 64
TOKENIZER_FILES = [
    {
        "path": filename,
        "sizeBytes": 1,
        "sha256": TOKENIZER_SHA256,
        "huggingFaceBlobID": "c" * 40,
    }
    for filename in ubuntu_pipeline.GLOBAL_TOKENIZER_SNAPSHOT_FILES
]
TOKENIZER_CLOSURE_SHA256 = ubuntu_pipeline.canonical_sha256(
    {
        "schemaVersion": "lumen.base-model-tokenizer-closure/1.0.0",
        "baseModelID": BASE_MODEL_ID,
        "baseModelRevision": BASE_MODEL_REVISION,
        "files": TOKENIZER_FILES,
    }
)


def _contract() -> dict[str, Any]:
    fields = train_sft.FLEET_LOSS_SHARE_FIELD_NAMES
    return {
        "schemaVersion": "lumen.fleet-loss-share/1.2.0",
        "enforcementRequired": True,
        "enforcementPhase": "post_tokenizer_load_pre_optimizer",
        "requiredLanes": ["sft", "dpo"],
        "authoritativeCapEncoding": "integer_basis_points",
        "basisPointDenominator": 10_000,
        "capsBasisPoints": {
            "supplementalStaticTotal": {"requested": 2_500, "hard": 3_000},
            "publicBehavioralTotal": {"requested": 3_500, "hard": 3_500},
            "eachSupplementalSourceFamily": {"hard": 1_000},
        },
        "exactTokenEvidenceContract": {
            "required": True,
            "schemaVersion": "lumen.fleet-loss-share-evidence/1.1.0",
            "statusAtGeneration": "pending_exact_tokenizer_preflight",
            "tokenizer": "pinned_qwen_tokenizer",
            "comparisonRule": (
                "numeratorTokenCount*basisPointDenominator<="
                "denominatorTokenCount*capBasisPoints"
            ),
            "lanes": copy.deepcopy(fields),
        },
        "failurePolicy": "abort_before_optimizer",
        "sourceSelectionProxy": {
            "status": "safety_budget_not_exact_token_count",
            "maximumPublicBehavioralShareBasisPoints": 3_000,
            "maximumSupplementalStaticShareBasisPoints": 1_500,
            "contract": {
                "schemaVersion": "lumen.source-token-proxy/1.0.0",
                "status": "source_side_selection_proxy_not_exact_token_count",
                "strategy": "max_whitespace_terms_utf8_byte_ceiling",
                "maxCharsPerToken": 4,
                "exactPinnedTokenizerAuthoritative": True,
                "authoritativeEnforcementPhase": (
                    "post_tokenizer_load_pre_optimizer"
                ),
            },
        },
        "dpoTokenizationPolicy": copy.deepcopy(
            train_sft.FLEET_DPO_TOKENIZATION_POLICY
        ),
        "rowMetadataContract": {
            "requiredCanonicalFields": ["sourceFamily", "taskType"],
            "missingOrUnknown": "hard_fail",
        },
        "sourceRoleRegistry": {
            "schemaVersion": "lumen.fleet-source-role/1.0.0",
            "unknownPairs": "hard_fail",
            "categories": [
                "behavioral_primary",
                "public_behavioral",
                "supplemental_static",
            ],
            "registeredPairs": [
                {
                    "sourceFamily": "adapter_ultra_specific",
                    "taskType": "delegation_protocol",
                    "category": "behavioral_primary",
                },
                {
                    "sourceFamily": "codebase_home_sft",
                    "taskType": "codebase_home_grounding",
                    "category": "supplemental_static",
                },
                {
                    "sourceFamily": "self_model_sft",
                    "taskType": "self_model_grounded_answer",
                    "category": "supplemental_static",
                },
            ],
            "publicBehavioralRule": {
                "sourceFamilyPrefix": "public_adapter_corpus_",
                "taskType": "public_capability_delegation",
                "requiresPublicCorpusLineage": True,
            },
        },
        "tokenizer": {
            "baseModelID": BASE_MODEL_ID,
            "baseModelRevision": BASE_MODEL_REVISION,
            "tokenizerSHA256": TOKENIZER_SHA256,
            "tokenizerClosureSHA256": TOKENIZER_CLOSURE_SHA256,
        },
        "tokenAccounting": {
            "sft": "assistant_mask_non_ignored_token_count",
            "dpo": (
                "rendered_chosen_completion_tokens_add_special_tokens_false_"
                "plus_one_trl_0_24_0_appended_eos"
            ),
        },
    }


def _public_contract() -> dict[str, Any]:
    return {
        "schemaVersion": "lumen.public-corpus-loss-share/1.0.0",
        "enforcementRequired": True,
        "enforcementPhase": "post_tokenizer_load_pre_optimizer",
        "requiredLanes": ["sft", "dpo"],
        "authoritativeCapEncoding": "integer_basis_points",
        "basisPointDenominator": 10_000,
        "capBasisPoints": {"requested": 3_500, "hard": 3_500},
        "dpoTokenizationPolicy": copy.deepcopy(
            train_sft.PUBLIC_CORPUS_DPO_TOKENIZATION_POLICY
        ),
        "exactTokenEvidenceContract": {
            "required": True,
            "schemaVersion": (
                "lumen.public-corpus-loss-share-evidence/1.0.0"
            ),
            "statusAtGeneration": "pending_exact_tokenizer_preflight",
            "tokenizer": "pinned_qwen_tokenizer",
            "comparisonRule": (
                "numeratorTokenCount*basisPointDenominator<="
                "denominatorTokenCount*capBasisPoints"
            ),
            "lanes": copy.deepcopy(
                train_sft.PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES
            ),
        },
        "failurePolicy": "abort_before_optimizer",
        "rowMetadataContract": {
            "publicSourceFamilyPrefix": "public_adapter_corpus_",
            "publicCorpusField": "publicCorpus",
            "classificationRule": "prefix_and_nonempty_lineage_required",
            "mismatch": "hard_fail",
        },
        "sourceSelectionProxy": {
            "status": "safety_budget_not_exact_token_count",
            "maximumPublicShareBasisPoints": 3_000,
            "contract": {
                "schemaVersion": "lumen.source-token-proxy/1.0.0",
                "status": "source_side_selection_proxy_not_exact_token_count",
                "strategy": "max_whitespace_terms_utf8_byte_ceiling",
                "maxCharsPerToken": 4,
                "exactPinnedTokenizerAuthoritative": True,
                "authoritativeEnforcementPhase": (
                    "post_tokenizer_load_pre_optimizer"
                ),
            },
        },
        "tokenizer": {
            "baseModelID": BASE_MODEL_ID,
            "baseModelRevision": BASE_MODEL_REVISION,
            "tokenizerSHA256": TOKENIZER_SHA256,
            "tokenizerClosureSHA256": TOKENIZER_CLOSURE_SHA256,
        },
        "tokenAccounting": {
            "sft": "assistant_mask_non_ignored_token_count",
            "dpo": (
                "rendered_chosen_completion_tokens_add_special_tokens_false_"
                "plus_one_trl_0_24_0_appended_eos"
            ),
        },
    }


def _config() -> dict[str, Any]:
    return {
        "agent": "fleet",
        "base_model_name": BASE_MODEL_ID,
        "baseModelID": BASE_MODEL_ID,
        "baseModelRevision": BASE_MODEL_REVISION,
        "baseModelTokenizerDigest": TOKENIZER_SHA256,
        "baseModelTokenizerFiles": copy.deepcopy(TOKENIZER_FILES),
        "baseModelTokenizerClosureSHA256": TOKENIZER_CLOSURE_SHA256,
        "fleetLossShareContract": _contract(),
        "publicCorpusLossShareContract": _public_contract(),
    }


def _metadata(kind: str) -> dict[str, Any]:
    if kind == "primary":
        return {
            "sourceFamily": "adapter_ultra_specific",
            "taskType": "delegation_protocol",
        }
    if kind == "supplemental_a":
        return {
            "sourceFamily": "codebase_home_sft",
            "taskType": "codebase_home_grounding",
        }
    if kind == "supplemental_b":
        return {
            "sourceFamily": "self_model_sft",
            "taskType": "self_model_grounded_answer",
        }
    if kind == "public":
        return {
            "sourceFamily": "public_adapter_corpus_fixture",
            "taskType": "public_capability_delegation",
            "publicCorpus": {"sourceArtifactSHA256": "c" * 64},
        }
    if kind == "unknown":
        return {"sourceFamily": "unknown", "taskType": "unknown"}
    raise AssertionError(kind)


def _words(count: int, prefix: str) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def _sft_rows(specification: list[tuple[str, int]]) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"request {index}"},
                {"role": "assistant", "content": _words(tokens, f"a{index}_")},
            ],
            "metadata": _metadata(kind),
        }
        for index, (kind, tokens) in enumerate(specification)
    ]


def _dpo_source_rows(
    specification: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [{"role": "user", "content": f"request {index}"}],
            "chosen": {
                "role": "assistant",
                "content": _words(tokens, f"c{index}_"),
            },
            "rejected": {"role": "assistant", "content": "reject"},
            "metadata": _metadata(kind),
        }
        for index, (kind, tokens) in enumerate(specification)
    ]


class _ExactMaskTokenizer:
    chat_template = "{% generation %}"
    eos_token_id = 151_645

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(value.split())))}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        enable_thinking: bool,
    ) -> Any:
        del add_generation_prompt, return_assistant_tokens_mask, enable_thinking
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        for message in messages:
            count = len(message["content"].split())
            input_ids.extend(range(len(input_ids), len(input_ids) + count))
            assistant_masks.extend(
                [1 if message["role"] == "assistant" else 0] * count
            )
        if tokenize and return_dict:
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "assistant_masks": assistant_masks,
            }
        if tokenize:
            return input_ids
        return "rendered"


class _ExactTextTokenizer:
    eos_token_id = 151_645

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(value.split())))}


def _render_preference(
    row: dict[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, str]:
    assert isinstance(tokenizer, (_ExactTextTokenizer, _ExactMaskTokenizer))

    def rendered_text(value: Any) -> str:
        if isinstance(value, list):
            return " ".join(
                str(item.get("content") or "")
                for item in value
                if isinstance(item, dict)
            )
        return str(value)

    return {
        "prompt": rendered_text(row["prompt"]),
        "chosen": rendered_text(row["chosen"]),
        "rejected": rendered_text(row["rejected"]),
    }


HAPPY_SPECIFICATION = [
    ("primary", 70),
    ("supplemental_a", 5),
    ("supplemental_b", 5),
    ("public", 20),
]


def _sft_preflight(
    train_specification: list[tuple[str, int]],
    validation_specification: list[tuple[str, int]] = HAPPY_SPECIFICATION,
) -> dict[str, Any]:
    config = _config()
    return train_sft._preflight_sft_token_lengths(
        {
            "train": (_sft_rows(train_specification), Path("train_sft.jsonl")),
            "validation": (
                _sft_rows(validation_specification),
                Path("val_sft.jsonl"),
            ),
        },
        tokenizer=_ExactMaskTokenizer(),
        max_sequence_length=512,
        agent="fleet",
        fleet_loss_share_contract=config["fleetLossShareContract"],
        public_corpus_loss_share_contract=config[
            "publicCorpusLossShareContract"
        ],
        fleet_config=config,
    )


def _dpo_preflight(
    train_specification: list[tuple[str, int]],
    validation_specification: list[tuple[str, int]] = HAPPY_SPECIFICATION,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    config = _config()
    source_splits = {
        "train": _dpo_source_rows(train_specification),
        "validation": _dpo_source_rows(validation_specification),
    }
    rendered_splits = {
        split: [
            {
                "prompt": "prompt",
                "chosen": row["chosen"]["content"],
                "rejected": row["rejected"]["content"],
            }
            for row in rows
        ]
        for split, rows in source_splits.items()
    }
    return (
        train_dpo._preflight_preference_token_lengths(
            rendered_splits,
            tokenizer=_ExactTextTokenizer(),
            render_preference=_render_preference,
            max_prompt_length=100,
            max_sequence_length=512,
            source_splits=source_splits,
            agent="fleet",
            fleet_loss_share_contract=config["fleetLossShareContract"],
            public_corpus_loss_share_contract=config[
                "publicCorpusLossShareContract"
            ],
            fleet_config=config,
        ),
        source_splits,
    )


def test_sft_and_dpo_happy_paths_record_exact_split_evidence() -> None:
    sft = _sft_preflight(HAPPY_SPECIFICATION)["fleetLossShareEvidence"]
    dpo, _ = _dpo_preflight(HAPPY_SPECIFICATION)
    dpo_evidence = dpo["fleetLossShareEvidence"]

    assert sft["lane"] == "sft"
    assert dpo_evidence["lane"] == "dpo"
    for evidence, denominator_field in (
        (sft, "assistantTargetTokenCount"),
        (dpo_evidence, "chosenTargetTokenCount"),
    ):
        assert evidence["status"] == "passed"
        assert set(evidence["splits"]) == {"train", "validation"}
        assert evidence["splits"]["train"]["capEnforcementStatus"] == (
            "optimizer_enforced"
        )
        assert evidence["splits"]["validation"]["capEnforcementStatus"] == (
            "observed_non_optimizer_split"
        )
        expected_denominator = 100 if evidence["lane"] == "sft" else 104
        expected_categories = (
            {
                "behavioral_primary": 70,
                "public_behavioral": 20,
                "supplemental_static": 10,
            }
            if evidence["lane"] == "sft"
            else {
                "behavioral_primary": 71,
                "public_behavioral": 21,
                "supplemental_static": 12,
            }
        )
        assert evidence["splits"]["train"][denominator_field] == expected_denominator
        assert evidence["splits"]["validation"][denominator_field] == (
            expected_denominator
        )
        assert (
            evidence["splits"]["train"]["targetTokenCountsByCategory"]
            == expected_categories
        )
    assert sft["dpoTokenizationPolicy"] is None
    assert dpo_evidence["dpoTokenizationPolicy"] == (
        train_dpo.DPO_COMPLETION_TOKENIZATION_POLICY
    )


def test_public_verifier_rejects_rehashed_self_consistent_false_counts(
    tmp_path: Path,
) -> None:
    config = _config()
    run_root = tmp_path / "run"
    dataset_dir = (
        run_root
        / "generated/fine_tuning/fleet/experiments/optimized"
    )
    config.update(
        {
            "dataset_dir": str(dataset_dir),
            "variant": "optimized",
            "max_seq_length": 512,
            "sft_minimum_sequence_margin_tokens": 128,
        }
    )
    rows = _sft_rows(HAPPY_SPECIFICATION)
    _write_jsonl(dataset_dir / "train_sft.jsonl", rows)
    _write_jsonl(dataset_dir / "val_sft.jsonl", rows)
    phase_evidence = copy.deepcopy(_sft_preflight(HAPPY_SPECIFICATION))
    evidence = phase_evidence["publicCorpusLossShareEvidence"]
    train = evidence["splits"]["train"]
    public_row = next(
        row for row in train["rowTokenEvidence"] if row["isPublicCorpus"]
    )
    original = public_row["targetTokenCount"]
    public_row["targetTokenCount"] = original - 1
    train["assistantTargetTokenCount"] -= 1
    train["publicAssistantTargetTokenCount"] -= 1

    # The forgery is arithmetically coherent and can be placed in a freshly
    # self-hashed audit, but it no longer matches the pinned tokenizer.
    unsigned_audit = {
        "schemaVersion": ubuntu_pipeline.GLOBAL_TOKENIZER_PREFLIGHT_SCHEMA,
        "status": "passed",
        "agents": [{"agent": "fleet", "sft": phase_evidence}],
    }
    forged_audit = {
        **unsigned_audit,
        "globalPreflightSHA256": ubuntu_pipeline.canonical_sha256(
            unsigned_audit
        ),
    }
    assert forged_audit["globalPreflightSHA256"] == (
        ubuntu_pipeline.canonical_sha256(
            {
                key: value
                for key, value in forged_audit.items()
                if key != "globalPreflightSHA256"
            }
        )
    )
    with pytest.raises(RuntimeError, match="exact-token row evidence drifted"):
        ubuntu_pipeline._verify_global_tokenizer_phase_evidence(
            run_root=run_root,
            agent="fleet",
            config=config,
            phase="sft",
            evidence=forged_audit["agents"][0]["sft"],
            tokenizer=_ExactMaskTokenizer(),
        )


def test_public_dpo_verifier_rejects_rehashed_false_chosen_counts(
    tmp_path: Path,
) -> None:
    config = _config()
    preflight, source_splits = _dpo_preflight(HAPPY_SPECIFICATION)
    _write_jsonl(tmp_path / "train_dpo.jsonl", source_splits["train"])
    _write_jsonl(
        tmp_path / "val_dpo.jsonl",
        source_splits["validation"],
    )
    evidence = copy.deepcopy(preflight["publicCorpusLossShareEvidence"])
    train = evidence["splits"]["train"]
    public_row = next(
        row for row in train["rowTokenEvidence"] if row["isPublicCorpus"]
    )
    public_row["targetTokenCount"] -= 1
    train["chosenTargetTokenCount"] -= 1
    train["publicChosenTargetTokenCount"] -= 1
    forged_sha256 = ubuntu_pipeline.canonical_sha256(evidence)
    assert len(forged_sha256) == 64

    with pytest.raises(RuntimeError, match="exact-token row evidence drifted"):
        ubuntu_pipeline._verify_public_corpus_loss_share_evidence(
            value=evidence,
            config=config,
            phase="preference",
            dataset_dir=tmp_path,
            tokenizer=_ExactTextTokenizer(),
            preference_renderer=_render_preference,
            require_exact_tokenizer_counts=True,
        )


def test_mouth_pilot_exact_37_29_percent_share_is_rejected() -> None:
    config = _config()
    config["agent"] = "mouth"
    config.pop("fleetLossShareContract")
    train_rows = _sft_rows([("primary", 1_643), ("public", 977)])
    validation_rows = _sft_rows([("primary", 80), ("public", 20)])

    with pytest.raises(
        RuntimeError,
        match=r"977\*10000 > 2620\*3500",
    ):
        train_sft._preflight_sft_token_lengths(
            {
                "train": (train_rows, Path("train_sft.jsonl")),
                "validation": (
                    validation_rows,
                    Path("val_sft.jsonl"),
                ),
            },
            tokenizer=_ExactMaskTokenizer(),
            max_sequence_length=4_096,
            agent="mouth",
            public_corpus_loss_share_contract=config[
                "publicCorpusLossShareContract"
            ],
            fleet_config=config,
        )


def test_small_validation_denominator_is_observed_without_false_optimizer_block(
    tmp_path: Path,
) -> None:
    validation = [("supplemental_a", 1)]
    sft = _sft_preflight(
        HAPPY_SPECIFICATION,
        validation,
    )["fleetLossShareEvidence"]
    dpo, _ = _dpo_preflight(HAPPY_SPECIFICATION, validation)

    assert sft["splits"]["validation"][
        "supplementalStaticAssistantTargetTokenCount"
    ] == 1
    assert dpo["fleetLossShareEvidence"]["splits"]["validation"][
        "supplementalStaticChosenTargetTokenCount"
    ] == 2
    assert sft["splits"]["validation"]["capEnforcementStatus"] == (
        "observed_non_optimizer_split"
    )
    dataset_dir = tmp_path / "fleet"
    _write_jsonl(dataset_dir / "train_sft.jsonl", _sft_rows(HAPPY_SPECIFICATION))
    _write_jsonl(dataset_dir / "val_sft.jsonl", _sft_rows(validation))
    assert ubuntu_pipeline._verify_fleet_loss_share_evidence(
        value=sft,
        config=_config(),
        phase="sft",
        dataset_dir=dataset_dir,
    ) == sft


@pytest.mark.parametrize(
    ("specification", "message"),
    [
        (
            [("primary", 60), ("supplemental_a", 15), ("supplemental_b", 15), ("public", 10)],
            "supplemental-static requested",
        ),
        (
            [("primary", 80), ("supplemental_a", 11), ("public", 9)],
            "supplemental source family",
        ),
        (
            [("primary", 59), ("supplemental_a", 5), ("public", 36)],
            "public-behavioral requested",
        ),
        (
            [("unknown", 70), ("supplemental_a", 5), ("public", 25)],
            "Unregistered Fleet source-role pair",
        ),
    ],
)
def test_sft_gate_rejects_cap_and_registry_violations(
    specification: list[tuple[str, int]],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _sft_preflight(specification)


@pytest.mark.parametrize(
    ("specification", "message"),
    [
        (
            [("primary", 60), ("supplemental_a", 15), ("supplemental_b", 15), ("public", 10)],
            "supplemental-static requested",
        ),
        (
            [("primary", 80), ("supplemental_a", 11), ("public", 9)],
            "supplemental source family",
        ),
        (
            [("primary", 59), ("supplemental_a", 5), ("public", 36)],
            "public-behavioral requested",
        ),
        (
            [("unknown", 70), ("supplemental_a", 5), ("public", 25)],
            "Unregistered Fleet source-role pair",
        ),
    ],
)
def test_dpo_gate_rejects_cap_and_registry_violations(
    specification: list[tuple[str, int]],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _dpo_preflight(specification)


def test_contract_and_metadata_are_fail_closed() -> None:
    config = _config()
    config["fleetLossShareContract"]["basisPointDenominator"] = 10_000.0
    with pytest.raises(RuntimeError, match="control fields drifted"):
        train_sft._preflight_sft_token_lengths(
            {
                "train": (_sft_rows(HAPPY_SPECIFICATION), Path("train.jsonl")),
                "validation": (_sft_rows(HAPPY_SPECIFICATION), Path("val.jsonl")),
            },
            tokenizer=_ExactMaskTokenizer(),
            max_sequence_length=512,
            agent="fleet",
            fleet_loss_share_contract=config["fleetLossShareContract"],
            public_corpus_loss_share_contract=config[
                "publicCorpusLossShareContract"
            ],
            fleet_config=config,
        )

    missing_metadata = _sft_rows(HAPPY_SPECIFICATION)
    del missing_metadata[0]["metadata"]["taskType"]
    config = _config()
    with pytest.raises(RuntimeError, match="canonical metadata"):
        train_sft._preflight_sft_token_lengths(
            {
                "train": (missing_metadata, Path("train.jsonl")),
                "validation": (_sft_rows(HAPPY_SPECIFICATION), Path("val.jsonl")),
            },
            tokenizer=_ExactMaskTokenizer(),
            max_sequence_length=512,
            agent="fleet",
            fleet_loss_share_contract=config["fleetLossShareContract"],
            public_corpus_loss_share_contract=config[
                "publicCorpusLossShareContract"
            ],
            fleet_config=config,
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(("lane", "phase"), [("sft", "sft"), ("dpo", "preference")])
def test_independent_verifier_reconstructs_sft_and_dpo_evidence(
    tmp_path: Path,
    lane: str,
    phase: str,
) -> None:
    config = _config()
    dataset_dir = tmp_path / "fleet"
    if lane == "sft":
        preflight = _sft_preflight(HAPPY_SPECIFICATION)
        rows = _sft_rows(HAPPY_SPECIFICATION)
        _write_jsonl(dataset_dir / "train_sft.jsonl", rows)
        _write_jsonl(dataset_dir / "val_sft.jsonl", rows)
    else:
        preflight, source_splits = _dpo_preflight(HAPPY_SPECIFICATION)
        _write_jsonl(dataset_dir / "train_dpo.jsonl", source_splits["train"])
        _write_jsonl(dataset_dir / "val_dpo.jsonl", source_splits["validation"])

    verified = ubuntu_pipeline._verify_fleet_loss_share_evidence(
        value=preflight["fleetLossShareEvidence"],
        config=config,
        phase=phase,
        dataset_dir=dataset_dir,
    )

    assert verified == preflight["fleetLossShareEvidence"]


def test_independent_verifier_rejects_missing_noninteger_zero_and_over_cap(
    tmp_path: Path,
) -> None:
    config = _config()
    dataset_dir = tmp_path / "fleet"
    rows = _sft_rows(HAPPY_SPECIFICATION)
    _write_jsonl(dataset_dir / "train_sft.jsonl", rows)
    _write_jsonl(dataset_dir / "val_sft.jsonl", rows)
    baseline = _sft_preflight(HAPPY_SPECIFICATION)["fleetLossShareEvidence"]

    with pytest.raises(RuntimeError, match="invalid schema"):
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=None,
            config=config,
            phase="sft",
            dataset_dir=dataset_dir,
        )

    noninteger = copy.deepcopy(baseline)
    noninteger["splits"]["train"]["rowTokenEvidence"][0][
        "targetTokenCount"
    ] = 70.0
    with pytest.raises(RuntimeError, match="row evidence drifted"):
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=noninteger,
            config=config,
            phase="sft",
            dataset_dir=dataset_dir,
        )

    zero_total = copy.deepcopy(baseline)
    zero_total["splits"]["train"]["assistantTargetTokenCount"] = 0
    with pytest.raises(RuntimeError, match="failed reconstruction"):
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=zero_total,
            config=config,
            phase="sft",
            dataset_dir=dataset_dir,
        )

    over_cap = copy.deepcopy(baseline)
    split = over_cap["splits"]["train"]
    row_counts = [40, 20, 20, 20]
    for row_evidence, count in zip(split["rowTokenEvidence"], row_counts):
        row_evidence["targetTokenCount"] = count
    split["targetTokenCountsByCategory"] = {
        "behavioral_primary": 40,
        "public_behavioral": 20,
        "supplemental_static": 40,
    }
    split["assistantTargetTokenCount"] = 100
    split["supplementalStaticAssistantTargetTokenCount"] = 40
    split["publicBehavioralAssistantTargetTokenCount"] = 20
    split["supplementalStaticAssistantTargetTokenCountsBySourceFamily"] = {
        "codebase_home_sft": 20,
        "self_model_sft": 20,
    }
    with pytest.raises(RuntimeError, match="total token cap failed"):
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=over_cap,
            config=config,
            phase="sft",
            dataset_dir=dataset_dir,
        )


def test_bound_preflight_verifier_checks_config_dataset_and_training_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    dataset_dir = run_root / "generated/fine_tuning/fleet/experiments/optimized"
    rows = _sft_rows(HAPPY_SPECIFICATION)
    _write_jsonl(dataset_dir / "train_sft.jsonl", rows)
    _write_jsonl(dataset_dir / "val_sft.jsonl", rows)
    dpo_rows = _dpo_source_rows(HAPPY_SPECIFICATION)
    _write_jsonl(dataset_dir / "train_dpo.jsonl", dpo_rows)
    _write_jsonl(dataset_dir / "val_dpo.jsonl", dpo_rows)
    _write_json(dataset_dir / "variant_manifest.json", {"variant": "optimized"})
    config_path = run_root / "configs/fleet.json"
    lineage_path = run_root / "checkpoint_lineage/fleet.sft.json"
    config = {
        **_config(),
        "variant": "optimized",
        "variantManifestSHA256": "d" * 64,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_root / "training/fleet"),
        "adapter_output_dir": str(run_root / "models/lora_qwen3_bootstrap/fleet"),
        "sftCheckpointLineagePath": str(lineage_path),
        "sftTokenLengthPreflightPath": str(
            run_root / "training/fleet/sft_token_length_preflight.json"
        ),
        "trainingCodeSHA256ByPhase": {
            "sft": "e" * 64,
            "dpo": "1" * 64,
        },
        "preference_trainer": "dpo",
        "resolvedTrainingEnvironmentSHA256": "f" * 64,
        "save_total_limit": 2,
        "max_seq_length": 512,
        "max_prompt_length": 256,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
        "dpo_learning_rate": 5e-6,
        "dpo_num_train_epochs": 1.0,
        "dpo_beta": 0.1,
        "gradient_checkpointing": True,
        "use_logits_to_keep": True,
        "precompute_ref_log_probs": True,
        "precompute_ref_batch_size": 1,
        "bf16": False,
        "fp16": True,
        "chatTemplateContract": {"schemaVersion": "fixture"},
    }
    _write_json(config_path, config)
    _write_json(
        lineage_path,
        train_sft._initial_sft_checkpoint_lineage(config, cfg_path=config_path),
    )
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    run_manifest = {
        "runManifestSHA256": "2" * 64,
        "agents": [{"agent": "fleet"}],
    }
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda _run_root: run_manifest,
    )
    global_result = ubuntu_pipeline.global_tokenizer_preflight(
        run_root=run_root,
        agents=("fleet",),
        tokenizer=_ExactMaskTokenizer(),
        tokenizer_file_sha256=TOKENIZER_SHA256,
        preference_renderer=_render_preference,
        chat_contract_verifier=lambda *_args, **_kwargs: None,
    )
    assert global_result["status"] == "global_tokenizer_preflight_passed"
    def verify_prepared() -> dict[str, object]:
        return ubuntu_pipeline._verified_prepared_global_tokenizer_preflight(
            run_root=run_root,
            agents=("fleet",),
            tokenizer=_ExactMaskTokenizer(),
            tokenizer_file_sha256=TOKENIZER_SHA256,
            preference_renderer=_render_preference,
            chat_contract_verifier=lambda *_args, **_kwargs: None,
        )

    prepared_audit = verify_prepared()
    assert prepared_audit["globalPreflightSHA256"] == global_result[
        "globalPreflightSHA256"
    ]
    global_audit_path = (
        run_root
        / "training"
        / ubuntu_pipeline.GLOBAL_TOKENIZER_PREFLIGHT_FILENAME
    )
    drifted_global = copy.deepcopy(prepared_audit)
    drifted_global["agents"][0]["sft"]["records"] += 1
    unsigned_global = dict(drifted_global)
    unsigned_global.pop("globalPreflightSHA256")
    drifted_global["globalPreflightSHA256"] = (
        ubuntu_pipeline.canonical_sha256(unsigned_global)
    )
    _write_json(global_audit_path, drifted_global)
    with pytest.raises(RuntimeError, match="exact tokenizer evidence drifted"):
        verify_prepared()

    fabricated_statistics = copy.deepcopy(prepared_audit)
    fabricated_statistics["agents"][0]["sft"]["totalTokens"]["p50"] += 1
    unsigned_global = dict(fabricated_statistics)
    unsigned_global.pop("globalPreflightSHA256")
    fabricated_statistics["globalPreflightSHA256"] = (
        ubuntu_pipeline.canonical_sha256(unsigned_global)
    )
    _write_json(global_audit_path, fabricated_statistics)
    with pytest.raises(RuntimeError, match="exact tokenizer evidence drifted"):
        verify_prepared()

    fabricated_fleet_row_tokens = copy.deepcopy(prepared_audit)
    fabricated_fleet_row_tokens["agents"][0]["sft"][
        "fleetLossShareEvidence"
    ]["splits"]["train"]["rowTokenEvidence"][0]["targetTokenCount"] += 1
    unsigned_global = dict(fabricated_fleet_row_tokens)
    unsigned_global.pop("globalPreflightSHA256")
    fabricated_fleet_row_tokens["globalPreflightSHA256"] = (
        ubuntu_pipeline.canonical_sha256(unsigned_global)
    )
    _write_json(global_audit_path, fabricated_fleet_row_tokens)
    with pytest.raises(RuntimeError, match="exact tokenizer evidence drifted"):
        verify_prepared()

    rebound_global = copy.deepcopy(prepared_audit)
    rebound_global["agents"][0]["sftDatasetFileSHA256"] = {
        "train_sft.jsonl": "0" * 64,
        "val_sft.jsonl": "0" * 64,
        "variant_manifest.json": "0" * 64,
    }
    unsigned_global = dict(rebound_global)
    unsigned_global.pop("globalPreflightSHA256")
    rebound_global["globalPreflightSHA256"] = (
        ubuntu_pipeline.canonical_sha256(unsigned_global)
    )
    _write_json(global_audit_path, rebound_global)
    with pytest.raises(RuntimeError, match="input binding drifted"):
        verify_prepared()
    _write_json(global_audit_path, prepared_audit)
    preflight = train_sft._preflight_sft_token_lengths(
        {
            "train": (rows, dataset_dir / "train_sft.jsonl"),
            "validation": (rows, dataset_dir / "val_sft.jsonl"),
        },
        tokenizer=_ExactMaskTokenizer(),
        max_sequence_length=512,
        agent="fleet",
        fleet_loss_share_contract=config["fleetLossShareContract"],
        public_corpus_loss_share_contract=config[
            "publicCorpusLossShareContract"
        ],
        fleet_config=config,
    )
    evidence = train_sft._bind_sft_token_length_preflight(
        config,
        cfg_path=config_path,
        preflight=preflight,
    )
    report = {
        "token_length_preflight": evidence,
        "token_length_preflight_path": config["sftTokenLengthPreflightPath"],
        "token_length_preflight_sha256": evidence["preflightSHA256"],
    }
    def verify_injected_global(**kwargs: Any) -> dict[str, Any]:
        return ubuntu_pipeline._verified_global_tokenizer_preflight_test_only(
            **kwargs,
            audit_snapshot=ubuntu_pipeline.read_object(global_audit_path),
            tokenizer=_ExactMaskTokenizer(),
            preference_renderer=_render_preference,
        )

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_global_tokenizer_preflight",
        verify_injected_global,
    )

    verified = ubuntu_pipeline._verify_training_token_length_preflight(
        run_root=run_root,
        agent="fleet",
        config=config,
        report=report,
        phase="sft",
    )
    assert verified == evidence

    drifted = copy.deepcopy(evidence)
    drifted["trainingCodeSHA256"] = "0" * 64
    unsigned = dict(drifted)
    unsigned.pop("preflightSHA256")
    drifted["preflightSHA256"] = ubuntu_pipeline.canonical_sha256(unsigned)
    _write_json(Path(config["sftTokenLengthPreflightPath"]), drifted)
    lineage = train_sft._read_sft_checkpoint_lineage(lineage_path)
    lineage["tokenLengthPreflightSHA256"] = drifted["preflightSHA256"]
    _write_json(
        lineage_path,
        train_sft._self_hashed_sft_checkpoint_record(lineage),
    )
    drifted_report = {
        "token_length_preflight": drifted,
        "token_length_preflight_path": config["sftTokenLengthPreflightPath"],
        "token_length_preflight_sha256": drifted["preflightSHA256"],
    }
    with pytest.raises(RuntimeError, match="failed verification"):
        ubuntu_pipeline._verify_training_token_length_preflight(
            run_root=run_root,
            agent="fleet",
            config=config,
            report=drifted_report,
            phase="sft",
        )


def test_global_tokenizer_snapshot_binds_complete_local_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_dir = tmp_path / "global_tokenizer_snapshot"
    snapshot_dir.mkdir(mode=0o700)
    payloads = {
        filename: f"fixture:{filename}".encode("utf-8")
        for filename in ubuntu_pipeline.GLOBAL_TOKENIZER_SNAPSHOT_FILES
    }
    for filename, payload in payloads.items():
        path = snapshot_dir / filename
        path.write_bytes(payload)
        path.chmod(0o400)
    snapshot_dir.chmod(0o700)
    config = {
        "base_model_name": BASE_MODEL_ID,
        "baseModelID": BASE_MODEL_ID,
        "baseModelRevision": BASE_MODEL_REVISION,
        "baseModelTokenizerDigest": hashlib.sha256(
            payloads["tokenizer.json"]
        ).hexdigest(),
        "baseModelTokenizerFiles": [
            {
                "path": filename,
                "sizeBytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "huggingFaceBlobID": ubuntu_pipeline._git_blob_sha1(payload),
            }
            for filename, payload in sorted(payloads.items())
        ],
    }
    config["baseModelTokenizerClosureSHA256"] = (
        ubuntu_pipeline.canonical_sha256(
            {
                "schemaVersion": "lumen.base-model-tokenizer-closure/1.0.0",
                "baseModelID": BASE_MODEL_ID,
                "baseModelRevision": BASE_MODEL_REVISION,
                "files": config["baseModelTokenizerFiles"],
            }
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    class StableAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            calls.append((path, kwargs))
            return types.SimpleNamespace(is_fast=True)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=StableAutoTokenizer),
    )
    tokenizer, contract = (
        ubuntu_pipeline._load_verified_global_tokenizer_snapshot(
            snapshot_dir=snapshot_dir,
            config=config,
        )
    )
    assert tokenizer.is_fast is True
    assert [item["path"] for item in contract["files"]] == list(
        ubuntu_pipeline.GLOBAL_TOKENIZER_SNAPSHOT_FILES
    )
    assert calls == [
        (
            str(snapshot_dir),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]

    class SwapAndRestoreAutoTokenizer:
        @staticmethod
        def from_pretrained(_path: str, **_kwargs: object) -> object:
            target = snapshot_dir / "tokenizer_config.json"
            original = target.read_bytes()
            target.chmod(0o600)
            target.write_bytes(b"temporary unbound tokenizer state")
            target.write_bytes(original)
            target.chmod(0o400)
            return types.SimpleNamespace(is_fast=True)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=SwapAndRestoreAutoTokenizer),
    )
    with pytest.raises(RuntimeError, match="changed while loading"):
        ubuntu_pipeline._load_verified_global_tokenizer_snapshot(
            snapshot_dir=snapshot_dir,
            config=config,
            expected_contract=contract,
        )


def test_loss_share_preflight_occurs_before_optimizer_owning_objects() -> None:
    tokenizer_loader_source = inspect.getsource(
        ubuntu_pipeline._load_verified_global_tokenizer_snapshot
    )
    assert "AutoTokenizer" in tokenizer_loader_source
    assert "local_files_only=True" in tokenizer_loader_source
    assert "AutoModel" not in tokenizer_loader_source
    assert "FastLanguageModel" not in tokenizer_loader_source

    sft_source = inspect.getsource(train_sft.main)
    assert sft_source.index("_preflight_sft_token_lengths(") < sft_source.index(
        "FastLanguageModel.get_peft_model("
    )

    dpo_source = inspect.getsource(train_dpo.main)
    assert dpo_source.index("_preflight_preference_token_lengths(") < dpo_source.index(
        "_load_sft_policy("
    )
    assert dpo_source.index("_preflight_preference_token_lengths(") < dpo_source.index(
        "_build_preference_trainer("
    )

    launcher = (
        Path(__file__).resolve().parents[4]
        / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")
    global_preflight = launcher.index("global-tokenizer-preflight")
    first_agent_loop = launcher.index('for agent in "${AGENTS[@]}"; do')
    prepare_only_exit = launcher.index('if [[ "$PREPARE_ONLY" == "1" ]]')
    assert global_preflight < prepare_only_exit < first_agent_loop
    assert (
        'log "prepared run manifest: $RUN_ROOT/aio_run_manifest.json"'
        in launcher
    )
    assert (
        'log "global tokenizer preflight audit: '
        '$RUN_ROOT/training/global_tokenizer_preflight.json"'
        in launcher
    )
