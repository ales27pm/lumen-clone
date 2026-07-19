from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tools.fine_tuning.unsloth import train_dpo, train_sft


REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = REPO_ROOT / "generated" / "fine_tuning"
OPTIMIZED_VARIANT = "internal_plus_public_optimized"
AGENTS = ("cortex", "executor", "fleet", "mimicry", "mouth", "rem")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


@pytest.mark.e2e
def test_all_six_optimized_lanes_pass_exact_pinned_public_target_caps() -> None:
    transformers = pytest.importorskip(
        "transformers",
        reason="Exact public-target cap proof requires transformers==4.57.6",
    )
    trl = pytest.importorskip(
        "trl",
        reason="Exact public-target DPO proof requires trl==0.24.0",
    )
    if transformers.__version__ != "4.57.6" or trl.__version__ != "0.24.0":
        pytest.skip("Exact public-target cap proof requires the pinned trainer lock")

    first_manifest = json.loads(
        (
            DATASET_ROOT
            / AGENTS[0]
            / "experiments"
            / OPTIMIZED_VARIANT
            / "variant_manifest.json"
        ).read_text(encoding="utf-8")
    )
    first_config = first_manifest["controlledTrainingConfig"]
    local_only = os.environ.get("LUMEN_ENABLE_NETWORK_TESTS") != "1"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            first_config["base_model_name"],
            revision=first_config["baseModelRevision"],
            local_files_only=local_only,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, ValueError):
        if not local_only:
            raise
        pytest.skip("Pinned Qwen tokenizer is not cached")

    from trl.data_utils import maybe_apply_chat_template

    for agent in AGENTS:
        variant_root = (
            DATASET_ROOT / agent / "experiments" / OPTIMIZED_VARIANT
        )
        manifest = json.loads(
            (variant_root / "variant_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        config = manifest["controlledTrainingConfig"]
        common: dict[str, Any] = {
            "agent": agent,
            "public_corpus_loss_share_contract": config[
                "publicCorpusLossShareContract"
            ],
            "fleet_config": config,
        }
        if agent == "fleet":
            common["fleet_loss_share_contract"] = config[
                "fleetLossShareContract"
            ]

        train_sft_rows = _read_jsonl(variant_root / "train_sft.jsonl")
        val_sft_rows = _read_jsonl(variant_root / "val_sft.jsonl")
        sft = train_sft._preflight_sft_token_lengths(
            {
                "train": (
                    train_sft_rows,
                    variant_root / "train_sft.jsonl",
                ),
                "validation": (
                    val_sft_rows,
                    variant_root / "val_sft.jsonl",
                ),
            },
            tokenizer=tokenizer,
            max_sequence_length=config["max_seq_length"],
            minimum_sequence_margin_tokens=config[
                "sft_minimum_sequence_margin_tokens"
            ],
            **common,
        )

        train_dpo_rows = _read_jsonl(variant_root / "train_dpo.jsonl")
        val_dpo_rows = _read_jsonl(variant_root / "val_dpo.jsonl")
        dpo = train_dpo._preflight_preference_token_lengths(
            {
                "train": [
                    train_dpo.row_to_preference(row)
                    for row in train_dpo_rows
                ],
                "validation": [
                    train_dpo.row_to_preference(row) for row in val_dpo_rows
                ],
            },
            tokenizer=tokenizer,
            render_preference=maybe_apply_chat_template,
            max_prompt_length=config["max_prompt_length"],
            max_sequence_length=config["max_seq_length"],
            minimum_prompt_margin_tokens=config[
                "preference_minimum_prompt_margin_tokens"
            ],
            minimum_sequence_margin_tokens=config[
                "preference_minimum_sequence_margin_tokens"
            ],
            source_splits={
                "train": train_dpo_rows,
                "validation": val_dpo_rows,
            },
            **common,
        )

        for evidence, denominator_field, public_field in (
            (
                sft["publicCorpusLossShareEvidence"],
                "assistantTargetTokenCount",
                "publicAssistantTargetTokenCount",
            ),
            (
                dpo["publicCorpusLossShareEvidence"],
                "chosenTargetTokenCount",
                "publicChosenTargetTokenCount",
            ),
        ):
            train = evidence["splits"]["train"]
            denominator = train[denominator_field]
            numerator = train[public_field]
            for cap in evidence["capBasisPoints"].values():
                assert numerator * 10_000 <= denominator * cap, (
                    agent,
                    evidence["lane"],
                    numerator,
                    denominator,
                    cap,
                )
