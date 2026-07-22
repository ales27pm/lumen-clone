from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any

from tools.fine_tuning.unsloth import train_dpo, train_sft, ubuntu_pipeline


REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_SOURCE = REPO_ROOT / "generated" / "fine_tuning"
FLEET_ROOT = DATASET_SOURCE / "fleet"
REQUIRED_GATE_ENV = "LUMEN_REQUIRE_FLEET_PINNED_TOKENIZER_GATE"
TOKENIZER_SNAPSHOT_ENV = "LUMEN_FLEET_TOKENIZER_SNAPSHOT"
TOKENIZER_BINDING_FIELDS = (
    "baseModelID",
    "base_model_name",
    "baseModelRevision",
    "baseModelTokenizerDigest",
    "baseModelTokenizerFiles",
    "baseModelTokenizerClosureSHA256",
    "chatTemplateContract",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise AssertionError(f"Expected non-empty JSONL objects: {path}")
    return rows


class FleetGeneratedExactTokenGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required_gate = os.environ.get(REQUIRED_GATE_ENV)
        if required_gate in {None, "0"}:
            raise unittest.SkipTest(
                f"Set {REQUIRED_GATE_ENV}=1 for the required pre-pilot gate"
            )
        if required_gate != "1":
            raise AssertionError(
                f"{REQUIRED_GATE_ENV} must be exactly 0 or 1"
            )
        snapshot_value = os.environ.get(TOKENIZER_SNAPSHOT_ENV, "").strip()
        if not snapshot_value:
            raise AssertionError(
                f"{TOKENIZER_SNAPSHOT_ENV} is required when the gate is enabled"
            )
        cls.snapshot_dir = Path(snapshot_value).expanduser().resolve()
        if not cls.snapshot_dir.is_dir():
            raise AssertionError(
                f"Pinned tokenizer snapshot is not a directory: {cls.snapshot_dir}"
            )
        if sys.version_info[:2] != (3, 10):
            raise AssertionError(
                "Exact Fleet gate requires the pinned Python 3.10 runtime"
            )
        versions = {
            package: importlib.metadata.version(package)
            for package in ("transformers", "trl", "tokenizers")
        }
        if versions != {
            "transformers": "4.57.6",
            "trl": "0.24.0",
            "tokenizers": "0.22.2",
        }:
            raise AssertionError(
                f"Pinned trainer runtime drifted: {versions}"
            )

        cls.root_config = _read_json(FLEET_ROOT / "unsloth_config.json")
        train_sft._resolve_training_precision(cls.root_config)
        train_dpo._validate_preference_training_config(cls.root_config)
        train_sft.verify_chat_template_contract(
            cls.root_config["chatTemplateContract"]
        )
        experiment_manifest = _read_json(
            FLEET_ROOT / "experiment_manifest.json"
        )
        variant_order = experiment_manifest.get("variantOrder")
        variants = experiment_manifest.get("variants")
        if (
            not isinstance(variant_order, list)
            or not variant_order
            or any(not isinstance(item, str) or not item for item in variant_order)
            or len(set(variant_order)) != len(variant_order)
            or not isinstance(variants, list)
            or [
                item.get("variant") if isinstance(item, dict) else None
                for item in variants
            ]
            != variant_order
        ):
            raise AssertionError("Fleet experiment variant order is invalid")
        experiment_directories = {
            path.name
            for path in (FLEET_ROOT / "experiments").iterdir()
            if path.is_dir()
        }
        if experiment_directories != set(variant_order):
            raise AssertionError(
                "Fleet experiment directories drifted from the advertised variants"
            )
        cls.variant_order = tuple(variant_order)
        cls.lane_configs: list[tuple[str, Path, dict[str, Any]]] = [
            ("root", FLEET_ROOT, cls.root_config)
        ]
        for variant in cls.variant_order:
            config, manifest, lane_root = ubuntu_pipeline.validate_variant(
                DATASET_SOURCE,
                agent="fleet",
                variant=variant,
                seed=cls.root_config["seed"],
                base_model_override=cls.root_config["base_model_name"],
            )
            if manifest.get("variant") != variant:
                raise AssertionError(
                    f"Fleet validated variant identity drifted: {variant}"
                )
            cls.lane_configs.append((variant, lane_root, config))
        cls.tokenizer, cls.tokenizer_closure = (
            ubuntu_pipeline._load_verified_global_tokenizer_snapshot(
                snapshot_dir=cls.snapshot_dir,
                config=cls.root_config,
            )
        )
        from trl.data_utils import maybe_apply_chat_template

        cls.render_preference = staticmethod(maybe_apply_chat_template)

    @classmethod
    def _lanes(cls) -> list[tuple[str, Path, dict[str, Any]]]:
        return list(cls.lane_configs)

    def _assert_tokenizer_binding(self, config: dict[str, Any]) -> None:
        for field in TOKENIZER_BINDING_FIELDS:
            self.assertEqual(
                config.get(field),
                self.root_config.get(field),
                msg=f"Fleet tokenizer binding drifted at {field}",
            )
        train_sft.verify_chat_template_contract(
            config["chatTemplateContract"],
            tokenizer=self.tokenizer,
        )

    def _source_rows(
        self,
        *,
        lane_root: Path,
        config: dict[str, Any],
        phase: str,
    ) -> dict[str, list[dict[str, Any]]]:
        if phase == "sft":
            return {
                "train": train_sft._limit_records(
                    _read_jsonl(lane_root / "train_sft.jsonl"),
                    config.get("max_train_records"),
                ),
                "validation": train_sft._limit_records(
                    _read_jsonl(lane_root / "val_sft.jsonl"),
                    config.get("max_val_records"),
                ),
            }
        return {
            "train": _read_jsonl(lane_root / "train_dpo.jsonl"),
            "validation": _read_jsonl(lane_root / "val_dpo.jsonl"),
        }

    def _assert_exact_family_band(
        self,
        evidence: dict[str, Any],
    ) -> None:
        band = evidence["optimizerFamilyShareBand"]
        train = evidence["splits"]["train"]
        numerator = train[band["numeratorEvidenceField"]]
        denominator = train[band["denominatorEvidenceField"]]
        self.assertGreaterEqual(
            numerator * 10_000,
            denominator * band["minimumBasisPoints"],
        )
        self.assertLessEqual(
            numerator * 10_000,
            denominator * band["maximumBasisPoints"],
        )

    def _run_sft(
        self,
        *,
        lane_root: Path,
        config: dict[str, Any],
    ) -> None:
        rows = self._source_rows(
            lane_root=lane_root,
            config=config,
            phase="sft",
        )
        result = train_sft._preflight_sft_token_lengths(
            {
                split: (
                    split_rows,
                    lane_root
                    / ("train_sft.jsonl" if split == "train" else "val_sft.jsonl"),
                )
                for split, split_rows in rows.items()
            },
            tokenizer=self.tokenizer,
            max_sequence_length=config["max_seq_length"],
            minimum_sequence_margin_tokens=config[
                "sft_minimum_sequence_margin_tokens"
            ],
            agent="fleet",
            fleet_loss_share_contract=config["fleetLossShareContract"],
            public_corpus_loss_share_contract=config[
                "publicCorpusLossShareContract"
            ],
            fleet_config=config,
        )
        evidence = result["fleetLossShareEvidence"]
        self._assert_exact_family_band(evidence)
        train = evidence["splits"]["train"]
        schedule = train["optimizerWindowSchedule"]
        band = evidence["optimizerFamilyShareBand"]
        rebuilt, _ = train_sft._build_fleet_sft_optimizer_window_schedule(
            row_token_evidence=train["rowTokenEvidence"],
            config=config,
            schedule_contract=config["fleetLossShareContract"][
                "sftOptimizerWindowScheduleContract"
            ],
            minimum_basis_points=band["minimumBasisPoints"],
            maximum_basis_points=band["maximumBasisPoints"],
        )
        self.assertEqual(schedule, rebuilt)
        self.assertTrue(schedule["allSourceRecordsCoveredAcrossConfiguredEpochs"])
        for epoch in schedule["epochs"]:
            share = epoch["windowNormalizedNativeShareBasisPoints"]
            self.assertGreaterEqual(share, band["minimumBasisPoints"])
            self.assertLessEqual(share, band["maximumBasisPoints"])
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=evidence,
            config=config,
            phase="sft",
            dataset_dir=lane_root,
            tokenizer=self.tokenizer,
            require_exact_tokenizer_counts=True,
        )
        ubuntu_pipeline._verify_public_corpus_loss_share_evidence(
            value=result["publicCorpusLossShareEvidence"],
            config=config,
            phase="sft",
            dataset_dir=lane_root,
            tokenizer=self.tokenizer,
            require_exact_tokenizer_counts=True,
        )

    def _run_dpo(
        self,
        *,
        lane_root: Path,
        config: dict[str, Any],
    ) -> None:
        rows = self._source_rows(
            lane_root=lane_root,
            config=config,
            phase="dpo",
        )
        result = train_dpo._preflight_preference_token_lengths(
            {
                split: [
                    train_dpo.row_to_preference(row) for row in split_rows
                ]
                for split, split_rows in rows.items()
            },
            tokenizer=self.tokenizer,
            render_preference=self.render_preference,
            max_prompt_length=config["max_prompt_length"],
            max_sequence_length=config["max_seq_length"],
            minimum_prompt_margin_tokens=config[
                "preference_minimum_prompt_margin_tokens"
            ],
            minimum_sequence_margin_tokens=config[
                "preference_minimum_sequence_margin_tokens"
            ],
            source_splits=rows,
            agent="fleet",
            fleet_loss_share_contract=config["fleetLossShareContract"],
            public_corpus_loss_share_contract=config[
                "publicCorpusLossShareContract"
            ],
            fleet_config=config,
        )
        evidence = result["fleetLossShareEvidence"]
        self._assert_exact_family_band(evidence)
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=evidence,
            config=config,
            phase="preference",
            dataset_dir=lane_root,
            tokenizer=self.tokenizer,
            preference_renderer=self.render_preference,
            require_exact_tokenizer_counts=True,
        )
        ubuntu_pipeline._verify_public_corpus_loss_share_evidence(
            value=result["publicCorpusLossShareEvidence"],
            config=config,
            phase="preference",
            dataset_dir=lane_root,
            tokenizer=self.tokenizer,
            preference_renderer=self.render_preference,
            require_exact_tokenizer_counts=True,
        )

    def test_root_and_all_variants_pass_exact_sft_and_dpo_contracts(self) -> None:
        for lane, lane_root, config in self._lanes():
            with self.subTest(lane=lane, phase="tokenizer_binding"):
                self._assert_tokenizer_binding(config)
            with self.subTest(lane=lane, phase="sft"):
                self._run_sft(lane_root=lane_root, config=config)
            with self.subTest(lane=lane, phase="dpo"):
                self._run_dpo(lane_root=lane_root, config=config)


if __name__ == "__main__":
    unittest.main()
