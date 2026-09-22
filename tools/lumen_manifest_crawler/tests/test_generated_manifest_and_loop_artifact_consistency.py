"""Consistency regression tests for committed generated artifacts touched by this PR.

These artifacts (``generated/agent_manifest/...`` and
``generated/agent_improvement_loop/...``) are produced by the
``lumen_manifest_crawler`` crawl/improve-loop commands and then committed. The
tests below do not re-run the (expensive, filesystem-wide) crawler; instead
they assert that the currently committed artifacts remain internally
consistent with one another, so a future regeneration that silently breaks
one of these bindings (a hash, a count, or an embedded fingerprint) fails
fast in CI instead of only being caught by manual review of a large diff.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


AGENT_MANIFEST_DIR = "generated/agent_manifest"
CROSS_MODEL_TRAINING_DIR = f"{AGENT_MANIFEST_DIR}/cross_model_training"
IMPROVEMENT_LOOP_DIR = "generated/agent_improvement_loop"


class TestAgentBehaviorManifestHashFiles:
    def test_sha256_file_matches_canonical_manifest_bytes(self) -> None:
        root = _repo_root()
        manifest_path = root / AGENT_MANIFEST_DIR / "AgentBehaviorManifest.json"
        sha_path = root / AGENT_MANIFEST_DIR / "AgentBehaviorManifest.sha256"
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert sha_path.read_text(encoding="utf-8").strip() == digest

    def test_sha256_file_is_a_single_lowercase_hex_digest_line(self) -> None:
        sha_path = _repo_root() / AGENT_MANIFEST_DIR / "AgentBehaviorManifest.sha256"
        text = sha_path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        (line,) = text.splitlines()
        assert len(line) == 64
        assert line == line.lower()
        int(line, 16)  # raises ValueError if not valid hex

    def test_incremental_sha256_file_is_a_single_lowercase_hex_digest_line(self) -> None:
        sha_path = _repo_root() / AGENT_MANIFEST_DIR / "AgentBehaviorManifest.incremental.sha256"
        text = sha_path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        (line,) = text.splitlines()
        assert len(line) == 64
        assert line == line.lower()
        int(line, 16)


class TestAgentBehaviorManifestSourceIntegrityConsistency:
    """The Markdown and pretty-JSON manifest renderings must agree with each other."""

    @pytest.fixture()
    def pretty_manifest(self) -> dict:
        path = _repo_root() / AGENT_MANIFEST_DIR / "AgentBehaviorManifest.pretty.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @pytest.fixture()
    def markdown_text(self) -> str:
        path = _repo_root() / AGENT_MANIFEST_DIR / "AgentBehaviorManifest.md"
        return path.read_text(encoding="utf-8")

    def test_source_integrity_fields_match_between_markdown_and_pretty_json(
        self, pretty_manifest: dict, markdown_text: str
    ) -> None:
        source_integrity = pretty_manifest["sourceIntegrity"]
        assert f"Base commit: `{source_integrity['baseCommit']}`" in markdown_text
        assert (
            f"Working-tree digest: `{source_integrity['workingTreeDigest']}`" in markdown_text
        )
        expected_dirty = "True" if source_integrity["dirtyState"] else "False"
        assert f"Dirty source state: `{expected_dirty}`" in markdown_text

    def test_source_integrity_is_currently_clean(self, pretty_manifest: dict) -> None:
        # Regression for this PR: the committed manifest snapshot moved from a
        # dirty working tree to a clean, fully-committed source state.
        assert pretty_manifest["sourceIntegrity"]["dirtyState"] is False

    def test_memory_freshness_classes_match_between_markdown_and_pretty_json(
        self, pretty_manifest: dict, markdown_text: str
    ) -> None:
        freshness_classes = {
            entry["id"]: entry for entry in pretty_manifest["memory"]["freshnessClasses"]
        }
        for scope_id in ("durable", "shortLived", "timeless", "volatile"):
            assert scope_id in freshness_classes
            entry = freshness_classes[scope_id]
            if entry["durable"]:
                expected_line = f"`{scope_id}`: durable; source: `{entry['source']}`"
            else:
                expected_line = (
                    f"`{scope_id}`: ttlSeconds={entry['ttlSeconds']}; source: `{entry['source']}`"
                )
            assert expected_line in markdown_text

    def test_short_lived_and_volatile_ttls_match_the_regenerated_memory_store_source(
        self, pretty_manifest: dict
    ) -> None:
        # Regression for this PR: both freshness classes moved from
        # ios/Lumen/Models/MemoryItem.swift-derived TTLs to the current
        # ios/Lumen/Services/MemoryStore.swift-derived TTLs.
        freshness_classes = {
            entry["id"]: entry for entry in pretty_manifest["memory"]["freshnessClasses"]
        }
        assert freshness_classes["shortLived"]["ttlSeconds"] == 21600
        assert freshness_classes["shortLived"]["source"] == "ios/Lumen/Services/MemoryStore.swift"
        assert freshness_classes["volatile"]["ttlSeconds"] == 2700
        assert freshness_classes["volatile"]["source"] == "ios/Lumen/Services/MemoryStore.swift"


class TestCrossModelTrainingArtifactConsistency:
    @pytest.fixture()
    def cross_model_dir(self) -> Path:
        return _repo_root() / CROSS_MODEL_TRAINING_DIR

    @pytest.fixture()
    def full_records(self, cross_model_dir: Path) -> list[dict]:
        return _load_jsonl(cross_model_dir / "cross_model_training.jsonl")

    def test_index_csv_counts_match_full_jsonl_grouped_counts(
        self, cross_model_dir: Path, full_records: list[dict]
    ) -> None:
        counts = Counter(
            (record["recordType"], record["agentRole"], record["taskType"])
            for record in full_records
        )
        with (cross_model_dir / "cross_model_training_index.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "cross_model_training_index.csv must not be empty"
        seen_keys = set()
        for row in rows:
            key = (row["recordType"], row["agentRole"], row["taskType"])
            seen_keys.add(key)
            assert counts.get(key, 0) == int(row["recordCount"]), row
        assert seen_keys == set(counts)

    def test_index_csv_total_matches_full_jsonl_record_count(
        self, cross_model_dir: Path, full_records: list[dict]
    ) -> None:
        with (cross_model_dir / "cross_model_training_index.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert sum(int(row["recordCount"]) for row in rows) == len(full_records)

    def test_fleet_dpo_orchestration_preference_count(self, full_records: list[dict]) -> None:
        # Regression for this PR's bump from 4 -> 16 fleet DPO
        # orchestration-preference rows.
        fleet_dpo = [
            record
            for record in full_records
            if record["recordType"] == "dpo"
            and record["agentRole"] == "fleet"
            and record["taskType"] == "fleet_orchestration_event_graph_preference"
        ]
        assert len(fleet_dpo) == 16

    def test_fleet_sft_orchestration_event_graph_count(self, full_records: list[dict]) -> None:
        # Regression for this PR's bump from 9 -> 36 fleet SFT
        # orchestration-event-graph rows.
        fleet_sft = [
            record
            for record in full_records
            if record["recordType"] == "sft"
            and record["agentRole"] == "fleet"
            and record["taskType"] == "fleet_orchestration_event_graph"
        ]
        assert len(fleet_sft) == 36

    def test_record_ids_are_unique(self, full_records: list[dict]) -> None:
        ids = [record["id"] for record in full_records]
        assert len(ids) == len(set(ids))

    def test_split_files_partition_the_full_dataset_by_record_type_and_split(
        self, cross_model_dir: Path, full_records: list[dict]
    ) -> None:
        dpo_full = [record for record in full_records if record["recordType"] == "dpo"]
        sft_full = [record for record in full_records if record["recordType"] == "sft"]
        dpo_train = _load_jsonl(cross_model_dir / "dpo_train_cross.jsonl")
        dpo_val = _load_jsonl(cross_model_dir / "dpo_val_cross.jsonl")
        sft_train = _load_jsonl(cross_model_dir / "train_sft_cross.jsonl")
        sft_val = _load_jsonl(cross_model_dir / "val_sft_cross.jsonl")

        assert len(dpo_train) + len(dpo_val) == len(dpo_full)
        assert len(sft_train) + len(sft_val) == len(sft_full)
        assert dpo_train and all(record["split"] == "train" for record in dpo_train)
        assert dpo_val and all(record["split"] == "validation" for record in dpo_val)
        assert sft_train and all(record["split"] == "train" for record in sft_train)
        assert sft_val and all(record["split"] == "validation" for record in sft_val)

    def test_dpo_validation_records_have_required_preference_shape(
        self, cross_model_dir: Path
    ) -> None:
        records = _load_jsonl(cross_model_dir / "dpo_val_cross.jsonl")
        assert records
        for record in records:
            assert record["recordType"] == "dpo"
            assert record["split"] == "validation"
            for key in ("id", "prompt", "chosen", "rejected", "agentRole", "taskType"):
                assert key in record
            assert record["chosen"]["role"] == "assistant"
            assert record["rejected"]["role"] == "assistant"
            chosen_payload = json.loads(record["chosen"]["content"])
            assert "handoff" in chosen_payload
            assert "targetSlotID" in chosen_payload["handoff"]
            rejected_payload = json.loads(record["rejected"]["content"])
            assert rejected_payload["tool"].endswith(".direct_private_call")

    def test_dpo_validation_record_ids_are_unique_and_reference_their_persona(
        self, cross_model_dir: Path
    ) -> None:
        records = _load_jsonl(cross_model_dir / "dpo_val_cross.jsonl")
        ids = [record["id"] for record in records]
        assert len(ids) == len(set(ids))
        for record in records:
            system_message = record["prompt"][0]["content"]
            match = re.match(r"You are (\w+)\.", system_message)
            assert match, system_message
            persona = match.group(1)
            assert persona in record["id"]

    def test_new_executor_delegate_dpo_validation_record_is_present(
        self, cross_model_dir: Path
    ) -> None:
        # Regression for this PR's addition of the executor-role delegate
        # validation example to dpo_val_cross.jsonl.
        records = _load_jsonl(cross_model_dir / "dpo_val_cross.jsonl")
        matching = [r for r in records if r["id"] == "fleet-delegate-dpo-executor-cortex"]
        assert len(matching) == 1
        record = matching[0]
        assert record["agentRole"] == "tool_executor"
        chosen_payload = json.loads(record["chosen"]["content"])
        assert chosen_payload["handoff"]["targetSlotID"] == "cortex"


class TestAgentImprovementLoopArtifactConsistency:
    @pytest.fixture()
    def loop_dir(self) -> Path:
        return _repo_root() / IMPROVEMENT_LOOP_DIR

    @pytest.fixture()
    def loop_state(self, loop_dir: Path) -> dict:
        return json.loads((loop_dir / "loop_state.json").read_text(encoding="utf-8"))

    def test_dataset_family_counts_sum_to_record_count(self, loop_state: dict) -> None:
        families = loop_state["dataset"]["families"]
        assert sum(families.values()) == loop_state["dataset"]["recordCount"]
        assert len(families) == loop_state["dataset"]["familyCount"]

    def test_embedding_pair_counts_are_internally_consistent(self, loop_state: dict) -> None:
        embedding = loop_state["dataset"]["embedding"]
        assert embedding["trainPairCount"] + embedding["valPairCount"] == embedding["pairCount"]
        assert (
            embedding["trainTripletCount"] + embedding["valTripletCount"]
            == embedding["tripletCount"]
        )
        assert embedding["hardNegativeCount"] == embedding["pairCount"]
        assert embedding["tripletCount"] == embedding["pairCount"]

    def test_reranker_pair_counts_are_internally_consistent(self, loop_state: dict) -> None:
        reranker = loop_state["dataset"]["reranker"]
        assert reranker["trainPairCount"] + reranker["valPairCount"] == reranker["pairCount"]
        assert reranker["hardNegativePairCount"] == reranker["pairCount"]

    def test_embedding_and_reranker_family_counts_match_their_summary_blocks(
        self, loop_state: dict
    ) -> None:
        families = loop_state["dataset"]["families"]
        embedding = loop_state["dataset"]["embedding"]
        reranker = loop_state["dataset"]["reranker"]
        assert families["embedding_corpus"] == embedding["corpusCount"]
        assert families["embedding_hard_negatives"] == embedding["hardNegativeCount"]
        assert families["embedding_train_pairs"] == embedding["trainPairCount"]
        assert families["embedding_val_pairs"] == embedding["valPairCount"]
        assert families["embedding_eval_retrieval"] == embedding["evalCount"]
        assert families["reranker_hard_negative_pairs"] == reranker["hardNegativePairCount"]
        assert families["reranker_train_pairs"] == reranker["trainPairCount"]
        assert families["reranker_val_pairs"] == reranker["valPairCount"]
        assert families["reranker_eval_reranking"] == reranker["evalCount"]

    def test_loop_report_dataset_records_matches_loop_state(self, loop_dir: Path, loop_state: dict) -> None:
        report_text = (loop_dir / "LOOP_REPORT.md").read_text(encoding="utf-8")
        assert f"Dataset records: `{loop_state['dataset']['recordCount']}`" in report_text

    def test_loop_report_gap_and_prompt_counts_match_loop_state(
        self, loop_dir: Path, loop_state: dict
    ) -> None:
        report_text = (loop_dir / "LOOP_REPORT.md").read_text(encoding="utf-8")
        assert f"Gaps: `{loop_state['gapCount']}`" in report_text
        assert f"Next action prompts: `{loop_state['nextActionPromptCount']}`" in report_text

    def test_testflight_runbook_build_identity_matches_loop_state_manifest(
        self, loop_dir: Path, loop_state: dict
    ) -> None:
        runbook_text = (loop_dir / "TESTFLIGHT_RUNBOOK.md").read_text(encoding="utf-8")
        fingerprint = loop_state["testFlight"]["manifestFingerprint"]
        base_commit = loop_state["manifest"]["baseCommit"]
        assert f"Manifest fingerprint: `{fingerprint}`" in runbook_text
        assert f"Manifest base commit: `{base_commit}`" in runbook_text

    def test_next_action_prompts_count_matches_loop_state(
        self, loop_dir: Path, loop_state: dict
    ) -> None:
        prompts = _load_jsonl(loop_dir / "next_action_prompts.jsonl")
        assert len(prompts) == loop_state["nextActionPromptCount"]
        assert len(prompts) == loop_state["gapCount"]

    def test_next_action_prompt_embeds_the_current_manifest_fingerprint(
        self, loop_dir: Path, loop_state: dict
    ) -> None:
        prompts = _load_jsonl(loop_dir / "next_action_prompts.jsonl")
        fingerprint = loop_state["testFlight"]["manifestFingerprint"]
        assert prompts
        for prompt in prompts:
            user_message = prompt["messages"][-1]["content"]
            assert fingerprint in user_message

    def test_next_action_prompt_schema_has_required_fields(self, loop_dir: Path) -> None:
        prompts = _load_jsonl(loop_dir / "next_action_prompts.jsonl")
        assert prompts
        for prompt in prompts:
            assert prompt["taskType"] in {"codebase_improvement", "testflight_runtime_audit", "loop_expansion"}
            assert prompt["priority"] in {"highest", "high", "medium", "low"}
            assert prompt["messages"][0]["role"] == "system"
            assert prompt["messages"][-1]["role"] == "user"
            assert "gapID" in prompt["metadata"]