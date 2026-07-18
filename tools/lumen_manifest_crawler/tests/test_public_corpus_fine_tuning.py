from __future__ import annotations

import json
from typing import Any

from lumen_manifest_crawler.dataset.fine_tuning import (
    AGENTS,
    FineTuningDatasetConfig,
    _build_experiment_variants,
    _cap_public_corpus_token_share,
    _experiment_public_group_limit,
    _public_validation_group_keys,
    _record_token_counts,
    _route_record_agents,
    _source_token_proxy_count,
    _stable_split,
    _unique_sorted_records,
    compile_agent_fine_tuning_datasets,
)
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, SourceIntegrity
from lumen_manifest_crawler.validators import _validate_public_corpus_metadata


def _provenance(*, target: str, group: str, row: str, repository: str = "OpenAssistant/oasst2") -> dict[str, Any]:
    return {
        "targetAdapter": target,
        "sourceRepository": repository,
        "sourceRevision": "a" * 40,
        "sourceLicense": "Apache-2.0",
        "sourceLicenseURL": "https://www.apache.org/licenses/LICENSE-2.0",
        "sourceURL": f"https://huggingface.co/datasets/{repository}",
        "sourcePath": f"train/{row}.json",
        "sourceContentSHA256": "b" * 64,
        "sourceArtifactSHA256": "c" * 64,
        "sourceGroupID": group,
        "partitionKind": "ml_split",
        "sourcePartition": "train",
        "transformationVersion": "public-corpus-v1",
        "transformedContentSHA256": "e" * 64,
        "attribution": "Public corpus authors",
    }


def _sft_record(*, target: str, group: str, row: str, repository: str = "OpenAssistant/oasst2") -> dict[str, Any]:
    public_corpus = _provenance(target=target, group=group, row=row, repository=repository)
    return {
        "sourceFamily": "public_adapter_corpus_dialogue",
        "taskType": "tool_call_generation",
        "agentRole": "executor",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Route this intent with strict JSON, then diagnose the tone "
                    f"and provide the final response for source row {row}."
                ),
            },
            {"role": "assistant", "content": f"Grounded response {row}."},
        ],
        "metadata": {"agent": "executor", "publicCorpus": public_corpus},
    }


def _all_sft(dataset: Any) -> list[dict[str, Any]]:
    return dataset.train_sft + dataset.val_sft


def _all_dpo(dataset: Any) -> list[dict[str, Any]]:
    return dataset.train_dpo + dataset.val_dpo


def test_public_records_route_only_to_explicit_target_and_preserve_provenance() -> None:
    manifest = AgentBehaviorManifest(sourceIntegrity=SourceIntegrity(commit="test-commit"))
    mouth_records = [
        _sft_record(target="mouth", group=group, row=f"{group}-{index}")
        for group in ("conversation-a", "conversation-b")
        for index in range(2)
    ]
    rejected_record = _sft_record(target="unknown", group="invalid", row="invalid")
    dpo_provenance = _provenance(
        target="rem",
        group="preference-a",
        row="preference-a",
        repository="grammarly/coedit",
    )
    dpo_record = {
        "sourceFamily": "public_adapter_corpus_preferences",
        "taskType": "dataset_repair",
        "prompt": [{"role": "user", "content": "Diagnose this response and select the grounded repair."}],
        "chosen": {"role": "assistant", "content": "Use the evidence-backed repair."},
        "rejected": {"role": "assistant", "content": "Invent a repair without evidence."},
        "metadata": {"agentRole": "mouth", "publicCorpus": dpo_provenance},
    }

    compiled = compile_agent_fine_tuning_datasets(
        manifest,
        {
            "public_adapter_corpus_mouth": mouth_records + [rejected_record],
            "public_adapter_corpus_preferences": [dpo_record],
        },
        config=FineTuningDatasetConfig(
            validation_ratio=0.5,
            max_public_corpus_token_share=None,
            include_unsloth_config=False,
        ),
    )

    mouth_public = [record for record in _all_sft(compiled["mouth"]) if "publicCorpus" in record["metadata"]]
    assert len(mouth_public) == 4
    assert all(record["metadata"]["publicCorpus"]["targetAdapter"] == "mouth" for record in mouth_public)
    assert all(record["metadata"]["publicCorpus"] in [item["metadata"]["publicCorpus"] for item in mouth_records] for record in mouth_public)

    for agent in set(AGENTS) - {"mouth", "rem"}:
        assert not [record for record in _all_sft(compiled[agent]) if "publicCorpus" in record["metadata"]]
        assert not [record for record in _all_dpo(compiled[agent]) if "publicCorpus" in record["metadata"]]

    rem_dpo = [record for record in _all_dpo(compiled["rem"]) if "publicCorpus" in record["metadata"]]
    assert len(rem_dpo) == 1
    assert rem_dpo[0]["metadata"]["publicCorpus"] == dpo_provenance
    assert rem_dpo[0]["metadata"]["sourceFamily"] == "public_adapter_corpus_preferences"
    assert rem_dpo[0]["metadata"]["taskType"] == "dataset_repair"

    all_public = [
        record["metadata"]["publicCorpus"]
        for agent in AGENTS
        for record in _all_sft(compiled[agent]) + _all_dpo(compiled[agent])
        if "publicCorpus" in record["metadata"]
    ]
    assert not any(item["sourcePath"] == "train/invalid.json" for item in all_public)

    mouth_card = compiled["mouth"].dataset_card["publicCorpus"]
    assert mouth_card["recordCounts"] == {
        "train_sft": 2,
        "val_sft": 2,
        "train_dpo": 0,
        "val_dpo": 0,
    }
    assert mouth_card["sourceCounts"] == {"OpenAssistant/oasst2": 4}
    assert mouth_card["licenses"] == ["Apache-2.0"]
    assert "public_adapter_corpus_dialogue" in compiled["mouth"].dataset_card["sourceFamilies"]
    assert "public_adapter_corpus_mouth" not in compiled["mouth"].dataset_card["sourceFamilies"]

    rem_card = compiled["rem"].dataset_card["publicCorpus"]
    assert rem_card["sourceCounts"] == {"grammarly/coedit": 2}
    assert rem_card["licenses"] == ["Apache-2.0"]


def test_public_groups_do_not_cross_sft_splits_and_internal_split_is_unchanged() -> None:
    config = FineTuningDatasetConfig(validation_ratio=0.5)
    internal = _unique_sorted_records([{"id": f"internal-{index}"} for index in range(6)])
    baseline_train, baseline_val = _stable_split(internal, config)

    public = _unique_sorted_records(
        [
            {
                "id": f"public-{group}-{index}",
                "metadata": {
                    "publicCorpus": _provenance(target="mouth", group=group, row=f"{group}-{index}")
                },
            }
            for group in ("conversation-a", "conversation-b")
            for index in range(2)
        ]
    )
    mixed_train, mixed_val = _stable_split(_unique_sorted_records(internal + public), config)

    assert {record["id"] for record in mixed_train if record["id"].startswith("internal-")} == {
        record["id"] for record in baseline_train
    }
    assert {record["id"] for record in mixed_val if record["id"].startswith("internal-")} == {
        record["id"] for record in baseline_val
    }

    train_groups = {
        record["metadata"]["publicCorpus"]["sourceGroupID"]
        for record in mixed_train
        if record["id"].startswith("public-")
    }
    val_groups = {
        record["metadata"]["publicCorpus"]["sourceGroupID"]
        for record in mixed_val
        if record["id"].startswith("public-")
    }
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)


def test_shared_public_group_has_one_global_split_across_adapters() -> None:
    config = FineTuningDatasetConfig(validation_ratio=0.5)
    records_by_agent = {
        "mouth": [
            {"id": "mouth-shared", "metadata": {"publicCorpus": _provenance(target="mouth", group="shared-group", row="mouth-shared")}},
            {"id": "mouth-only", "metadata": {"publicCorpus": _provenance(target="mouth", group="mouth-only", row="mouth-only")}},
        ],
        "fleet": [
            {"id": "fleet-shared", "metadata": {"publicCorpus": _provenance(target="fleet", group="shared-group", row="fleet-shared")}},
            {"id": "fleet-only", "metadata": {"publicCorpus": _provenance(target="fleet", group="fleet-only", row="fleet-only")}},
        ],
    }
    validation_groups = _public_validation_group_keys(
        [record for records in records_by_agent.values() for record in records],
        config,
    )

    lanes: dict[str, str] = {}
    for agent, records in records_by_agent.items():
        train, validation = _stable_split(
            records,
            config,
            public_validation_group_keys=validation_groups,
        )
        for lane, split_records in (("train", train), ("validation", validation)):
            if any(
                (record.get("metadata") or {}).get("publicCorpus", {}).get("sourceGroupID")
                == "shared-group"
                for record in split_records
            ):
                lanes[agent] = lane
    assert set(lanes) == {"mouth", "fleet"}
    assert lanes["mouth"] == lanes["fleet"]


def test_public_split_stratifies_each_source_family() -> None:
    config = FineTuningDatasetConfig(validation_ratio=0.2)
    records: list[dict[str, Any]] = []
    for repository, stratum in (("AmazonScience/massive", "weather"), ("json-schema-org/JSON-Schema-Test-Suite", "type.json")):
        for index in range(10):
            provenance = _provenance(
                target="rem",
                group=f"{index:064x}",
                row=f"{stratum}-{index}",
                repository=repository,
            )
            provenance["stratum"] = stratum
            records.append(
                {
                    "id": f"{repository}-{index}",
                    "metadata": {"publicCorpus": provenance},
                }
            )

    train, validation = _stable_split(_unique_sorted_records(records), config)
    train_sources = {
        record["metadata"]["publicCorpus"]["sourceRepository"] for record in train
    }
    validation_sources = {
        record["metadata"]["publicCorpus"]["sourceRepository"] for record in validation
    }
    assert train_sources == validation_sources == {
        "AmazonScience/massive",
        "json-schema-org/JSON-Schema-Test-Suite",
    }


def test_explicit_adapter_role_suppresses_incidental_text_heuristics() -> None:
    manifest = AgentBehaviorManifest(sourceIntegrity=SourceIntegrity(commit="test-commit"))
    records = {
        "role_directory_samples": [
            {
                "agentRole": "cortex",
                "taskType": "custom_role_directory",
                "messages": [
                    {"role": "user", "content": "Explain the fleet role directory, final user-facing response, and tone policy."},
                    {"role": "assistant", "content": "Cortex routes the request without taking over peer roles."},
                ],
            },
            {
                "sourceFamily": "adapter_ultra_specific",
                "agentRole": "fleet",
                "taskType": "ultra_specific_fleet_slot_directory",
                "messages": [
                    {"role": "user", "content": "Describe tone and the final user-facing stage."},
                    {
                        "role": "assistant",
                        "content": '{"scope":"fleet_boundaries_only"}',
                    },
                ],
            },
        ]
    }

    compiled = compile_agent_fine_tuning_datasets(
        manifest,
        records,
        config=FineTuningDatasetConfig(include_unsloth_config=False),
    )
    fixture_prompts = {
        "Explain the fleet role directory, final user-facing response, and tone policy.",
        "Describe tone and the final user-facing stage.",
    }
    custom_by_agent = {
        agent: [
            record
            for record in _all_sft(compiled[agent])
            if record["messages"][1]["content"] in fixture_prompts
        ]
        for agent in AGENTS
    }

    assert len(custom_by_agent["cortex"]) == 1
    assert len(custom_by_agent["fleet"]) == 1
    assert not custom_by_agent["mouth"]
    assert not custom_by_agent["mimicry"]
    assert not custom_by_agent["executor"]
    assert not custom_by_agent["rem"]


def test_fleet_routing_uses_structured_ownership_without_bypassing_role_locks() -> None:
    manifest = AgentBehaviorManifest(sourceIntegrity=SourceIntegrity(commit="test-commit"))
    role_targets = {
        "orchestrator": ("cortex", "fleet_delegation"),
        "tool_executor": ("executor", "fleet_peer_source_knowledge"),
        "user_response": ("mouth", "source_code_self_knowledge"),
        "tone_adapter": ("mimicry", "fleet_peer_knowledge"),
        "idle_reflection": ("rem", "fleet_peer_knowledge"),
    }
    peer_records = [
        {
            "agentRole": role,
            "taskType": task_type,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fleet peer. Read slotID, model directory, and delegation boundaries.",
                },
                {
                    "role": "user",
                    "content": f"Describe the full model fleet from the {target} source boundary.",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"sourceBoundary": target},
                        sort_keys=True,
                    ),
                },
            ],
        }
        for role, (target, task_type) in role_targets.items()
    ]
    text_only = {
        "taskType": "text_only_role_words",
        "messages": [
            {
                "role": "system",
                "content": "You are Fleet. Inspect slotID, model directory, peer role, and delegation.",
            },
            {"role": "user", "content": "This text has no structured slot ownership."},
            {"role": "assistant", "content": "Do not route from serialized message text."},
        ],
    }
    structured_fleet = {
        **text_only,
        "sourceFamily": "adapter_ultra_specific",
        "taskType": "ultra_specific_fleet_slot_directory",
        "messages": [
            *text_only["messages"][:-1],
            {
                "role": "assistant",
                "content": '{"ownership":"structured_fleet"}',
            },
        ],
        "metadata": {"agentRole": "fleet"},
    }
    # This routing-focused fixture includes enough role-native primary loss to
    # retain all five compact cross-model rows while keeping their shared
    # source family below the production 5% source-proxy safety ceiling. Short
    # synthetic primary targets need more rows than the real Fleet corpus.
    fleet_primary_support = [
        {
            "sourceFamily": "adapter_ultra_specific",
            "taskType": "ultra_specific_fleet_delegation",
            "messages": [
                {
                    "role": "user",
                    "content": f"Return primary Fleet routing support row {index}.",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"routingSupport": index},
                        sort_keys=True,
                    ),
                },
            ],
            "metadata": {"agentRole": "fleet"},
        }
        for index in range(144)
    ]

    compiled = compile_agent_fine_tuning_datasets(
        manifest,
        {
            "cross_model_training": peer_records,
            "unowned_text_samples": [
                text_only,
                structured_fleet,
                *fleet_primary_support,
            ],
        },
        config=FineTuningDatasetConfig(
            include_unsloth_config=False,
            max_supplemental_sft_ratio=0.75,
        ),
    )

    for target in AGENTS:
        peer_samples = [
            record
            for record in _all_sft(compiled[target])
            if record["messages"][1]["content"].startswith(
                "Describe the full model fleet from the "
            )
        ]
        assert len(peer_samples) == (len(peer_records) if target == "fleet" else 0)
    assert not [
        record
        for agent in AGENTS
        for record in _all_sft(compiled[agent])
        if record["metadata"].get("taskType") == "text_only_role_words"
    ]
    structured_fleet_samples = [
        record
        for record in _all_sft(compiled["fleet"])
        if record["messages"][1]["content"]
        == "This text has no structured slot ownership."
    ]
    assert len(structured_fleet_samples) == 1


def test_known_fleet_slot_metadata_without_adapter_mapping_routes_to_fleet() -> None:
    assert _route_record_agents(
        source_family="custom_slot_contract",
        record={"metadata": {"slotRole": "planner"}},
        task_type="custom_slot_contract",
        tool_ids=[],
        slot_ids={"planner-slot"},
        slot_roles={"planner"},
    ) == ["fleet"]
    assert _route_record_agents(
        source_family="custom_slot_contract",
        record={"metadata": {"slotID": "planner-slot"}},
        task_type="custom_slot_contract",
        tool_ids=[],
        slot_ids={"planner-slot"},
        slot_roles={"planner"},
    ) == ["fleet"]


def test_public_corpus_provenance_validation_fails_closed() -> None:
    valid = _provenance(target="mouth", group="d" * 64, row="safe-row")
    failures: list[Any] = []
    _validate_public_corpus_metadata(
        valid,
        agent="mouth",
        path="fine_tuning.mouth.sft.0.metadata.publicCorpus",
        failures=failures,
    )
    assert failures == []

    invalid = {
        **valid,
        "targetAdapter": "fleet",
        "sourceLicense": "CC-BY-NC-4.0",
        "sourceRevision": "short",
        "sourceContentSHA256": "not-a-digest",
        "sourcePartition": "validation",
        "partitionKind": "unknown_partition",
        "user_id": "raw-upstream-user",
    }
    failures = []
    _validate_public_corpus_metadata(
        invalid,
        agent="mouth",
        path="fine_tuning.mouth.sft.0.metadata.publicCorpus",
        failures=failures,
    )
    codes = {failure.code for failure in failures}
    assert {
        "public_corpus_adapter_mismatch",
        "public_corpus_license_not_allowed",
        "public_corpus_revision_not_pinned",
        "public_corpus_invalid_digest",
        "public_corpus_invalid_partition_kind",
        "public_corpus_raw_identifier_leak",
    }.issubset(codes)

    heldout = {**valid, "sourcePartition": "validation"}
    failures = []
    _validate_public_corpus_metadata(
        heldout,
        agent="mouth",
        path="fine_tuning.mouth.sft.0.metadata.publicCorpus",
        failures=failures,
    )
    assert "public_corpus_heldout_split_ingested" in {failure.code for failure in failures}


def test_sft_deduplicates_exact_messages_and_prefers_lumen_native_record() -> None:
    manifest = AgentBehaviorManifest(sourceIntegrity=SourceIntegrity(commit="test-commit"))
    messages = [
        {"role": "user", "content": "Summarize the approved observation."},
        {"role": "assistant", "content": "The approved operation completed successfully."},
    ]
    public_record = {
        "sourceFamily": "public_adapter_corpus_dialogue",
        "taskType": "public_grounded_response_finalization",
        "messages": messages,
        "metadata": {
            "agent": "mouth",
            "publicCorpus": _provenance(
                target="mouth",
                group="duplicate-public-group",
                row="duplicate-public",
            ),
        },
    }
    native_record = {
        "sourceFamily": "mouth_responses",
        "taskType": "user_response_generation",
        "messages": messages,
        "metadata": {"agent": "mouth"},
    }

    compiled = compile_agent_fine_tuning_datasets(
        manifest,
        {
            "mouth_responses": [native_record],
            "public_adapter_corpus_mouth": [public_record],
        },
        config=FineTuningDatasetConfig(include_unsloth_config=False),
    )
    matches = [record for record in _all_sft(compiled["mouth"]) if record["messages"][1:] == messages]
    assert len(matches) == 1
    assert "publicCorpus" not in matches[0]["metadata"]


def test_public_corpus_total_and_target_token_shares_are_capped_per_lane() -> None:
    manifest = AgentBehaviorManifest(sourceIntegrity=SourceIntegrity(commit="test-commit"))
    internal_records = [
        {
            "sourceFamily": "mouth_responses",
            "taskType": "user_response_generation",
            "messages": [
                {"role": "user", "content": f"Summarize approved observation {index}."},
                {
                    "role": "assistant",
                    "content": f"Observation {index} completed with a verified result for the user.",
                },
            ],
            "metadata": {"agent": "mouth"},
        }
        for index in range(24)
    ]
    public_records = []
    for index in range(40):
        long_target = " ".join([f"supported-{index}"] * 30)
        public_records.append(
            {
                "sourceFamily": "public_adapter_corpus_dialogue",
                "taskType": "public_grounded_response_finalization",
                "messages": [
                    {"role": "user", "content": f"Use only source observation {index}."},
                    {"role": "assistant", "content": long_target},
                ],
                "metadata": {
                    "agent": "mouth",
                    "publicCorpus": _provenance(
                        target="mouth",
                        group=f"public-group-{index}",
                        row=f"public-{index}",
                    ),
                },
            }
        )

    cap = 0.30
    compiled = compile_agent_fine_tuning_datasets(
        manifest,
        {
            "mouth_responses": internal_records,
            "public_adapter_corpus_mouth": public_records,
        },
        config=FineTuningDatasetConfig(
            validation_ratio=0.25,
            max_public_corpus_token_share=cap,
            include_unsloth_config=False,
        ),
    )
    public_card = compiled["mouth"].dataset_card["publicCorpus"]
    assert 0 < sum(public_card["recordCounts"].values()) < len(public_records)
    assert public_card["maxSFTTokenProxyShare"] == cap
    proxy_contract = public_card["selectionContract"][
        "sourceTokenProxyContract"
    ]
    assert proxy_contract == {
        "schemaVersion": "lumen.source-token-proxy/1.0.0",
        "status": "source_side_selection_proxy_not_exact_token_count",
        "strategy": "max_whitespace_terms_utf8_byte_ceiling",
        "maxCharsPerToken": 4,
        "exactPinnedTokenizerAuthoritative": True,
        "authoritativeEnforcementPhase": "post_tokenizer_load_pre_optimizer",
    }
    for lane in ("train_sft", "val_sft"):
        assert public_card["tokenProxyShares"][lane]["total"] <= cap
        assert public_card["tokenProxyShares"][lane]["target"] <= cap


def test_source_token_proxy_counts_minified_json_without_claiming_exact_tokens() -> None:
    minified_json = (
        '{"action":{"tool":"calendar.create","args":{"title":"Quarterly '
        'planning","startsAt":"2026-08-01T09:00:00-04:00"}}}'
    )
    prose = "Create the quarterly planning event at nine tomorrow morning."

    assert len(minified_json.split()) == 2
    assert _source_token_proxy_count(minified_json, max_chars_per_token=4) == (
        len(minified_json.encode("utf-8")) + 3
    ) // 4
    assert _source_token_proxy_count(minified_json) > len(minified_json.split())
    assert _source_token_proxy_count(
        minified_json,
        max_chars_per_token=2,
    ) > _source_token_proxy_count(minified_json, max_chars_per_token=4)
    assert _source_token_proxy_count(prose) >= len(prose.split())
    assert _source_token_proxy_count("éééé", max_chars_per_token=4) == 2

    record = {
        "messages": [
            {"role": "user", "content": prose},
            {"role": "assistant", "content": minified_json},
        ]
    }
    total_proxy, target_proxy = _record_token_counts(record)
    assert target_proxy == _source_token_proxy_count(minified_json)
    assert total_proxy == target_proxy + _source_token_proxy_count(prose)


def test_public_share_cap_rejects_minified_json_that_whitespace_count_would_admit() -> None:
    internal = {
        "messages": [
            {"role": "user", "content": "Ground this internal request carefully."},
            {
                "role": "assistant",
                "content": "one two three four five six seven eight nine ten",
            },
        ],
        "metadata": {"agent": "executor"},
    }
    minified = (
        '{"action":{"tool":"calendar.create","args":{"title":"Planning",'
        '"startsAt":"2026-08-01T09:00:00-04:00"}}}'
    )
    public = {
        "messages": [
            {"role": "user", "content": "Route it."},
            {"role": "assistant", "content": minified},
        ],
        "metadata": {
            "agent": "executor",
            "publicCorpus": _provenance(
                target="executor",
                group="minified-json",
                row="minified-json",
            ),
        },
    }

    selected = _cap_public_corpus_token_share(
        [internal, public],
        0.20,
        max_chars_per_token=4,
    )
    assert selected == [internal]
    assert _record_token_counts(public)[1] > len(minified.split())


def test_public_token_cap_balances_sources_before_source_strata() -> None:
    internal = [
        {
            "messages": [
                {"role": "user", "content": f"Internal request {index} with grounded context."},
                {"role": "assistant", "content": f"Internal grounded answer {index} is complete."},
            ],
            "metadata": {"agent": "rem"},
        }
        for index in range(30)
    ]
    public: list[dict[str, Any]] = []
    for source, stratum_count in (("source/many-strata", 10), ("source/one-stratum", 1)):
        for index in range(20):
            provenance = _provenance(
                target="rem",
                group=f"{source}-{index}",
                row=f"{source}-{index}",
                repository=source,
            )
            provenance["stratum"] = f"stratum-{index % stratum_count}"
            public.append(
                {
                    "messages": [
                        {"role": "user", "content": f"Repair public candidate {source} {index}."},
                        {"role": "assistant", "content": f"Public repair {source} {index} is valid."},
                    ],
                    "metadata": {"agent": "rem", "publicCorpus": provenance},
                }
            )

    selected = _cap_public_corpus_token_share(internal + public, 0.30)
    source_counts: dict[str, int] = {}
    for record in selected:
        provenance = (record.get("metadata") or {}).get("publicCorpus")
        if isinstance(provenance, dict):
            source = provenance["sourceRepository"]
            source_counts[source] = source_counts.get(source, 0) + 1
    assert set(source_counts) == {"source/many-strata", "source/one-stratum"}
    assert abs(source_counts["source/many-strata"] - source_counts["source/one-stratum"]) <= 1


def test_public_token_cap_prefers_higher_selection_score_within_stratum() -> None:
    internal = [{
        "messages": [
            {"role": "user", "content": " ".join(["internal-user"] * 100)},
            {"role": "assistant", "content": " ".join(["internal-target"] * 100)},
        ],
        "metadata": {"agent": "mouth"},
    }]
    public = []
    for name, score in (("low", 0.1), ("high", 0.9)):
        provenance = _provenance(
            target="mouth",
            group=f"quality-{name}",
            row=f"quality-{name}",
        )
        provenance["stratum"] = "grounded-final"
        provenance["selectionScore"] = {"overall": score}
        public.append({
            "messages": [
                {"role": "user", "content": " ".join([f"{name}-user"] * 10)},
                {"role": "assistant", "content": " ".join([f"{name}-target"] * 10)},
            ],
            "metadata": {"agent": "mouth", "publicCorpus": provenance},
        })

    selected = _cap_public_corpus_token_share([*internal, *public], 0.10)
    selected_public = [
        record for record in selected
        if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
    ]
    assert len(selected_public) == 1
    assert selected_public[0]["metadata"]["publicCorpus"]["selectionScore"]["overall"] == 0.9


def test_experiment_variants_separate_internal_baseline_and_quality_optimized_corpora() -> None:
    internal = [{
        "messages": [
            {"role": "user", "content": " ".join(["internal-user"] * 100)},
            {"role": "assistant", "content": " ".join(["internal-target"] * 100)},
        ],
        "metadata": {"agent": "mouth"},
    }]
    public = []
    for index in range(10):
        provenance = _provenance(
            target="mouth",
            group=f"variant-group-{index}",
            row=f"variant-{index}",
        )
        provenance["stratum"] = "grounded-final"
        provenance["selectionScore"] = {"overall": 0.5}
        public.append({
            "messages": [
                {"role": "user", "content": " ".join([f"public-user-{index}"] * 10)},
                {"role": "assistant", "content": " ".join([f"public-target-{index}"] * 10)},
            ],
            "metadata": {"agent": "mouth", "publicCorpus": provenance},
        })

    baseline_probe = _cap_public_corpus_token_share(
        [*internal, *public],
        0.10,
        prefer_quality=False,
    )
    baseline_group = next(
        record["metadata"]["publicCorpus"]["sourceGroupID"]
        for record in baseline_probe
        if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
    )
    for index, record in enumerate(public):
        public_metadata = record["metadata"]["publicCorpus"]
        public_metadata["selectionScore"]["overall"] = (
            0.0 if public_metadata["sourceGroupID"] == baseline_group else index + 1.0
        )

    optimized = _cap_public_corpus_token_share([*internal, *public], 0.10)
    variants, experiment = _build_experiment_variants(
        agent="mouth",
        available_train_sft=[*internal, *public],
        available_val_sft=[],
        available_train_dpo=[],
        available_val_dpo=[],
        evaluation_records=[],
        training_config={"base_model_name": "Qwen/Qwen3-1.7B", "seed": 42},
        max_public_share=0.10,
    )

    internal_only = variants["internal_only"]
    assert internal_only["train_sft"] == internal
    assert not any(
        isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
        for record in internal_only["train_sft"]
    )
    assert variants["internal_plus_public_optimized"]["train_sft"] == optimized
    assert (
        variants["internal_plus_public_baseline"]["variant_manifest"]["trainingCorpusSHA256"]
        != variants["internal_plus_public_optimized"]["variant_manifest"]["trainingCorpusSHA256"]
    )
    assert experiment["variantOrder"] == [
        "internal_only",
        "internal_plus_public_baseline",
        "internal_plus_public_optimized",
    ]


def test_experiment_selection_policies_produce_distinct_corpora_for_all_adapters() -> None:
    for agent in AGENTS:
        internal = [{
            "messages": [
                {"role": "user", "content": " ".join([f"{agent}-internal-user"] * 100)},
                {"role": "assistant", "content": " ".join([f"{agent}-internal-target"] * 100)},
            ],
            "metadata": {"agent": agent},
        }]
        public: list[dict[str, Any]] = []
        for index in range(10):
            provenance = _provenance(
                target=agent,
                group=f"{agent}-policy-group-{index}",
                row=f"{agent}-policy-{index}",
            )
            provenance["stratum"] = f"stratum-{index % 2}"
            provenance["selectionScore"] = {"overall": 0.5}
            public.append({
                "messages": [
                    {"role": "user", "content": " ".join([f"{agent}-public-user-{index}"] * 10)},
                    {"role": "assistant", "content": " ".join([f"{agent}-public-target-{index}"] * 10)},
                ],
                "metadata": {"agent": agent, "publicCorpus": provenance},
            })

        group_limit = _experiment_public_group_limit([*internal, *public])
        assert group_limit == 8
        baseline_probe = _cap_public_corpus_token_share(
            [*internal, *public],
            0.80,
            prefer_quality=False,
            max_public_groups=group_limit,
        )
        baseline_groups = {
            record["metadata"]["publicCorpus"]["sourceGroupID"]
            for record in baseline_probe
            if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
        }
        for record in public:
            public_metadata = record["metadata"]["publicCorpus"]
            public_metadata["selectionScore"]["overall"] = (
                0.0 if public_metadata["sourceGroupID"] in baseline_groups else 1.0
            )

        variants, experiment = _build_experiment_variants(
            agent=agent,
            available_train_sft=[*internal, *public],
            available_val_sft=[],
            available_train_dpo=[],
            available_val_dpo=[],
            evaluation_records=[],
            training_config={"base_model_name": "Qwen/Qwen3-1.7B", "seed": 42},
            max_public_share=0.80,
        )
        baseline_manifest = variants["internal_plus_public_baseline"]["variant_manifest"]
        optimized_manifest = variants["internal_plus_public_optimized"]["variant_manifest"]
        assert baseline_manifest["trainingCorpusSHA256"] != optimized_manifest["trainingCorpusSHA256"]
        assert baseline_manifest["publicSelectionPolicy"]["qualityScorePreference"] is False
        assert optimized_manifest["publicSelectionPolicy"]["qualityScorePreference"] is True
        assert (
            baseline_manifest["publicSelectionPolicy"]["lanePublicGroupLimits"]
            == optimized_manifest["publicSelectionPolicy"]["lanePublicGroupLimits"]
        )
        assert experiment["comparisonEligibility"] == {
            "status": "eligible",
            "promotionEligible": True,
            "promotionProhibited": False,
            "reason": "distinct_public_selection_corpora",
            "publicRecordCount": 10,
            "baselineTrainingCorpusSHA256": baseline_manifest["trainingCorpusSHA256"],
            "optimizedTrainingCorpusSHA256": optimized_manifest["trainingCorpusSHA256"],
        }


def test_identical_public_variant_corpora_are_marked_not_applicable_for_promotion() -> None:
    internal = [{
        "messages": [
            {"role": "user", "content": "internal request"},
            {"role": "assistant", "content": "internal response"},
        ],
        "metadata": {"agent": "mouth"},
    }]
    provenance = _provenance(target="mouth", group="only-public-group", row="only-public")
    provenance["selectionScore"] = {"overall": 1.0}
    public = [{
        "messages": [
            {"role": "user", "content": "public request"},
            {"role": "assistant", "content": "public response"},
        ],
        "metadata": {"agent": "mouth", "publicCorpus": provenance},
    }]

    variants, experiment = _build_experiment_variants(
        agent="mouth",
        available_train_sft=[*internal, *public],
        available_val_sft=[],
        available_train_dpo=[],
        available_val_dpo=[],
        evaluation_records=[],
        training_config={"base_model_name": "Qwen/Qwen3-1.7B", "seed": 42},
        max_public_share=0.80,
    )

    baseline_manifest = variants["internal_plus_public_baseline"]["variant_manifest"]
    optimized_manifest = variants["internal_plus_public_optimized"]["variant_manifest"]
    assert baseline_manifest["trainingCorpusSHA256"] == optimized_manifest["trainingCorpusSHA256"]
    assert experiment["comparisonEligibility"]["status"] == "not_applicable"
    assert experiment["comparisonEligibility"]["promotionEligible"] is False
    assert experiment["comparisonEligibility"]["promotionProhibited"] is True
    assert (
        experiment["comparisonEligibility"]["reason"]
        == "identical_baseline_and_optimized_training_corpora"
    )
