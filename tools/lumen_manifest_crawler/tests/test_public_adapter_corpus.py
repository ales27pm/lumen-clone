from __future__ import annotations

import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from lumen_manifest_crawler.dataset import public_adapter_corpus as corpus


def _lumen_manifest(path: Path) -> Path:
    payload = {
        "tools": [
            {"id": "alarm.list", "arguments": []},
            {"id": "calendar.list", "arguments": []},
            {"id": "reminders.list", "arguments": []},
            {
                "id": "weather",
                "arguments": [
                    {"name": "city", "type": "string", "required": False, "allowedValues": None},
                ],
            },
            {
                "id": "maps.search",
                "arguments": [
                    {"name": "query", "type": "string", "required": True, "allowedValues": None},
                ],
            },
            {
                "id": "contacts.search",
                "arguments": [
                    {"name": "query", "type": "string", "required": True, "allowedValues": None},
                ],
            },
            {
                "id": "web.search",
                "arguments": [
                    {"name": "query", "type": "string", "required": True, "allowedValues": None},
                ],
            },
        ],
        "intents": [
            {"id": intent, "allowedToolIDs": tools}
            for intent, tools in (
                ("alarm", ["alarm.list"]),
                ("calendar", ["calendar.list"]),
                ("chat", []),
                ("contactSearch", ["contacts.search"]),
                ("emailDraft", []),
                ("maps", ["maps.search"]),
                ("outlook", []),
                ("reminder", ["reminders.list"]),
                ("weather", ["weather"]),
                ("webSearch", ["web.search"]),
            )
        ],
        "agentProtocols": {
            "cortexOutput": {
                "requiredFields": [
                    "intent",
                    "selectedToolID",
                    "requiresApproval",
                    "nextModel",
                    "reasoningSummary",
                ]
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source(
    source_id: str,
    artifact: Path,
    *,
    artifact_format: str,
    adapter_caps: dict[str, int],
    **extra: object,
) -> dict[str, object]:
    return {
        "id": source_id,
        "datasetID": "example/public",
        "revision": "a" * 40,
        "sourceURL": "https://example.test/source/tree/" + "a" * 40,
        "artifactURL": "https://example.test/artifact",
        "artifactFormat": artifact_format,
        "artifactSHA256": corpus._sha256_file(artifact),
        "partitionKind": "ml_split",
        "sourcePartition": "train",
        "license": "MIT",
        "licenseURL": "https://opensource.org/license/mit",
        "attribution": "Example public source.",
        "adapterCaps": adapter_caps,
        **extra,
    }


def _source_manifest(path: Path, sources: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "lumen.public-adapter-corpus-sources/1.0.0",
                "selectionPolicyVersion": "test.1",
                "allowedLicenses": ["MIT"],
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def _massive_tar(path: Path) -> Path:
    rows = [
        {
            "id": "1",
            "locale": "en-US",
            "partition": "train",
            "scenario": "weather",
            "intent": "weather_query",
            "utt": "will it rain in Montreal tomorrow",
            "annot_utt": "will it rain in [place_name : Montreal] tomorrow",
        },
        {
            "id": "2",
            "locale": "en-US",
            "partition": "train",
            "scenario": "qa",
            "intent": "qa_definition",
            "utt": "what does luminescence mean",
            "annot_utt": "what does [definition_word : luminescence] mean",
        },
        {
            "id": "3",
            "locale": "en-US",
            "partition": "validation",
            "scenario": "qa",
            "intent": "qa_definition",
            "utt": "reserved validation question",
            "annot_utt": "reserved validation question",
        },
        {
            "id": "4",
            "locale": "en-US",
            "partition": "train",
            "scenario": "music",
            "intent": "play_music",
            "utt": "play music",
            "annot_utt": "play music",
        },
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    info = tarfile.TarInfo("1.1/data/en-US.jsonl")
    info.size = len(payload)
    info.mtime = 0
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    return path


def _oasst_labels(**overrides: float) -> dict[str, list[object]]:
    labels = {
        "spam": 0.0,
        "fails_task": 0.0,
        "lang_mismatch": 0.0,
        "pii": 0.0,
        "not_appropriate": 0.0,
        "hate_speech": 0.0,
        "sexual_content": 0.0,
        "quality": 0.9,
        "toxicity": 0.0,
        "helpfulness": 0.9,
        "violence": 0.0,
        **overrides,
    }
    return {"name": list(labels), "value": list(labels.values()), "count": [3] * len(labels)}


def test_pinned_source_manifest_has_only_approved_licensed_sources() -> None:
    manifest = corpus.load_public_corpus_source_manifest()

    assert {source["datasetID"] for source in manifest["sources"]} == {
        "AmazonScience/massive",
        "OpenAssistant/oasst2",
        "grammarly/coedit",
        "json-schema-org/JSON-Schema-Test-Suite",
    }
    assert {source["license"] for source in manifest["sources"]} <= set(manifest["allowedLicenses"])
    assert all(len(source["revision"]) == 40 for source in manifest["sources"])
    assert all(len(source["artifactSHA256"]) == 64 for source in manifest["sources"])
    assert {source["partitionKind"] for source in manifest["sources"]} == {
        "ml_split",
        "reference_corpus",
    }
    oasst = next(source for source in manifest["sources"] if source["datasetID"] == "OpenAssistant/oasst2")
    assert oasst["artifactFormat"] == "parquet"
    assert oasst["partitionKind"] == "ml_split"
    assert oasst["sourcePartition"] == "train"
    assert "validation" not in oasst["artifactURL"]
    reference = next(
        source
        for source in manifest["sources"]
        if source["datasetID"] == "json-schema-org/JSON-Schema-Test-Suite"
    )
    assert reference["partitionKind"] == "reference_corpus"
    assert reference["sourcePartition"] == "draft2020-12"


def test_source_manifest_rejects_unknown_license_and_short_revision(tmp_path: Path) -> None:
    artifact = tmp_path / "source.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    source = _source("coedit-test", artifact, artifact_format="jsonl", adapter_caps={"rem": 1})
    source["license"] = "unknown"
    source["revision"] = "main"
    path = _source_manifest(tmp_path / "sources.json", [source])

    with pytest.raises(corpus.PublicCorpusError):
        corpus.load_public_corpus_source_manifest(path)


@pytest.mark.parametrize("partition", ["validation", "test", "future-holdout"])
def test_source_manifest_fails_closed_for_unapproved_ml_partitions(
    tmp_path: Path,
    partition: str,
) -> None:
    artifact = tmp_path / "source.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    source = _source("coedit-test", artifact, artifact_format="jsonl", adapter_caps={"rem": 1})
    source["sourcePartition"] = partition
    path = _source_manifest(tmp_path / "sources.json", [source])

    with pytest.raises(corpus.PublicCorpusError, match="not approved for training"):
        corpus.load_public_corpus_source_manifest(path)


def test_canonical_lumen_contract_hash_ignores_manifest_format_and_catalog_order(tmp_path: Path) -> None:
    first_path = _lumen_manifest(tmp_path / "first.json")
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    payload["tools"].reverse()
    payload["intents"].reverse()
    payload["tools"][0]["source"] = "a non-contract source location"
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    assert corpus.load_lumen_contract(first_path).sha256 == corpus.load_lumen_contract(second_path).sha256


def test_manifest_tool_validation_rejects_missing_extra_and_wrong_types(tmp_path: Path) -> None:
    contract = corpus.load_lumen_contract(_lumen_manifest(tmp_path / "lumen.json"))

    assert corpus._validate_tool_call(
        {"tool": "maps.search", "arguments": {"query": "coffee"}}, contract.tools
    )
    assert not corpus._validate_tool_call({"tool": "maps.search", "arguments": {}}, contract.tools)
    assert not corpus._validate_tool_call(
        {"tool": "maps.search", "arguments": {"query": 3}}, contract.tools
    )
    assert not corpus._validate_tool_call(
        {"tool": "maps.search", "arguments": {"query": "coffee", "radius": 3}}, contract.tools
    )
    assert not corpus._validate_tool_call(
        {"tool": "foreign.weather", "arguments": {}}, contract.tools
    )


def test_massive_transform_uses_only_lumen_intents_and_valid_tool_envelopes(tmp_path: Path) -> None:
    artifact = _massive_tar(tmp_path / "massive.tar.gz")
    source = _source(
        "massive-test",
        artifact,
        artifact_format="tar.gz-jsonl",
        artifactMember="1.1/data/en-US.jsonl",
        adapter_caps={"cortex": 10, "executor": 10, "fleet": 10},
        locale="en-US",
    )
    contract = corpus.load_lumen_contract(_lumen_manifest(tmp_path / "lumen.json"))

    records = corpus._transform_massive(source, artifact, "test.1", contract)

    assert {record["metadata"]["agent"] for record in records} == {"cortex", "executor", "fleet"}
    assert all("reserved validation question" not in json.dumps(record) for record in records)
    assert all("play_music" not in json.dumps(record) for record in records)
    for record in records:
        agent = record["metadata"]["agent"]
        if agent == "cortex":
            target = json.loads(record["messages"][1]["content"])
            assert corpus._validate_cortex_target(target, contract)
            assert set(contract.cortex_required_fields) <= set(target)
            if target["selectedToolID"] is None:
                assert target["intent"] == "chat"
                assert target["nextModel"] == "mouth"
                assert "actionStep" not in target
            else:
                assert target["selectedToolID"] in contract.intents[target["intent"]]["allowedToolIDs"]
                assert target["actionStep"] == {
                    "type": "tool_call",
                    "toolID": target["selectedToolID"],
                    "mustPersistBeforeFinal": True,
                }
        if agent == "executor":
            call = json.loads(record["messages"][1]["content"])
            assert corpus._validate_tool_call(call, contract.tools)
            assert call["tool"] in contract.tools


def test_massive_cortex_target_fails_closed_for_non_manifest_tool_mapping(tmp_path: Path) -> None:
    contract = corpus.load_lumen_contract(_lumen_manifest(tmp_path / "lumen.json"))

    assert corpus._massive_cortex_target("weather_query", "weather", contract) == {
        "intent": "weather",
        "selectedToolID": "weather",
        "requiresApproval": False,
        "nextModel": "executor",
        "reasoningSummary": "The manifest allows weather for weather; persist the tool action before finalization.",
        "actionStep": {"type": "tool_call", "toolID": "weather", "mustPersistBeforeFinal": True},
    }
    assert corpus._massive_cortex_target("alarm_set", "alarm", contract) is None


def test_massive_normalizes_wake_words_and_rejects_cross_domain_list_labels() -> None:
    assert corpus._clean_massive_utterance("Olly, find coffee near me") == "find coffee near me"
    playlist = {
        "intent": "lists_query",
        "utt": "what is on my rap playlist",
        "annot_utt": "what is on my [list_name : rap] playlist",
    }
    assert corpus._massive_lumen_intent(playlist, playlist["utt"]) is None
    assert corpus._massive_lumen_tool_call(playlist) is None


def test_oasst_transform_creates_grounded_noncopy_final_without_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "id": "oasst2-test",
        "datasetID": "OpenAssistant/oasst2",
        "revision": "b" * 40,
        "sourceURL": "https://example.test/oasst2",
        "artifactSHA256": "c" * 64,
        "artifactPath": "data/train.parquet",
        "partitionKind": "ml_split",
        "sourcePartition": "train",
        "license": "Apache-2.0",
        "licenseURL": "https://www.apache.org/licenses/LICENSE-2.0",
        "attribution": "OpenAssistant contributors.",
        "adapterCaps": {"mouth": 2},
        "languageCaps": {"en": 1},
    }
    messages = [
        {
            "message_id": "11111111-1111-4111-8111-111111111111",
            "parent_id": None,
            "message_tree_id": "11111111-1111-4111-8111-111111111111",
            "user_id": "99999999-9999-4999-8999-999999999999",
            "role": "prompter",
            "lang": "en",
            "text": "How can I organize a focused work session?",
            "review_count": 3,
            "review_result": True,
            "tree_state": "ready_for_export",
            "deleted": False,
            "synthetic": False,
            "labels": _oasst_labels(),
            "detoxify": {
                "toxicity": 0.01,
                "severe_toxicity": 0.01,
                "obscene": 0.01,
                "identity_attack": 0.01,
                "insult": 0.01,
                "threat": 0.01,
                "sexual_explicit": 0.01,
            },
        },
        {
            "message_id": "22222222-2222-4222-8222-222222222222",
            "parent_id": "11111111-1111-4111-8111-111111111111",
            "message_tree_id": "11111111-1111-4111-8111-111111111111",
            "user_id": "88888888-8888-4888-8888-888888888888",
            "role": "assistant",
            "lang": "en",
            "text": (
                "Choose one outcome before the session begins. Remove distractions from the workspace. "
                "Work in a bounded interval and review the result afterward."
            ),
            "review_count": 3,
            "review_result": True,
            "tree_state": "ready_for_export",
            "deleted": False,
            "synthetic": False,
            "rank": 0,
            "labels": _oasst_labels(),
            "detoxify": {
                "toxicity": 0.01,
                "severe_toxicity": 0.01,
                "obscene": 0.01,
                "identity_attack": 0.01,
                "insult": 0.01,
                "threat": 0.01,
                "sexual_explicit": 0.01,
            },
        },
    ]
    monkeypatch.setattr(corpus, "_read_parquet_rows", lambda _: iter(messages))

    records = corpus._transform_oasst2(source, tmp_path / "unused.parquet", "test.1")

    assert len(records) == 1
    prompt = records[0]["messages"][0]["content"]
    target = records[0]["messages"][1]["content"]
    assert "Source observations" in prompt
    assert "trusted" not in prompt.casefold()
    assert target == (
        "- Choose one outcome before the session begins.\n"
        "- Remove distractions from the workspace."
    )
    assert target != messages[1]["text"]
    assert len(target) < len(messages[1]["text"])
    assert records[0]["taskType"] == "public_grounded_response_finalization"
    assert records[0]["metadata"]["publicCorpus"]["quality"] == {
        "expertAnnotated": False,
        "humanReviewed": False,
        "piiRegexScreened": True,
        "synthetic": False,
        "upstreamHumanReviewed": True,
        "upstreamRank": 0,
        "upstreamReviewCountMinimum": 3,
    }
    serialized = json.dumps(records[0])
    for raw_field in ("message_id", "parent_id", "message_tree_id", "user_id"):
        assert raw_field not in serialized
    for raw_id in (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        "88888888-8888-4888-8888-888888888888",
        "99999999-9999-4999-8999-999999999999",
    ):
        assert raw_id not in serialized
    provenance = records[0]["metadata"]["publicCorpus"]
    assert provenance["sourceGroupID"] == corpus._opaque_group_hash(
        "oasst2-test", "11111111-1111-4111-8111-111111111111"
    )
    assert provenance["sourcePath"] == "data/train.parquet"
    assert provenance["sourceContentSHA256"] != provenance["transformedContentSHA256"]


def test_oasst_quality_requires_three_reviews_and_low_target_detox() -> None:
    message = {
        "role": "assistant",
        "text": "Choose one outcome, remove distractions, and review the work afterward.",
        "review_count": 3,
        "review_result": True,
        "tree_state": "ready_for_export",
        "deleted": False,
        "synthetic": False,
        "rank": 0,
        "labels": _oasst_labels(),
        "detoxify": {
            "toxicity": 0.01,
            "severe_toxicity": 0.01,
            "obscene": 0.01,
            "identity_attack": 0.01,
            "insult": 0.01,
            "threat": 0.01,
            "sexual_explicit": 0.01,
        },
    }
    assert corpus._oasst_message_is_eligible(message, assistant=True)
    message["review_count"] = 2
    assert not corpus._oasst_message_is_eligible(message, assistant=True)
    message["review_count"] = 3
    message["detoxify"]["toxicity"] = 0.051
    assert not corpus._oasst_message_is_eligible(message, assistant=True)


@pytest.mark.parametrize(
    "text",
    [
        "As OpenAssistant, I am a language model created to answer questions.",
        "A doctor can diagnose these symptoms and recommend a medication dosage.",
        "Here is current investment advice and the best stock to buy right now.",
        "Use this password exploit to bypass authentication on the network.",
        "Write song lyrics in the style of Taylor Swift.",
    ],
)
def test_oasst_rejects_identity_high_stakes_security_and_style_imitation(text: str) -> None:
    message = {
        "role": "assistant",
        "text": text,
        "review_count": 3,
        "review_result": True,
        "tree_state": "ready_for_export",
        "deleted": False,
        "synthetic": False,
        "rank": 0,
        "labels": _oasst_labels(),
        "detoxify": {
            "toxicity": 0.01,
            "severe_toxicity": 0.01,
            "obscene": 0.01,
            "identity_attack": 0.01,
            "insult": 0.01,
            "threat": 0.01,
            "sexual_explicit": 0.01,
        },
    }

    assert not corpus._oasst_message_is_eligible(message, assistant=True)


def test_oasst_grounding_requires_three_safe_claims_and_never_copies_source() -> None:
    assert corpus._grounded_oasst_example(
        "How should I arrange a focused work session?",
        "Choose one outcome first. Remove distractions from the desk. Review the result after the session.",
        "en",
    ) is not None
    assert corpus._grounded_oasst_example(
        "How should I arrange a focused work session?",
        "Choose one outcome first. Remove distractions from the desk.",
        "en",
    ) is None


def test_oasst_grounding_rejects_questions_fragments_and_non_answers() -> None:
    assert corpus._grounded_oasst_example(
        "Quel est le niveau d'automatisation de l'automobile dans les années 2020 ?",
        (
            "Le degré d'automatisation de la conduite d'une voiture ? "
            "Le degré d'automatisation des voitures elles-mêmes ? "
            "Le degré d'automatisation de la fabrication des voitures ?"
        ),
        "fr",
    ) is None
    assert corpus._grounded_oasst_example(
        "Explain the requested conversion.",
        (
            "I'm sorry, but do you mean the Laplace transform? "
            "Please clarify which conversion you want. "
            "I need more information before answering."
        ),
        "en",
    ) is None


def test_oasst_grounding_skips_questions_before_selecting_observations() -> None:
    result = corpus._grounded_oasst_example(
        "How should I prepare a focused work session?",
        (
            "Do you mean a solo work session? "
            "Choose one concrete outcome before starting. "
            "Remove unrelated materials from the desk. "
            "Review the completed work after the session."
        ),
        "en",
    )

    assert result is not None
    prompt, final_answer = result
    assert "Do you mean" not in prompt
    assert "?" not in final_answer


def test_coedit_routes_style_edits_to_mimicry_and_gec_only_to_rem(tmp_path: Path) -> None:
    artifact = tmp_path / "coedit.jsonl"
    rows = [
        {
            "_id": "1",
            "task": "clarity",
            "src": "Make this text clearer: The team completed the final report in a timely manner.",
            "tgt": "The team completed the final report promptly.",
        },
        {
            "_id": "2",
            "task": "gec",
            "src": "Fix grammar in the sentence: The reports was delivered by the analyst.",
            "tgt": "The reports were delivered by the analyst.",
        },
        {
            "_id": "3",
            "task": "paraphrase",
            "src": "Paraphrase: A completely unrelated source sentence appears here.",
            "tgt": "This excluded task must never enter the snapshot.",
        },
    ]
    artifact.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source = _source(
        "coedit-test",
        artifact,
        artifact_format="jsonl",
        adapter_caps={"mimicry": 10, "rem": 10},
        artifactPath="train.jsonl",
    )

    records = corpus._transform_coedit(source, artifact, "test.1")

    assert Counter(record["metadata"]["agent"] for record in records) == Counter({"mimicry": 1, "rem": 1})
    assert all("excluded task" not in json.dumps(record) for record in records)
    rem = next(record for record in records if record["metadata"]["agent"] == "rem")
    assert json.loads(rem["messages"][1]["content"])["diagnosis"] == "grammar_or_usage_error"
    rem = next(record for record in records if record["metadata"]["agent"] == "rem")
    rem_target = json.loads(rem["messages"][1]["content"])
    assert rem_target == {
        "diagnosis": "grammar_or_usage_error",
        "preserveMeaning": True,
        "repair": "The reports were delivered by the analyst.",
    }


def test_coedit_quality_preserves_numbers_urls_and_negation_and_rejects_code() -> None:
    assert corpus._meaning_preserving_edit(
        "The team completed the final report in a timely manner.",
        "The team completed the final report promptly.",
    )
    assert not corpus._meaning_preserving_edit(
        "The 2025 report is not available at https://example.test/a.",
        "The 2026 report is available at https://example.test/b.",
    )
    assert not corpus._meaning_preserving_edit(
        "The code defines function parse(value) for the current payload.",
        "The code defines function parse(item) for the current payload.",
    )


def test_coedit_quality_rejects_artifacts_and_preserves_semantic_anchors() -> None:
    assert not corpus._meaning_preserving_edit(
        "The first Atlas test covered three kilometers in 12 minutes.",
        "The Atlas test covered two kilometers in 12 minutes.",
    )
    assert not corpus._meaning_preserving_edit(
        "The Atlas Research Group completed the report on Monday.",
        "The research group completed the report on Monday.",
    )
    assert not corpus._meaning_preserving_edit(
        "Garlic knots are bread with garlic from New York City.",
        "Garlic knots are bread with garlic. <SEP> They are from New York City.",
    )
    assert not corpus._meaning_preserving_edit(
        "Kirby wrote the referenced book on early English kings.",
        "Jump up ^ Kirby, Early English Kings, p. 63.",
    )
    assert not corpus._meaning_preserving_edit(
        "The team completed a detailed report for the board.",
        "The team completed a detailed report for the",
    )


def test_coedit_quality_preserves_epistemic_and_factual_status_anchors() -> None:
    assert not corpus._meaning_preserving_edit(
        "She supports the false belief that the claim is accurate.",
        "She supports the belief that the claim is accurate.",
    )
    assert not corpus._meaning_preserving_edit(
        "The source states that the procedure avoids this effect.",
        "The source states that the procedure may avoid this effect.",
    )
    assert not corpus._meaning_preserving_edit(
        "The witness described the event in the report.",
        "The witness confirmed the event in the report.",
    )
    assert corpus._meaning_preserving_edit(
        "The video asserts that the group changed the final schedule.",
        "The video indicates that the group changed the final schedule.",
    )


def test_coedit_quality_preserves_normative_modality() -> None:
    assert not corpus._meaning_preserving_edit(
        "The constitution takes precedence over the regional policy.",
        "The constitution should take precedence over the regional policy.",
    )
    assert not corpus._meaning_preserving_edit(
        "The policy permits staff to use the shared workspace.",
        "The policy prohibits staff from using the shared workspace.",
    )
    assert corpus._meaning_preserving_edit(
        "The policy requires the team to retain the signed record.",
        "The policy mandates that the team retain the signed record.",
    )


@pytest.mark.parametrize(
    "term",
    [
        "vaccines",
        "vaccination",
        "autism",
        "asthma",
        "ADHD",
        "autoimmune disorders",
        "diseases",
        "drugs",
        "diagnoses",
        "infection",
        "tumours",
    ],
)
def test_coedit_rejects_broad_medical_and_high_stakes_terms(term: str) -> None:
    assert not corpus._meaning_preserving_edit(
        f"The article describes disputed claims about {term} in detail.",
        f"The article explains the disputed claims about {term} in detail.",
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            "The doctor described a medical treatment for the patient's symptoms.",
            "The physician described treatment for the patient's symptoms.",
        ),
        (
            "This is the latest investment recommendation for the current market.",
            "This is the newest investment recommendation for today's market.",
        ),
        (
            "Rewrite the password exploit instructions in a clearer form.",
            "Rewrite the authentication exploit instructions more clearly.",
        ),
        (
            "Imitate the writing style of a living artist in this paragraph.",
            "Copy the living artist's writing style in this paragraph.",
        ),
    ],
)
def test_coedit_rejects_high_stakes_current_security_and_imitation(
    source: str, target: str
) -> None:
    assert not corpus._meaning_preserving_edit(source, target)


def test_json_schema_tests_are_rem_fail_closed_repairs_not_executor_targets(tmp_path: Path) -> None:
    artifact = tmp_path / "schema.tar.gz"
    payload = json.dumps(
        [
            {
                "description": "string values only",
                "schema": {"type": "string"},
                "tests": [
                    {"description": "a string is valid", "data": "ok", "valid": True},
                    {"description": "an integer is invalid", "data": 7, "valid": False},
                ],
            }
        ]
    ).encode("utf-8")
    info = tarfile.TarInfo("suite/tests/draft2020-12/type.json")
    info.size = len(payload)
    info.mtime = 0
    with tarfile.open(artifact, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    source = _source(
        "json-schema-test-suite-test",
        artifact,
        artifact_format="tar.gz-json",
        adapter_caps={"rem": 10},
        artifactMemberPrefix="tests/draft2020-12/",
        selectedFiles=["type.json"],
        partitionKind="reference_corpus",
        sourcePartition="draft2020-12",
    )

    records = corpus._transform_json_schema_tests(source, artifact, "test.1")

    assert len(records) == 1
    assert records[0]["metadata"]["agent"] == "rem"
    assert records[0]["metadata"]["publicCorpus"]["partitionKind"] == "reference_corpus"
    diagnosis = json.loads(records[0]["messages"][1]["content"])
    assert diagnosis["decision"] == "reject"
    assert diagnosis["repair"]["knownValidExample"] == "ok"


def test_snapshot_build_is_deterministic_and_loader_rejects_tampering(tmp_path: Path) -> None:
    artifact = _massive_tar(tmp_path / "massive.tar.gz")
    source = _source(
        "massive-test",
        artifact,
        artifact_format="tar.gz-jsonl",
        artifactMember="1.1/data/en-US.jsonl",
        adapter_caps={"cortex": 10, "executor": 10, "fleet": 10},
        locale="en-US",
    )
    source_manifest = _source_manifest(tmp_path / "sources.json", [source])
    lumen_manifest = _lumen_manifest(tmp_path / "lumen.json")
    first = tmp_path / "first"
    second = tmp_path / "second"

    result_one = corpus.build_public_adapter_corpus(
        first,
        cache_dir=tmp_path / "cache",
        lumen_manifest_path=lumen_manifest,
        source_manifest_path=source_manifest,
        offline=True,
        artifact_paths={"massive-test": artifact},
    )
    result_two = corpus.build_public_adapter_corpus(
        second,
        cache_dir=tmp_path / "cache",
        lumen_manifest_path=lumen_manifest,
        source_manifest_path=source_manifest,
        offline=True,
        artifact_paths={"massive-test": artifact},
    )

    assert result_one.records_sha256 == result_two.records_sha256
    assert (first / "records.jsonl").read_bytes() == (second / "records.jsonl").read_bytes()
    contract = corpus.load_lumen_contract(lumen_manifest)
    grouped = corpus.load_public_adapter_corpus(first, lumen_contract=contract)
    assert set(grouped) == {"cortex", "executor", "fleet"}
    provenance = grouped["executor"][0]["metadata"]["publicCorpus"]
    assert json.loads((first / "manifest.json").read_text(encoding="utf-8"))["lumenContractSHA256"] == contract.sha256
    assert provenance["partitionKind"] == "ml_split"
    assert provenance["sourcePartition"] == "train"
    assert provenance["sourcePath"] == "1.1/data/en-US.jsonl"
    assert provenance["sourceContentSHA256"] != provenance["transformedContentSHA256"]

    changed_payload = json.loads(lumen_manifest.read_text(encoding="utf-8"))
    changed_payload["tools"] = [tool for tool in changed_payload["tools"] if tool["id"] != "weather"]
    changed_manifest_path = tmp_path / "changed-lumen.json"
    changed_manifest_path.write_text(json.dumps(changed_payload), encoding="utf-8")
    changed_contract = corpus.load_lumen_contract(changed_manifest_path)
    with pytest.raises(corpus.PublicCorpusError, match="does not match the current manifest"):
        corpus.load_public_adapter_corpus(first, lumen_contract=changed_contract)

    snapshot_manifest_path = first / "manifest.json"
    original_snapshot_manifest = snapshot_manifest_path.read_bytes()
    snapshot_manifest = json.loads(original_snapshot_manifest)
    snapshot_manifest["lumenContractSHA256"] = changed_contract.sha256
    snapshot_manifest_path.write_text(json.dumps(snapshot_manifest), encoding="utf-8")
    with pytest.raises(corpus.PublicCorpusError, match="manifest-valid Lumen tool envelope"):
        corpus.load_public_adapter_corpus(first, lumen_contract=changed_contract)
    snapshot_manifest_path.write_bytes(original_snapshot_manifest)

    with (first / "records.jsonl").open("ab") as handle:
        handle.write(b" \n")
    with pytest.raises(corpus.PublicCorpusError, match="hash mismatch"):
        corpus.load_public_adapter_corpus(first, lumen_contract=contract)


def test_pii_screen_rejects_direct_identifiers() -> None:
    assert corpus._contains_pii("Contact jane.person@example.com for the private details")
    assert corpus._contains_pii("Call +1 (514) 555-0123 immediately")
    assert corpus._contains_pii("The account number is 4111 1111 1111 1111")
    assert not corpus._contains_pii("Find a coffee shop near me tomorrow")
