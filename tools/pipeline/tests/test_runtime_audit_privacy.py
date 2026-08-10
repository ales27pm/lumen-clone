import importlib.util
import json
from pathlib import Path
import subprocess


def load_checker():
    script = Path(__file__).resolve().parents[2] / "check_runtime_audit_privacy.py"
    spec = importlib.util.spec_from_file_location("check_runtime_audit_privacy", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def initialize_repository(repository: Path) -> None:
    repository.mkdir()
    run_git(repository, "init", "--initial-branch=main")
    run_git(repository, "config", "user.name", "Privacy Guard Test")
    run_git(repository, "config", "user.email", "privacy-guard@example.invalid")


def commit_all(repository: Path, message: str) -> None:
    run_git(repository, "add", "--all")
    run_git(repository, "commit", "-m", message)


def test_rejects_private_export_reachable_only_from_history(tmp_path):
    checker = load_checker()
    repository = tmp_path / "repository"
    initialize_repository(repository)
    export = repository / "exports" / "lumen-live-e2e-report-private.json"
    export.parent.mkdir()
    export.write_text('{"prompt":"synthetic private value"}', encoding="utf-8")
    commit_all(repository, "add private export")
    export.unlink()
    commit_all(repository, "delete private export")

    failures = checker.check_git_history_privacy(repository)

    assert failures == [
        "Git history revision 'HEAD' exposes forbidden private export path: "
        "exports/lumen-live-e2e-report-private.json"
    ]


def test_accepts_history_without_private_export_paths(tmp_path):
    checker = load_checker()
    repository = tmp_path / "repository"
    initialize_repository(repository)
    readme = repository / "README.md"
    readme.write_text("safe\n", encoding="utf-8")
    commit_all(repository, "add safe source")

    assert checker.check_git_history_privacy(repository) == []


def test_history_scan_fails_closed_for_unknown_revision(tmp_path):
    checker = load_checker()
    repository = tmp_path / "repository"
    initialize_repository(repository)
    readme = repository / "README.md"
    readme.write_text("safe\n", encoding="utf-8")
    commit_all(repository, "add safe source")

    failures = checker.check_git_history_privacy(
        repository,
        revision="refs/heads/missing",
    )

    assert len(failures) == 1
    assert "could not inspect Git history revision" in failures[0]


def test_rejects_legacy_live_e2e_filename(tmp_path):
    checker = load_checker()
    legacy = tmp_path / "lumen-live-e2e-report-legacy.json"
    legacy.write_text("{}", encoding="utf-8")

    failures = checker.check_runtime_audits(tmp_path)

    assert any("legacy raw-content" in failure for failure in failures)


def test_rejects_legacy_trace_and_grounding_package_filenames(tmp_path):
    checker = load_checker()
    (tmp_path / "agent-behavior-traces.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "lumen-testflight-agent-grounding-old.json").write_text("{}", encoding="utf-8")

    failures = checker.check_runtime_audits(tmp_path)

    assert len([failure for failure in failures if "legacy raw-content" in failure]) == 2


def test_accepts_redacted_live_e2e_report(tmp_path):
    checker = load_checker()
    report = tmp_path / "lumen-live-e2e-report-redacted-v1-test.json"
    placeholder = "[redacted sha256=" + ("a" * 64) + " chars=12]"
    report.write_text(
        json.dumps(
            {
                "payload": {
                    "results": [
                        {
                            "prompt": placeholder,
                            "finalText": placeholder,
                            "events": [{"message": placeholder}],
                            "metadata": {
                                "failureKind": "toolObservationLeak",
                                "metadata_" + ("b" * 16): placeholder,
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert checker.check_runtime_audits(tmp_path) == []


def test_accepts_interactive_model_tool_metadata_and_event_phase(tmp_path):
    checker = load_checker()
    report = tmp_path / "e2e-results-redacted-v1.jsonl"
    placeholder = "[redacted sha256=" + ("a" * 64) + " chars=12]"
    report.write_text(
        json.dumps(
            {
                "events": [{"phase": "tool-result", "message": placeholder}],
                "metadata": {
                    "attributableModelToolEvidence": "true",
                    "modelFinalMatchesNativeObservation": "true",
                    "modelFinalTraceCount": "1",
                    "nativeToolObservationStepCount": "1",
                    "nativeToolResultEvidenceCount": "1",
                    "primaryAgentJSONActionTraceCount": "1",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert checker.check_runtime_audits(tmp_path) == []


def test_interactive_metadata_allowlist_is_exact_and_preserves_canary_detection(tmp_path):
    checker = load_checker()
    safe_keys = (
        "attributableModelToolEvidence",
        "modelFinalMatchesNativeObservation",
        "modelFinalTraceCount",
        "nativeToolObservationStepCount",
        "nativeToolResultEvidenceCount",
        "primaryAgentJSONActionTraceCount",
    )

    for index, safe_key in enumerate(safe_keys):
        case = tmp_path / str(index)
        case.mkdir()
        report = case / "e2e-results-redacted-v1.jsonl"
        report.write_text(
            json.dumps(
                {
                    "events": [{"phase": "tool-result"}],
                    "metadata": {f"{safe_key}Canary": "privacy-canary@example.invalid"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        failures = checker.check_runtime_audits(case)

        assert any("non-allowlisted metadata key" in failure for failure in failures)
        assert any("raw sensitive identifier" in failure for failure in failures)


def test_phase_allowlist_preserves_sensitive_value_canary_detection(tmp_path):
    checker = load_checker()
    report = tmp_path / "e2e-results-redacted-v1.jsonl"
    report.write_text(
        json.dumps({"events": [{"phase": "privacy-canary@example.invalid"}]}) + "\n",
        encoding="utf-8",
    )

    failures = checker.check_runtime_audits(tmp_path)

    assert not any("non-allowlisted JSON key" in failure for failure in failures)
    assert any("raw sensitive identifier" in failure for failure in failures)


def test_accepts_only_versioned_redacted_json_companion_names(tmp_path):
    checker = load_checker()
    companion_prefixes = (
        "accepted_training-redacted-v1",
        "agent-parse-failures-redacted-v1",
        "agent-parse-noise-redacted-v1",
        "quarantined_samples-redacted-v1",
        "regression_tests-redacted-v1",
    )
    for prefix in companion_prefixes:
        (tmp_path / f"{prefix}.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"{prefix}-2026-08-10T03-20-38Z.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )

    assert checker.check_runtime_audits(tmp_path) == []


def test_rejects_mutated_companion_name_canaries(tmp_path):
    checker = load_checker()
    mutated_names = (
        "accepted_training.jsonl",
        "accepted_training-redacted-v2.jsonl",
        "quarantined_samples-redacted-v1.txt",
        "regression_tests-redacted-v1private.jsonl",
    )
    for name in mutated_names:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    failures = checker.check_runtime_audits(tmp_path)

    assert len([failure for failure in failures if "unrecognized or unversioned" in failure]) == 4


def test_rejects_text_summary_and_legacy_agent_parse_names(tmp_path):
    checker = load_checker()
    (tmp_path / "latest-e2e-report-redacted-v1.txt").write_text(
        "synthetic private summary",
        encoding="utf-8",
    )
    (tmp_path / "agent-parse-failures.jsonl").write_text("{}\n", encoding="utf-8")

    failures = checker.check_runtime_audits(tmp_path)

    assert len([failure for failure in failures if "unrecognized or unversioned" in failure]) == 2


def test_accepts_hash_summaries_in_redacted_grounding_package(tmp_path):
    checker = load_checker()
    report = tmp_path / "lumen-testflight-agent-grounding-redacted-v1-test.json"
    report.write_text(
        json.dumps(
            {
                "runtimeManifestAudit": {
                    "failures": [
                        {
                            "expected": ["expected0_chars=12;sha256=" + ("a" * 16)],
                            "actual": "actual_chars=9;sha256=" + ("b" * 16),
                            "problem": "problem_chars=7;sha256=" + ("c" * 16),
                        }
                    ],
                    "recommendedDatasetRepairs": [
                        "repairRecommendation_chars=4;sha256=" + ("d" * 16)
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    assert checker.check_runtime_audits(tmp_path) == []


def test_rejects_raw_self_model_text_inside_redacted_trace(tmp_path):
    checker = load_checker()
    trace = tmp_path / "agent-behavior-traces-redacted-v1.jsonl"
    trace.write_text(
        json.dumps(
            {
                "promptPrefix": "prompt_chars=4;sha256=" + ("a" * 16),
                "rawOutputPrefix": "rawOutput_chars=4;sha256=" + ("b" * 16),
                "selfModel": {
                    "included": True,
                    "schemaVersion": "private injected value",
                    "sourceIDs": ["sourceIDs0_chars=8;sha256=" + ("c" * 16)],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    failures = checker.check_runtime_audits(tmp_path)

    assert any("non-redacted self-model text" in failure for failure in failures)


def test_rejects_raw_content_inside_redacted_filename(tmp_path):
    checker = load_checker()
    report = tmp_path / "e2e-results-redacted-v1.jsonl"
    report.write_text(
        json.dumps({"prompt": "private calendar title", "metadata": {"private": "raw value"}})
        + "\n",
        encoding="utf-8",
    )

    failures = checker.check_runtime_audits(tmp_path)

    assert any("non-redacted free-form text" in failure for failure in failures)
    assert any("non-redacted metadata" in failure for failure in failures)


def test_rejects_raw_free_form_text_nested_inside_dictionary(tmp_path):
    checker = load_checker()
    report = tmp_path / "lumen-live-e2e-report-redacted-v1-test.json"
    report.write_text(
        json.dumps(
            {
                "payload": {
                    "results": [
                        {
                            "prompt": {
                                "nestedPrivateLabel": "synthetic private calendar title"
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    failures = checker.check_runtime_audits(tmp_path)

    assert any("non-redacted free-form text" in failure for failure in failures)
    assert any("non-allowlisted JSON key" in failure for failure in failures)


def test_rejects_arbitrary_json_key_even_when_value_is_redacted(tmp_path):
    checker = load_checker()
    report = tmp_path / "lumen-live-e2e-report-redacted-v1-test.json"
    placeholder = "[redacted sha256=" + ("a" * 64) + " chars=12]"
    report.write_text(
        json.dumps(
            {
                "payload": {
                    "results": [
                        {"syntheticPrivateTitleKey": placeholder}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    failures = checker.check_runtime_audits(tmp_path)

    assert any("non-allowlisted JSON key" in failure for failure in failures)


def test_rejects_every_unversioned_evidence_filename(tmp_path):
    checker = load_checker()
    raw_parse_trace = tmp_path / "agent-parse-noise.jsonl"
    raw_parse_trace.write_text(
        json.dumps(
            {
                "systemPromptPrefix": "private system prompt",
                "userTurnPrefix": "private user prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    structural_but_unversioned = tmp_path / "persistent-runtime-diagnostics-export.json"
    structural_but_unversioned.write_text(json.dumps({"passed": True}), encoding="utf-8")

    failures = checker.check_runtime_audits(tmp_path)

    assert len([failure for failure in failures if "unrecognized or unversioned" in failure]) == 2


def test_rejects_sensitive_or_nonallowlisted_metadata_keys(tmp_path):
    checker = load_checker()
    report = tmp_path / "lumen-live-e2e-report-redacted-v1-test.json"
    placeholder = "[redacted sha256=" + ("a" * 64) + " chars=12]"
    report.write_text(
        json.dumps(
            {
                "payload": {
                    "results": [
                        {
                            "prompt": placeholder,
                            "metadata": {
                                "987-65-4321": placeholder,
                                "privatePayload": placeholder,
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    failures = checker.check_runtime_audits(tmp_path)

    assert any("sensitive identifier in a JSON key" in failure for failure in failures)
    assert any("non-allowlisted metadata key" in failure for failure in failures)


def test_rejects_raw_correlation_fields_and_caller_tokens(tmp_path):
    checker = load_checker()
    report = tmp_path / "e2e-results-redacted-v1.jsonl"
    report.write_text(
        json.dumps(
            {
                "e2eRunID": "11111111-1111-4111-8111-111111111111",
                "correlationToken": "caller-controlled-987-65-4321",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    failures = checker.check_runtime_audits(tmp_path)

    assert any("raw correlation identifier" in failure for failure in failures)
    assert any("non-opaque correlation token" in failure for failure in failures)
