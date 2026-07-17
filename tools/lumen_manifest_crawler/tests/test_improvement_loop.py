"""Tests for the manifest improvement loop runner."""

# pylint: disable=missing-function-docstring,line-too-long

import json
from pathlib import Path

import pytest

from lumen_manifest_crawler.improvement_loop import AgentImprovementLoopConfig, run_agent_improvement_loop

pytestmark = pytest.mark.slow


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_improvement_loop_writes_state_gaps_prompts_and_testflight_artifacts(tmp_path: Path):
    output = tmp_path / "agent_manifest"
    loop_output = tmp_path / "loop"

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=output,
            loop_output=loop_output,
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            test_command=("python", "--version"),
            generate_agent_fine_tuning=True,
            generate_system_prompts=True,
            testflight_build_label="1.0.0-build-99",
            testflight_scenario_limit=12,
        )
    )

    assert (output / "AgentBehaviorManifest.json").exists()
    assert (output / "dataset" / "train_sft.jsonl").exists()
    assert (output / "fine_tuning" / "adapter_runtime_manifest.json").exists()
    assert (output / "fine_tuning" / "cortex" / "train_sft.jsonl").exists()
    assert (output / "fine_tuning" / "cortex" / "adapter_export_plan.json").exists()
    assert (output / "fine_tuning" / "cortex" / "unsloth_config.json").exists()
    assert (loop_output / "loop_state.json").exists()
    assert (loop_output / "loop_gaps.json").exists()
    assert (loop_output / "next_action_prompts.jsonl").exists()
    assert (loop_output / "testflight_scenarios.jsonl").exists()
    assert (loop_output / "TESTFLIGHT_RUNBOOK.md").exists()
    assert (loop_output / "LOOP_REPORT.md").exists()
    assert (loop_output / "gap_triage.json").exists()
    assert (loop_output / "GAP_TRIAGE.md").exists()

    runtime_manifest = json.loads((output / "fine_tuning" / "adapter_runtime_manifest.json").read_text(encoding="utf-8"))
    cortex_plan = json.loads((output / "fine_tuning" / "cortex" / "adapter_export_plan.json").read_text(encoding="utf-8"))
    cortex_config = json.loads((output / "fine_tuning" / "cortex" / "unsloth_config.json").read_text(encoding="utf-8"))

    assert runtime_manifest["mode"] == "adapter_first"
    assert runtime_manifest["runtimeStrategy"]["loadBaseModelOnce"] is True
    assert runtime_manifest["runtimeStrategy"]["selectAdapterByAgentSlot"] is True
    assert runtime_manifest["runtimeStrategy"]["mergeAdaptersByDefault"] is False
    assert runtime_manifest["releaseBakePolicy"]["enabledByDefault"] is False
    assert cortex_plan["mode"] == "adapter_first"
    assert cortex_plan["exportPolicy"]["defaultArtifact"] == "adapter"
    assert cortex_plan["exportPolicy"]["mergeAdaptersByDefault"] is False
    assert cortex_config["artifactMode"] == "adapter_first"
    assert cortex_config["defaultExportArtifact"] == "lora_adapter"
    assert cortex_config["adapterExport"]["saveAdapterByDefault"] is True
    assert cortex_config["adapterExport"]["mergeAdaptersByDefault"] is False
    assert cortex_config["mergeExport"]["enabledByDefault"] is False

    assert result.state["schemaVersion"] == "1.1.0"
    assert result.state["manifest"]["toolCount"] >= 0
    assert result.state["dataset"]["recordCount"] > 0
    assert result.state["dataset"]["embedding"]["pairCount"] > 0
    assert result.state["dataset"]["reranker"]["pairCount"] > 0
    assert result.state["dataset"]["reranker"]["hardNegativePairCount"] > 0
    assert result.state["dataset"]["reranker"]["enabledByDefault"] is False
    assert result.state["dataset"]["agentFineTuning"]["cortex"]["trainSFT"] > 0
    assert result.state["testFlight"]["status"] == "awaiting-testflight-runtime-audit"
    assert result.state["testFlight"]["buildLabel"] == "1.0.0-build-99"
    assert len(result.testflight_scenarios) <= 12
    assert any(scenario.get("sourceFamily") == "trace_export_coverage" for scenario in result.testflight_scenarios)
    assert any(scenario.get("sourceFamily") == "trace_integrity" for scenario in result.testflight_scenarios)
    assert result.next_prompts
    assert result.state["triage"]["totalGaps"] == len(result.gaps)


def test_fine_tuning_only_loop_preserves_complete_fleet_evaluation_corpus(tmp_path: Path):
    output = tmp_path / "agent_manifest"

    run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=output,
            loop_output=tmp_path / "loop",
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=True,
            generate_system_prompts=False,
        )
    )

    fleet_eval_path = output / "fine_tuning" / "fleet" / "eval.jsonl"
    records = [
        json.loads(line)
        for line in fleet_eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    orchestration_scenarios = {
        record["metadata"]["scenarioID"]
        for record in records
        if record.get("metadata", {}).get("evalType")
        == "fleet_orchestration_event_graph_eval"
    }

    assert len(records) == 15
    assert orchestration_scenarios == {
        "no-delegation",
        "sequential-dependencies",
        "parallel-dependencies",
        "context-handoff",
        "duplicate-suppression",
        "aggregation-owner",
        "approval-boundary",
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }


def test_improvement_loop_can_require_testflight_runtime_audit(tmp_path: Path):
    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            require_testflight_runtime_audit=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    assert any(gap["category"] == "testflight_runtime_pending" for gap in result.gaps)
    assert any(gap["severity"] == "error" for gap in result.gaps if gap["category"] == "testflight_runtime_pending")
    assert result.passed is False


def test_improvement_loop_flags_stale_testflight_runtime_audit_build(tmp_path: Path):
    report = tmp_path / "lumen-live-e2e-report.json"
    report.write_text(
        json.dumps({
            "schemaVersion": "1.0.0",
            "generatedAt": "2026-06-29T00:00:00Z",
            "app": {
                "name": "Lumen",
                "bundleIdentifier": "com.27pm.lumenclone",
                "shortVersion": "1.0.0",
                "buildNumber": "20260628223733",
            },
            "exportPolicy": {
                "format": "live-e2e-test-report-json",
                "sourceLayer": "e2eTestReport",
                "ownsLiveE2EScenarios": True,
                "includesDeterministicStaticScenarios": False,
            },
            "payload": {
                "id": "66666666-6666-6666-6666-666666666666",
                "startedAt": "2026-06-29T00:00:00Z",
                "finishedAt": "2026-06-29T00:00:10Z",
                "passed": 0,
                "failed": 1,
                "results": [
                    {
                        "id": "77777777-7777-7777-7777-777777777777",
                        "scenarioID": "self-model-evidence",
                        "kind": "standard",
                        "title": "Self-model evidence",
                        "prompt": "What evidence supports your claim?",
                        "expectedIntent": "rag",
                        "actualIntent": "rag",
                        "requiresAgentRun": False,
                        "passed": False,
                        "failures": ["stale build response failure"],
                        "finalText": "Unsupported stale build response.",
                        "events": [],
                    }
                ],
            },
        }),
        encoding="utf-8",
    )
    repairs = tmp_path / "agent_manifest" / "dataset" / "runtime_audit_repairs.jsonl"
    repairs.parent.mkdir(parents=True)
    repairs.write_text('{"stale":true}\n', encoding="utf-8")

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(report,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
            testflight_build_label="20260629054657",
        )
    )

    mismatch_gaps = [gap for gap in result.gaps if gap["category"] == "testflight_runtime_build_mismatch"]
    assert len(mismatch_gaps) == 1
    assert mismatch_gaps[0]["severity"] == "error"
    assert mismatch_gaps[0]["evidence"]["expectedBuildLabel"] == "20260629054657"
    assert mismatch_gaps[0]["evidence"]["mismatchedReportCount"] == 1
    assert mismatch_gaps[0]["evidence"]["mismatchedReports"][0]["appBuildNumber"] == "20260628223733"
    assert not any(gap["category"] == "e2e_runtime_failure" for gap in result.gaps)
    assert not any(gap["category"] == "testflight_runtime_pending" for gap in result.gaps)
    assert not any(gap["title"] == "Empty dataset family: runtime_audit_repairs" for gap in result.gaps)
    assert result.state["runtime"]["reportCount"] == 0
    assert result.state["runtime"]["totalReportCount"] == 1
    assert result.state["runtime"]["staleReportCount"] == 1
    assert result.state["runtime"]["failureCount"] == 0
    assert result.state["runtime"]["rawFailureCount"] == 0
    assert result.state["testFlight"]["status"] == "runtime-audit-stale"
    assert result.state["testFlight"]["runtimeAuditProvided"] is True
    assert result.state["testFlight"]["currentRuntimeAuditProvided"] is False
    assert result.state["testFlight"]["staleRuntimeAuditReportCount"] == 1
    assert result.state["triage"]["rawRuntimeFailureCount"] == 0
    assert result.state["triage"]["freshRuntimeFailureCount"] == 0
    assert result.state["triage"]["rootCauseCounts"]["stale_audit_evidence"] == 1
    assert result.state["triage"]["groupCounts"]["Runtime audit export does not prove the current TestFlight build"] == 1
    assert result.state["triage"]["totalGaps"] == 1
    assert not repairs.exists()
    assert result.passed is False


def test_improvement_loop_ignores_stale_testflight_reports_when_current_build_is_present(tmp_path: Path):
    for name, build_number in {
        "stale-e2e.json": "20260628223733",
        "current-e2e.json": "20260629054657",
    }.items():
        (tmp_path / name).write_text(
            json.dumps({
                "schemaVersion": "1.0.0",
                "generatedAt": "2026-06-29T00:00:00Z",
                "app": {
                    "name": "Lumen",
                    "bundleIdentifier": "com.27pm.lumenclone",
                    "shortVersion": "1.0.0",
                    "buildNumber": build_number,
                },
                "exportPolicy": {
                    "format": "live-e2e-test-report-json",
                    "sourceLayer": "e2eTestReport",
                    "ownsLiveE2EScenarios": True,
                    "includesDeterministicStaticScenarios": False,
                },
                "payload": {
                    "id": build_number,
                    "passed": 1,
                    "failed": 0,
                    "results": [],
                },
            }),
            encoding="utf-8",
        )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(tmp_path,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
            testflight_build_label="20260629054657",
        )
    )

    assert not any(gap["category"] == "testflight_runtime_build_mismatch" for gap in result.gaps)
    assert result.state["runtime"]["reportCount"] == 1
    assert result.state["runtime"]["totalReportCount"] == 2
    assert result.state["runtime"]["staleReportCount"] == 1
    assert result.state["testFlight"]["status"] == "runtime-audit-ingested"


def test_improvement_loop_records_failed_command_as_critical_gap(tmp_path: Path):
    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            deterministic=True,
            strict=False,
            dry_run_commands=False,
            test_command=("python", "-c", "import sys; sys.exit(7)"),
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    assert any(gap["category"] == "command_failure" for gap in result.gaps)
    assert any(gap["severity"] == "critical" for gap in result.gaps)
    assert result.passed is False


def test_improvement_loop_reclassifies_skipped_live_model_evidence(tmp_path: Path):
    report = tmp_path / "live-e2e.json"
    report.write_text(
        json.dumps({
            "kind": "lumen_e2e_test_report",
            "passed": False,
            "failed": 1,
            "scenarios": [
                {
                    "name": "Live outlook.folders.list direct",
                    "passed": False,
                    "prompt": "List my Outlook folders.",
                    "intent": "outlook",
                    "expectedIntent": "outlook",
                    "requiresAgentRun": True,
                    "failures": ["Live E2E scenario did not run: no chat model loaded"],
                    "final": "Outlook tool output could not be validated.",
                    "events": [
                        {"phase": "models", "message": "no chat model loaded"},
                        {"phase": "model-evidence", "message": "AgentService model path was not entered; reason=model not loaded; scenarioID=Live outlook.folders.list direct,e2eRunID=11111111-1111-4111-8111-111111111111"},
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(tmp_path,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    runtime_gaps = [gap for gap in result.gaps if gap["title"] == "e2e_architecture_finalizer_failure"]
    assert runtime_gaps == []
    assert result.state["runtime"]["rawFailureCount"] == 1
    assert result.state["runtime"]["failureCount"] == 0
    assert result.state["runtime"]["skippedLiveModelGenerationCount"] == 1
    assert result.passed is True


def test_improvement_loop_keeps_deterministic_live_evidence_non_blocking(tmp_path: Path):
    report = tmp_path / "live-e2e.json"
    report.write_text(
        json.dumps({
            "kind": "lumen_e2e_test_report",
            "passed": 1,
            "failed": 0,
            "scenarios": [
                {
                    "name": "Weather here must not create events",
                    "kind": "regression",
                    "passed": True,
                    "prompt": "What is the weather here?",
                    "intent": "weather",
                    "expectedIntent": "weather",
                    "requiresAgentRun": True,
                    "failures": [],
                    "final": "Weather summary.",
                    "events": [
                        {
                            "phase": "model-evidence",
                            "message": "runtime=deterministic-compatibility, kind=policy-first-deterministic, stage=compatibility-final",
                        }
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(tmp_path,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    runtime_gaps = [gap for gap in result.gaps if gap["category"] == "deterministic_compatibility_not_live_evidence"]
    assert runtime_gaps == []
    assert result.state["runtime"]["rawFailureCount"] == 1
    assert result.state["runtime"]["failureCount"] == 1
    assert result.state["runtime"]["skippedLiveModelGenerationCount"] == 0
    assert "deterministic_compatibility_not_live_evidence" not in result.state["triage"]["rootCauseCounts"]
    assert result.passed is True


def test_improvement_loop_uses_persistent_diagnostic_remediation_action(tmp_path: Path):
    report = tmp_path / "persistent-runtime-diagnostics-export.json"
    report.write_text(
        json.dumps({
            "exportedAt": "2026-06-24T00:00:00Z",
            "appVersion": "1.0.0",
            "ndjson": "",
            "state": {
                "records": [
                    {
                        "id": "manual-skip",
                        "scenario": "liveAgentStream",
                        "status": "skipped",
                        "remediationProposals": [
                            {
                                "id": "manual-scenario-foreground",
                                "title": "Run the diagnostic from the foreground control",
                                "rationale": "This scenario requires explicit user action and should not run unattended.",
                                "action": "Open Runtime Diagnostics and start the matching manual probe from the foreground UI.",
                                "severity": "info",
                            }
                        ],
                    }
                ]
            },
        }),
        encoding="utf-8",
    )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(tmp_path,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    runtime_gaps = [gap for gap in result.gaps if gap["title"] == "persistent_diagnostics_scenario_not_passed"]
    assert len(runtime_gaps) == 1
    assert runtime_gaps[0]["severity"] == "warning"
    assert runtime_gaps[0]["category"] == "persistent_diagnostics_remediation"
    assert runtime_gaps[0]["recommendedAction"] == "Open Runtime Diagnostics and start the matching manual probe from the foreground UI."
    assert runtime_gaps[0]["evidence"]["remediationProposals"][0]["id"] == "manual-scenario-foreground"
    assert result.state["runtime"]["failureCount"] == 1
    assert result.state["runtime"]["allFailureTypes"] == {"persistent_diagnostics_scenario_not_passed": 1}


def test_improvement_loop_surfaces_agent_json_completed_without_text_from_sidecars(tmp_path: Path):
    report = tmp_path / "latest-e2e-report.json"
    report.write_text(
        json.dumps({
            "kind": "lumen_e2e_test_report",
            "passed": False,
            "failed": 1,
            "scenarios": [
                {
                    "name": "Training eval: pure chat quality",
                    "passed": False,
                    "prompt": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
                    "intent": "chat",
                    "expectedIntent": "chat",
                    "requiresAgentRun": True,
                    "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                    "final": "Precision is exactness; recall is coverage.",
                    "events": [{"phase": "model-evidence", "message": "missing fresh AgentBehaviorTrace modelTurn"}],
                }
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "agent-behavior-traces.jsonl").write_text(
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": "empty",
            "emptyOutputReason": "completedWithoutText",
            "streamStarted": True,
            "firstChunkReceived": False,
            "textChunkCount": 0,
            "finalChunkReceived": True,
            "streamTerminationReason": "stop",
            "rawOutputPrefix": "",
            "promptPrefix": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
        }) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "agent-parse-failures.jsonl").write_text(
        json.dumps({
            "modelName": "agent-json",
            "parseError": "empty",
            "rawOutputPrefix": "",
            "userTurnPrefix": "User request:\nExplain tradeoffs between precision and recall in retrieval systems in plain English.",
        }) + "\n",
        encoding="utf-8",
    )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(report,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    runtime_gaps = [gap for gap in result.gaps if gap["evidence"].get("rootCauseCategory") == "agent_json_completed_without_text"]
    assert len(runtime_gaps) == 1
    assert runtime_gaps[0]["category"] == "agent_json_completed_without_text"
    assert runtime_gaps[0]["severity"] == "error"
    assert result.state["runtime"]["failureCount"] == 1
    assert result.state["runtime"]["skippedLiveModelGenerationCount"] == 0
    assert result.state["triage"]["rootCauseCounts"]["agent_json_completed_without_text"] == 1


def test_improvement_loop_groups_agent_json_resource_budget_denied(tmp_path: Path):
    report = tmp_path / "latest-e2e-report.json"
    prompt = "Search the web for two recent Swift concurrency best practices and summarize them."
    report.write_text(
        json.dumps({
            "kind": "lumen_e2e_test_report",
            "passed": False,
            "failed": 1,
            "scenarios": [
                {
                    "name": "Training eval: web research synthesis",
                    "passed": False,
                    "prompt": prompt,
                    "intent": "webSearch",
                    "expectedIntent": "webSearch",
                    "requiresAgentRun": True,
                    "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                    "final": "Search failed.",
                    "events": [{"phase": "model-evidence", "message": "found primary agent-json modelTurn but agent-json emitted empty output"}],
                }
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "agent-behavior-traces.jsonl").write_text(
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": "empty",
            "emptyOutputReason": "cancelledBeforeFirstToken",
            "streamStarted": False,
            "firstChunkReceived": False,
            "textChunkCount": 0,
            "finalChunkReceived": False,
            "streamTerminationReason": "resource-budget-denied-before-prompt-eval",
            "rawOutputPrefix": "",
            "promptPrefix": prompt,
        }) + "\n",
        encoding="utf-8",
    )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(tmp_path,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    runtime_gaps = [
        gap for gap in result.gaps
        if gap["evidence"].get("rootCauseCategory") == "runtime_environment_deferred"
    ]
    assert runtime_gaps == []
    assert result.state["runtime"]["rawFailureCount"] == 1
    assert "runtime_environment_deferred" not in result.state["triage"]["rootCauseCounts"]
    assert result.passed is True


def test_improvement_loop_quarantines_non_trainable_finalizer_failure(tmp_path: Path):
    report = tmp_path / "latest-e2e-report.json"
    report.write_text(
        json.dumps({
            "kind": "lumen_e2e_test_report",
            "passed": False,
            "failed": 1,
            "results": [
                {
                    "scenarioID": "training-web-research",
                    "kind": "training",
                    "title": "Training eval: web research synthesis",
                    "passed": False,
                    "prompt": "Search the web for two recent Swift concurrency best practices and summarize them.",
                    "actualIntent": "webSearch",
                    "expectedIntent": "webSearch",
                    "requiresAgentRun": True,
                    "failures": ["Live agent returned fallback/error text instead of completing the scenario"],
                    "finalText": "No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.",
                    "events": [],
                }
            ],
        }),
        encoding="utf-8",
    )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(report,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    runtime_gaps = [gap for gap in result.gaps if gap["title"] == "e2e_architecture_finalizer_failure"]
    assert runtime_gaps == []
    assert result.state["runtime"]["rawFailureCount"] == 1
    assert result.state["runtime"]["failureCount"] == 1
    assert result.passed is True


def test_improvement_loop_groups_agent_json_context_overflow(tmp_path: Path):
    report = tmp_path / "latest-e2e-report.json"
    prompt = "What is the weather here and should I carry an umbrella?"
    report.write_text(
        json.dumps({
            "kind": "lumen_e2e_test_report",
            "passed": False,
            "failed": 1,
            "scenarios": [
                {
                    "name": "Training eval: weather stays grounded",
                    "passed": False,
                    "prompt": prompt,
                    "intent": "weather",
                    "expectedIntent": "weather",
                    "requiresAgentRun": True,
                    "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                    "final": "Weather tool output could not be validated.",
                    "events": [{"phase": "model-evidence", "message": "found AgentBehaviorTrace modelTurn but parseError=noJSONObject; stage=agent-repair"}],
                }
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "agent-behavior-traces.jsonl").write_text(
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": "contextWindowExceeded",
            "rawOutputPrefix": "Prompt exceeded context window before generation",
            "promptPrefix": prompt,
        }) + "\n",
        encoding="utf-8",
    )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=_repo_root(),
            output=tmp_path / "agent_manifest",
            loop_output=tmp_path / "loop",
            runtime_audit_paths=(report,),
            deterministic=True,
            strict=False,
            dry_run_commands=True,
            generate_agent_fine_tuning=False,
            generate_system_prompts=False,
        )
    )

    runtime_gaps = [gap for gap in result.gaps if gap["evidence"].get("rootCauseCategory") == "agent_json_context_overflow"]
    assert len(runtime_gaps) == 1
    assert runtime_gaps[0]["category"] == "prompt_budget_overflow"
    assert runtime_gaps[0]["severity"] == "error"
    assert result.state["runtime"]["failureCount"] == 1
    assert result.state["runtime"]["skippedLiveModelGenerationCount"] == 0
    assert result.state["triage"]["rootCauseCounts"]["agent_json_context_overflow"] == 1
