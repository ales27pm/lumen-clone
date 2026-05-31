"""Tests for the manifest improvement loop runner."""

# pylint: disable=missing-function-docstring,line-too-long

import json
from pathlib import Path

from lumen_manifest_crawler.improvement_loop import AgentImprovementLoopConfig, run_agent_improvement_loop


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
    assert result.state["dataset"]["agentFineTuning"]["cortex"]["trainSFT"] > 0
    assert result.state["testFlight"]["status"] == "awaiting-testflight-runtime-audit"
    assert result.state["testFlight"]["buildLabel"] == "1.0.0-build-99"
    assert len(result.testflight_scenarios) <= 12
    assert any(scenario.get("sourceFamily") == "trace_export_coverage" for scenario in result.testflight_scenarios)
    assert any(scenario.get("sourceFamily") == "trace_integrity" for scenario in result.testflight_scenarios)
    assert result.next_prompts


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
