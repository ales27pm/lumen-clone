# Lumen Developer Workflow

The canonical developer workflow is:

```bash
python3 -m lumen_manifest_crawler developer-cycle --root . --portable
```

or through the shell wrapper:

```bash
bash scripts/run-lumen-developer-cycle.sh --portable
```

This command coordinates existing validation, manifest, runtime-audit, improvement-loop, Xcode, and training/HF jobs. It does not replace the lower-level tools; it records them as phases with explicit commands, skip reasons, outputs, and pass/fail state.

## Phases

1. **Environment detection** records whether the root is a non-git ZIP/export or git checkout, whether the host is Linux/Codex/static, macOS without Xcode, or macOS with Xcode, and writes the result to `generated/developer_framework/developer_cycle_report.json` and `generated/developer_framework/DEVELOPER_CYCLE_REPORT.md`.
2. **Static source validation** runs the existing static gates: agent kernel boundary checks, adapter runtime invariants, iOS LoRA hardening invariants, MSAL release config, iOS signing/capability checks, shell-subprocess security checks, iOS build-readiness, and `git diff --check` when inside a git worktree.
3. **Manifest and dataset generation** calls the existing manifest crawler/improvement-loop path. It writes `generated/agent_manifest/AgentBehaviorManifest.json`, `AgentBehaviorManifest.md`, manifest validation, dataset manifest/index, fleet prompts, and fine-tuning outputs when enabled by the canonical loop.
4. **Runtime-audit/report ingestion** reads exported evidence from `exports/`, `runtime-audits/`, and each repeatable `--runtime-audit PATH`. It does not treat generated loop reports/logs as runtime evidence. It writes `generated/developer_framework/framework_report.json` and `runtime_report_index.json`.
5. **Improvement-loop preparation** writes the standard loop outputs in `generated/agent_improvement_loop/`: `LOOP_REPORT.md`, `loop_state.json`, `loop_gaps.json`, `GAP_TRIAGE.md`, `gap_triage.json`, `TESTFLIGHT_RUNBOOK.md`, and `testflight_scenarios.jsonl`. Persistent runtime diagnostics exports keep bounded local remediation proposals attached to loop gaps, while Live E2E remains the only source that owns scenario pass/fail. The developer-cycle report includes `improvementLoopOutputContractPassed` to confirm these outputs match the next improve-loop handoff contract.
6. **Optional macOS/Xcode validation** runs `scripts/validate_lumen_ios.sh` only when Xcode is available, or when `--with-xcode` requires it. Portable/Linux/ZIP environments mark this phase as skipped, not passed.
7. **Optional training/HF profile** remains opt-in. `--with-training-plan` prints the `ubuntu-preflight`, `train-adapters`, `convert-adapters`, `hf-resolve`, and `hf-upload-dry-run` jobs. `--run-training` is required to execute them.

## Learned Local Defaults

- Use `build-for-testing` as the deterministic Xcode checkpoint before simulator execution.
- For focused simulator proof, reuse the compiled `.xctestrun` with `test-without-building`; do not recompile the app for every narrow rerun.
- Prefer the dedicated `Lumen Focused Test iPhone` simulator. If `bootstatus -b` stalls at System App but the device is Booted and SpringBoard/backboardd are running, use the readiness-probe path and keep the run bounded.
- Keep release workflow evidence strict: App Store Connect submission is complete only after `UPLOAD SUCCEEDED with no errors` and a `Delivery UUID`; archive/export success alone is not upload success.
- When a run exposes a repeatable failure mode, update the docs or agent instructions before treating the work as done.

## Pass/Fail Semantics

The final report always separates:

- `staticValidationPassed`
- `manifestValidationPassed`
- `runtimeEvidencePresent`
- `improvementLoopPassed`
- `improvementLoopOutputContractPassed`
- `xcodeValidationStatus`: `passed`, `failed`, or `skipped`
- `trainingStatus`: `not_requested`, `planned`, `running`, `failed`, or `passed`
- `overallPortablePassed`
- `overallReleaseCandidatePassed`

Portable pass does not imply release-candidate pass. A release-candidate pass requires Xcode validation and runtime evidence. Missing runtime evidence is recorded as missing evidence, not converted into success. Skipped live model generation remains separate from raw runtime failures and cannot make the loop look like live model proof passed.

## Useful Profiles

```bash
python3 -m lumen_manifest_crawler developer-cycle --root . --portable
python3 -m lumen_manifest_crawler developer-cycle --root . --runtime-audit runtime-audits/latest.json
python3 -m lumen_manifest_crawler developer-cycle --root . --with-xcode --fail-on-static --fail-on-validation
python3 -m lumen_manifest_crawler developer-cycle --root . --with-training-plan
python3 -m lumen_manifest_crawler developer-cycle --root . --run-training
```

Use `--json` for machine-readable console output. Use `--dry-run` to write the report shape without executing commands. Use `--skip-generation` or `--skip-improvement-loop` only when inspecting existing artifacts.
