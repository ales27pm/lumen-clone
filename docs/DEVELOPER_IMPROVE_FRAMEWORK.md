# Lumen Developer Improve Framework

This document is the canonical developer framework for testing, debugging,
diagnosing, improving, and training Lumen. It unifies the on-device Developer
Console, Swift diagnostics, Python manifest crawler, adapter-first training
loop, TestFlight runtime audit flow, and generated improvement artifacts into
one repeatable system.

The framework loop is:

```text
Observe -> Diagnose -> Plan -> Change -> Validate -> Learn
```

The primary live-runtime source of truth is the shipped app on device through
TestFlight exports. Local checks, simulator checks, static manifests, and
diagnostic exports are required support layers, but they do not replace
TestFlight/device evidence.

## Framework Contract

Every improvement pass must answer six questions:

1. What did the current source and generated manifest say should happen?
2. What did static/local validation prove before running the app?
3. What did the shipped app observe on device?
4. Which gaps are code defects, runtime wiring defects, model behavior defects,
   dataset gaps, or workflow gaps?
5. What change removes the gap without weakening privacy, runtime invariants,
   or evidence ownership?
6. Which new artifacts should the next loop learn from?

The loop is intentionally one auditable pass at a time. External automation may
repeat the pass, but each iteration must leave behind state, gaps, reports, and
next actions.

## Evidence Layers

Evidence is routed by layer. Do not infer evidence ownership from filenames or
visual success alone.

| Layer | Trust role | Typical source | Owns live scenario pass/fail |
|---|---|---|---|
| `static_source` | Source-of-truth extraction | Swift source, manifest crawler | No |
| `local_validation` | Preflight correctness | pytest, build readiness, invariant scripts | No |
| `simulator_validation` | Optional runtime preflight | Xcode simulator build/test/smoke | No |
| `device_runtime` | Shipped-runtime diagnostic truth | TestFlight Agent Grounding exports, runtime traces | No |
| `live_e2e` | Shipped-runtime scenario truth | Live E2E report JSON | Yes |
| `training_feedback` | Learning inputs | repair samples, SFT/DPO/eval datasets | No |

Rules:

- `live_e2e` is the only layer that can claim a real scenario passed or failed.
- Agent Grounding exports are runtime evidence, grounding evidence, behavior
  audit evidence, and trace evidence.
- Static scenario checks are useful diagnostics, but they are non-live.
- Empty runtime trace exports are gaps. They must not be hidden by passing
  static checks.
- The ingester must route by `exportPolicy.sourceLayer` and
  `exportPolicy.ownsLiveE2EScenarios`.

See `docs/RUNTIME_AUDIT_BOUNDARIES.md` for the exact export contract.

## Loop Phases

### 1. Observe

Collect current source, generated artifacts, diagnostics, runtime exports, and
E2E evidence.

Primary inputs:

```text
ios/Lumen/
ios/LumenTests/
tools/lumen_manifest_crawler/
scripts/
exports/
runtime-audits/
generated/agent_manifest/
generated/agent_improvement_loop/
```

The observation phase must keep privacy boundaries intact. Diagnostics may use
bounded counters, statuses, trace prefixes, manifest-derived records, repair
samples, and explicit runtime audit JSON. They must not become unrestricted raw
prompts, transcripts, contacts, files, photos, memory bodies, or tool payload
bodies.

### 2. Diagnose

Convert evidence into gaps. A gap should identify the failing layer and the
next useful action.

Common gap categories:

```text
validation
validation_warning
command_failure
runtime_drift
dataset_coverage
agent_fine_tuning_coverage
agent_eval_coverage
eval_coverage
testflight_runtime_pending
```

Severity policy:

- `critical`: command failure, unsafe runtime drift, or evidence that blocks
  trusting the pass.
- `error`: validation failure, required coverage missing, or required
  TestFlight/runtime evidence missing when enforced.
- `warning`: useful coverage or workflow gap that should not block all local
  iteration.

### 3. Plan

Choose the smallest change that removes the gap while preserving:

- adapter-first Qwen3 runtime shape;
- evidence-layer ownership;
- deterministic generated artifacts;
- privacy boundaries;
- TestFlight/device authority for live behavior.

Plans should target one or more of:

```text
Swift runtime or diagnostics wiring
manifest extraction
runtime ingest and normalization
dataset compiler
eval/scenario generation
validation scripts
training/export workflow
docs and runbooks
```

### 4. Change

Make source changes only after the gap and target layer are clear.

Implementation rules:

- Do not invent tool IDs or runtime schemas.
- Do not make Agent Grounding static checks look like live E2E results.
- Do not weaken `agent_grounding_no_recent_model_traces`.
- Do not silently switch the Qwen3 default back to role-baked full GGUFs.
- Do not commit model binaries, LoRA adapters, checkpoints, or release-baked
  GGUFs to GitHub.

### 5. Validate

Validation is layered. The minimum useful pass depends on the change:

```bash
python tools/check_adapter_runtime_invariants.py
python -m pytest tools/lumen_manifest_crawler/tests
scripts/check-ios-build-readiness.sh
scripts/validate_lumen_ios.sh
```

On macOS with Xcode, use the simulator as a preflight:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build
```

For live confidence, run the generated TestFlight runbook on device and ingest
the exported JSON into the next loop.

### 6. Learn

Feed valid runtime evidence into the next manifest/dataset/training pass.

Expected learning outputs:

```text
generated/agent_manifest/dataset/runtime_audit_repairs.jsonl
generated/agent_manifest/dataset/eval_scenarios.jsonl
generated/agent_manifest/dataset/dpo_preference_pairs.jsonl
generated/agent_manifest/fine_tuning/
generated/fine_tuning/
generated/agent_improvement_loop/next_action_prompts.jsonl
```

Runtime failures should become repair samples, regression evals, stricter
schemas, stronger traces, or targeted code fixes.

## Canonical Commands

Use the repo's Python 3.11 environment for crawler and training commands. The
commands below use `python` to mean the active virtualenv interpreter, not
Xcode's system `python3`.

Generate manifest, datasets, gaps, runbook, and TestFlight queue:

```bash
python -m lumen_manifest_crawler improve-loop \
  --root . \
  --output generated/agent_manifest \
  --loop-output generated/agent_improvement_loop \
  --generate-system-prompts \
  --generate-agent-fine-tuning \
  --testflight-build-label "<version-build>"
```

Require TestFlight/device evidence for the pass:

```bash
python -m lumen_manifest_crawler improve-loop \
  --root . \
  --require-testflight-runtime-audit \
  --fail-on-validation
```

Ingest one or more runtime exports:

```bash
python -m lumen_manifest_crawler improve-loop \
  --root . \
  --runtime-audit exports/lumen-agent-grounding-audit-testflight.json \
  --runtime-audit exports/lumen-live-e2e-report-testflight.json
```

Run the full adapter-first terminal workflow when the training environment is
ready:

```bash
python tools/lumen_terminal_improve_loop.py \
  --mode full \
  --resume \
  --state-file generated/agent_improvement_loop/pipeline_state.json \
  --config-dir tools/fine_tuning/unsloth/configs_qwen3_bootstrap \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --base-model-id Qwen/Qwen3-1.7B \
  --seed 42 \
  --assistant-only-loss \
  --hf-private \
  --fail-if-missing-qwen3-config \
  --stop-on-error
```

Use the visual loop for inspection and orchestration, not as a separate source
of truth:

```bash
python tools/run_visual_improve_loop_v2.py
python tools/serve_visual_improve_loop.py --open
```

Use the consolidated developer framework CLI and local UI:

```bash
python -m lumen_manifest_crawler framework status --root .
python -m lumen_manifest_crawler framework plan --root . --environment macos
python -m lumen_manifest_crawler framework serve --root . --environment macos --open
python tools/lumen_dev.py status --root .
```

On Ubuntu training hosts:

```bash
python tools/lumen_dev.py plan --root . --environment ubuntu
python tools/lumen_dev.py train --root . --dry-run
```

## On-Device Developer Console

The app exposes one consolidated Developer Console from Settings. It is the
shipped-runtime developer surface for:

- diagnostics snapshots;
- evidence-layer ownership;
- Agent Grounding runtime audit package export;
- live E2E report export;
- persistent runtime diagnostics;
- bounded logs and diagnostic text;
- export-packet guidance for the offline loop.

The console intentionally labels Agent Grounding outputs as diagnostic/runtime
evidence and Live E2E outputs as the scenario pass/fail owner.

## Required Artifacts

The framework is healthy when a pass can produce and explain these artifacts:

```text
generated/agent_improvement_loop/loop_state.json
generated/agent_improvement_loop/loop_gaps.json
generated/agent_improvement_loop/next_action_prompts.jsonl
generated/agent_improvement_loop/testflight_scenarios.jsonl
generated/agent_improvement_loop/TESTFLIGHT_RUNBOOK.md
generated/agent_improvement_loop/LOOP_REPORT.md
generated/agent_manifest/AgentBehaviorManifest.json
generated/agent_manifest/dataset_manifest.json
generated/agent_manifest/dataset/
generated/agent_manifest/fine_tuning/
```

Artifact meanings:

- `loop_state.json`: machine-readable state for the pass.
- `loop_gaps.json`: prioritized diagnosis queue.
- `next_action_prompts.jsonl`: implementation prompts for the next pass.
- `testflight_scenarios.jsonl`: shipped-app scenario queue.
- `TESTFLIGHT_RUNBOOK.md`: human/device execution protocol.
- `LOOP_REPORT.md`: compact pass summary.
- manifest and dataset outputs: learning inputs and runtime grounding artifacts.

## Pass And Failure Rules

A local pass is not a live-runtime pass. A loop can be locally clean while still
awaiting TestFlight evidence.

The loop should fail or create blocking gaps when:

- required commands fail;
- manifest or fine-tuning validation fails;
- required dataset families are empty;
- required TestFlight runtime evidence is missing;
- E2E evidence claims model success while reporting no loaded model or
  routing-only execution;
- runtime traces are empty after a claimed live interaction batch;
- a selected tool is outside the manifest-allowed tool set;
- forbidden sentinels or hidden reasoning leak into user-visible output.

The loop may warn when:

- TestFlight evidence is pending but not required for the current local pass;
- eval coverage is below target;
- an agent has limited eval coverage;
- diagnostics expose a non-blocking visibility gap.

## Existing Documents

This document owns the workflow model. Supporting docs own narrower contracts:

- `docs/RUNTIME_AUDIT_BOUNDARIES.md`: evidence export and ingestion ownership.
- `docs/ADAPTER_RUNTIME_IMPROVE_LOOP.md`: Qwen3 adapter runtime and training
  invariants.
- `docs/VISUAL_IMPROVE_LOOP.md`: visual dashboard and UI orchestration.
- `docs/DIAGNOSTICS_UI.md`: privacy-safe diagnostics surfaces.
- `docs/HF_ARTIFACT_WORKFLOW.md`: Hugging Face artifact publication.

## CLI Consolidation

The framework now exposes a consolidated CLI that wraps existing commands
without changing evidence semantics:

```text
python -m lumen_manifest_crawler framework status
python -m lumen_manifest_crawler framework plan
python -m lumen_manifest_crawler framework run <job-id>
python -m lumen_manifest_crawler framework serve
python -m lumen_manifest_crawler framework diagnose
python -m lumen_manifest_crawler framework ingest
python -m lumen_manifest_crawler framework train
```

`tools/lumen_dev.py` is a thin entry point to the same framework commands.
