# Audit-to-adapter pipeline deep dive

## Evidence status

- **Label:** `static_analysis`
- **What this document proves:** the static implementation map, package fields, inspection rules, and failure modes that the pipeline must check before trusting audit inputs.
- **What this document does not prove:** that the app emitted fresh device traces, that adapter evidence appeared in a real run, or that any live E2E scenario passed.

This is the deeper implementation map for the Lumen Qwen3 adapter-first loop. It complements `docs/AUDIT_TO_ADAPTER_PIPELINE.md`.

## Why the first contract was not enough

A shallow pipeline can say:

```text
read JSON -> generate datasets -> train adapters -> convert/upload -> app installs adapters
```

That is not enough for Lumen. The real risk is that the pipeline quietly trains from weak or stale signals while the iOS runtime appears to work through fallback behavior.

The deeper contract has to prove four things:

1. The app exported a real `LumenInAppDatasetPackage`, not just any JSON file.
2. The package contains live `recentTraces` from actual model turns.
3. Qwen3 role turns show adapter evidence: `runtimePath=sharedAdapter`, `adapterSlot`, and `adapterApplied=true`.
4. Generated adapter training/export plans use the same artifact paths that the app runtime expects.

## App export producer

The app producer is:

```text
ios/Lumen/Services/AgentGrounding/InAppDatasetPackageExporter.swift
```

The exported package schema is `1.2.0`. It includes:

```text
runtimeManifestAudit
behaviorAudit
scenarioResults
recentTraces
improveLoop
exportPolicy
```

Important package policy values:

```text
format: agent-grounding-runtime-json-package
sourceLayer: agentGroundingRuntimeAudit
```

The package writer also emits JSONL sidecars:

```text
accepted_training-<timestamp>.jsonl
quarantined_samples-<timestamp>.jsonl
regression_tests-<timestamp>.jsonl
```

These sidecars are useful, but the JSON package remains the canonical audit input because it preserves runtime trace evidence and export policy.

## Audit package inspector

New file:

```text
tools/pipeline/audit_package_inspector.py
```

It understands:

```text
schemaVersion
exportPolicy.format
exportPolicy.sourceLayer
usedRuntimeFallback
recentTraces
behaviorAudit.violations
behaviorAudit.repairSamples
improveLoop.acceptedTraining
improveLoop.quarantinedSamples
improveLoop.regressionTests
```

It counts:

```text
trace_count
model_turn_count
qwen3_model_turn_count
shared_adapter_runtime_turn_count
adapter_applied_true_count
adapter_applied_false_count
adapter_applied_missing_count
trace_parse_error_count
trace_selected_tool_allowed_count
accepted_training_count
quarantined_sample_count
regression_test_count
adapter_slots_seen
slots_seen
```

Strict checks are exposed through:

```text
assert_audit_requirements(...)
```

## Strict audit inspection CLI

New file:

```text
tools/pipeline/inspect_audit_to_adapter_inputs.py
```

Recommended use before ingest:

```bash
python tools/pipeline/inspect_audit_to_adapter_inputs.py \
  --require-runtime-audit \
  --require-adapter-traces \
  --require-training-signals \
  --write generated/agent_improvement_loop/audit_input_inspection.json
```

This command fails if:

```text
no audit files exist
no in-app dataset package exists
no adapterApplied/adapterSlot evidence exists
no improve-loop training/regression/repair signals exist
```

For a looser check:

```bash
python tools/pipeline/inspect_audit_to_adapter_inputs.py
```

For machine-readable output:

```bash
python tools/pipeline/inspect_audit_to_adapter_inputs.py --json
```

## Deep pipeline validator

New file:

```text
tools/pipeline/validate_audit_to_adapter_pipeline_deep.py
```

Recommended strict validation:

```bash
python tools/pipeline/validate_audit_to_adapter_pipeline_deep.py \
  --require-runtime-audit \
  --require-adapter-traces \
  --require-training-signals
```

This runs the repository contract alignment checks and the audit package inspection together.

## Runtime audit locations

The canonical audit glob set now includes:

```text
exports/*.json
runtime-audits/**/*.json
generated/runtime_audits/*.json
generated/runtime_audit/*.json
generated/testflight_exports/*.json
generated/agent_improvement_loop/runtime_audits/*.json
Diagnostics/LumenDatasetExports/*.json
```

The `runtime-audits/**/*.json` path matters because committed second-loop evidence already lives under `runtime-audits/...`.

## Generated adapter export path fix

The older fine-tuning export planner used:

```text
models/lora/<agent>
```

That was too generic and no longer matched the Qwen3 contract. It is now aligned to:

```text
models/lora_qwen3_bootstrap/<agent>
models/lora_qwen3_gguf/lumen-<agent>-lora.gguf
```

Updated file:

```text
tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/adapter_export.py
```

The generated `adapter_runtime_manifest.json`, `adapter_export_plan.json`, and per-agent `unsloth_config.json` now carry:

```text
sharedBaseRepoID: ales27pm/lumen-qwen3-bootstrap-gguf
sharedBaseFileName: lumen-qwen3-fast-shared-q4_k_m.gguf
adapterRepoID: ales27pm/lumen-qwen3-bootstrap-adapters-gguf
adapterDirectory: models/lora_qwen3_bootstrap/<agent>
adapterGGUFArtifact: models/lora_qwen3_gguf/lumen-<agent>-lora.gguf
```

## Updated test expectation

Updated file:

```text
tools/lumen_manifest_crawler/tests/test_agent_fine_tuning.py
```

The test now asserts the Qwen3-specific adapter paths and repo IDs instead of the stale generic `models/lora/<agent>` path.

## Recommended end-to-end command sequence

Run the audit/input validators locally, then use the canonical Ubuntu launcher
for every mutating training, evaluation, conversion, and optional upload stage.
The legacy `run_audit_to_adapter_pipeline.py --mode train-adapters` and
terminal improve-loop training modes are not supported operator paths.

```bash
cd /absolute/path/to/lumen-clone

python tools/pipeline/validate_audit_to_adapter_pipeline_deep.py

python tools/pipeline/inspect_audit_to_adapter_inputs.py \
  --require-runtime-audit \
  --require-adapter-traces \
  --require-training-signals \
  --write generated/agent_improvement_loop/audit_input_inspection.json

bash scripts/ubuntu_train_lumen_full_pipeline.sh
```

For an explicit private upload, add `--upload --token-file
/secure/path/lumen-hf-token`. Ordinary upload requires a complete frozen
quality pass; `--no-gguf` remains qualified as `complete_without_gguf`. Smoke
or unevaluated publication requires the additional
`--allow-diagnostic-upload` acknowledgement and is isolated below
`diagnostic-runs/` with `promotionEligible=false`. See
`docs/UBUNTU_TRAINING.md`; no training result is an iOS runtime promotion or
real-device proof.

## What still needs real-device proof

Static checks can validate contracts, paths, and audit structure. They cannot prove that iOS/Metal actually applied the adapter at runtime.

Before calling a model release good, export a fresh app audit after real prompts and verify:

```text
runtimePath=sharedAdapter
modelFamily=qwen3
adapterSlot=<expected live slot>
adapterApplied=true
adapterFailureReason is nil/empty
```
