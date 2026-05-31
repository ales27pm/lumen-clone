# Lumen Manifest Crawler

The Lumen Manifest Crawler is the deterministic bridge between the Swift runtime, the in-app audit loop, and the model fine-tuning pipeline.

It extracts the real source-of-truth from the Lumen / monGARS codebase and writes an `AgentBehaviorManifest.json` plus role-specific, compiled, runtime-repair, and fleet self-knowledge artifacts for:

- Cortex
- Tool Executor
- Mouth
- Mimicry
- REM
- SFT train / validation
- DPO preference pairs
- eval scenarios
- runtime audit repair records
- fleet system prompts
- cross-model self/peer/delegation training
- source-code self-awareness records
- unified fleet identity records
- a compact Markdown manifest for RAG or prompt injection

## Why this exists

LLM agents drift when they are trained on stale examples, invented tool names, outdated memory scopes, or synthetic schemas that no longer match the app. This crawler prevents that by deriving the training source from Swift files that actually define the runtime.

The pipeline combines three truth sources:

1. **Static Swift source analysis** from the repository.
2. **In-app runtime audit JSON** exported from the Lumen app, when provided.
3. **E2E evaluation reports** from TestFlight/runtime evals, when provided as JSON, `.txt`, `.md`, or `.log` files.

It now also emits fleet self-knowledge artifacts so every model slot can learn:

1. **Who am I?** Role, responsibilities, tools, memory scopes, output contract, and source origin.
2. **Who are the others?** Public peer slot directory, source origin, purpose, input/output signatures, and private-state boundaries.
3. **How do we act as one?** Routing rules, topology, delegation boundaries, shared memory policy, and unified Lumen identity.
4. **What code map defines us?** Manifest-derived source files, hashes, extracted domains, tool origins, slot origins, and routing origins.

## What it extracts

- model fleet slots and role contracts
- fleet topology and peer call graph
- source-code map with hashed source-of-truth files
- tool IDs, argument schemas, and source origins
- approval requirements
- permission keys
- `UserIntent` cases and source origins
- intent-to-tool routing
- supported JSON value types
- memory scopes
- TTL and freshness classes
- Mimicry style hints
- REM report hints
- forbidden sentinels such as `<private_reasoning>` and `<tool_json>`

## What the dataset compiler adds

The raw role datasets are preserved, then compiled into higher-grade training artifacts:

- `train_sft.jsonl` — canonical chat-message SFT records with stable IDs, role labels, task labels, curriculum tags, risk labels, manifest grounding, and privacy metadata.
- `validation_sft.jsonl` — deterministic validation split.
- `eval_scenarios.jsonl` — manifest adherence tests for routing, tool schemas, sentinel suppression, and hallucinated tool rejection.
- `dpo_preference_pairs.jsonl` — chosen/rejected preference pairs built from negative tool-call samples.
- `tool_schema_cards.jsonl` — exact immutable tool-schema grounding cards.
- `manifest_grounding_cards.jsonl` — fleet, memory, protocol, and sentinel policy cards.
- `runtime_audit_repairs.jsonl` — REM-style repair samples produced from in-app audit failures and E2E evaluation reports.
- `dataset_manifest.json` — dataset lineage, counts, hashes, source policy, split policy, and privacy policy.
- `dataset_index.csv` — compact overview of every emitted dataset family.

## Fleet self-knowledge artifacts

When enabled, the crawler also writes:

- `fleet_system_prompts.json` — one deterministic prompt and compact context payload per model slot.
- `AgentBehaviorManifest.md` — compact LLM-readable Markdown manifest for RAG or prompt injection.
- `cross_model_training/cross_model_training.jsonl` — SFT and DPO records for self-knowledge, peer-knowledge, delegation, source-code awareness, and private-state boundaries.
- `cross_model_training/cross_model_training_index.csv` — compact count summary by record type, role, and task.

The fleet artifacts now include these source-aware task families:

- `fleet_whole_system_identity` — teaches every slot that Lumen is one logical agent composed of specialized slots.
- `fleet_self_knowledge` — teaches each slot its own role, source, responsibilities, memory scope, and boundary.
- `fleet_peer_knowledge` — teaches each slot the public role and contract of every other slot.
- `fleet_peer_source_knowledge` — teaches each slot where peer roles come from and how to coordinate without crossing private-state boundaries.
- `source_code_self_knowledge` — teaches the manifest-derived source map, hashed source files, and code-awareness limits.
- `source_tool_registry_knowledge` — teaches every valid tool, argument schema, permission boundary, approval boundary, and source origin.
- `source_routing_knowledge` — teaches intent routing, allowed tools, forbidden tools, and source-derived routing constraints.
- `fleet_delegation` / `fleet_delegation_preference` — teaches manifest-compliant routing and rejects invented peer/tool calls.
- `fleet_private_state_boundary` — teaches that peer-private caches, hidden reasoning, and runtime internals cannot be exposed or fabricated.

These artifacts are derived from the same manifest and are intended to make the fleet behave like one coherent Lumen agent instead of isolated models.

## Run locally

From the repo root:

```bash
cd tools/lumen_manifest_crawler
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m lumen_manifest_crawler generate --root ../.. --output ../../generated/agent_manifest --pretty
```

Or from the repo root after installation:

```bash
python -m lumen_manifest_crawler generate --root . --output generated/agent_manifest --pretty
```

## Run one closed improvement-loop cycle

The `improve-loop` command performs one auditable cycle of the static/TestFlight-runtime/training loop:

1. optionally run a local validation command;
2. scan Swift source and regenerate the manifest;
3. ingest one or more in-app dataset package JSON files from a previous TestFlight run;
4. ingest one or more E2E evaluation reports from a previous runtime eval;
5. compile base datasets, fleet artifacts, and per-agent fine-tuning datasets;
6. optionally run build/archive/training commands;
7. write a loop state file, gap report, Markdown report, TestFlight runbook, TestFlight scenario queue, and next-action prompts for the next pass.

```bash
python -m lumen_manifest_crawler improve-loop \
  --root . \
  --output generated/agent_manifest \
  --loop-output generated/agent_improvement_loop \
  --runtime-audit runtime-audits/latest-testflight-export.json \
  --runtime-audit runtime-audits/latest-e2e-report.txt \
  --generate-system-prompts \
  --generate-agent-fine-tuning \
  --testflight-build-label "1.0.0-build-42"
```

The loop writes:

```text
generated/agent_improvement_loop/
├── loop_state.json
├── loop_gaps.json
├── next_action_prompts.jsonl
├── testflight_scenarios.jsonl
├── TESTFLIGHT_RUNBOOK.md
└── LOOP_REPORT.md
```

Use `TESTFLIGHT_RUNBOOK.md` and `testflight_scenarios.jsonl` for the real in-app phase. Install the TestFlight build, run scenario prompts through the normal app UI, open Agent Grounding, run the audit, export the in-app dataset package JSON, then feed that JSON into the next loop with `--runtime-audit`.

To force the loop to fail when a TestFlight export has not yet been ingested:

```bash
python -m lumen_manifest_crawler improve-loop \
  --root . \
  --require-testflight-runtime-audit \
  --fail-on-validation
```

To include commands without executing them:

```bash
python -m lumen_manifest_crawler improve-loop \
  --root . \
  --dry-run-commands \
  --test-command "python -m pytest tools/lumen_manifest_crawler/tests" \
  --build-command "xcodebuild -version" \
  --train-command "python tools/fine_tuning/unsloth/train.py"
```

Run the command repeatedly from CI, a local shell loop, or a Codex pass. Each iteration should either ingest a fresh TestFlight export, remove a gap, or expand runtime coverage with a new TestFlight scenario family, trace field, adversarial dataset family, or quality gate.

## Ingest E2E reports into the next fine-tuning cycle

`--runtime-audit` now accepts structured runtime JSON and text-based E2E reports with suffixes `.txt`, `.md`, `.markdown`, or `.log`.

Example text report shape:

```text
E2E Test Report
Passed: 5
Failed: 2

Training signals for next run:
• response-quality: 2 issues
• Capture failed prompts + final outputs into next fine-tuning dataset.

❌ Training eval: memory save/recall
Prompt: Remember that I prefer concise bullet points, then tell me what you remembered.
Intent: memory / expected memory
Failures: Required final hint missing: remember
Final: Saved: I prefer concise bullet points.
```

The ingest layer converts failed E2E scenarios into `runtime_audit_repairs` records with:

- original prompt
- intended/expected intent
- bad final output
- missing required hint
- corrected output
- lesson
- curriculum label

This lets failed runtime evals become direct supervised repair samples for the next fine-tuning dataset instead of being manually appended to generated JSONL.

## Generate fleet self-knowledge artifacts

```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --pretty \
  --generate-system-prompts
```

This writes:

```text
generated/agent_manifest/
├── fleet_system_prompts.json
├── AgentBehaviorManifest.md
└── cross_model_training/
    ├── cross_model_training.jsonl
    └── cross_model_training_index.csv
```

To write cross-model training elsewhere:

```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --generate-system-prompts \
  --cross-model-train-dir generated/cross_model_training
```

To export only the Markdown manifest plus normal outputs:

```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --export-md
```

## Include in-app runtime audit data

Export one or more in-app dataset package JSON files from the iPhone app, then pass each file or a directory containing JSON/text reports:

```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --pretty \
  --runtime-audit ./runtime-audits/latest-audit.json \
  --runtime-audit ./runtime-audits/latest-e2e-report.txt
```

Multiple inputs are allowed:

```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --runtime-audit ./runtime-audits/device-a.json \
  --runtime-audit ./runtime-audits/device-b.json \
  --runtime-audit ./runtime-audits/archive/
```

The compiler only ingests explicit audit/package/report files. It does not scrape free-form user logs, full conversations, private text, photos, contacts, calendar bodies, files, or unrestricted tool payloads.

## Determinism

By default, generated timestamps and splits are deterministic so CI can detect drift cleanly. For local exploratory builds, use:

```bash
python -m lumen_manifest_crawler generate --root . --output generated/agent_manifest --non-deterministic
```

## CI drift prevention

Use `--fail-on-change` to make CI fail when regenerated outputs differ from the current git working tree:

```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --pretty \
  --generate-system-prompts \
  --cross-model-train-dir generated/cross_model_training \
  --fail-on-change
```

That forces every tool, intent, model slot, fleet topology, memory policy, or runtime repair change to refresh the self-descriptive fleet artifacts.

## Full output

```text
generated/agent_manifest/
├── AgentBehaviorManifest.json
├── AgentBehaviorManifest.pretty.json
├── AgentBehaviorManifest.sha256
├── AgentBehaviorManifest.md
├── fleet_system_prompts.json
├── manifest_validation_report.json
├── dataset_manifest.json
├── dataset_index.csv
├── routing_matrix.csv
├── tool_registry.csv
├── cross_model_training/
│   ├── cross_model_training.jsonl
│   └── cross_model_training_index.csv
└── dataset/
    ├── cortex_routing.jsonl
    ├── executor_tool_calls.jsonl
    ├── mouth_responses.jsonl
    ├── mimicry_style.jsonl
    ├── rem_reflection.jsonl
    ├── negative_samples.jsonl
    ├── approval_boundary_samples.jsonl
    ├── train_sft.jsonl
    ├── validation_sft.jsonl
    ├── eval_scenarios.jsonl
    ├── dpo_preference_pairs.jsonl
    ├── tool_schema_cards.jsonl
    ├── manifest_grounding_cards.jsonl
    └── runtime_audit_repairs.jsonl
```

## Agent usage

### Cortex

Use:

- `cortex_routing.jsonl`
- `routing_matrix.csv`
- `eval_scenarios.jsonl`
- `fleet_system_prompts.json` entry for the Cortex/router slot
- `cross_model_training.jsonl` records with `taskType=fleet_delegation`, `fleet_peer_knowledge`, `source_routing_knowledge`, and `fleet_whole_system_identity`

Cortex should classify intent, select only allowed manifest tools or slots, and delegate out-of-scope work rather than improvising.

### Tool Executor

Use:

- `executor_tool_calls.jsonl`
- `approval_boundary_samples.jsonl`
- `tool_schema_cards.jsonl`
- `negative_samples.jsonl`
- `dpo_preference_pairs.jsonl`
- `cross_model_training.jsonl` records with `source_tool_registry_knowledge`, `source_code_self_knowledge`, and private-state boundary tasks
- its `fleet_system_prompts.json` entry

Tool Executor should emit exact manifest-valid tool JSON, respect required arguments, and stop at approval/permission boundaries.

### Mouth

Use:

- `mouth_responses.jsonl`
- `eval_scenarios.jsonl` sentinel cases
- `manifest_grounding_cards.jsonl`
- `cross_model_training.jsonl` records with `fleet_whole_system_identity`, `fleet_peer_knowledge`, and `fleet_private_state_boundary`
- `runtime_audit_repairs.jsonl` records from failed E2E response-quality reports
- its `fleet_system_prompts.json` entry

Mouth should produce only user-facing text and must never expose private reasoning, raw tool JSON, forbidden sentinels, fabricated source knowledge, or peer-private runtime state.

### Mimicry

Use:

- `mimicry_style.jsonl`
- `train_sft.jsonl` records with `agentRole=mimicry`
- `cross_model_training.jsonl` records with `fleet_self_knowledge`, `fleet_peer_source_knowledge`, and `source_code_self_knowledge`
- its `fleet_system_prompts.json` entry

Mimicry should adapt style, language, and formatting without changing facts, tool IDs, source-code boundaries, safety boundaries, or approval requirements.

### REM

Use:

- `rem_reflection.jsonl`
- `runtime_audit_repairs.jsonl`
- `manifest_grounding_cards.jsonl`
- `dataset_manifest.json`
- `cross_model_training.jsonl` records with `source_code_self_knowledge`, `source_routing_knowledge`, `fleet_private_state_boundary`, and `fleet_whole_system_identity`
- its `fleet_system_prompts.json` entry

REM should diagnose drift, produce repair samples, classify memory/freshness decisions, and keep the fleet aligned with the manifest, TestFlight runtime exports, and E2E reports.

## Adding a new tool safely

1. Add the Swift tool definition to `ToolDefinition.swift`.
2. Include a stable tool ID.
3. Include explicit argument names and JSON-compatible types.
4. Set `requiresApproval` correctly.
5. Add `permissionKey` when touching iOS permissions.
6. Run the crawler with `--generate-system-prompts`.
7. Review `tool_registry.csv`, `tool_schema_cards.jsonl`, `fleet_system_prompts.json`, and `AgentBehaviorManifest.md`.
8. Commit generated manifest/dataset/fleet artifact changes.

## Adding a new slot safely

1. Add the slot to the Swift model fleet source.
2. Include role and responsibilities.
3. Run the crawler with `--generate-system-prompts`.
4. Review `AgentBehaviorManifest.json.fleet`, `fleetTopology`, `fleet_system_prompts.json`, and `cross_model_training.jsonl`.
5. Verify delegation examples and private-state-boundary examples were generated.
6. Commit generated artifacts.

## Adding a new intent safely

1. Add the `UserIntent` case.
2. Add deterministic routing to valid tool IDs.
3. Run the crawler with `--generate-system-prompts`.
4. Check `routing_matrix.csv`.
5. Check `eval_scenarios.jsonl`.
6. Check relevant fleet system prompts.
7. Commit the updated manifest/datasets/artifacts.

## iPhone runtime audit vs source crawling

The Python crawler runs on the build/dev machine and reads Swift source.

The iPhone app does not crawl raw Swift source. It loads the bundled manifest and compares it against the live runtime tool registry and deterministic scenarios. This verifies the shipped app still matches the training truth.

When an audit or E2E report fails, its JSON/text report can be fed back into the dataset compiler. The compiler converts failures into REM/Mouth repair records so the model learns how to diagnose drift and produce corrected final responses in the next fine-tuning cycle.

## Validation behaviour

Hard failures block generation when the manifest would train agents on impossible behaviour. Examples:

- duplicate tool IDs
- intent references unknown tool
- unsupported JSON argument type
- approval-required tool without approval samples
- sentinel leak in user-facing or compiled model-visible dataset text
- executor dataset references unknown tool
- compiled dataset record missing stable ID or canonical chat messages
- malformed DPO preference pair

Warnings flag quality issues that should be cleaned up, such as missing descriptions or ambiguous intent routing.
