# Visual Improve-Loop Runner

`tools/run_visual_improve_loop_v2.py` is the hardened one-command visual orchestrator for the Lumen improvement loop.

It wraps the existing `lumen_manifest_crawler improve-loop` command, runs the adapter-first fine-tuning output pass, writes the TestFlight handoff queue, runs the release-bake manifest pass, writes first-class Qwen3 embedding retrieval datasets, and generates a standalone visual dashboard.

The v2 runner is repo-rooted: every relative output path is resolved against `--root`, not against the shell's current working directory. This prevents generated loop artifacts from being scattered outside the repository when the script is invoked from another directory.

> `tools/run_visual_improve_loop.py` remains the first visual draft. Prefer `tools/run_visual_improve_loop_v2.py` for real use.

## Artifact distribution policy

Lumen uses a Hugging Face-first artifact workflow. The local PC runs the loop and training, but heavy runtime artifacts are published to Hugging Face and downloaded by the app.

```text
local PC improve-loop
→ generate datasets
→ fine-tune adapters
→ evaluate gates
→ upload selected base / adapter / embedding / release artifacts to Hugging Face
→ update catalog or artifact manifest metadata
→ app downloads artifacts from Hugging Face
→ TestFlight/on-device runtime audit exports results
→ next local improve-loop cycle ingests runtime exports
```

GitHub remains source-only: code, scripts, manifests, checksums, dataset cards, and metadata. Do not commit model binaries, adapters, checkpoints, or release-baked model files.

See:

```text
docs/HF_ARTIFACT_WORKFLOW.md
tools/hf_artifacts/lumen_hf_artifact_manifest.template.json
tools/hf_artifacts/publish_hf_artifacts.py
```

## Default adapter-first run

Run from the repository root:

```bash
python tools/run_visual_improve_loop_v2.py
```

The same command also works from outside the repository when `--root` is explicit:

```bash
python /path/to/Lumen/tools/run_visual_improve_loop_v2.py --root /path/to/Lumen
```

Default outputs, all rooted under `--root`:

```text
generated/agent_manifest/
generated/agent_manifest/embedding/
generated/agent_improvement_loop/
generated/fine_tuning/
generated/fine_tuning/release_bake_gguf_manifest.json
generated/visual_improve_loop/index.html
generated/visual_improve_loop/pipeline.svg
generated/visual_improve_loop/visual_improve_loop_summary.json
```

The default run is adapter-first. It does **not** merge LoRA adapters into full GGUF models. It writes a release-bake manifest that explicitly says the GGUF bake was skipped by default. After training, publish the selected adapter artifacts to Hugging Face with the HF artifact helper instead of committing them to GitHub.

## Local web control page

Run a localhost web control panel:

```bash
python tools/serve_visual_improve_loop.py --open
```

The web page can trigger only preconfigured server-start commands. It does not accept arbitrary shell commands from HTTP requests.

With a configured training command:

```bash
python tools/serve_visual_improve_loop.py \
  --open \
  --train-command "python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/cortex.json"
```

Useful flow:

1. click **Run visual improve-loop**;
2. inspect the generated dashboard;
3. click **Run fine-tuning command** when the dataset output is ready;
4. publish selected artifacts to Hugging Face;
5. update catalog or artifact manifest metadata for the app;
6. run the app/TestFlight layer and export runtime evidence;
7. rerun the loop with `--runtime-audit`.

## Publish trained artifacts to Hugging Face

Prepare or edit:

```text
tools/hf_artifacts/lumen_hf_artifact_manifest.template.json
```

Validate and write a resolved manifest without uploading:

```bash
python tools/hf_artifacts/publish_hf_artifacts.py \
  --manifest tools/hf_artifacts/lumen_hf_artifact_manifest.template.json \
  --skip-upload
```

Dry-run the HF upload commands:

```bash
python tools/hf_artifacts/publish_hf_artifacts.py \
  --manifest tools/hf_artifacts/lumen_hf_artifact_manifest.template.json \
  --dry-run
```

Real upload:

```bash
export HF_TOKEN="..."
python tools/hf_artifacts/publish_hf_artifacts.py \
  --manifest tools/hf_artifacts/lumen_hf_artifact_manifest.template.json
```

This writes:

```text
generated/hf_artifacts/lumen_hf_artifact_manifest.resolved.json
```

The resolved manifest contains artifact sizes, SHA-256 hashes, and Hugging Face download URLs.

## Embedding dataset stage

The loop now generates a dedicated embedding dataset for:

```text
Qwen/Qwen3-Embedding-0.6B
```

It is retrieval/ranking data, not chat SFT. Planned/generated files:

```text
generated/agent_manifest/embedding/
├── corpus.jsonl
├── train_pairs.jsonl
├── val_pairs.jsonl
├── train_triplets.jsonl
├── val_triplets.jsonl
├── hard_negatives.jsonl
├── eval_retrieval.jsonl
└── dataset_card.json
```

The same records are also indexed in the generic dataset family folder as:

```text
generated/agent_manifest/dataset/embedding_corpus.jsonl
generated/agent_manifest/dataset/embedding_train_pairs.jsonl
generated/agent_manifest/dataset/embedding_val_pairs.jsonl
generated/agent_manifest/dataset/embedding_train_triplets.jsonl
generated/agent_manifest/dataset/embedding_val_triplets.jsonl
generated/agent_manifest/dataset/embedding_hard_negatives.jsonl
generated/agent_manifest/dataset/embedding_eval_retrieval.jsonl
generated/agent_manifest/dataset/embedding_dataset_card.jsonl
```

The corpus includes tool schemas, intents, routing rules, fleet slots, memory scopes, source-map entries, manifest grounding cards, eval scenarios, and runtime repair samples. The pair/triplet data is built for query → relevant document retrieval and hard-negative ranking.

## Open the visual dashboard automatically

```bash
python tools/run_visual_improve_loop_v2.py --open-dashboard
```

## Run without local tests

Useful when the environment does not have `pytest` installed yet:

```bash
python tools/run_visual_improve_loop_v2.py --skip-tests
```

## Ingest a TestFlight / in-app audit export

After running the app on device and exporting the Agent Grounding runtime audit package or the live E2E report JSON:

```bash
python tools/run_visual_improve_loop_v2.py \
  --runtime-audit exports/lumen-agent-grounding-audit-testflight.json
```

Multiple audit files or directories are supported:

```bash
python tools/run_visual_improve_loop_v2.py \
  --runtime-audit exports/ \
  --runtime-audit generated/testflight_exports/
```

The runner also auto-discovers likely runtime audit JSON files under:

```text
generated/runtime_audits/
generated/runtime_audit/
generated/agent_improvement_loop/runtime_audits/
generated/testflight_exports/
exports/
```

Disable discovery with:

```bash
python tools/run_visual_improve_loop_v2.py --no-auto-discover-runtime-audit
```

## Treat missing TestFlight audit as hard failure

```bash
python tools/run_visual_improve_loop_v2.py --require-testflight-runtime-audit --fail-on-gaps
```

## Record build/train commands in the loop state

```bash
python tools/run_visual_improve_loop_v2.py \
  --build-command "xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -configuration Debug build" \
  --train-command "python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/cortex.json" \
  --dry-run-commands
```

Remove `--dry-run-commands` when the build/train environment is ready and the commands should execute.

## Explicit GGUF release bake

Only run this after adapter eval gates pass or when the runtime cannot load adapters dynamically:

```bash
python tools/run_visual_improve_loop_v2.py \
  --release-bake \
  --release-bake-python .venv-unsloth/bin/python \
  --skip-release-bake-existing
```

With Hugging Face upload:

```bash
python tools/run_visual_improve_loop_v2.py \
  --release-bake \
  --release-bake-python .venv-unsloth/bin/python \
  --hf-repo-id ales27pm/lumen-fleet-gguf
```

## Important runtime boundary

The script automates the repository-side loop. It cannot execute the iOS app on a physical device. The live-app stage is handled by generated TestFlight artifacts:

```text
generated/agent_improvement_loop/testflight_scenarios.jsonl
generated/agent_improvement_loop/TESTFLIGHT_RUNBOOK.md
```

Run those scenarios in the real app, export the Agent Grounding runtime audit package and/or live E2E report JSON, then feed that JSON back with `--runtime-audit`.

## Quick CI-friendly command

```bash
python tools/run_visual_improve_loop_v2.py \
  --quiet-commands \
  --fail-on-gaps \
  --require-testflight-runtime-audit
```

## Dashboard contents

The generated HTML dashboard includes:

- pipeline step status;
- command output tails;
- dataset family record counts;
- per-agent fine-tuning record counts;
- embedding dataset family counts;
- gap severity distribution;
- TestFlight scenario count;
- next-action prompt count;
- adapter runtime manifest summary;
- release-bake manifest summary;
- runtime audit summary.

## Regression coverage

`tools/lumen_manifest_crawler/tests/test_visual_improve_loop_runner.py` validates:

- relative output paths are rooted under `--root`;
- the improve-loop command receives repo-rooted output paths;
- GGUF release bake requires explicit `--release-bake`;
- realistic TestFlight exports are auto-discoverable;
- loop-state JSON is not mistaken for a runtime audit export;
- dynamic dashboard content is HTML-escaped.

`tools/lumen_manifest_crawler/tests/test_embedding_dataset.py` validates:

- embedding dataset families are generated;
- records are retrieval/ranking records, not chat SFT messages;
- core object types are present;
- the dedicated embedding output directory is written;
- embedding generation is deterministic.
