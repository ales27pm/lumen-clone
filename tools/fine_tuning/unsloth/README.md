# Lumen Agent Fine-Tuning (Unsloth)

This directory contains per-agent training scripts and configs for **Unsloth**.

## Workflow

For a full Ubuntu NVIDIA run, use the pinned container launcher. It prepares an
isolated variant snapshot, trains SFT then DPO for every selected role, verifies
both phase boundaries, runs frozen deterministic evaluation, and creates
adapter-only GGUFs by default:

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh
```

See [`docs/UBUNTU_TRAINING.md`](../../../docs/UBUNTU_TRAINING.md) for host
prerequisites, variants, capacity estimates, resume/overwrite behavior, output
layout, and upload safety. The default is the optimized variant, all six roles,
full frozen evaluation, no upload, private visibility, and no runtime promotion.
Run it as a regular non-root user: the image maps that account's UID/GID to a
real container passwd/group entry and validates its writable home before model
training. The launcher requires a clean Git worktree, hashes the full Ubuntu
orchestration closure, and bakes the verified source and frozen inputs into the
image. Neither the training nor credential container mounts the host checkout.

The commands below describe the individual components. Use them for inspection
and targeted development; the full Ubuntu launcher is the canonical operator
entry point because it supplies the pinned environment and run lineage.

1. Generate datasets.
```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --generate-agent-fine-tuning \
  --fine-tuning-output generated/fine_tuning \
  --pretty \
  --strict
```

2. Inspect each `generated/fine_tuning/<agent>/dataset_card.json`.

3. Train SFT with a run-scoped config that binds the explicit experiment variant, base-model
lineage, phase-specific training-code digest, dependency lock, source Git commit, and observed
container environment. The source contract also binds the canonical working-tree and Ubuntu
orchestration digests. The full Ubuntu host launcher builds the controlled image, derives its
local image ID, and prepares those configs. The inner launcher handles one experiment variant at a
time; `baseline-and-optimized` and `all` are expanded into isolated fail-fast variant batches by
the host wrapper and do not themselves make a comparison/promotion decision. Direct
`--resume-from-checkpoint` calls without the repository's run/checkpoint contract fail closed.
Before any model or PEFT state is loaded, preparation verifies the exact pinned five-file tokenizer
closure (`config.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`, and `vocab.json`),
reconstructs the SFT and preference token-ID transcripts, and checks Fleet's exact optimizer-token
share. `--prepare-only` is independently rechecked from an exact read-only run-root mount; nested
mounts, path substitution, insufficient descriptor headroom, and any input mutation fail closed.
After the prepare-only early-exit boundary, a full or resumed execution invokes the exact
`train_sft --runtime-binding-smoke` loader before converter setup or any PEFT/trainer construction.
It runs once per distinct prepared runtime-load contract (one load for the current six roles) and
atomically retains a self-hashed `training/runtime_binding_smoke.json`; resume re-verifies that
report against every current config rather than silently replacing it. The load proof binds the
explicit compute dtype and sequence length, all 196 expected CUDA NF4 projections, every model
parameter on CUDA, and a finite fixed four-token forward through the patched Qwen/BitsAndBytes
kernel path. Verification reconstructs the expected module and parameter inventories from the
pinned model config instead of trusting the runtime's self-hash alone.

After regenerating Fleet artifacts and before rebuilding an image or launching a pilot, run the
required exact-token gate against a previously verified tokenizer snapshot. The source proxy is a
construction heuristic only: dense minified event-graph JSON can tokenize differently enough that
the proxy passes while the authoritative aggregate or optimizer-window schedule fails. This gate
loads the snapshot through the production closure verifier and independently checks root plus every
advertised Fleet variant, both SFT and DPO, the strict four-epoch SFT schedule, and public/Fleet
loss-share evidence. Required mode never converts missing dependencies or a missing snapshot into a
skip:

```bash
ROOT="$(pwd -P)"
SNAPSHOT="$ROOT/.local/ubuntu_finetune_runs/<verified-run>/training/global_tokenizer_snapshot"
IMAGE="sha256:<verified-local-image-id>"
docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g,mode=1777 \
  -v "$ROOT:/workspace:ro" \
  -v "$SNAPSHOT:/run/lumen-tokenizer:ro" \
  -e PYTHONPATH=/workspace \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e XDG_CACHE_HOME=/tmp/cache \
  -e LUMEN_REQUIRE_FLEET_PINNED_TOKENIZER_GATE=1 \
  -e LUMEN_FLEET_TOKENIZER_SNAPSHOT=/run/lumen-tokenizer \
  -w /workspace \
  --entrypoint python \
  "$IMAGE" \
  -m unittest -v \
  tools.fine_tuning.unsloth.tests.test_fleet_generated_exact_token_gate
```

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --variant internal_plus_public_baseline \
  --output-dir "$PWD/.local/ubuntu_finetune_runs/baseline"
```

4. Train DPO/ORPO per agent from the verified finalized SFT artifact. This phase is part of the
full Ubuntu pipeline. DPO writes a
separate `sft_dpo` adapter and records the parent SFT digest. The parent boundary verifies the
self-hashed finalized SFT manifest, canonical adapter bytes and base-model declaration, effective
seed, experiment/source manifest, full base-model index/shard/tokenizer contract, environment and
dependency locks, requirements digest, runtime kind, and SFT phase code digest. Preference records
remain structured message lists so pinned TRL 0.24 applies the Qwen chat template and assistant-turn
boundaries.
```bash
python tools/fine_tuning/unsloth/train_dpo.py \
  --config "$LUMEN_AIO_RUN_ROOT/configs/cortex.json" \
  --sft-adapter-dir "$LUMEN_AIO_RUN_ROOT/models/lora_qwen3_bootstrap/cortex" \
  --sft-finalized-variant-manifest "$LUMEN_AIO_RUN_ROOT/training/cortex/finalized_variant_manifest.json"
```

5. Merged-model release bake is intentionally outside the one-click Ubuntu contract. The
run-scoped `<agent>.final.json` files bind the evaluated preference adapters but contain container
paths. The older exporter selects `<agent>.json` from a directory and would therefore select the
superseded SFT parents. Do not point it at the run config directory. A future merged-artifact
workflow must consume explicit final configs inside the pinned image and rerun evaluation and
release approval for the newly created artifact.

6. Optional Hub upload is owned by the full Ubuntu launcher. Use `--upload`; it keeps the
destination private by default, scopes credentials to a separate upload container, re-verifies
the allowlisted evidence, and requires `--public` for public visibility. Ordinary upload requires
a full quality-passed evaluation. A full pass with conversion disabled remains qualified as
`complete_without_gguf`; smoke or unevaluated publication additionally requires
`--allow-diagnostic-upload`, uses `diagnostic-runs/`, and records `promotionEligible=false`.
An intentionally SFT-only run (`preference=false`) is a separate diagnostic state:
`status=sft_only_diagnostic_complete`, `trainingScope=sft_only`, no evaluation or GGUF
boundary, and `promotionEligible=false`. It can be published only with the same explicit
diagnostic override, under `diagnostic-sft-runs/`; that path re-verifies and publishes SFT
evidence only and never probes preference, evaluation, or GGUF paths.
The uploader executes only the attested image copy under isolated Python, with no host source
mount or repository `PYTHONPATH`; run artifacts are read-only and only the receipt path is
writable. Before publication it independently re-verifies the pre-training runtime-binding
smoke report and each included phase report, then binds their exact file digests and compact
model, tokenizer, PEFT-base, adapter-tokenizer, and private-snapshot identities into the upload
intent and receipt.
Here `promotionEligible` means eligible for normal artifact publication after the frozen local
quality gate. It is not proof of installation or behavior on an iPhone. Release promotion still
requires fresh device/TestFlight evidence showing the intended shared-adapter path, adapter slot,
and `adapterApplied=true` without fallback.

7. Evaluate the final preference adapter. The Ubuntu launcher creates the
final lineage config and runs this for every selected role automatically:
```bash
python -m tools.fine_tuning.unsloth.evaluate_adapter \
  --config "$LUMEN_AIO_RUN_ROOT/configs/cortex.final.json" \
  --eval-jsonl "$LUMEN_AIO_RUN_ROOT/generated/fine_tuning/cortex/eval.jsonl" \
  --behavior-manifest "$LUMEN_AIO_RUN_ROOT/generated/agent_manifest/AgentBehaviorManifest.json" \
  --output-dir "$LUMEN_AIO_RUN_ROOT/evaluation/cortex"
```
The evaluator writes candidate outputs, a scored report, and a self-hashed run
manifest. Malformed output and full-suite quality failures return nonzero.
Before advancing to each next selected case it atomically and durably commits
the completed candidate and every raw generation attempt to the private,
self-hashed `evaluation_checkpoint.json`. Missing private directory components
are created with an inode fsync followed by a parent-directory fsync before the
initial journal write. That journal binds the exact adapter,
config, evaluator code, frozen suite, behavior/tool manifest, execution plan,
generation settings, and selected-record order. Resume accepts only an exact
verified prefix; tamper, duplicates, reordering, or any binding drift fail
closed. A complete verified journal is scored and finalized without reloading
the model. The canonical candidate/report/run-manifest trio is published only
after every selected case is present, then the journal is durably removed.
If interruption lands between the three atomic final-file publications, the
complete journal plus only the known private final-file subset is recoverable;
those files are deterministically reconstructed and overwritten. After the
existing journal verifies, resume may also discard an owned mode-0600 regular
temp file whose basename exactly matches this evaluator's atomic writer for the
journal or one of those three finals. Unsafe lookalikes, temps without a valid
journal, any other unknown entry, or an incomplete journal mixed with final
files are rejected. Resume independently reconstructs a terminal evidence trio
without requiring it to have passed quality; verified failures are retained and
stop the run. Any unclassified, invalid, or operational verifier failure also
preserves the directory and aborts rather than authorizing deletion.
Training completion alone is not a model-quality pass, and no local result is
a TestFlight/device pass. `--eval-smoke N` is bounded smoke evidence only; the
prepared `N` is bound into the evaluation evidence and must be smaller than the
frozen suite.

8. Never train on private app exports unless explicitly sanitized.

## ZeroGPU authorization and resume

Use `scripts/hf_zerogpu_train_lumen_adapters_aio.sh` with separate
`LUMEN_ZERO_GPU_ADMIN_TOKEN` and fine-grained `LUMEN_ZERO_GPU_HUB_TOKEN` credentials. The Space,
dataset repository, and adapter/model repository are private by default. Their only public
overrides are `LUMEN_ZERO_GPU_PUBLIC_SPACE=1`, `LUMEN_ZERO_GPU_PUBLIC_DATASET=1`, and
`LUMEN_ZERO_GPU_PUBLIC_ADAPTERS=1`, respectively. Each changes only its named repository, and a
public Space still requires the admin header. Dataset uploads are pinned by their returned Hub
commit SHA. Existing repositories with matching visibility are reused without a settings mutation;
a mismatch stops deployment unless the operator sets that repository's explicit
`LUMEN_ZERO_GPU_CONFIRM_*_VISIBILITY_CHANGE=1` migration confirmation. Confirmed changes and new
repositories are read back before any upload. The browser page does not expose a training button
because browser events cannot attach the required header. Use the authenticated launcher.
The Space also rechecks adapter-repository visibility without mutating it immediately before
uploading trained artifacts.

At Space startup, Lumen hashes the complete installed distribution environment once and records its
digest plus scan timing/count/byte metrics. Trainer subprocesses reuse only that exact manifest via
a process-local HMAC; direct trainer use without the cache key rescans the environment. The
configured ZeroGPU size must match the deployed decorator before allocation, duration must not be
clamped, and the observed CUDA inventory remains unverified audit evidence compared across
experimental variants. Startup scan timing and cache signatures remain audit evidence outside the
immutable resume hash; a restarted process can authorize its new signature only against the same
persisted resolved-environment digest.

`LUMEN_ZERO_GPU_RESUME=1` is accepted only when the existing self-hashed run manifest, original
local dataset snapshot, prepared configs, checkpoint-lineage records, and at least one checkpoint
all match the requested lineage. Fresh runs reject existing workspaces unless
`LUMEN_ZERO_GPU_DESTRUCTIVE_RESET=1` is explicit. Resume and destructive reset are mutually
exclusive.

When the launcher splits agents across multiple batches, resume additionally requires
`LUMEN_ZERO_GPU_RESUME_BATCH=<1-based-batch-number>` and invokes only that batch. Ambiguous
multi-batch resume is rejected. Space-local checkpoints survive only while that deployment's local
disk remains intact; restart-safe resume requires external checkpoint persistence.

The built Space deploys a stable package and invokes:

```bash
python -m lumen_training.train_sft --help
python -m lumen_training.train_dpo --help
```

`lumen_training.train_dpo` selects DPO or ORPO from the config and imports shared SFT helpers through
the same package. The deployed process does not rely on the repository source tree being on
`PYTHONPATH`.

Before model loading, the runtime verifies the full deployed code closure: `app.py`, requirements,
the entire `lumen_training` package, and all covered Python and runtime-loaded JSON/text/config
resources in `lumen_manifest_crawler`. The manifest's closure policy is checked bidirectionally,
so both declared-file drift and unexpected behavior files fail. Explicit volatile run files do not
change the controlled digest. Separate SFT, DPO, and ORPO digests are retained under one bundle
digest, alongside the direct-dependency lock and requirements hash. A runtime-resolved distribution
manifest additionally binds every installed package, safe direct/VCS provenance, and the
behavior-bearing files declared by its `RECORD`; controlled comparisons require the same
`resolvedTrainingEnvironmentSHA256`. Unhashed self-`RECORD` rows are excluded only after canonical
containment checks. Generated bytecode is always hashed from its installed bytes and must map to one
uniquely SHA-256-attested source file. If an installer records the same generated bytecode twice—one
empty generated-file row plus one stale wheel SHA-256 row—the verifier recognizes only that exact
canonical pair, hashes the installed bytecode once, and rejects every other duplicate.

The Space README front matter is separately canonicalized and verified as Gradio with `app.py` on
Python 3.10. Unknown runtime fields, a changed entrypoint, or README-selected hardware fail before
training. ZeroGPU hardware is requested through the Hub API rather than front matter.

Runtime-source lineage keeps the expected uploaded Space revision, authenticated repository-head
observation, platform runtime observation, binding status, and method as separate fields. A
repository-head match does not prove what the container executes; when no trusted platform runtime
metadata exists the source binding remains operator-declared and unverified. Parent SFT audit
evidence, DPO frozen-reference evidence, and the new preference-training runtime are likewise kept
separate. These fields propagate into reports and finalized manifests, but promotion is still
unsupported until a trusted runtime-image attestation exists.

## Deployment Notes

- The app can use LoRA adapters differently per slot if the runtime supports it.
- If using one small base model on-device, train separate LoRA adapters per slot.
- If runtime cannot hot-swap LoRA, merge strongest common adapters or train a unified fleet adapter.
- Local training and adapter conversion do not authorize changing the shipped runtime artifact
  pointer. Runtime-image promotion remains unsupported without the separate attestation,
  compatibility, evaluation, and live-device gates.
