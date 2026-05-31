# Improve-loop hardening — handoff

This change hardens the terminal AIO improve-loop (`tools/lumen_terminal_improve_loop.py`)
and the SFT trainer (`tools/fine_tuning/unsloth/train_sft.py`). It does **not**
touch the iOS runtime, the dataset crawler, or the export policy.

## Scope (Wave 1 + Wave 2)

### Wave 1 — blocking correctness
- Hugging Face CLI: `hf repo create` → `hf repos create` (with `--exist-ok`).
- LoRA→GGUF conversion: now requires an explicit base via either
  `--base-model-id` (default `Qwen/Qwen3-1.7B`) or `--base-model-dir`. The
  legacy implicit fallback through `adapter_config.json` is gone.
- Strict Qwen3 preflight: `validate_qwen3_configs()` rejects any agent config
  in `tools/fine_tuning/unsloth/configs_qwen3_bootstrap/` that
  - is missing,
  - references a Qwen2.x base, or
  - sets `merge_adapters_by_default` / `release_bake_enabled_by_default` to
    true. Hard-fail with `--fail-if-missing-qwen3-config` or
    `--stop-on-error`.
- `tools/check_adapter_runtime_invariants.py` now also scans the terminal
  loop, the trainer, and the bootstrap configs.
- Resumable `pipeline_state.json`: every stage logs argv, input paths,
  input hash (size + mtime), output paths, returncode, elapsed and ISO
  timestamps. With `--resume`, stages with unchanged inputs and a prior
  `ok` status are skipped.

### Wave 2 — reproducibility
- `train_sft.py --seed` (CLI > `LUMEN_TRAIN_SEED` env > config > 42),
  threaded into Python `random`, NumPy, PyTorch, Transformers, PEFT
  `random_state`, TRL `seed` and `data_seed`.
- `train_sft.py --resume-from-checkpoint` (auto-discovers latest
  `checkpoint-*` in `output_dir`).
- `train_sft.py --assistant-only-loss` (forwarded to TRL `SFTConfig`).
- Each training run writes both `training_report.json` (with metrics) and
  `train_manifest.json` (without) including: agent, base model, config
  sha256, train/val sha256, record counts, seed + seed source, git SHA,
  Python/system platform, package versions for `torch`, `transformers`,
  `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes`, `unsloth`.
- The terminal loop now forwards `--seed`, `--resume`, `--assistant-only-loss`
  to `train_sft.py` and exports `PYTHONHASHSEED` / `LUMEN_TRAIN_SEED`.

## New / changed CLI surface

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

New flags added: `--state-file`, `--resume`, `--seed`, `--assistant-only-loss`,
`--require-adapter-traces`, `--fail-if-missing-qwen3-config`, `--base-model-id`,
`--base-model-dir`, `--large-folder-upload`.

## Validation done locally

- `python tools/check_adapter_runtime_invariants.py`
- `python -m pytest tools/lumen_manifest_crawler/tests`
- `python tools/lumen_terminal_improve_loop.py --mode preflight --dry-run --skip-pytest`

## Validation still required by a human

- `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16 Pro' build`
- Device smoke test of the adapter runtime per
  `docs/ADAPTER_RUNTIME_IMPROVE_LOOP.md` §"Human local validation".

## Out of scope (deferred Wave 3)

- GitHub Actions workflows (`.github/workflows/improve-loop-guards.yml`).
- Per-role hyperparameter divergence beyond what configs already encode.
- Cross-run comparative training summaries.
- Privacy preflight (`git diff --check`, large-file scan, secret regex).
