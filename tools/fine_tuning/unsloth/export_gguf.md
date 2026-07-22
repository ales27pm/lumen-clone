# Export GGUF

Use the automated exporter to create explicit, deterministic per-agent release-baked GGUF artifacts.

Run release bake inside the pinned training container with the same `/outputs`
mount used to prepare the run. The prepared configs bind absolute snapshot paths
and filesystem verification evidence; do not copy or rebase them onto the host.
When `--config` and `--config-dir` are omitted, release-bake mode resolves
`$LUMEN_AIO_RUN_ROOT/configs` and requires `<agent>.final.json` for every selected
agent. It never falls back to the pending/SFT `<agent>.json` config.
Before creating any output, the exporter independently verifies the canonical
completed run summary and requires full `quality_gate_passed` evaluation evidence
bound to every selected final adapter. Smoke, interrupted, and unevaluated runs
are not release-bake sources.

```bash
export LUMEN_AIO_RUN_ROOT=/outputs/<exact-run-id-and-variant>
export LUMEN_RELEASE_BAKE_ROOT=/outputs/release-bake/<new-export-id>
install -d -m 700 "$LUMEN_RELEASE_BAKE_ROOT"

/opt/lumen-venv/bin/python -m tools.fine_tuning.unsloth.export_gguf \
  --release-bake \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root "$LUMEN_RELEASE_BAKE_ROOT/models" \
  --manifest-output "$LUMEN_RELEASE_BAKE_ROOT/manifest.json" \
  --skip-upload
```

The export destination is separate from the frozen training run, and the
release-bake process receives no Hugging Face credential. Evaluate the merged
artifact, approve it independently, then publish it through a credential-scoped
verified uploader that writes a receipt. Direct token-backed upload from the
GPU/export container is not a supported release path.

## Output Naming

Each exported file is normalized to:

`lumen-<agent>-release-bake-<quantization>.gguf`

Examples:

- `lumen-cortex-release-bake-q4_k_m.gguf`
- `lumen-executor-release-bake-q4_k_m.gguf`
- `lumen-mouth-release-bake-q4_k_m.gguf`
- `lumen-mimicry-release-bake-q4_k_m.gguf`
- `lumen-rem-release-bake-q4_k_m.gguf`
- `lumen-fleet-release-bake-q4_k_m.gguf`

The generated manifest includes size and SHA256 for each artifact at the
explicit `--manifest-output` path.

Without `--release-bake`, the default source remains the checked-in
`generated/fine_tuning/<agent>/unsloth_config.json` tree and the exporter writes
only the adapter-first skipped manifest. Those portable configs intentionally do
not contain run-scoped snapshot or finalized-adapter evidence.
