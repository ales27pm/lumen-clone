# Export GGUF

Use the automated exporter to create explicit, deterministic per-agent release-baked GGUF artifacts.

```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --release-bake \
  --config-dir "$LUMEN_AIO_RUN_ROOT/configs" \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

## Upload to Hugging Face

```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --release-bake \
  --config-dir "$LUMEN_AIO_RUN_ROOT/configs" \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --hf-repo-id ales27pm/lumen-fleet-gguf \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

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

The generated manifest includes size and SHA256 for each artifact:

`generated/fine_tuning/merged_gguf_manifest.json`
