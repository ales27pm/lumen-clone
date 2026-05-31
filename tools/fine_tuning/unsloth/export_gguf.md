# Export GGUF

Use the automated exporter to create deterministic, per-agent merged GGUF artifacts.

```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --config-dir tools/fine_tuning/unsloth/configs \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

## Upload to Hugging Face

```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --config-dir tools/fine_tuning/unsloth/configs \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --hf-repo-id ales27pm/lumen-fleet-gguf \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

## Output Naming

Each exported file is normalized to:

`lumen-<agent>-merged-<quantization>.gguf`

Examples:

- `lumen-cortex-merged-q4_k_m.gguf`
- `lumen-executor-merged-q4_k_m.gguf`
- `lumen-mouth-merged-q4_k_m.gguf`
- `lumen-mimicry-merged-q4_k_m.gguf`
- `lumen-rem-merged-q4_k_m.gguf`
- `lumen-fleet-merged-q4_k_m.gguf`

The generated manifest includes size and SHA256 for each artifact:

`generated/fine_tuning/merged_gguf_manifest.json`
