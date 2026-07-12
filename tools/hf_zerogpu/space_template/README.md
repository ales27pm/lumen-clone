---
title: Lumen ZeroGPU Adapter Trainer
sdk: gradio
app_file: app.py
python_version: "3.10"
suggested_hardware: zero-a10g
---

# Lumen ZeroGPU Adapter Trainer

Automated Gradio Space for Lumen adapter-first fine-tuning on Hugging Face ZeroGPU.

- Space repo: `{{SPACE_REPO}}`
- Dataset repo: `{{DATASET_REPO}}`
- Adapter repo: `{{ADAPTER_REPO}}`
- ZeroGPU size: `{{GPU_SIZE}}`
- GPU duration: `{{GPU_DURATION_SECONDS}}` seconds

The app trains LoRA adapters from one of the three controlled experiment datasets in the uploaded snapshot and pushes adapter artifacts back to the configured Hugging Face model repo. The default is `internal_plus_public_optimized`; `internal_only` and `internal_plus_public_baseline` remain available for controlled comparisons. Missing or mismatched variant manifests and datasets stop training before model execution. It keeps merge/release-bake disabled by default.
