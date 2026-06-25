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

The app trains LoRA adapters from the uploaded dataset snapshot and pushes adapter artifacts back to the configured Hugging Face model repo. It keeps merge/release-bake disabled by default.
