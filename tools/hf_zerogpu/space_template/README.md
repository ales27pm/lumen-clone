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

The app trains LoRA adapters from one of the three controlled experiment datasets in the uploaded snapshot and pushes adapter artifacts back to the configured Hugging Face model repo. No experiment variant is preselected: the operator must select and confirm one before each run. Missing or mismatched variant manifests, datasets, immutable base-model lineage, declared container digest, or software lock stop training before model execution. The declared digest is recorded as `manual_validation_required` because Gradio ZeroGPU does not expose trusted runtime-image binding; it is not promotion evidence by itself. Merge/release-bake remains disabled by default.
