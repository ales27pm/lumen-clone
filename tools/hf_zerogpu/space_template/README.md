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

The app trains LoRA adapters from one of the three controlled experiment datasets in an immutable, commit-pinned dataset snapshot and pushes adapter artifacts back to the configured Hugging Face model repo. No experiment variant is preselected: the operator must select and confirm one before each run.

The Space, immutable dataset repository, and adapter/model repository are private by default. Public visibility requires the repository-specific `--public-space`, `--public-dataset`, or `--public-adapters` operator override. Every API request additionally requires the dedicated `LUMEN_ZERO_GPU_ADMIN_TOKEN` in `X-Lumen-Admin-Token`; the separate `LUMEN_ZERO_GPU_HUB_TOKEN` must be a fine-grained token scoped only to the required Space, dataset, and adapter repositories. Only the authorization wrapper is exposed through Gradio. It obtains the single-operation process lock before invoking the GPU-decorated trainer, and external failures return only a correlation ID and stable error code.

Fresh runs refuse to replace an existing workspace unless destructive reset is explicitly enabled. Resumes validate the self-hashed run, dataset, config, environment, code, and checkpoint lineage before any snapshot download, config rewrite, or recursive deletion. Missing or mismatched variant manifests, datasets, immutable base-model lineage, declared container digest, software dependency lock, deployed training-code digest, runtime source revision, or checkpoint evidence stop training before model execution.

The Space executes trainers as `python -m lumen_training.train_sft` and `python -m lumen_training.train_dpo`. Its code bundle covers the complete deployed `lumen_training` and `lumen_manifest_crawler` behavior closure plus `app.py` and `requirements.txt`. Preflight rejects missing or changed declared files and unexpected behavior-affecting files under covered trees; only explicit volatile run state is excluded. Preference training accepts only a fully verified finalized SFT parent, keeps parent and frozen DPO reference evidence separate, and records the DPO/ORPO runtime independently.

Runtime-source lineage separates the expected Space commit, observed repository head, observed platform runtime revision, binding method, and binding status. Repository-head equality is supplemental audit evidence and never proves which revision the running container executes. Without trusted platform runtime metadata, the binding remains explicitly unverified.

The declared container digest remains recorded as `manual_validation_required` because Gradio ZeroGPU does not expose trusted runtime-image binding; it is not promotion evidence by itself. Merge/release-bake remains disabled by default.
