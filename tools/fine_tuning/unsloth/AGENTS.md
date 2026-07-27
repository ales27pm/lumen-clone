# AGENTS.md

## Scope

Governs `tools/fine_tuning/unsloth/`: controlled Ubuntu/CUDA container definitions, SFT/DPO training, evaluation, GGUF export/merge, runtime-binding gates, lineage/resume logic, role configs, and unit tests. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree consumes frozen generated datasets/configs and produces evaluated, lineage-bound model adapters/artifacts. It is a heavyweight external/GPU pipeline; successful training is not proof that the iOS app loaded or used the artifact.

## Key Files And Entry Points

- `Dockerfile.ubuntu-cu128`: controlled CUDA/Ubuntu environment.
- `train_sft.py`, `train_dpo.py`: training entry points.
- `ubuntu_pipeline.py`, `ubuntu_postcondition.py`, `ubuntu_source_integrity.py`, `ubuntu_uploader.py`: orchestration, completion, source, and optional upload controls.
- `training_lineage.py`, `adapter_artifact.py`: immutable source/model/tokenizer/dataset/artifact evidence.
- `evaluate_adapter.py`, `runtime_binding_smoke_gate.py`: quality/runtime-binding gates.
- `merge_lora.py`, `export_gguf.py`, `export_gguf.md`: artifact conversion.
- `configs/` and `configs_qwen3_bootstrap/`: role-specific controlled inputs.
- `tests/`: resume, lineage, tokenizer, dataset caps, completion, Ubuntu, and runtime-binding coverage.
- `scripts/ubuntu_train_lumen_full_pipeline.sh`: repository launcher.

## Public Interfaces

Config keys, CLI arguments, checkpoint/resume state, lineage schema, completion evidence, artifact paths/hashes, evaluation metrics, upload visibility/revision, and runtime-binding reports are consumed by operators, HF tooling, generated artifact docs, and release decisions.

## Internal Structure

Validate source/image/model/tokenizer/dataset/config identity -> prepare exact role rows -> train/resume with recorded sampler/tokenizer state -> evaluate -> merge/export -> verify hashes/runtime binding -> optionally upload under explicit policy -> emit completion lineage.

## Incoming Dependencies

Generated fine-tuning datasets/manifests, public corpus provenance, pinned base-model revisions, controlled container image/source snapshot, and explicit operator configuration feed the pipeline.

## Outgoing Dependencies

Local/GPU storage, Docker/CUDA/Unsloth/Transformers stack, artifact manifests, optional private Hugging Face upload, and downstream iOS conversion/runtime verification.

## Data And Control Flow

Frozen inputs -> preflight/integrity -> SFT/DPO -> checkpoint -> evaluation -> artifact/export -> runtime-binding smoke gate -> immutable lineage -> optional upload. Resume is allowed only when saved lineage/tokenizer/sampler/config matches current inputs.

## Local Invariants

- Record immutable source commit/snapshot, container image, base model and revision, tokenizer identity, dataset/config hashes, seed, and output hashes.
- Do not resume across mismatched lineage or reconstruct missing completion evidence.
- Keep role datasets/configs and exact row caps/splits deterministic.
- Upload is off and private by default unless the operator explicitly requests another policy and confirms visibility.
- Credentials exist only in the isolated upload boundary and never in logs/manifests.
- Training/evaluation/export success does not establish iOS deployment or live model evidence.
- One deployed role adapter must match the shared base/tokenizer/runtime contract.

## Coordinated Changes

Training config/schema changes require crawler-generated inputs, contamination/eval gates, relevant unit tests, lineage/resume handling, artifact docs, and iOS runtime compatibility review. Export changes require model loader/catalog, SwiftLlama/GGUF compatibility, hash/revision records, and live runtime validation.

## Safe Editing Rules

Do not run heavyweight training/upload as routine validation. Keep preflight before GPU allocation. Preserve checkpoints on failure. Fail closed on missing/mismatched lineage. Avoid downloading mutable `latest` artifacts; resolve immutable revisions.

## Validation

From the repository root:

```bash
python3 -m compileall tools/fine_tuning/unsloth
uv run --python 3.12 --with pytest pytest tools/fine_tuning/unsloth/tests/test_training_lineage.py tools/fine_tuning/unsloth/tests/test_runtime_binding_smoke_gate.py tools/fine_tuning/unsloth/tests/test_ubuntu_pipeline.py
```

Run broader affected tests under `tools/fine_tuning/unsloth/tests/` when dependencies are available. Run `bash scripts/ubuntu_train_lumen_full_pipeline.sh` only for an explicitly requested controlled training run.

## Common Failure Modes

- Resume uses a checkpoint from different data/tokenizer/source.
- A mutable model revision makes an artifact irreproducible.
- Upload succeeds but visibility/revision/hash is not confirmed.
- Evaluation result is reported as deployed iOS behavior.
- A conversion changes tokenizer/chat-template/runtime compatibility.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Generated inputs: [`../../../generated/AGENTS.md`](../../../generated/AGENTS.md). Public corpus: [`../../../datasets/AGENTS.md`](../../../datasets/AGENTS.md). No child file is needed.
