# Hugging Face Artifact Workflow

Lumen uses a Hugging Face-first artifact workflow.

GitHub is for source code, manifests, dataset generators, tests, and small JSON metadata. Hugging Face is the registry for heavy runtime artifacts.

## Canonical flow

```text
local PC improve-loop
→ generate datasets
→ fine-tune adapters
→ evaluate gates
→ optionally release-bake GGUF/CoreML artifacts
→ upload selected artifacts to Hugging Face
→ update app/catalog manifest metadata
→ app downloads artifacts from Hugging Face
→ TestFlight/on-device runtime audit exports results
→ next local improve-loop cycle ingests runtime exports
```

## What stays in GitHub

```text
source code
training scripts
improve-loop scripts
dataset generators
small dataset cards
model catalog metadata
artifact manifests
checksums
Hugging Face repo IDs
Hugging Face file names
```

## What does not go into GitHub

```text
base GGUF/CoreML model binaries
LoRA/adaptor binaries
merged/release-baked model binaries
training checkpoints
large generated inference artifacts
```

Those are published to Hugging Face instead.

## Existing app direction

The app catalog already follows this structure. `CatalogModel` stores:

```text
repoId
fileName
```

and builds the download URL as:

```text
https://huggingface.co/<repoId>/resolve/main/<fileName>?download=true
```

So the intended runtime path remains:

```text
Models screen → Download → Hugging Face artifact → local app storage → load model/adaptor
```

## Local PC role

Running locally does not mean artifacts are local-only.

Your PC is the workstation that executes:

```text
dataset generation
fine-tuning
evaluation
artifact preparation
Hugging Face upload
```

After that, the app downloads from Hugging Face using the catalog/manifest metadata.

## Recommended HF repos

Use separate repos to avoid mixing base models, adapters, and release-baked outputs:

```text
ales27pm/lumen-agent-base
ales27pm/lumen-agent-adapters
ales27pm/lumen-embedding
ales27pm/lumen-release-gguf
ales27pm/lumen-runtime-manifests
```

Suggested contents:

```text
lumen-agent-base/
└── qwen3-1.7b/<base model files>

lumen-agent-adapters/
├── cortex/<adapter files>
├── executor/<adapter files>
├── mouth/<adapter files>
├── rem/<adapter files>
├── mimicry/<adapter files>
└── fleet/<adapter files>

lumen-embedding/
└── qwen3-embedding-0.6b/<embedding model or adapter files>

lumen-release-gguf/
└── release-candidates/<merged/baked artifacts if needed>

lumen-runtime-manifests/
└── artifact_manifest.json
```

## Artifact manifest contract

The app and loop should exchange a small manifest like:

```json
{
  "schemaVersion": "1.0.0",
  "artifactRegistry": "huggingface",
  "baseModel": {
    "repoId": "ales27pm/lumen-agent-base",
    "fileName": "qwen3-1.7b/base.gguf",
    "sha256": "...",
    "role": "agentBase"
  },
  "embedding": {
    "repoId": "ales27pm/lumen-embedding",
    "fileName": "qwen3-embedding-0.6b/model.gguf",
    "sha256": "...",
    "role": "embedding"
  },
  "adapters": {
    "cortex": {
      "repoId": "ales27pm/lumen-agent-adapters",
      "fileName": "cortex/adapter.safetensors",
      "sha256": "..."
    }
  }
}
```

## Adapter-first policy

Default policy:

```text
one base model
multiple role-specific adapters
no merge phase by default
upload adapters to HF
app downloads base + adapter artifacts
release-bake only when adapter runtime is impossible or for selected release candidates
```

## Release-bake policy

Only bake/merge when:

```text
adapter runtime cannot load reliably on device
release candidate requires simpler runtime packaging
performance improves enough to justify larger artifacts
```

Baked outputs still go to Hugging Face, not GitHub.

## Required local secrets

Do not commit tokens. Store them locally:

```bash
export HF_TOKEN="..."
```

or use:

```bash
huggingface-cli login
```

## GitHub hygiene

The repo ignores generated artifact directories such as:

```text
models/base/
models/adapters/
models/embedding/
models/release/
artifacts/
checkpoints/
generated/fine_tuning/checkpoints/
```

This keeps GitHub fast and source-only while preserving the original HF download structure.
