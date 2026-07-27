# AGENTS.md

## Scope

Governs `tools/hf_zerogpu/`, the builder and template for an external Hugging Face ZeroGPU Space. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This subtree assembles a separate hosted demonstration/evaluation surface from immutable model artifacts. It is not part of the iOS app runtime and cannot establish local-device compatibility.

## Key Files And Entry Points

- `build_lumen_zerogpu_space.py`: resolves/builds the Space payload.
- `space_template/app.py`: hosted application template.
- `space_template/requirements.txt`: hosted Python dependencies.
- `space_template/README.md`: generated/hosted usage context.
- `tools/hf_artifacts/publish_hf_artifacts.py`: adjacent artifact publication/resolution owner.

## Public Interfaces

Builder arguments, Space file layout, model repository/revision, visibility, dependency versions, and hosted app inputs/outputs are consumed by operators and Hugging Face Spaces.

## Internal Structure

Operator supplies artifact identity/config -> builder resolves immutable revisions and template -> emits Space payload -> optional external publication occurs through explicit HF tooling -> operator confirms revision/visibility/runtime.

## Incoming Dependencies

Lineage-bound model artifacts and explicit operator configuration feed the builder.

## Outgoing Dependencies

The output targets Hugging Face Hub/Spaces and its ZeroGPU runtime. Network/API credentials are external and must not enter generated source.

## Data And Control Flow

Artifact manifest/revision -> validation/resolution -> template assembly -> local payload -> explicit upload/deploy -> remote status confirmation. A built payload is not a deployed/running Space.

## Local Invariants

- Resolve immutable model/artifact revisions and record hashes.
- Keep upload/deploy opt-in and confirm repository visibility explicitly.
- Never write tokens into template, README, logs, or manifests.
- Keep hosted capability claims separate from iOS local runtime claims.
- Treat remote build/runtime failure as explicit; do not infer success from file creation.

## Coordinated Changes

Template dependency/API changes require builder, requirements, README, artifact manifest, and remote compatibility review. Model input changes require Unsloth/HF lineage and iOS compatibility documentation when claims cross surfaces.

## Safe Editing Rules

Prefer local payload generation before any network action. Pin dependencies/revisions intentionally. Do not auto-create or overwrite a remote Space without explicit user direction.

## Validation

From the repository root:

```bash
python3 -m compileall tools/hf_zerogpu
```

No dedicated checked-in ZeroGPU unit tests were found. Validate builder help/dry-run or generated payload only when the current CLI explicitly supports a non-network mode; do not invent flags or deploy during routine validation.

## Common Failure Modes

- A branch/default revision moves after validation.
- Local payload generation is reported as remote deployment.
- Hosted success is reported as on-device runtime proof.
- A token leaks into generated Space files.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Training lineage follows [`../fine_tuning/unsloth/AGENTS.md`](../fine_tuning/unsloth/AGENTS.md). No child guidance is needed.
