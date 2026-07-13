# ZeroGPU training-lineage hardening

This hardening pass closes four independent gaps in the adapter-training audit chain.

| Root cause | Enforced invariant |
| --- | --- |
| A mutable Hub dataset revision could be recopied over a resumable workspace. | Every real run uses the immutable dataset upload commit. Resume validates the existing self-hashed run and checkpoint lineage before any download, copy, deletion, config write, or training operation. |
| Preference rows were flattened before TRL could apply the pinned tokenizer chat template. | DPO and ORPO receive validated conversational message lists. Missing, malformed, empty, or equivalent preferences fail closed; no fallback prompt or completion is synthesized. |
| The GPU-decorated Gradio function was directly addressable and shared the repository credential boundary. | A non-GPU wrapper authenticates a dedicated administrative secret before invoking a single-operation GPU worker. Hub access uses a separate repository-scoped token, conflicting calls fail deterministically, and external errors contain only stable codes and correlation IDs. |
| Environment lineage omitted the deployed trainer/finalizer bytes, Space/source revision, and part of the direct dependency set. | A sorted phase-specific code manifest and a complete direct-dependency lock are verified before model loading. Their digests flow through run, checkpoint, training, finalization, evaluation, and comparison evidence; the runtime source commit remains separate audit evidence. |
| `trl==0.24.0` was paired with `transformers==5.5.0`, whose changed optional-package probe made TRL treat absent integrations as installed and prevented DPO/ORPO trainer imports. | The coherent lock uses `transformers==4.57.6`, which is within the pinned Unsloth revision's supported range and preserves TRL 0.24's expected probe contract. A pinned-package API test binds both trainer constructor signatures. |

The operator-declared container digest is still not a trusted runtime-image attestation. Code and
dependency hashing make training logic reproducible, but they do not prove which container image
the platform executed. Automated promotion therefore remains unsupported until an independently
verifiable platform attestation is available.
