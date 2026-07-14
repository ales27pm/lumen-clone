# ZeroGPU training-lineage hardening

This hardening pass closes independent gaps in the adapter-training audit chain.

| Root cause | Enforced invariant |
| --- | --- |
| A mutable Hub dataset revision could be recopied over a resumable workspace. | Every real run uses the immutable dataset upload commit. Resume validates the existing self-hashed run and checkpoint lineage before any download, copy, deletion, config write, or training operation. |
| Preference rows were flattened before TRL could apply the pinned tokenizer chat template. | DPO and ORPO receive validated conversational message lists. Missing, malformed, empty, or equivalent preferences fail closed; no fallback prompt or completion is synthesized. |
| The GPU-decorated Gradio function was directly addressable and shared the repository credential boundary. | A non-GPU wrapper authenticates a dedicated administrative secret before invoking a single-operation GPU worker. Hub access uses a separate repository-scoped token, conflicting calls fail deterministically, and external errors contain only stable codes and correlation IDs. |
| Environment lineage omitted the deployed trainer/finalizer bytes, Space/source revision, and part of the direct dependency set. | A sorted phase-specific code manifest and a complete direct-dependency lock are verified before model loading. Their digests flow through run, checkpoint, training, finalization, evaluation, and comparison evidence; the runtime source commit remains separate audit evidence. |
| `trl==0.24.0` was paired with `transformers==5.5.0`, whose changed optional-package probe made TRL treat absent integrations as installed and prevented DPO/ORPO trainer imports. | The coherent lock uses `transformers==4.57.6`, which is within the pinned Unsloth revision's supported range and preserves TRL 0.24's expected probe contract. A pinned-package API test binds both trainer constructor signatures. |
| The code digest covered a curated subset while the Space copied the complete crawler package. | The controlled bundle discovers the full deployed `lumen_training` and `lumen_manifest_crawler` behavior closure plus the Space app and requirements. Its closure policy is verified in both directions, so missing, changed, or newly introduced covered files fail preflight. SFT, DPO, and ORPO retain separate phase digests under one bundle digest. |
| The Space renamed flat trainer files while preference training imported the original module name. | The deployed package has stable `lumen_training.train_sft` and `lumen_training.train_dpo` module entrypoints and package-relative shared-helper imports. Clean-Space execution does not depend on repository paths. |
| Dataset and adapter repositories could become public through omitted opt-in privacy flags. | Space, dataset, and adapter/model repositories are private by default. Visibility changes require the repository-specific `--public-*` override, and a public Space still requires the independent admin secret. |
| A syntactically valid expected Space SHA was treated as if it identified the executing container. | Expected source revision, observed repository head, observed platform runtime revision, binding status, and binding method are distinct lineage fields. Repository-head equality never upgrades the source binding; without trusted platform runtime metadata it remains operator-declared and unverified. |
| Preference training validated only part of its SFT parent lineage. | One finalized-parent boundary checks manifest integrity, artifact bytes, adapter base-model identity, seed, variant/source manifest, complete model shard contract, environment/dependency/requirements locks, runtime kind, and SFT code digest. Parent, frozen DPO reference, and preference-runtime evidence remain distinct. |

The complete code closure includes all covered Python package initializers and transitively imported
helpers, runtime-loaded evaluation source/fingerprint JSON, trainer and artifact-lineage modules,
`app.py`, and `requirements.txt`. Its explicit policy excludes only the generated defaults/run
manifests and volatile `.git`, log, checkpoint, output, upload, `__pycache__`, and bytecode state.
Credentials remain environment secrets and are never deployed as files. Adding an unlisted
behavior-affecting file to a covered tree fails just as a declared-file mutation does.

The three Hub repository boundaries are independent. `--public-space`, `--public-dataset`, and
`--public-adapters` each alter only the named repository; omission keeps all three private. Endpoint
authorization and repository credentials remain separate regardless of repository visibility.

The operator-declared container digest is still not a trusted runtime-image attestation. Code and
dependency hashing make training logic reproducible, but they do not prove which container image
the platform executed. Automated promotion therefore remains unsupported until an independently
verifiable platform attestation is available.
