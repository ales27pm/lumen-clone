# AGENTS.md

## Repository Purpose

Lumen is a native SwiftUI iOS application that runs a local-first AI assistant. The shipping runtime combines an on-device language-model path, a structured agent kernel, permission-gated native tools, SwiftData-backed conversations/memory/RAG, voice and background execution, diagnostics, and an optional Microsoft Graph integration. This repository also contains the Python manifest/dataset generator, model-training and artifact-publication tooling, validation scripts, generated grounding/training artifacts, and historical runtime evidence.

The iOS app is the only product runtime found in the tracked tree. The Python and shell programs are build, validation, generation, training, release, or operational tools; they are not an application backend.

## Instruction Scope And Precedence

- This file governs the whole repository.
- A nested `AGENTS.md` refines this guidance for its directory and descendants. The nearest applicable file takes precedence for local details.
- Explicit user instructions override repository guidance. System and platform safety constraints still apply.
- Treat executable configuration and current implementation as stronger evidence than prose. When code, Xcode settings, scripts, and documentation disagree, investigate the discrepancy rather than repeating a stale claim.
- Do not infer shipped capability from generated artifacts, historical audits, build success, or test fixtures.

## Repository Map

| Path | Ownership | Local guidance |
| --- | --- | --- |
| `.github/` | Pull-request and push workflows for integration and MSAL linkage checks | [`.github/AGENTS.md`](.github/AGENTS.md) |
| `assets/` | Checked-in source images and visual assets; no generator was identified for the top-level PNGs | Inherits this file |
| `codebase_txt_chunks/` | Generated or historical codebase text snapshots; not authoritative source | Inherits this file |
| `datasets/` | Public adapter corpus records, provenance, licenses, and hashes | [`datasets/AGENTS.md`](datasets/AGENTS.md) |
| `docs/` | Architecture, runtime, validation, artifact, training, and release documentation | [`docs/AGENTS.md`](docs/AGENTS.md) |
| `evals/` | Small checked-in evaluation inputs/configuration consumed by tooling | Inherits this file and `tools/AGENTS.md` when changing consumers |
| `generated/` | Regenerable manifests, datasets, reports, prompts, training inputs, and visualizations | [`generated/AGENTS.md`](generated/AGENTS.md) |
| `ios/` | Xcode project, Swift app, unit tests, UI tests, scheme, entitlements, and bundled resources | [`ios/AGENTS.md`](ios/AGENTS.md) |
| `lumen_manifest_crawler/` | Compatibility import shim that redirects to the package under `tools/` | Inherits this file; implementation guidance is in the crawler file below |
| `runtime-audits/` | Historical runtime/evidence exports; snapshots are not current proof | [`runtime-audits/AGENTS.md`](runtime-audits/AGENTS.md) |
| `scripts/` | Local gates, simulator runners, generation sync, release/upload, training launchers, and exceptional repair/installer scripts | [`scripts/AGENTS.md`](scripts/AGENTS.md) |
| `tools/` | Python hardening checks, crawler/generator, training, Hugging Face, classifier, and artifact tooling | [`tools/AGENTS.md`](tools/AGENTS.md) |

Ignored local directories such as `build/`, `.cache/`, `.local/`, `.venv/`, `.venv-hf-zerogpu/`, `exports/`, and local model storage are caches, environments, build products, exports, or large machine-local artifacts. Do not crawl, edit, clean, or commit them unless the task explicitly targets them.

## Architecture

### Runtime topology and entry points

- `ios/Lumen/LumenApp.swift` is the `@main` SwiftUI entry point. It constructs the SwiftData `ModelContainer`, installs it in `SharedContainer`, starts bounded bootstrap work through `AppStartupCoordinator`, and coordinates scene transitions and background registration.
- `ios/Lumen/LumenAppDelegate.swift` connects application lifecycle callbacks, MetricKit, and Microsoft authentication URL callbacks.
- UI entry points under `ios/Lumen/Views/`, App Intents, voice, CarPlay, and headless diagnostics submit `AgentKernelRequest` values to `AssistantKernel`; they must not create alternate production agent loops.
- `ios/Lumen/Assistant/AssistantKernel+Streaming.swift` and `StructuredAgentKernelExecutor.swift` own the model-to-tool-to-final-answer loop. Tool calls cross `SecureToolRegistry` only after schema, availability, policy, permission, and approval checks.
- `ios/Lumen/Services/LlamaService.swift` owns the shared SwiftLlama/llama.cpp context and role-adapter activation. `ModelLoader.swift` loads models on demand; app startup and background maintenance must not eagerly load them.
- SwiftData persistence is split between model declarations in `ios/Lumen/Models/` and stores/services such as `MemoryStore`, `RAGStore`, and `VectorIndex` under `ios/Lumen/Services/`.
- Voice and CarPlay route recognized text back through the same kernel. Background tasks perform bounded maintenance and trigger scanning, not prompt generation.
- Microsoft Graph is an optional external boundary implemented under `ios/Lumen/Services/MicrosoftGraph/`; local assistant operation must not depend on Graph availability.

### Build-time and evidence topology

- `tools/lumen_manifest_crawler/` crawls Swift source and emits deterministic manifests, prompts, datasets, audits, and developer-cycle products under `generated/`.
- `scripts/sync-agent-manifest-resource.sh` mirrors the deterministic generated behavior manifest into `ios/Lumen/AgentBehaviorManifest.json`. `ios/Lumen/Scripts/copy_agent_grounding_resources.sh` verifies/copies grounding resources into the built app.
- The Xcode scheme runs the grounding-resource copy script after build. Generated resources are inputs to runtime grounding, not proof that a live model or tool run succeeded.
- `tools/fine_tuning/unsloth/` and `scripts/ubuntu_train_lumen_full_pipeline.sh` form the controlled Ubuntu/GPU training path. Training outputs require explicit lineage, tokenizer, revision, hash, and runtime-binding evidence before they can be associated with the app.
- `runtime-audits/` and generated improve-loop reports are evidence products with freshness limits. `docs/ARTIFACT_STATUS.md` defines their interpretation.

### Dependency direction

- UI, App Intents, voice, CarPlay, and developer surfaces depend inward on kernel contracts; the kernel does not depend on view implementations.
- The kernel depends on runtime adapters, grounding, stores, and secure tool interfaces. Native tools depend on permission and Apple framework adapters, not on UI.
- Persistence logic depends on `@Model` types; model types do not call stores or views.
- Background and scene coordinators may cancel or defer runtime work but must not become alternate inference owners.
- Python tooling reads source/configuration and writes generated artifacts. Production Swift must not import Python outputs except through explicitly bundled resource formats.
- Tests may use internal app symbols through the test target. Production source must not depend on test fixtures or test-only deterministic behavior.

### Important control flows

1. User input: `ChatView` or another surface creates a kernel request -> `AssistantKernel` builds context -> the runtime streams model events -> structured actions are validated -> `SecureToolRegistry` executes approved native tools -> trusted observations feed finalization -> sanitized text and metadata are persisted.
2. Memory/RAG: stores read SwiftData -> embedding uses the shared local service -> retrieval combines semantic and lexical evidence -> context builders pass bounded context to the kernel. Empty results and operational failures remain distinct.
3. Voice: explicit permission -> audio/recognition lifecycle -> transcript -> `VoiceCommandRouter` -> `AssistantKernel` -> event reducer -> synthesis. Cancellation and audio interruptions terminate active work.
4. Background: `TriggerScheduler` registers exact BGTask identifiers -> `BackgroundOrchestrator` acquires bounded leases -> maintenance/trigger scans run under resource policy -> expiration cancels work. No model load or prompt generation belongs in this flow.
5. Generation: crawler reads current Swift/tool/runtime definitions -> canonical artifacts are written under `generated/` -> sync/copy scripts verify the app resource mirror -> runtime auditors compare live registration to the bundled manifest.

### Authoritative sources

| Concern | Source of truth |
| --- | --- |
| Xcode targets, packages, build settings, synchronized groups | `ios/Lumen.xcodeproj/project.pbxproj` and `ios/Lumen.xcodeproj/xcshareddata/xcschemes/Lumen.xcscheme` |
| App startup and SwiftData schema registration | `ios/Lumen/LumenApp.swift` |
| Kernel request/event contract | `ios/Lumen/Assistant/AgentKernelContracts.swift` |
| Production runtime routing | `ios/Lumen/Assistant/AssistantRuntimeRouter.swift` and `ios/Lumen/Services/LLM/LLMEngineFactory.swift` |
| Structured action validation/execution | `ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift`, `ios/Lumen/Tools/ToolSchemaBridge.swift`, and `ios/Lumen/Tools/ToolRegistry.swift` |
| Canonical tool catalog and IDs | `ios/Lumen/Models/ToolDefinition.swift`, `ios/Lumen/Tools/ToolID.swift`, and `ios/Lumen/Services/ToolExecutor.swift` |
| Persistent model declarations | `ios/Lumen/Models/MemoryItem.swift`, `RAGChunk.swift`, `Conversation.swift`, `ChatMessage.swift`, and other `@Model` files registered by `LumenApp.swift` |
| Generated artifact status and freshness | `docs/ARTIFACT_STATUS.md` |
| Python package metadata | `tools/lumen_manifest_crawler/pyproject.toml` |
| Current validation claims | `FEATURE_COMPLETE_VALIDATION.md`, `docs/VALIDATION.md`, and current runtime status documents, checked against code |

## Cross-Cutting Invariants

- Release runtime routing fails closed. DEBUG-only deterministic, unavailable, experimental, or compatibility backends must not become selectable in Release.
- Structured actions are untrusted model output. Canonicalize the tool ID, require enabled registration, reject unknown/extra/wrongly typed arguments, validate enums and required fields, apply bounded repair, then enforce policy, permission, and approval before execution.
- Do not add fake, placeholder, staged-as-shipped, or empty-success production behavior. Preserve typed distinctions among empty data, unavailable capability, permission denial, cancellation, persistence failure, corrupt state, and runtime failure.
- Do not log raw prompts, documents, memory, tool arguments, tokens, OAuth material, or personal data. Normal diagnostics use hashes, counts, categories, timing, and redacted metadata.
- Model loading is on demand. App launch, scene changes, background maintenance, diagnostics discovery, and headless runner construction must not load or prompt the model.
- Keep the shared llama context and adapter lifecycle serialized by their actor. Heavy inference/load work stays off the main actor; UI, SwiftData contexts, permissions, audio lifecycle, and scene state obey their explicit main-actor ownership.
- Background work is bounded, cancellable, resource-gated, and non-interactive. It must not silently request permissions, approvals, or user input.
- Generated manifests and historical evidence do not establish live runtime success. Compile, static audit, simulator test, model-backed run, device run, signed archive, and App Store upload are separate evidence layers.
- Generated outputs are changed through their generator and reviewed as a coherent scope. Do not hand-edit canonical JSONL, SHA sidecars, resource mirrors, or training lineage files.
- Preserve local-first operation. Optional network integrations must expose typed unavailability and must not become required for chat, memory, RAG, or local tools.
- Do not weaken tests, gates, privacy, approval, evidence, or Release routing to make a check pass.

## Change Impact Matrix

| Change | Inspect and coordinate |
| --- | --- |
| Kernel request/events or streaming semantics | `ios/Lumen/Assistant/`, `Views/ChatKernelEventReducer.swift`, `Voice/VoiceKernelEventReducer.swift`, `HeadlessAgentKernelRunner.swift`, E2E runner, and kernel/reducer tests |
| Tool ID, schema, alias, argument, or result | `Models/ToolDefinition.swift`, `Tools/ToolID.swift`, `Tools/ToolSchemaBridge.swift`, `Tools/ToolRegistry.swift`, `Services/ToolExecutor.swift`, generated manifest/datasets, grounding resource mirror, and tool contract tests |
| Runtime backend, model format, or adapter policy | `AssistantRuntimeRouter.swift`, `Services/LlamaService.swift`, `Services/ModelLoader.swift`, `Services/LLM/`, Xcode Swift package pin, hardening checks, model/runtime tests, and runtime status docs |
| Persistent `@Model` shape | `ios/Lumen/Models/`, the schema in `LumenApp.swift`, owning store, export/diagnostic code, fixtures, and `PersistenceAuditTests.swift`. No versioned migration plan was found; do not assume automatic compatibility is sufficient |
| Memory or RAG embedding identity/dimension | `Memory/`, `RAG/`, `Services/MemoryStore.swift`, `RAGStore.swift`, `VectorIndex.swift`, model metadata, stale-index behavior, and retrieval tests |
| Permission domain or sensitive native action | `Permissions/`, corresponding `Tools/Builtin/` adapter, approval policy, entitlements/Info settings when applicable, privacy UI/diagnostics, and permission/tool policy tests |
| Background task identifier or policy | `Services/TriggerScheduler.swift`, `Background/`, `LumenApp.swift`, project entitlements/configuration, and background registration/no-model-load tests |
| Voice/audio lifecycle | `Voice/`, `Services/VoiceService.swift`, app cancellation/scene coordination, privacy usage descriptions, reducers/views, and voice lifecycle/audio tests |
| Microsoft Graph auth or mail API | `Services/MicrosoftGraph/`, callback handling in `LumenAppDelegate.swift`, Xcode URL/config settings, Outlook tools/views, cache/sign-out behavior, and MSAL linkage workflow |
| Behavior manifest or grounding format | crawler source/writer, `generated/agent_manifest/`, sync/copy scripts, `ios/Lumen/AgentBehaviorManifest.json`, `Services/AgentGrounding/`, and grounding regression tests |
| Training dataset/config/output | crawler dataset/fine-tuning modules, `datasets/`, `generated/fine_tuning/`, Unsloth configs/tests, contamination/evaluation gates, and artifact/runtime-binding lineage |
| Release/upload mechanics | Xcode build number/settings, archive/upload scripts, entitlements, signing/export configuration, release docs, and manual App Store/device evidence |

## Development Environment

- Host: macOS with Xcode capable of the iOS 18.0 deployment target. The project uses Swift 5 and Swift Package Manager through Xcode.
- Python: the crawler declares Python `>=3.11`; `.python-version` currently records `3.11.9`; repository validation and CI-style commands intentionally use `uv --python 3.12`.
- Shell: scripts use Bash even when the interactive shell is zsh. Invoke repository scripts with `bash path/to/script.sh` unless their documentation requires another interpreter.
- No checked-in CocoaPods, npm, or top-level SwiftPM package manifest was found. Do not add a package manager merely to run existing work.
- Never place credentials, OAuth secrets, signing material, Hugging Face tokens, or model licenses in tracked files.

## Build, Test, Lint, And Validation

Run from the repository root unless a command starts with `cd`.

### Fast local checks

```bash
git diff --check
python3 -m compileall tools scripts
python3 tools/check_release_hardening.py
python3 tools/check_agent_kernel_boundary.py --strict
python3 tools/check_adapter_runtime_invariants.py
python3 tools/check_ios_lora_hardening_invariants.py
python3 scripts/check-generated-jsonl-artifacts.py
bash scripts/check-lumen-integration-gate.sh
```

These respectively cover patch sanity, Python syntax, Release routing/static policy, kernel ownership, adapter runtime rules, iOS LoRA/runtime pinning, generated JSONL structure, and the repository integration gate. Do not substitute one for another.

### Python package checks

```bash
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"
cd tools/lumen_manifest_crawler && uv run --python 3.12 --with pytest pytest --collect-only
```

The first command runs the non-slow/non-E2E package suite from the root. The second verifies collection from the package directory. Remove `tools/lumen_manifest_crawler/uv.lock` if `uv` created it only as an unrequested validation byproduct.

### Stable iOS compile checkpoint

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  build-for-testing \
  CODE_SIGNING_ALLOWED=NO
```

Use the dedicated simulator when it exists. Build first; reuse its `.xctestrun` with the bounded focused runner for repeated tests.

### Focused and full simulator execution

```bash
bash scripts/run_focused_simulator_tests.sh
```

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  test \
  CODE_SIGNING_ALLOWED=NO
```

Run executed XCTest only when required and CoreSimulator is healthy. A successful `build-for-testing` is compile evidence, not executed-test evidence. Treat install, launch, or test-manager stalls as environment boundaries rather than silently reporting success.

### Release validation

Static and simulator checks do not prove signed archive/export, entitlement payload, privacy manifest handling, TestFlight behavior, real-device model load, live tools/RAG/memory/voice/App Intent flows, or App Store acceptance. Record these as manual gaps unless fresh evidence exists. Run `bash scripts/build_and_submit_appstoreconnect.sh` only when the user explicitly requests submission.

## Generated, Vendored, And Restricted Files

- Do not edit `generated/` outputs directly. Change the crawler/generator, regenerate the intended complete scope, and review provenance/hash changes.
- Do not edit `ios/Lumen/AgentBehaviorManifest.json` directly; it is the app mirror managed by `scripts/sync-agent-manifest-resource.sh`.
- Do not hand-edit `datasets/public_adapter_corpus/records.jsonl`; regenerate it from the crawler's public-corpus builder and review its manifest, licenses, attribution, revisions, and hashes together.
- `generated/fine_tuning/` is the canonical fine-tuning output root. `generated/agent_manifest/fine_tuning/` is a forbidden stale duplicate location, not a second source of truth.
- `runtime-audits/`, generated improve-loop reports, and `codebase_txt_chunks/` are snapshots. Do not use them as substitutes for current source or fresh runtime evidence.
- Generated visual resources and bundled `p5.min.js` are marked as generated/vendored in `.gitattributes`; edit their generator or upstream source rather than minified output.
- Xcode/SwiftPM build products, DerivedData, virtual environments, caches, local models, exports, and credentials are machine-local and must remain untracked.
- No tracked SwiftData migration directory or tracked privacy manifest was found. Treat both as explicit verification gaps rather than inventing paths or guarantees.

## Security And Sensitive Areas

- `ios/Lumen/Permissions/`, `ios/Lumen/Tools/`, `ios/Lumen/Services/MicrosoftGraph/`, persistent stores, diagnostics exporters, entitlements, and release scripts are security-sensitive.
- Preserve least privilege, explicit user initiation, approval-before-side-effect, secure cache protection, sign-out purge, and redacted telemetry.
- OAuth state/verifier/token material belongs in the Keychain or process-local protected state, never logs, fixtures, generated datasets, or AGENTS files.
- Tool output is untrusted external/local data. Bound its size and keep observations separate from model-authored final text.
- Do not expose secret values while documenting configuration. Name variables/keys only when already public and necessary.

## Git And Pull Request Guidance

- Preserve unrelated user changes and untracked files. Stage explicit paths in a mixed worktree.
- Do not commit generated manifests, datasets, audit exports, lockfiles, or validation byproducts unless they are part of the requested coherent deliverable.
- Run `git diff --check` before commit or handoff. Add component-specific checks in proportion to risk.
- Do not trigger or rerun GitHub Actions unless explicitly requested. A push to a workflow-matched branch may itself consume Actions; ask before pushing when that matters.
- Draft pull requests are the default when a PR is requested. State changed contracts, local evidence, unexecuted checks, manual device/release gaps, and generated scopes.
- Do not equate a clean static gate with live, device, model-backed, signed, or uploaded validation.

## Known Hazards

- `scripts/archive_lumen_stable.sh` and release helpers can patch project/linker settings while archiving; they are not read-only validation commands.
- `scripts/tripleboot_aio.sh` is a separate destructive removable-media installer utility. Never run it as part of Lumen validation and never infer a target device.
- Hotfix/repair scripts may modify production source or Xcode settings. Inspect them before execution and run only for an explicit repair task.
- DEBUG-only fallback code can make local tests look healthy while Release has no valid backend. Always inspect conditional compilation and Release routing.
- Historical migration documents can describe superseded mixed/legacy paths. Current implementation and current status matrices take precedence.
- Python compatibility is `>=3.11`, but reproducible repository validation uses 3.12; switching interpreters can create misleading environment-only failures.
- App Store upload tools can exit without making a successful upload. Only terminal output containing an error-free success and Delivery UUID is authoritative.
- Synchronized Xcode groups reduce manual project-file membership edits, but build phases, packages, entitlements, configuration, and non-source resources still require project review.

## Nested AGENTS.md Index

- [`.github/AGENTS.md`](.github/AGENTS.md): workflow triggers and local-vs-hosted validation boundary.
- [`ios/AGENTS.md`](ios/AGENTS.md): Xcode, Swift, lifecycle, project, and test rules.
- [`ios/Lumen/Assistant/AGENTS.md`](ios/Lumen/Assistant/AGENTS.md): production kernel and structured execution.
- [`ios/Lumen/AppIntents/AGENTS.md`](ios/Lumen/AppIntents/AGENTS.md): Shortcuts/App Intent policy and headless routing.
- [`ios/Lumen/Background/AGENTS.md`](ios/Lumen/Background/AGENTS.md): bounded background leases and no-model-load policy.
- [`ios/Lumen/CarPlay/AGENTS.md`](ios/Lumen/CarPlay/AGENTS.md): CarPlay voice lifecycle and policy.
- [`ios/Lumen/Developer/AGENTS.md`](ios/Lumen/Developer/AGENTS.md): developer console and evidence-layer presentation.
- [`ios/Lumen/Memory/AGENTS.md`](ios/Lumen/Memory/AGENTS.md): memory extraction, scoring, context, and capture.
- [`ios/Lumen/Models/AGENTS.md`](ios/Lumen/Models/AGENTS.md): shared value and SwiftData model contracts.
- [`ios/Lumen/Permissions/AGENTS.md`](ios/Lumen/Permissions/AGENTS.md): permission state, request, and diagnostics boundary.
- [`ios/Lumen/RAG/AGENTS.md`](ios/Lumen/RAG/AGENTS.md): indexing, chunking, retrieval, and context policy.
- [`ios/Lumen/Services/AGENTS.md`](ios/Lumen/Services/AGENTS.md): runtime/persistence/platform service ownership.
- [`ios/Lumen/Services/AgentGrounding/AGENTS.md`](ios/Lumen/Services/AgentGrounding/AGENTS.md): behavior manifest, trace, and audit contracts.
- [`ios/Lumen/Services/Diagnostics/AGENTS.md`](ios/Lumen/Services/Diagnostics/AGENTS.md): evidence-envelope export boundary.
- [`ios/Lumen/Services/LLM/AGENTS.md`](ios/Lumen/Services/LLM/AGENTS.md): backend interfaces, GGUF bridge, model storage, and device policy.
- [`ios/Lumen/Services/MicrosoftGraph/AGENTS.md`](ios/Lumen/Services/MicrosoftGraph/AGENTS.md): OAuth/MSAL, mail API, protected cache, and Outlook tools.
- [`ios/Lumen/System/AGENTS.md`](ios/Lumen/System/AGENTS.md): cancellation, budgets, pressure, metrics, and scene safety.
- [`ios/Lumen/Tools/AGENTS.md`](ios/Lumen/Tools/AGENTS.md): secure native tool schema/policy/execution.
- [`ios/Lumen/Views/AGENTS.md`](ios/Lumen/Views/AGENTS.md): SwiftUI presentation and kernel event reduction.
- [`ios/Lumen/Voice/AGENTS.md`](ios/Lumen/Voice/AGENTS.md): recognition, synthesis, session state, and kernel routing.
- [`ios/LumenTests/AGENTS.md`](ios/LumenTests/AGENTS.md): unit-test conventions and focused execution.
- [`docs/AGENTS.md`](docs/AGENTS.md): shipped-state and evidence documentation.
- [`scripts/AGENTS.md`](scripts/AGENTS.md): operational script risk and invocation.
- [`tools/AGENTS.md`](tools/AGENTS.md): Python/tooling topology and generated ownership.
- [`tools/lumen_manifest_crawler/AGENTS.md`](tools/lumen_manifest_crawler/AGENTS.md): deterministic crawler/generator package.
- [`tools/fine_tuning/unsloth/AGENTS.md`](tools/fine_tuning/unsloth/AGENTS.md): controlled GPU training and lineage.
- [`tools/hf_zerogpu/AGENTS.md`](tools/hf_zerogpu/AGENTS.md): external ZeroGPU Space assembly.
- [`generated/AGENTS.md`](generated/AGENTS.md): derived artifact provenance and freshness.
- [`datasets/AGENTS.md`](datasets/AGENTS.md): public corpus provenance and licensing.
- [`runtime-audits/AGENTS.md`](runtime-audits/AGENTS.md): historical evidence semantics.
