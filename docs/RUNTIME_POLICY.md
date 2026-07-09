# Runtime Policy

`ComputePolicy` consumes `ComputePolicyInput` and returns deterministic limits:
- Background: `maxTokens=256`, heavy runtime disabled.
- Foreground low-power or serious/critical thermal: `maxTokens=512`, heavy runtime disabled.
- Foreground nominal: `maxTokens=1024`, heavy runtime enabled.

`AssistantRuntimeRouter` uses this decision and task kind to select:
- CoreML for embedding/safety tasks only when the runtime capability matrix marks CoreML embeddings as selectable.
- FoundationModels for preferred foreground chat-like tasks only when the runtime capability matrix marks FoundationModels generation as selectable and policy allows.
- llama when FoundationModels is unavailable or constrained.
- deterministic fallback for constrained or unavailable scenarios.

`AssistantRuntimeCapabilityMatrix` is the source of truth for staged adapter capability reporting. FoundationModels and CoreML may appear in diagnostics with framework or model status, but staged implementations must report `generationSelectable=false` or `embeddingSelectable=false` until their real generation/embedding paths are implemented. Diagnostics should show both the status and selectability so a staged adapter cannot look production-capable by mere framework availability.

## ResourceBudgetGate model-load policy

`ResourceBudgetGate` is the lifecycle/resource gate for expensive local model work:
- only explicit `userChat` and `userVoice` intents may start model/tokenizer/runtime loading;
- `appStartup`, `diagnostics`, and `background` intents degrade without loading model assets;
- inactive/background scenes and serious/critical/unknown thermal state deny heavy work; memory warnings deny heavy work during the cooldown window;
- Low Power Mode denies passive heavy work but still permits explicit foreground user chat/voice turns when other resource checks pass.

`BackgroundTaskPolicy` may allow a background trigger scan so due local tasks can be inspected and background-safe tools can run, but its background decisions keep `allowModelLoading=false`. A background scan is permission to coordinate already-safe local work, not permission to load chat/runtime assets. `BackgroundToolExecutionPolicy` must prove the routed tools are background-capable and allowed by current policy before the headless secure tool path runs; otherwise the trigger records a skip reason and does not fall through to unsafe tool or model execution. Background model housekeeping is limited to unloading optional resident chat slots and is allowed in Low Power Mode because it releases resources instead of consuming model compute.

Queued memory captures may be stored from user-initiated App Intents when local embeddings are unavailable, but promotion into vector-backed memory is policy-gated. Background memory maintenance passes its `allowModelLoading` decision into `MemoryConsolidator`; when that decision is false, queued captures remain local and pending instead of loading embedding runtime assets.

FoundationModels availability is treated as a cheap metadata status. Diagnostics and capability profiling must not instantiate FoundationModels tokenizer/model assets.

## Remote model escalation

Remote/cloud model records are denied by default by `DeviceModelPolicy`. A remote backend may be considered only when the caller constructs an explicit `RemoteModelAccessPolicy` that allows remote models, and the built-in approved policy is foreground-only. Background work, diagnostics, and passive maintenance must not silently select remote backends as a fallback for missing local models.
