# Runtime Policy

`ComputePolicy` consumes `ComputePolicyInput` and returns deterministic limits:
- Background: `maxTokens=256`, heavy runtime disabled.
- Foreground low-power or serious/critical thermal: `maxTokens=512`, heavy runtime disabled.
- Foreground nominal: `maxTokens=1024`, heavy runtime enabled.

`AssistantRuntimeRouter` uses this decision and task kind to select:
- CoreML for embedding/safety tasks when available.
- FoundationModels for preferred foreground chat-like tasks when available and policy allows.
- llama when FoundationModels is unavailable or constrained.
- deterministic fallback for constrained or unavailable scenarios.

## ResourceBudgetGate model-load policy

`ResourceBudgetGate` is the lifecycle/resource gate for expensive local model work:
- only explicit `userChat` and `userVoice` intents may start model/tokenizer/runtime loading;
- `appStartup`, `diagnostics`, and `background` intents degrade without loading model assets;
- inactive/background scenes, serious/critical/unknown thermal state, and recent memory warnings deny heavy work;
- Low Power Mode denies passive heavy work but still permits explicit foreground user chat/voice turns when other resource checks pass.

FoundationModels availability is treated as a cheap metadata status. Diagnostics and capability profiling must not instantiate FoundationModels tokenizer/model assets.
