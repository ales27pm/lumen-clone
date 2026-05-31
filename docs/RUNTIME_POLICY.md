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
