# Tool Security Model

Implemented secure tools:
- device.status
- memory.search
- rag.search.secure
- calendar.read
- contacts.lookup
- location.snapshot
- notify.local
- open.url

Rules enforced:
- Deterministic `ToolApprovalPolicy` decides allow/deny/requiresApproval.
- Permission-read tools gate through `PermissionRegistry` + `PermissionGate`.
- Sensitive/user-visible actions return `requiresApproval` for model-proposed invocations.
- Foreground user initiation is not treated as approval. Approval-gated tools require a confirmed `userApproved` invocation source produced by the UI confirmation path.
- Approval confirmations are one-time, tool-bound, and short-lived; missing, malformed, expired, or mismatched pending approval tokens are denied instead of executed.
- Background tool visibility is restricted to tools that explicitly support background execution, remain read-only/permission-read safe, and do not require user approval.
- Background tool execution cannot initiate permission prompts. Missing or not-yet-granted permissions return a denied/degraded result until the user opens the foreground app and grants access.
- Output is bounded by `SafeToolOutputLimiter`.
- Tool metrics are recorded to `RuntimeMetricsStore` without raw payload logging.

Legacy bridge:
- Legacy `Services/ToolExecutor.swift` remains active for existing agent pipelines.
- New `ToolRegistry` is integrated through `AssistantKernel.executeTool(...)` as the migration path.
- No silent schema divergence: mapping remains explicit by tool IDs while migration proceeds.

Deferred tools:
- camera/microphone capture and file-import tools are intentionally deferred until explicit foreground user flows + approval UI integration are completed.

## Legacy bridge status
Headless path now maps secure tool definitions via `LegacyToolSchemaBridge` and injects only background-safe tool definitions. Legacy `ToolExecutor` remains active for backward compatibility and is the main remaining migration risk.

## Legacy secure execution
`SecureToolRegistry.executeLegacyTool(...)` wraps remaining legacy tool execution:
- mapped secure tools route via `ToolRegistry`/`ToolInvocation`
- unknown sensitive/network/destructive-looking tool IDs are denied
- only explicit read-only allowlist can run through legacy fallback

Legacy bypass status:
- Reduced: migrated execution points call `SecureToolRegistry.executeLegacyTool(...)`.
- Remaining: unmigrated legacy prompt/request paths may still expose legacy tool metadata until full coordinator adoption.
