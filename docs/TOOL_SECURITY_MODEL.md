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

Secure execution:
- `SecureToolRegistry` is the execution boundary for kernel, headless, approval-confirmed, and migrated agent tool calls.
- Tool command execution canonicalizes tool IDs and arguments before calling `ToolRegistry`/`ToolInvocation`.
- No silent schema divergence: mapping remains explicit by tool IDs and schema validation must pass before execution.

Deferred tools:
- camera/microphone capture and file-import tools are intentionally deferred until explicit foreground user flows + approval UI integration are completed.

## Background Tool Execution Status
Headless path maps secure tool definitions through `ToolSchemaBridge` and exposes only background-safe tool definitions. Requests that need foreground approval, personal-data permissions, external network tools, or clarification are skipped with explicit diagnostic reasons.

## Secure Command Execution
`SecureToolRegistry.executeToolCommand(...)` wraps string-command tool requests:
- mapped secure tools route via `ToolRegistry`/`ToolInvocation`
- unknown sensitive/network/destructive-looking tool IDs are denied by registry or approval policy
- background execution is limited to explicit read-only or permission-read tools that declare background support

Bypass status:
- Migrated execution points call `SecureToolRegistry.executeToolCommand(...)` or native `SecureToolRegistry.execute(...)`.
- Release-visible headless/background paths use secure tool definitions and do not fall through to unchecked tool execution.
