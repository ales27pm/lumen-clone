# Permissions

`PermissionRegistry` centralizes runtime permission state and explicit request calls.

Behavior:
- No permission requests at startup.
- Background/headless flows do not prompt; not-determined states are denied in background.
- Permission-read tools check `PermissionGate` before execution.
- `networkAccess` is app-controlled and defaults disabled.
- Local network domain reports capability/status conservatively based on Info.plist/runtime constraints.
