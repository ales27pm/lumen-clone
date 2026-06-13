# Agent Kernel Migration PR10: Native Communication LocalTools

PR10 continues PR F by moving the communication and Outlook tool group out of the temporary `LegacyToolExecutorLocalTool` adapter and into native `LocalTool` implementations registered directly with `SecureToolRegistry`.

## Native tool IDs

- `contacts.search`
- `messages.draft`
- `mail.draft`
- `phone.call`
- `outlook.status`
- `outlook.folders.list`
- `outlook.messages.list`
- `outlook.messages.search`
- `outlook.message.read`
- `outlook.attachments.list`
- `outlook.draft.create`
- `outlook.mail.send`
- `outlook.message.mark_read`
- `outlook.message.mark_unread`
- `outlook.message.move`
- `outlook.message.archive`
- `outlook.message.delete`
- `outlook.message.reply`
- `outlook.message.reply_all`
- `outlook.message.forward`

## New path

```text
SecureToolRegistry.execute(... communication tool ...)
  -> CommunicationLocalTool.execute(...)
  -> ToolRouteGuard approval + permission gates
  -> existing ContactsTools / OutlookTools helpers
```

## Behavioral parity

The underlying platform helper functions are preserved. `CommunicationLocalTool` keeps the same legacy sequence before helper execution:

1. `ToolRouteGuard.canExecuteTool(...)`
2. `ToolRouteGuard.ensurePermissionIfNeeded(...)`
3. helper dispatch

This keeps explicit approval behavior for drafts, sends, calls, replies, forwards, message mutation, and destructive Outlook actions.

## Registry impact

`LegacyToolExecutorLocalTool.all` now excludes both `ProductivityLocalTool.nativeToolIDs` and `CommunicationLocalTool.nativeToolIDs`, preventing duplicate secure tool IDs while shrinking the legacy adapter surface area.

## Validation

```bash
python3 tools/check_agent_kernel_boundary.py
swiftc -parse \
  ios/Lumen/Tools/Builtin/CommunicationLocalTools.swift \
  ios/Lumen/Tools/Builtin/ProductivityLocalTools.swift \
  ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift \
  ios/Lumen/Tools/ToolRegistry.swift
```

`ios/LumenTests/SecureToolRegistryTests.swift` imports XCTest and the app test module, so it should be validated through the normal Xcode test action rather than plain `swiftc -parse`.

## Next PR

`feat(kernel): port location and media tools to native LocalTool implementations`
