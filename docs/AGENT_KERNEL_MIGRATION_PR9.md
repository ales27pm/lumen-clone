# Agent Kernel Migration PR9: Native Productivity LocalTools

PR9 continues PR F by moving the productivity tool group out of the temporary `LegacyToolExecutorLocalTool` adapter and into native `LocalTool` implementations registered directly with `SecureToolRegistry`.

## Native tool IDs

- `calendar.create`
- `calendar.list`
- `reminders.create`
- `reminders.list`
- `trigger.create`
- `trigger.list`
- `trigger.cancel`
- `alarm.authorization_status`
- `alarm.request_authorization`
- `alarm.schedule`
- `alarm.countdown`
- `alarm.list`
- `alarm.pause`
- `alarm.resume`
- `alarm.stop`
- `alarm.snooze`
- `alarm.cancel`

## New path

```text
SecureToolRegistry.execute(... productivity tool ...)
  -> ProductivityLocalTool.execute(...)
  -> existing CalendarTools / TriggerTools / AlarmTools helper
```

## Why this is still incremental

The underlying platform helper functions are preserved to keep behavioral parity. The important migration step is that these tool IDs are no longer served by the broad legacy adapter and are now individually registered in `SecureToolRegistry`.

## Registry impact

`LegacyToolExecutorLocalTool.all` now excludes the productivity IDs owned by `ProductivityLocalTool`, preventing duplicate secure tool IDs while shrinking the legacy adapter surface area.

## Validation

```bash
python3 tools/check_agent_kernel_boundary.py
swiftc -parse \
  ios/Lumen/Tools/Builtin/ProductivityLocalTools.swift \
  ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift \
  ios/Lumen/Tools/ToolRegistry.swift \
  ios/LumenTests/SecureToolRegistryTests.swift
```

## Next PR

`feat(kernel): port communication tools to native LocalTool implementations`
