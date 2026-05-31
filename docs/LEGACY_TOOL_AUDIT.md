# Legacy Tool Audit

Tool IDs from legacy `ToolExecutor` are classified below.

## Migrated to secure ToolRegistry (direct/bridged)
- `memory.recall` -> `memory.search`
- `rag.search` -> `rag.search.secure`
- `contacts.search` -> `contacts.lookup`
- `location.current` -> `location.snapshot`

## Wrapped by LegacySecureToolExecutor (allowlisted read-only)
- `weather`
- `maps.search`
- `files.read`
- `trigger.list`

## Denied pending migration (sensitive/network/destructive patterns)
- IDs containing: `delete`, `send`, `open`, `call`, `mail`, `message`, `web`

## Still legacy risk
- Legacy-only tool IDs not mapped and not in allowlist remain blocked or fallback legacy depending on classification; full migration pending for entire ToolExecutor matrix.
