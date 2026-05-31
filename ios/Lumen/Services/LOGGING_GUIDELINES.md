# Runtime Logging Guidelines

## Production logging policy

- Do **not** use `print(...)` in production app/service code.
- Use `Logger` from `OSLog` with stable `subsystem` + `category` names.
- Prefer machine-parseable key/value fields for runtime failures.

## Llama/runtime naming

- `subsystem`: `com.lumen.runtime`
- `category`: `llama.service`

## Required fields for runtime failures

Emit an error log line with these fields where applicable:

- `event`: stable event name (for example `llama.embedding.failure`)
- `severity`: `error` or `fault`
- `error_code`: taxonomy value (`network`, `decode`, `model-load`, `timeout`, `runtime`)
- `request_id` / `correlation_id`: UUID or propagated request identifier
- contextual dimensions useful for triage (for example model slot, token count, requested dimensions)
