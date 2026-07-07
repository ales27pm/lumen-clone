# AGENTS.md

## Documentation Scope

This directory describes Lumen architecture, runtime state, validation, privacy, and release readiness. Documentation must match current code and validation evidence.

## Shipped-State Rules

- Do not claim a Release feature is complete unless code and tests prove it.
- Do not describe shipped Release behavior as partial, planned, staged, unavailable, temporary, not implemented, or compatibility-bridge backed.
- Keep experimental or DEBUG-only behavior clearly separated from shipped Release behavior.
- When a capability needs physical hardware, Apple credentials, TestFlight, real local model artifacts, or live runtime exports, document it as manual validation still required.
- Prefer exact states and recovery actions over vague wording. For runtime readiness, use states such as model missing, downloading, verifying, ready, corrupt, incompatible, failed to load, degraded, offline usable, or offline blocked.

## Validation Docs

Keep these files aligned when validation requirements change:

- `VALIDATION.md`
- `RUNTIME_STATUS_MATRIX.md`
- `AGENT_KERNEL_MIGRATION_STATUS.md`
- `../FEATURE_COMPLETE_VALIDATION.md`

Validation summaries should say what actually ran, what passed, what failed, and what was skipped. If `python` is unavailable and `python3` was used, record both facts.

## Documentation Checks

Useful local checks:

| Command | Purpose |
| --- | --- |
| `git diff --check` | Catch Markdown whitespace problems |
| `python3 tools/check_release_hardening.py` | Catch shipped-status wording and Release-routing regressions |
| `rg -n "partial|planned|compatibility bridge|staged|not implemented" docs README.md FEATURE_COMPLETE_VALIDATION.md` | Manual review for language that may overstate shipped status |

Do not update documentation to make a feature sound complete unless the implementation and validation evidence changed in the same task.
