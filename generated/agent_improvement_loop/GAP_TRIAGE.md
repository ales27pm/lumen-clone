# Agent Improvement Gap Triage

- Total gaps: `33`
- Raw runtime failures: `42`
- Fresh runtime failures: `41`
- Skipped live model generation: `1`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `stale_or_unclassified_runtime_evidence`: `33`

## Failure Groups

### outlook

- Count: `32`
- Status: `deferred`
- Root causes: `{"stale_or_unclassified_runtime_evidence": 32}`
- Categories: `{"runtime_drift": 32}`
- Fresh runtime failures: `32`
- Skipped live model generation: `0`
- Recommended action: Quarantine this architecture/runtime/finalizer failure from SFT; add a deterministic regression test or runtime diagnostic instead.

  - `stale_or_unclassified_runtime_evidence` | skipped=`False` | prompt: Archive my latest Outlook email. | actual: Outlook is not configured correctly for this build. Verify Microsoft Graph MSAL client ID, redirect URI, and bundle identifier.
  - `stale_or_unclassified_runtime_evidence` | skipped=`False` | prompt: Archive my latest Outlook email. | actual: Outlook is not configured correctly for this build. Verify Microsoft Graph MSAL client ID, redirect URI, and bundle identifier.
  - `stale_or_unclassified_runtime_evidence` | skipped=`False` | prompt: Trash the latest Outlook email. | actual: Outlook is not configured correctly for this build. Verify Microsoft Graph MSAL client ID, redirect URI, and bundle identifier.

### rag

- Count: `1`
- Status: `deferred`
- Root causes: `{"stale_or_unclassified_runtime_evidence": 1}`
- Categories: `{"runtime_drift": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `stale_or_unclassified_runtime_evidence` | skipped=`False` | prompt: Search my files for architecture notes and summarize key modules. | actual: Summary [1] - [C0E44F7F] Photos 2026-07 | score=0.28 | Photos (2026-07): 96 items between Jul 1, 2026 and Jul 11, 2026. 0 favorites, 2 videos, 72 screenshots, 0 selfies, 0 live photos, 0 portraits, 0 with location. - [ECE05887] Photos 2026-
