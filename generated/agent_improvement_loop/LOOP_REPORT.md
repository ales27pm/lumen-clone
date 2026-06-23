# Lumen Agent Improvement Loop Report

- Passed: `False`
- Tools: `53`
- Intents: `22`
- Model slots: `6`
- Dataset records: `39983`
- Runtime audit reports: `4`
- Runtime failures: `33`
- Raw runtime failures: `46`
- Skipped live model generation: `13`
- TestFlight status: `runtime-audit-ingested`
- TestFlight scenarios: `120`
- Gaps: `46`
- Next action prompts: `46`

## TestFlight handoff

Run `TESTFLIGHT_RUNBOOK.md` in the real TestFlight app, export the in-app dataset package JSON, then rerun this command with `--runtime-audit <exported-json>`.

## Top gaps

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — argument_mismatch

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — missing_live_argument

- Category: `manifest_mismatch`
- Recommendation: Regenerate executor schema cards and add missing-argument clarification examples.

### ERROR — unmanifested_live_argument

- Category: `manifest_mismatch`
- Recommendation: Regenerate the manifest from Swift source, then add unknown-tool DPO contrast samples.

### ERROR — persistent_diagnostics_scenario_not_passed

- Category: `runtime_drift`
- Recommendation: Fix the diagnostics scenario or app runtime path, then rerun persistent diagnostics before using the artifact.
