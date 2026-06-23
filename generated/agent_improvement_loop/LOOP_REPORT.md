# Lumen Agent Improvement Loop Report

- Passed: `False`
- Tools: `53`
- Intents: `22`
- Model slots: `6`
- Dataset records: `34936`
- Runtime audit reports: `20`
- Runtime failures: `38`
- Raw runtime failures: `51`
- Skipped live model generation: `13`
- TestFlight status: `runtime-audit-ingested`
- TestFlight scenarios: `120`
- Gaps: `51`
- Next action prompts: `51`

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

### ERROR — agent_grounding_no_recent_model_traces

- Category: `runtime_drift`
- Recommendation: Fix runtime trace instrumentation or rerun the app before exporting; do not train from empty-trace evidence.
