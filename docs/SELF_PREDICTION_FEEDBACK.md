# Observed tool feedback and controlled perturbations

Status: source candidate. Component tests, complete iOS compilation, model-backed
adaptation and installed-device behavior are separate evidence layers. The user
reports Gemma working on their device; that report does not validate this new code.

## Observable behavior

Before each structured model turn, the kernel reads feedback from **its own secure
tool registry's ledger**, including when the registry is explicitly injected. The
same ledger already records the forecast before execution and the host's typed
result afterward. There is no additional model request, alternate agent loop,
permission change, tool retry, training, or mutation of a saved forecast.

The next structured request can receive recent observations such as a tool's last
status, its successes among the retained results, and a change between the latest
two eligible statuses. A `success -> denied` transition describes two observations.
It does not establish why the tool was denied or whether permission remains denied.
The current tool catalog, permission checks and approval policy remain authoritative.

## Retained evidence

`SelfPredictionFeedback` derives a snapshot from the existing validated ledger:

- Filter by exact runtime label and the canonical tool IDs allowed in the current
  structured request. A runtime label is not a model-version identity.
- Retain outcomes at most 24 hours old, including exactly the 24-hour boundary.
  Future timestamps are excluded; sequence numbers determine logical order.
- For each tool, keep at most the latest eight eligible, noncancelled outcomes.
  Retain at most three tools, ordered by latest outcome sequence, then tool ID.
- Count unresolved predictions, cancelled outcomes, stale records and future
  records separately. Freshness is classified first, so these categories do not
  overlap. No unresolved or cancelled action becomes a failed observation.
- Retain typed outcomes even when the original probability was missing, invalid
  or host-repaired: observations and model confidence are different data. The
  projection supplies no probability, learned capability score or calibrated rate.

These are observations within the stored window, not a complete history of app
actions. Changing runtime or the allowed tool set changes which history is visible.
Restarting the app reuses valid saved evidence until it ages out. A missing ledger
means no recorded observations; unreadable/corrupt storage means unavailable
feedback. After a recording failure in the current registry session, feedback is
unavailable instead of silently serving an incomplete recent history.

The existing `self_prediction_ledger.v1` file/export is unchanged. No new persistent
schema, retention deletion, automatic repair or reset is introduced.

## Prompt and privacy boundary

Host feedback contains only bounded tool/runtime identifiers, typed statuses,
counts, sequence and age metadata. It contains no arguments, tool output,
conversation, memory, document text or model-authored explanation.

The host snapshot keeps its existing 400-character limit. Historical feedback has
a separate 640-character limit including its data-only wrapper. Rendering drops
whole tool entries if needed. Prompt assembly keeps the current snapshot ahead of
history and omits history as a whole when it cannot fit without displacing the
existing allocation for caller instructions or tool definitions. It never cuts a
history record or enlarges the global prompt limits. In particular, a full llama
agent-JSON budget may omit history even when it exists.

Compaction cannot recreate omitted feedback from user/history text. Both runtime
preflights derive `predictionFeedbackIncluded` from the assembled system context;
an unknown test adapter does not attest inclusion. Structured diagnostics expose
that boolean when available, separately from snapshot inclusion. It proves prompt
presence, not that the model used the information or improved its prediction.

## Perturbation and ablation contract

The native component tests exercise the real ledger and secure registry with
isolated temporary storage and synthetic local tool adapters. They cover observed
removal/recovery and policy changes, durable pre-action forecasts, missing outcomes,
cancellation, restart, runtime/allowlist isolation and unavailable storage.

Controls omit recording or omit feedback while preserving the tool action. The
comparison tests the contribution of each data path. No test changes the user's
real permissions, registers a synthetic Release tool, or trains an artificial
predictor to manufacture an improvement. Tests of the production kernel capture
the next request to verify it receives the preceding registry outcome.

On an Apple executor with Swift 6.2+, run:

```bash
bash scripts/run_self_prediction_tests.sh
```

For the complete app's kernel/context integration, also run the focused
`HostSelfModelGroundingTests` with the repository's Xcode test procedure. A portable
component runner is not a substitute for the complete iOS target or Gemma execution.
Use the existing `score-self-prediction-ledger` CLI for explicitly exported device
forecasts. No eligible model observations means no calibration result, not success.

## Remaining acceptance boundary

Actual improvement in Gemma's future predictions requires fresh model-backed,
matched trials with and without feedback and an external perturbation schedule.
The component suite demonstrates evidence transport and state updates only. It
does not establish general causal self-modeling, task completion or consciousness.
