# Action predictions: second self-model slice

Status: source implementation and component validation candidate. This is not a
claim of signed release, iPhone execution, calibrated model confidence, or a causal
self-model. The first slice grounds host facts; this slice records forecasts and
their observed tool statuses. Hidden perturbations, ablations, capability updates,
and feedback-driven training remain separate work.

## Forecast target and coverage

On a structured action turn the existing model request asks for optional metadata:

```json
{"action":{"tool":"device.status","args":{}},"prediction":{"successProbability":0.72}}
```

The target is exactly **the tool registry returning `ToolResult.status == .success`**,
including availability, permissions and approval policy. It is not completion of
the user's task. The forecast never grants permission or changes tool arguments.
There is no second inference call. Old responses without metadata remain valid;
no confidence is inferred from prose, task complexity or historical examples.

The structured kernel supplies a context to `SecureToolRegistry.execute`. Before
lookup, policy or native execution, the registry awaits the ledger's durable
`begin` acknowledgement. After execution it records the app's typed result,
whether the tool body started, and cancellation. Model-written success text is
not ground truth. Cancellation is checked again after the added suspension and
permission lookup. The tool result is bound to the host's invocation ID.

Coverage is deliberately `structured_tool_invocations_stored_window`. Model parse
failures, invalid actions, early approval boundaries in the kernel, text-only
answers, and other callers that do not supply prediction context are outside this
window. A missing outcome remains unresolved after termination or storage failure;
it is never converted to failure or success. Neither an export nor its hash proves
that all app actions were captured.

## Provenance and durable storage

`SelfPredictionLedger` is an actor separate from the transient conversation/tool
ledger and best-effort trace recorder. It receives only typed metadata:

- `model`: an unchanged model action with a finite probability in `[0, 1]`.
- `missing`: an unchanged model action without prediction metadata.
- `invalid`: an unchanged model action with malformed prediction metadata.
- `host_repair`: an action changed or synthesized by the host; probability absent.

Structural action equality is checked before argument coercion. A deduplication
key is insufficient to prove model authorship. Host-forced continuations do not
inherit the preceding forecast. Stored IDs and sequence numbers come from the host.
The ledger contains no messages, documents, argument values, outputs, or hashes of
those private values. Tool IDs and runtime labels are bounded metadata.

The JSONL file is limited to 2 MiB and 4096 events, with capacity reserved for
pending outcomes. There is no silent pruning, rotation, repair, or automatic retry
of a tool. Loading rejects corrupt or inconsistent existing storage. An append is
acknowledged only after synchronization. If writing fails, the original tool flow
can continue: the registry records a bounded diagnostic and does not pretend the
forecast was saved. Exports may therefore have coverage gaps even when valid.

Settings → **Action predictions → Export action predictions** produces a local JSON
document in Release as well as DEBUG. Opening Settings does not load a model. The
user chooses the export destination; nothing is uploaded automatically. Full or
unreadable storage prevents new recording and must not be interpreted as an empty,
healthy run. This version does not automatically clear the ledger.

## Export contract and scoring

The envelope contains exactly `schemaVersion: "self_prediction_ledger.v1"`,
`coverageScope: "structured_tool_invocations_stored_window"`, and `events`.
Each event has `kind`, contiguous one-based `seq`, `id`, and host ISO8601 `timestamp`.
Prediction events also have `runID`, `toolID`, `runtime`, `probabilityOrigin`, and
`probability` only for model forecasts. Outcome events have `status`,
`toolBodyStarted`, and `cancelled`. An outcome follows its prediction exactly once.
Sequence order, rather than wall-clock monotonicity, establishes logical ordering.

From the repository root, score an explicitly exported file:

```bash
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler \
  python -m lumen_manifest_crawler score-self-prediction-ledger \
  --ledger /path/to/lumen-action-predictions.json \
  --output /path/to/prediction-score.json
```

The scorer validates the whole bounded export before measuring anything. Duplicate
keys, malformed types, NaN, out-of-range probabilities, duplicate/orphan outcomes,
and events reordered ahead of their prediction fail explicitly.

For the `registry_status_success` target, a matched, noncancelled model forecast is
scored as `1` for success and `0` for failed, denied, requiresApproval or unavailable.
The report separately measures calls whose tool body started, so policy refusal is
not presented as a failed native execution. It reports Brier score, ten-bin ECE,
success base rate, probability range, and denominators. Missing/invalid/host-repair
forecasts, cancellations and unresolved predictions are counted, not scored. No
eligible observations produces null metrics, not a perfect zero score.

These metrics describe the retained sample, with its policy filtering and missing
data. Brier measures probabilistic accuracy as well as calibration; ECE depends on
the binning and sample size. Neither is by itself a release acceptance threshold.
The report does not set `allPassed`, authenticate a device, promote data to live E2E
evidence, update capability estimates, or prove subjective experience.

## Validation boundaries

On an Apple executor with Swift 6.2+, from an independent checkout of the reviewed
commit, run the standalone native component suite:

```bash
bash scripts/run_self_prediction_tests.sh
```

The runner compiles the real ledger, parser, forecast schema/provenance logic,
registry, approval policy and export view. Platform/permission adapters and
telemetry are explicit test doubles. It uses a temporary Swift package and removes
that package afterward; it neither signs nor uploads an app.

Python tests cover hand-computed metrics, malformed exports, ordering, backwards
wall-clock changes, missing/censored outcomes, and CLI behavior. Native component
tests cover write-before-body ordering, typed outcomes, cancellation, provenance,
corruption/capacity, and restart/export behavior. Consult the validation report for
which tests were actually executed on a given commit. A standalone component runner
does not compile the complete iOS target, execute real native permissions/tools, or
establish behavior of the installed Gemma model.
