# Adapter-specialized, policy-first pipeline

Lumen keeps the adapter-first deployment economics while moving control authority to deterministic code.

## Authority split

1. `IntentRouter` owns intent classification for production safety-critical routing.
2. `DeterministicToolPlanner` owns canonical tool actions and argument normalization.
3. `ToolRouteGuard` / executor code owns permissions, approval boundaries, and tool availability.
4. `FinalIntentValidator` owns output hygiene and cross-intent leak filtering.
5. Adapters specialize language behavior by role:
   - Cortex: compact advisory decision / JSON discipline.
   - Executor: argument extraction when deterministic planning delegates to it.
   - Mouth: user-visible response from approved observations.
   - Mimicry: meaning-preserving tone adaptation.
   - REM: trace compression and repair proposal generation.

Adapters never become the source of truth for tool permission or canonical route validity.

## Improve-loop gates

The improve loop now emits three datasets:

- `accepted_training-*.jsonl`
- `quarantined_samples-*.jsonl`
- `regression_tests-*.jsonl`

Only accepted samples are candidates for LoRA training. Quarantined samples preserve evidence but are blocked from training when they contain stale grounding, legacy tool IDs, resource fallback text, validator fallbacks, or uncorrected raw model output. Regression samples are for tests and prompt contracts, not gradient updates.

## Fine-tuning rule

Train role behavior, not policy truth. Tool IDs in accepted data must already be canonicalized. Runtime policy remains enforced by code.
