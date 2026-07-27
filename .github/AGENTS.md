# AGENTS.md

## Scope

This file governs `.github/`, currently the hosted workflows under `.github/workflows/`. It refines the repository-wide rules in [`../AGENTS.md`](../AGENTS.md).

## Role In The System

The workflows provide remote integration and linkage checks for pushes and pull requests. They do not define production runtime behavior and must not be treated as the only validation path; repository policy requires local evidence unless the user explicitly requests GitHub Actions.

## Key Files And Entry Points

- `.github/workflows/lumen-integration.yml`: repository integration workflow and its local script/tool invocations.
- `.github/workflows/msal-link-validation.yml`: validates Microsoft Authentication Library linkage in the Xcode project/build.
- `scripts/check-lumen-integration-gate.sh`: local integration entry point referenced by repository validation.
- `ios/Lumen.xcodeproj/project.pbxproj`: authoritative package/product linkage inspected by the MSAL check.

## Public Interfaces

Workflow names, triggers, required checks, job outputs, environment names, and artifact names are interfaces consumed by GitHub branch protection and contributors. Changing them can break required-check matching even when job steps remain valid.

## Internal Structure And Dependencies

Incoming events are GitHub push/pull-request events. Jobs check out the tree, install or select toolchains, and call repository scripts/Xcode checks. The workflows depend on paths outside this subtree; production code does not depend on workflow YAML.

## Data And Control Flow

GitHub event -> workflow trigger/filter -> toolchain setup -> repository command -> check/status result. A local command passing does not prove the workflow's toolchain/setup still works, while a hosted check passing does not replace required local runtime evidence.

## Local Invariants

- Do not add a network secret to YAML or print secret context.
- Keep branch/path filters and required-check names compatible unless their consumers are intentionally updated.
- Do not weaken a gate, remove Release coverage, or replace a real command with an unconditional success.
- Do not trigger, rerun, or push merely to exercise Actions unless the user explicitly requests it.

## Coordinated Changes

When changing a workflow, inspect every referenced script, working directory, Xcode destination, Python version, and artifact path. Changes to MSAL linkage must also inspect the project file and Microsoft Graph guidance. Renaming a job/check requires checking branch-protection expectations outside the repository.

## Safe Editing Rules

Keep hosted orchestration thin and move reusable logic into a local script only when that script is itself tested and documented. Pin third-party actions deliberately. Do not infer that a newer action/toolchain is compatible without checking its inputs and repository deployment target.

## Validation

From the repository root:

```bash
git diff --check -- .github
bash scripts/check-lumen-integration-gate.sh
```

Run the concrete local commands referenced by any edited job. No dedicated checked-in workflow YAML parser was found; do not claim hosted setup was validated unless Actions actually ran at the user's request.

## Common Failure Modes

- A changed job name no longer satisfies branch protection.
- A relative path works locally but not from the workflow's `working-directory`.
- A push intended only to update documentation consumes an unwanted workflow run.
- MSAL appears in Swift source but is missing from the linked Xcode product.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). No child `AGENTS.md` files are currently needed under `.github/workflows/`; both workflows share the same trigger and remote-validation risks.
