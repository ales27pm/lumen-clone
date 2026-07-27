#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_check() {
  local title="$1"
  shift
  printf '\n== %s ==\n' "$title"
  "$@"
}

run_git_diff_check() {
  printf '\n== Git diff whitespace check ==\n'
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'Running git diff --check in git worktree.\n'
    git diff --check
  else
    printf 'Skipping git diff --check: not a git worktree\n'
  fi
}

run_check "Agent kernel boundary" python3 tools/check_agent_kernel_boundary.py
run_check "Ubuntu training launcher syntax" bash -n \
  scripts/ubuntu_run_fleet_canary.sh \
  scripts/ubuntu_train_lumen_adapters_aio.sh \
  scripts/ubuntu_train_lumen_full_pipeline.sh
run_check "Agent kernel boundary strict" python3 tools/check_agent_kernel_boundary.py --strict
run_check "Adapter runtime invariants" python3 tools/check_adapter_runtime_invariants.py
run_check "Release hardening guard" python3 tools/check_release_hardening.py
run_check "Generated JSONL artifacts" python3 scripts/check-generated-jsonl-artifacts.py
run_check "iOS LoRA hardening invariants" python3 tools/check_ios_lora_hardening_invariants.py
run_check "MSAL iOS release config" python3 scripts/validate-msal-ios-release-config.py
run_check "iOS signing capabilities" python3 scripts/validate_ios_signing_capabilities.py
run_check "No shell subprocess security check" python3 tools/security/check_no_shell_subprocess.py
if [[ -z "${LUMEN_BUILT_APP_PATH:-}" ]]; then
  latest_built_app="$(find build -maxdepth 2 \( -name '*.xcarchive' -o -name '*.ipa' \) -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$latest_built_app" ]]; then
    export LUMEN_BUILT_APP_PATH="$latest_built_app"
  fi
fi
run_check "iOS build readiness" bash scripts/check-ios-build-readiness.sh
run_git_diff_check

printf '\nLumen integration gate passed.\n'
