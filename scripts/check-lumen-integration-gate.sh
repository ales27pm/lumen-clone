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

run_check "Agent kernel boundary" python3 tools/check_agent_kernel_boundary.py
run_check "Agent kernel boundary strict" python3 tools/check_agent_kernel_boundary.py --strict
run_check "Adapter runtime invariants" python3 tools/check_adapter_runtime_invariants.py
run_check "iOS LoRA hardening invariants" python3 tools/check_ios_lora_hardening_invariants.py
run_check "MSAL iOS release config" python3 scripts/validate-msal-ios-release-config.py
run_check "No shell subprocess security check" python3 tools/security/check_no_shell_subprocess.py
run_check "iOS build readiness" bash scripts/check-ios-build-readiness.sh
run_check "Git diff whitespace check" git diff --check

printf '\nLumen integration gate passed.\n'
