#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE="0x100d88000"
DEFAULT_OFFSETS=(0x184E5C 0x24F488 0x11AAD 0xBE81 0x6E3FD)

usage() {
  cat <<USAGE
Usage: $0 --dsym PATH_TO_Lumen.app.dSYM|--archive PATH_TO.xcarchive [--base 0xLOAD_ADDRESS] [offset ...]

Symbolicates Lumen crash offsets by adding each offset to the app load address
and passing the resulting absolute addresses to atos.

Defaults from TestFlight build 1.0.0 (2):
  --base ${DEFAULT_BASE}
  offsets: ${DEFAULT_OFFSETS[*]}

Examples:
  $0 --dsym ./Lumen.app.dSYM
  $0 --archive ./Lumen.xcarchive --base 0x100d88000 0x184E5C 0x24F488
USAGE
}

DSYM=""
ARCHIVE=""
BASE="$DEFAULT_BASE"
OFFSETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dsym) DSYM="${2:-}"; shift 2 ;;
    --archive) ARCHIVE="${2:-}"; shift 2 ;;
    --base) BASE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) OFFSETS+=("$1"); shift ;;
  esac
done

if ! command -v atos >/dev/null 2>&1; then
  echo "error: atos not found. Run on macOS with Xcode command line tools installed." >&2
  exit 2
fi

if [[ -n "$ARCHIVE" ]]; then
  if [[ ! -d "$ARCHIVE" ]]; then echo "error: archive not found: $ARCHIVE" >&2; exit 2; fi
  DSYM=$(find "$ARCHIVE/dSYMs" -maxdepth 1 -name 'Lumen.app.dSYM' -print -quit)
fi

if [[ -z "$DSYM" ]]; then echo "error: provide --dsym or --archive." >&2; usage >&2; exit 2; fi
if [[ ! -d "$DSYM" ]]; then echo "error: dSYM not found: $DSYM" >&2; exit 2; fi

DWARF="$DSYM/Contents/Resources/DWARF/Lumen"
if [[ ! -f "$DWARF" ]]; then echo "error: DWARF binary missing at $DWARF" >&2; exit 2; fi

if [[ ${#OFFSETS[@]} -eq 0 ]]; then OFFSETS=("${DEFAULT_OFFSETS[@]}"); fi

python3 - "$BASE" "${OFFSETS[@]}" <<'PY' > /tmp/lumen-atos-addresses.$$
import sys
base = int(sys.argv[1], 16)
for off in sys.argv[2:]:
    print(f"{off} {hex(base + int(off, 16))}")
PY

ADDRESSES=($(awk '{print $2}' /tmp/lumen-atos-addresses.$$))
SYMBOLS=$(atos -o "$DWARF" -l "$BASE" "${ADDRESSES[@]}" || true)
if [[ -z "$SYMBOLS" ]]; then echo "error: atos produced no symbols." >&2; rm -f /tmp/lumen-atos-addresses.$$; exit 3; fi

echo "Lumen symbolication report"
echo "dSYM: $DSYM"
echo "load address: $BASE"
echo
paste /tmp/lumen-atos-addresses.$$ <(printf '%s\n' "$SYMBOLS") | awk '{off=$1; addr=$2; $1=""; $2=""; sub(/^  */, ""); printf "%s -> %s -> %s\n", off, addr, $0}'
rm -f /tmp/lumen-atos-addresses.$$
