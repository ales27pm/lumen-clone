#!/usr/bin/env bash
set -euo pipefail

REAL_ACTOOL="${LUMEN_REAL_ACTOOL:-/Users/ales27pm/Downloads/Xcode.app/Contents/Developer/usr/bin/actool}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHED_APP="${LUMEN_CACHED_ASSETS_APP:-$REPO_ROOT/build/Lumen-20260630-211948.xcarchive/Products/Applications/Lumen.app}"

has_arg() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$needle" ]] && return 0
  done
  return 1
}

value_after() {
  local needle="$1"
  shift
  local previous=""
  local arg
  for arg in "$@"; do
    if [[ "$previous" == "$needle" ]]; then
      printf '%s\n' "$arg"
      return 0
    fi
    previous="$arg"
  done
  return 1
}

if has_arg "--version" "$@" || has_arg "--generate-swift-asset-symbols" "$@"; then
  exec "$REAL_ACTOOL" "$@"
fi

compile_dir="$(value_after "--compile" "$@" || true)"
if [[ -z "$compile_dir" ]]; then
  exec "$REAL_ACTOOL" "$@"
fi

assets_car="$CACHED_APP/Assets.car"
[[ -f "$assets_car" ]] || {
  printf 'error: cached Assets.car not found: %s\n' "$assets_car" >&2
  exit 1
}

mkdir -p "$compile_dir"
cp "$assets_car" "$compile_dir/Assets.car"
for icon in "$CACHED_APP"/AppIcon*.png; do
  [[ -f "$icon" ]] || continue
  cp "$icon" "$compile_dir/$(basename "$icon")"
done

partial_info="$(value_after "--output-partial-info-plist" "$@" || true)"
if [[ -n "$partial_info" ]]; then
  mkdir -p "$(dirname "$partial_info")"
  cat > "$partial_info" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIcons</key>
  <dict>
    <key>CFBundlePrimaryIcon</key>
    <dict>
      <key>CFBundleIconFiles</key>
      <array>
        <string>AppIcon60x60</string>
      </array>
      <key>CFBundleIconName</key>
      <string>AppIcon</string>
    </dict>
  </dict>
  <key>CFBundleIcons~ipad</key>
  <dict>
    <key>CFBundlePrimaryIcon</key>
    <dict>
      <key>CFBundleIconFiles</key>
      <array>
        <string>AppIcon60x60</string>
        <string>AppIcon76x76</string>
      </array>
      <key>CFBundleIconName</key>
      <string>AppIcon</string>
    </dict>
  </dict>
</dict>
</plist>
PLIST
fi

dependency_info="$(value_after "--export-dependency-info" "$@" || true)"
if [[ -n "$dependency_info" ]]; then
  mkdir -p "$(dirname "$dependency_info")"
  : > "$dependency_info"
fi

printf '/* com.apple.actool.compilation-results */\n'
printf '%s\n' "$compile_dir/Assets.car"
find "$compile_dir" -maxdepth 1 -type f -name 'AppIcon*.png' -print | sort
[[ -z "$partial_info" ]] || printf '%s\n' "$partial_info"
