#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${LUMEN_REAL_ACTOOL:-}" ]]; then
  REAL_ACTOOL="$LUMEN_REAL_ACTOOL"
else
  REAL_ACTOOL="$(xcrun --find actool)"
fi
[[ -x "$REAL_ACTOOL" ]] || {
  printf 'error: actool executable not found: %s\n' "$REAL_ACTOOL" >&2
  exit 1
}
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ -L "$SCRIPT_PATH" ]]; then
  resolved_path="$(readlink "$SCRIPT_PATH")"
  if [[ "$resolved_path" != /* ]]; then
    resolved_path="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/$resolved_path"
  fi
  SCRIPT_PATH="$resolved_path"
fi
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
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

first_asset_catalog() {
  local arg
  for arg in "$@"; do
    [[ "$arg" == *.xcassets ]] && {
      printf '%s\n' "$arg"
      return 0
    }
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
  asset_catalog="$(first_asset_catalog "$@" || true)"
  {
    printf '\0actool-cached-assets\0'
    [[ -z "$asset_catalog" ]] || printf '\020%s\0' "$asset_catalog"
    [[ -z "$partial_info" ]] || printf '@%s\0' "$partial_info"
    find "$compile_dir" -maxdepth 1 -type f \( -name 'Assets.car' -o -name 'AppIcon*.png' \) -print \
      | sort \
      | while IFS= read -r output_file; do
          printf '@%s\0' "$output_file"
        done
  } > "$dependency_info"
fi

printf '/* com.apple.actool.compilation-results */\n'
printf '%s\n' "$compile_dir/Assets.car"
find "$compile_dir" -maxdepth 1 -type f -name 'AppIcon*.png' -print | sort
[[ -z "$partial_info" ]] || printf '%s\n' "$partial_info"
