#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT_DIR/generated/agent_manifest/AgentBehaviorManifest.json"
DEST_DIR="$ROOT_DIR/ios/Lumen/Resources"
DEST="$DEST_DIR/AgentBehaviorManifest.json"

if [[ ! -f "$SOURCE" ]]; then
  echo "Missing generated manifest: $SOURCE" >&2
  echo "Run: PYTHONPATH=tools/lumen_manifest_crawler python -m lumen_manifest_crawler generate --root . --output generated/agent_manifest --pretty" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SOURCE" "$DEST"
echo "Synced $SOURCE -> $DEST"
