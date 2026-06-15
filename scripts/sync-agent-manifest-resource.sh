#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT_DIR/generated/agent_manifest/AgentBehaviorManifest.json"
DEST_DIR="$ROOT_DIR/ios/Lumen"
DEST="$DEST_DIR/AgentBehaviorManifest.json"

if [[ ! -f "$SOURCE" ]]; then
  echo "Missing generated manifest: $SOURCE" >&2
  echo "Run: PYTHONPATH=tools/lumen_manifest_crawler python -m lumen_manifest_crawler generate --root . --output generated/agent_manifest --pretty" >&2
  exit 1
fi

python3 - "$SOURCE" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = manifest.get("artifactStatus") or {}
if status.get("runtimeEvidence") is not False:
    raise SystemExit("AgentBehaviorManifest.json must declare artifactStatus.runtimeEvidence=false before sync")
if status.get("deterministicBuild") is not True:
    raise SystemExit("AgentBehaviorManifest.json must declare artifactStatus.deterministicBuild=true before sync")
PY

mkdir -p "$DEST_DIR"
cp "$SOURCE" "$DEST"
echo "Synced $SOURCE -> $DEST"
