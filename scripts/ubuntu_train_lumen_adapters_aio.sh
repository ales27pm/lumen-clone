#!/usr/bin/env bash
set -Eeuo pipefail

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${LUMEN_AIO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${LUMEN_AIO_RUN_ROOT:-$ROOT/.local/ubuntu_finetune_runs/$RUN_ID}"
DATASET_SOURCE="${LUMEN_AIO_DATASET_SOURCE:-$ROOT/generated/agent_manifest/fine_tuning}"
AGENTS_CSV="${LUMEN_AIO_AGENTS:-cortex,executor,mouth,mimicry,rem,fleet}"
BASE_MODEL_OVERRIDE="${LUMEN_AIO_BASE_MODEL:-}"
SEED="${LUMEN_AIO_SEED:-42}"
VENV="${LUMEN_AIO_VENV:-$ROOT/.venv-unsloth}"
PYTHON_BIN="${LUMEN_AIO_PYTHON:-python3}"
TORCH_INDEX_URL="${LUMEN_AIO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
ASSISTANT_ONLY_LOSS="${LUMEN_AIO_ASSISTANT_ONLY_LOSS:-1}"
RESUME="${LUMEN_AIO_RESUME:-0}"
SKIP_INSTALL="${LUMEN_AIO_SKIP_INSTALL:-0}"
USE_ACTIVE_PYTHON="${LUMEN_AIO_USE_ACTIVE_PYTHON:-0}"
REQUIRE_CUDA="${LUMEN_AIO_REQUIRE_CUDA:-1}"
CONVERT_GGUF="${LUMEN_AIO_CONVERT_GGUF:-1}"
UPLOAD="${LUMEN_AIO_UPLOAD:-0}"
HF_PRIVATE="${LUMEN_AIO_HF_PRIVATE:-0}"
OVERWRITE="${LUMEN_AIO_OVERWRITE:-0}"
PREPARE_ONLY="${LUMEN_AIO_PREPARE_ONLY:-0}"

log() {
  printf '[lumen-aio] %s\n' "$*"
}

die() {
  printf '[lumen-aio] ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

if [[ ! -d "$DATASET_SOURCE" && -d "$ROOT/generated/fine_tuning" ]]; then
  DATASET_SOURCE="$ROOT/generated/fine_tuning"
fi
[[ -d "$DATASET_SOURCE" ]] || die "missing fine-tuning dataset source: $DATASET_SOURCE"
[[ -f "$DATASET_SOURCE/adapter_runtime_manifest.json" ]] || die "dataset source is missing adapter_runtime_manifest.json: $DATASET_SOURCE"

if [[ -e "$RUN_ROOT" ]]; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$RUN_ROOT"
  else
    die "run root already exists: $RUN_ROOT (set LUMEN_AIO_OVERWRITE=1 to replace it)"
  fi
fi

mkdir -p "$RUN_ROOT/generated/fine_tuning" "$RUN_ROOT/configs" "$RUN_ROOT/logs" "$RUN_ROOT/models/lora_qwen3_bootstrap" "$RUN_ROOT/models/lora_qwen3_gguf"
cp -a "$DATASET_SOURCE/." "$RUN_ROOT/generated/fine_tuning/"

log "repo root: $ROOT"
log "run root: $RUN_ROOT"
log "dataset source: $DATASET_SOURCE"
log "agents: $AGENTS_CSV"

if [[ "$USE_ACTIVE_PYTHON" == "1" ]]; then
  TRAIN_PY="$PYTHON_BIN"
else
  "$PYTHON_BIN" -m venv "$VENV"
  TRAIN_PY="$VENV/bin/python"
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  log "installing/updating Python training dependencies"
  "$TRAIN_PY" -m pip install -U pip setuptools wheel packaging ninja cmake
  "$TRAIN_PY" -m pip install -U torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
  "$TRAIN_PY" -m pip install -U \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    trl datasets transformers accelerate peft bitsandbytes sentencepiece protobuf \
    huggingface_hub hf_transfer
else
  log "LUMEN_AIO_SKIP_INSTALL=1; using existing Python environment: $TRAIN_PY"
fi

if [[ "$REQUIRE_CUDA" == "1" ]]; then
  have nvidia-smi || die "nvidia-smi not found. Install NVIDIA drivers or set LUMEN_AIO_REQUIRE_CUDA=0."
  "$TRAIN_PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch. Check the Torch wheel/CUDA driver match.")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY
fi

"$TRAIN_PY" - "$ROOT" "$RUN_ROOT" "$AGENTS_CSV" "$BASE_MODEL_OVERRIDE" "$SEED" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
run_root = Path(sys.argv[2]).resolve()
agents = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
base_override = sys.argv[4].strip()
seed = int(sys.argv[5])
src_root = run_root / "generated" / "fine_tuning"

runtime_manifest = json.loads((src_root / "adapter_runtime_manifest.json").read_text(encoding="utf-8"))
base_by_agent = {
    item["agent"]: item.get("baseModelID") or runtime_manifest.get("sharedBaseModelID") or "Qwen/Qwen3-1.7B"
    for item in runtime_manifest.get("adapters", [])
    if isinstance(item, dict) and item.get("agent")
}
adapter_repo = runtime_manifest.get("adapterRepoID") or "ales27pm/lumen-qwen3-bootstrap-adapters-gguf"

prepared = []
for agent in agents:
    agent_dir = src_root / agent
    cfg_path = agent_dir / "unsloth_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing generated config for {agent}: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = base_override or base_by_agent.get(agent) or cfg.get("base_model_name") or "Qwen/Qwen3-1.7B"
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
    adapter_gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
    release_bake = run_root / "models" / "gguf_release_bake_qwen3_bootstrap" / f"{agent}_merged_gguf"

    cfg["base_model_name"] = base
    cfg["baseModelID"] = base
    cfg["dataset_dir"] = str(agent_dir)
    cfg["output_dir"] = str(adapter_dir)
    cfg["adapter_output_dir"] = str(adapter_dir)
    cfg["adapter_gguf_output_path"] = str(adapter_gguf)
    cfg["gguf_output_dir"] = str(release_bake)
    cfg["seed"] = seed
    cfg["merge_adapters_by_default"] = False
    cfg["release_bake_enabled_by_default"] = False
    cfg.setdefault("adapterExport", {})
    cfg["adapterExport"]["trainBaseModelWeights"] = False
    cfg["adapterExport"]["mergeAdaptersByDefault"] = False
    cfg["adapterExport"]["adapterArtifact"] = str(adapter_dir)
    cfg["adapterExport"]["adapterDirectory"] = str(adapter_dir)
    cfg["adapterExport"]["adapterGGUFArtifact"] = str(adapter_gguf)

    out = run_root / "configs" / f"{agent}.json"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prepared.append({
        "agent": agent,
        "config": str(out),
        "dataset_dir": str(agent_dir),
        "base_model_name": base,
        "adapter_dir": str(adapter_dir),
        "adapter_gguf": str(adapter_gguf),
    })

run_manifest = {
    "schema": "lumen.ubuntu_train_adapters_aio/1.0.0",
    "fresh_run": True,
    "resume_default": False,
    "adapter_first": True,
    "train_base_model_weights": False,
    "adapter_repo": adapter_repo,
    "source_dataset_root": str(src_root),
    "agents": prepared,
}
(run_root / "aio_run_manifest.json").write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True))
PY

"$TRAIN_PY" - "$RUN_ROOT" "$AGENTS_CSV" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
agents = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
bad = []
for agent in agents:
    for split in ("train_sft.jsonl", "val_sft.jsonl"):
        path = run_root / "generated" / "fine_tuning" / agent / split
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages") or []
            assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            if not str(assistant).strip() or str(assistant).strip().lower() in {"null", "none"}:
                bad.append(f"{path}:{lineno}")
if bad:
    raise SystemExit("Refusing to train on empty/null assistant outputs:\n" + "\n".join(bad[:20]))
print("dataset null-output scan passed")
PY

if [[ "$PREPARE_ONLY" == "1" ]]; then
  log "LUMEN_AIO_PREPARE_ONLY=1; stopping after isolated dataset/config preparation"
  log "prepared run manifest: $RUN_ROOT/aio_run_manifest.json"
  exit 0
fi

TRAIN_ARGS=()
if [[ "$ASSISTANT_ONLY_LOSS" == "1" ]]; then
  TRAIN_ARGS+=(--assistant-only-loss)
fi
if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint)
  log "resume enabled by LUMEN_AIO_RESUME=1"
else
  log "fresh adapter training: resume disabled"
fi

while IFS= read -r agent; do
  [[ -n "$agent" ]] || continue
  log "training adapter: $agent"
  "$TRAIN_PY" "$ROOT/tools/fine_tuning/unsloth/train_sft.py" \
    --config "$RUN_ROOT/configs/$agent.json" \
    --seed "$SEED" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/train_$agent.log"
done < <(printf '%s' "$AGENTS_CSV" | tr ',' '\n')

if [[ "$CONVERT_GGUF" == "1" ]]; then
  CONVERTER="${LUMEN_AIO_LORA_CONVERTER:-$HOME/.unsloth/llama.cpp/convert_lora_to_gguf.py}"
  if [[ ! -f "$CONVERTER" ]]; then
    log "LoRA converter not found at $CONVERTER; cloning llama.cpp into run workspace"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$RUN_ROOT/llama.cpp"
    CONVERTER="$RUN_ROOT/llama.cpp/convert_lora_to_gguf.py"
  fi
  [[ -f "$CONVERTER" ]] || die "missing convert_lora_to_gguf.py"

  while IFS= read -r agent; do
    [[ -n "$agent" ]] || continue
    base_model="$("$TRAIN_PY" - "$RUN_ROOT/configs/$agent.json" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["base_model_name"])
PY
)"
    adapter_dir="$RUN_ROOT/models/lora_qwen3_bootstrap/$agent"
    outfile="$RUN_ROOT/models/lora_qwen3_gguf/lumen-$agent-lora.gguf"
    log "converting adapter to GGUF: $agent"
    "$TRAIN_PY" "$CONVERTER" "$adapter_dir" \
      --outfile "$outfile" \
      --base-model-id "$base_model" \
      2>&1 | tee "$RUN_ROOT/logs/convert_$agent.log"
  done < <(printf '%s' "$AGENTS_CSV" | tr ',' '\n')
fi

"$TRAIN_PY" - "$RUN_ROOT" "$AGENTS_CSV" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
agents = [item.strip() for item in sys.argv[2].split(",") if item.strip()]

def sha(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

summary = {
    "schema": "lumen.ubuntu_train_adapters_aio.summary/1.0.0",
    "run_root": str(run_root),
    "agents": {},
}
for agent in agents:
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
    gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
    report = adapter_dir / "training_report.json"
    summary["agents"][agent] = {
        "adapter_dir": str(adapter_dir),
        "adapter_dir_exists": adapter_dir.exists(),
        "training_report": str(report),
        "training_report_exists": report.exists(),
        "adapter_gguf": str(gguf),
        "adapter_gguf_exists": gguf.exists(),
        "adapter_gguf_sha256": sha(gguf),
        "adapter_gguf_size_bytes": gguf.stat().st_size if gguf.exists() else 0,
    }
(run_root / "aio_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

if [[ "$UPLOAD" == "1" ]]; then
  HF_CLI="$VENV/bin/hf"
  if [[ "$USE_ACTIVE_PYTHON" == "1" ]]; then
    HF_CLI="$(command -v hf || true)"
  fi
  [[ -n "$HF_CLI" && -x "$HF_CLI" ]] || die "hf CLI not found; install huggingface_hub or disable upload"
  adapter_repo="$("$TRAIN_PY" - "$RUN_ROOT/generated/fine_tuning/adapter_runtime_manifest.json" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read()).get("adapterRepoID") or "ales27pm/lumen-qwen3-bootstrap-adapters-gguf")
PY
)"
  create_args=(repos create "$adapter_repo" --type model --exist-ok)
  if [[ "$HF_PRIVATE" == "1" ]]; then
    create_args+=(--private)
  fi
  log "uploading adapter GGUFs to Hugging Face repo: $adapter_repo"
  "$HF_CLI" "${create_args[@]}"
  "$HF_CLI" upload "$adapter_repo" "$RUN_ROOT/models/lora_qwen3_gguf" "." --repo-type model
fi

log "done"
log "summary: $RUN_ROOT/aio_summary.json"
