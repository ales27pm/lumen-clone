#!/usr/bin/env bash
set -Eeuo pipefail

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LUMEN_AIO_EXPERIMENT_VARIANT:?Select an explicit experiment variant}"
: "${LUMEN_AIO_CONTAINER_IMAGE_DIGEST:?Declare the intended training container image sha256 digest for manual verification}"
EXPERIMENT_VARIANT="$LUMEN_AIO_EXPERIMENT_VARIANT"
CONTAINER_IMAGE_DIGEST="$LUMEN_AIO_CONTAINER_IMAGE_DIGEST"
RUN_ID_BASE="${LUMEN_AIO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "$RUN_ID_BASE" == *"-${EXPERIMENT_VARIANT}" ]]; then
  RUN_ID="$RUN_ID_BASE"
else
  RUN_ID="${RUN_ID_BASE}-${EXPERIMENT_VARIANT}"
fi
RUN_ROOT="${LUMEN_AIO_RUN_ROOT:-$ROOT/.local/ubuntu_finetune_runs/$RUN_ID}"
DATASET_SOURCE="${LUMEN_AIO_DATASET_SOURCE:-$ROOT/generated/agent_manifest/fine_tuning}"
AGENTS_CSV="${LUMEN_AIO_AGENTS:-cortex,executor,mouth,mimicry,rem,fleet}"
BASE_MODEL_OVERRIDE="${LUMEN_AIO_BASE_MODEL:-}"
SEED="${LUMEN_AIO_SEED:-42}"
VENV="${LUMEN_AIO_VENV:-$ROOT/.venv-unsloth}"
PYTHON_BIN="${LUMEN_AIO_PYTHON:-python3}"
TORCH_INDEX_URL="${LUMEN_AIO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
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

case "$EXPERIMENT_VARIANT" in
  internal_only|internal_plus_public_baseline|internal_plus_public_optimized) ;;
  *) die "unsupported experiment variant: $EXPERIMENT_VARIANT (expected internal_only, internal_plus_public_baseline, or internal_plus_public_optimized)" ;;
esac
[[ "$CONTAINER_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "LUMEN_AIO_CONTAINER_IMAGE_DIGEST must be sha256:<64 lowercase hex characters>"

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

mkdir -p "$RUN_ROOT/generated/fine_tuning" "$RUN_ROOT/configs" "$RUN_ROOT/logs" "$RUN_ROOT/training" "$RUN_ROOT/models/lora_qwen3_bootstrap" "$RUN_ROOT/models/lora_qwen3_dpo" "$RUN_ROOT/models/lora_qwen3_gguf"
cp -a "$DATASET_SOURCE/." "$RUN_ROOT/generated/fine_tuning/"

log "repo root: $ROOT"
log "run root: $RUN_ROOT"
log "dataset source: $DATASET_SOURCE"
log "agents: $AGENTS_CSV"
log "experiment variant: $EXPERIMENT_VARIANT"

if [[ "$USE_ACTIVE_PYTHON" == "1" ]]; then
  TRAIN_PY="$PYTHON_BIN"
else
  "$PYTHON_BIN" -m venv "$VENV"
  TRAIN_PY="$VENV/bin/python"
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  log "installing/updating Python training dependencies"
  "$TRAIN_PY" -m pip install pip==26.1.1 setuptools==80.9.0 wheel==0.46.3
  "$TRAIN_PY" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$TORCH_INDEX_URL"
  "$TRAIN_PY" -m pip install -r "$ROOT/tools/hf_zerogpu/space_template/requirements.txt"
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

"$TRAIN_PY" - "$ROOT" "$RUN_ROOT" "$AGENTS_CSV" "$BASE_MODEL_OVERRIDE" "$SEED" "$EXPERIMENT_VARIANT" "$CONTAINER_IMAGE_DIGEST" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
run_root = Path(sys.argv[2]).resolve()
agents = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
base_override = sys.argv[4].strip()
seed = int(sys.argv[5])
variant = sys.argv[6]
container_image_digest = sys.argv[7]
src_root = run_root / "generated" / "fine_tuning"

allowed_variants = {
    "internal_only",
    "internal_plus_public_baseline",
    "internal_plus_public_optimized",
}
if variant not in allowed_variants:
    raise SystemExit(f"Unsupported experiment variant: {variant}")

required_dataset_files = (
    "train_sft.jsonl",
    "val_sft.jsonl",
    "train_dpo.jsonl",
    "val_dpo.jsonl",
)
uncontrolled_config_fields = {
    "adapterExport", "adapter_gguf_output_path", "adapter_output_dir", "dataset_dir", "dpo_output_dir",
    "gguf_output_dir", "gguf_repo_id", "mergeExport", "output_dir",
}
runtime_lineage_config_fields = {"variant", "variantAttestation", "variantManifestSHA256"}
runtime_lineage_config_fields.update({
    "trainingContainerImageDigest",
    "trainingContainerImageDigestSource",
    "trainingRuntimeImageBindingStatus",
    "trainingRuntimeImageBindingVerified",
    "trainingEnvironmentSHA256",
})
def canonical_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def read_jsonl(path):
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise SystemExit(f"Expected JSON object at {path}:{lineno}")
        records.append(record)
    return records

def require_dataset_contract(manifest, key, records, manifest_path):
    datasets = manifest.get("datasets")
    contract = datasets.get(key) if isinstance(datasets, dict) else None
    if not isinstance(contract, dict):
        raise SystemExit(f"Experiment variant manifest is missing datasets.{key}: {manifest_path}")
    if type(contract.get("count")) is not int or contract["count"] != len(records):
        raise SystemExit(f"Experiment variant dataset count mismatch for datasets.{key}: {manifest_path}")
    if contract.get("sha256") != canonical_sha256(records):
        raise SystemExit(f"Experiment variant dataset hash mismatch for datasets.{key}: {manifest_path}")

def load_variant_manifest(agent, variant_dir):
    for filename in required_dataset_files:
        path = variant_dir / filename
        if not path.is_file():
            raise SystemExit(f"Missing generated experiment dataset for {agent}/{variant}: {path}")
    path = variant_dir / "variant_manifest.json"
    if not path.is_file():
        raise SystemExit(f"Missing generated experiment variant manifest for {agent}/{variant}: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit(f"Experiment variant manifest is not an object: {path}")
    if manifest.get("agent") != agent or manifest.get("variant") != variant:
        raise SystemExit(f"Experiment variant manifest identity mismatch: {path}")
    expected_sha = manifest.get("variantManifestSHA256")
    unsigned = dict(manifest)
    unsigned.pop("variantManifestSHA256", None)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or canonical_sha256(unsigned) != expected_sha:
        raise SystemExit(f"Experiment variant manifest integrity check failed: {path}")
    lanes = {
        filename.removesuffix(".jsonl"): read_jsonl(variant_dir / filename)
        for filename in required_dataset_files
    }
    require_dataset_contract(manifest, "trainSFT", lanes["train_sft"], path)
    require_dataset_contract(manifest, "validationSFT", lanes["val_sft"], path)
    datasets = manifest.get("datasets")
    if "trainDPO" in datasets or "validationDPO" in datasets:
        require_dataset_contract(manifest, "trainDPO", lanes["train_dpo"], path)
        require_dataset_contract(manifest, "validationDPO", lanes["val_dpo"], path)
    else:
        require_dataset_contract(manifest, "dpo", [*lanes["train_dpo"], *lanes["val_dpo"]], path)
    training_corpus = [
        *lanes["train_sft"], *lanes["val_sft"], *lanes["train_dpo"], *lanes["val_dpo"]
    ]
    if manifest.get("trainingCorpusSHA256") != canonical_sha256(training_corpus):
        raise SystemExit(f"Experiment variant training-corpus hash mismatch: {path}")
    return manifest

def training_attestation(cfg, manifest):
    controlled = manifest["controlledTrainingConfig"]
    effective_controlled = {key: cfg.get(key) for key in controlled}
    unexpected_fields = set(cfg) - set(controlled) - uncontrolled_config_fields - runtime_lineage_config_fields
    if effective_controlled != controlled or unexpected_fields:
        raise SystemExit("Effective training configuration drifted from the controlled variant")
    return {
        "schema": "lumen.training-variant-attestation/1.0.0",
        "variant": manifest["variant"],
        "variantManifestSHA256": manifest["variantManifestSHA256"],
        "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
        "laneHashes": {
            name: contract["sha256"]
            for name, contract in sorted(manifest["datasets"].items())
            if isinstance(contract, dict) and isinstance(contract.get("sha256"), str)
        },
        "effectiveTrainingConfigSHA256": canonical_sha256(effective_controlled),
        "baseModelRevision": manifest["baseModelRevision"],
        "baseModelIndexDigest": manifest["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": manifest["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": manifest["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": manifest["baseModelArtifactDigest"],
        "baseModelWeightShards": manifest["baseModelWeightShards"],
        "baseModelTokenizerDigest": manifest["baseModelTokenizerDigest"],
        "trainingEnvironmentLockSHA256": manifest["trainingEnvironmentLockSHA256"],
        "trainingEnvironmentSHA256": cfg["trainingEnvironmentSHA256"],
        "runtimeImageBindingStatus": cfg["trainingRuntimeImageBindingStatus"],
        "runtimeImageBindingVerified": cfg["trainingRuntimeImageBindingVerified"],
    }

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
    variant_dir = agent_dir / "experiments" / variant
    variant_manifest = load_variant_manifest(agent, variant_dir)
    cfg_path = agent_dir / "unsloth_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing generated config for {agent}: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    controlled = variant_manifest.get("controlledTrainingConfig")
    controlled_keys = set(controlled) if isinstance(controlled, dict) else set()
    unexpected_fields = set(cfg) - controlled_keys - uncontrolled_config_fields if isinstance(cfg, dict) else set()
    if (
        not isinstance(cfg, dict)
        or not isinstance(controlled, dict)
        or variant_manifest.get("trainingConfigSHA256") != canonical_sha256(controlled)
        or any(cfg.get(key) != value for key, value in controlled.items())
        or unexpected_fields
    ):
        raise SystemExit(f"Generated training config is not bound to the variant manifest: {cfg_path}")
    base = base_override or base_by_agent.get(agent) or cfg.get("base_model_name") or "Qwen/Qwen3-1.7B"
    if variant_manifest.get("baseModelID") != base:
        raise SystemExit(f"Base-model override would break the controlled variant for {agent}: {base}")
    if variant_manifest.get("seed") != seed:
        raise SystemExit(f"Seed override would break the controlled variant for {agent}: {seed}")
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
    training_dir = run_root / "training" / agent
    dpo_adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
    adapter_gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
    release_bake = run_root / "models" / "gguf_release_bake_qwen3_bootstrap" / f"{agent}_merged_gguf"

    cfg["base_model_name"] = base
    cfg["baseModelID"] = base
    for field in (
        "baseModelRevision",
        "baseModelIndexDigest",
        "baseModelIndexReferencedShardNames",
        "baseModelIndexShardBindingSHA256",
        "baseModelArtifactDigest",
        "baseModelWeightShards",
        "baseModelTokenizerDigest",
        "trainingEnvironmentLock",
    ):
        if cfg.get(field) != variant_manifest.get(field):
            raise SystemExit(f"{field} drifted from the controlled variant for {agent}")
    environment = {
        "schemaVersion": "lumen.adapter-training-environment/1.0.0",
        "containerImageDigest": container_image_digest,
        "containerImageDigestSource": "operator_declared",
        "runtimeImageBindingStatus": "manual_validation_required",
        "runtimeImageBindingVerified": False,
        "effectiveSeed": seed,
        "environmentLock": variant_manifest["trainingEnvironmentLock"],
    }
    cfg["trainingContainerImageDigest"] = container_image_digest
    cfg["trainingContainerImageDigestSource"] = environment["containerImageDigestSource"]
    cfg["trainingRuntimeImageBindingStatus"] = environment["runtimeImageBindingStatus"]
    cfg["trainingRuntimeImageBindingVerified"] = environment["runtimeImageBindingVerified"]
    cfg["trainingEnvironmentSHA256"] = canonical_sha256(environment)
    cfg["dataset_dir"] = str(variant_dir)
    cfg["variant"] = variant
    cfg["variantManifestSHA256"] = variant_manifest["variantManifestSHA256"]
    cfg["output_dir"] = str(training_dir)
    cfg["adapter_output_dir"] = str(adapter_dir)
    cfg["dpo_output_dir"] = str(dpo_adapter_dir)
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
    attestation = training_attestation(cfg, variant_manifest)
    cfg["variantAttestation"] = attestation

    out = run_root / "configs" / f"{agent}.json"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prepared.append({
        "agent": agent,
        "config": str(out),
        "dataset_dir": str(variant_dir),
        "variant": variant,
        "variantManifestSHA256": variant_manifest["variantManifestSHA256"],
        "variantAttestation": attestation,
        "base_model_name": base,
        "adapter_dir": str(adapter_dir),
        "training_dir": str(training_dir),
        "finalized_variant_manifest": str(training_dir / "finalized_variant_manifest.json"),
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
    "variant": variant,
    "agents": prepared,
}
(run_root / "aio_run_manifest.json").write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True))
PY

"$TRAIN_PY" - "$RUN_ROOT" "$AGENTS_CSV" "$EXPERIMENT_VARIANT" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
agents = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
variant = sys.argv[3]
bad = []
for agent in agents:
    for split in ("train_sft.jsonl", "val_sft.jsonl"):
        path = run_root / "generated" / "fine_tuning" / agent / "experiments" / variant / split
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

"$TRAIN_PY" - "$ROOT" "$RUN_ROOT" "$AGENTS_CSV" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
agents = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
sys.path.insert(0, str(root))
from tools.fine_tuning.unsloth.adapter_artifact import verify_adapter_artifact

def canonical_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

for agent in agents:
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
    finalized_path = run_root / "training" / agent / "finalized_variant_manifest.json"
    finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
    config_path = run_root / "configs" / f"{agent}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(finalized, dict) or not isinstance(config, dict):
        raise SystemExit(f"Finalized manifest or config is not a JSON object for {agent}")
    finalized_sha = finalized.get("variantManifestSHA256")
    unsigned = dict(finalized)
    unsigned.pop("variantManifestSHA256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(finalized_sha or "")) is None
        or canonical_sha256(unsigned) != finalized_sha
    ):
        raise SystemExit(f"Finalized variant manifest integrity check failed: {finalized_path}")
    if (
        finalized.get("agent") != agent
        or finalized.get("variant") != config.get("variant")
        or finalized.get("sourceVariantManifestSHA256")
        != config.get("variantManifestSHA256")
    ):
        raise SystemExit(f"Finalized variant manifest identity or source lineage mismatch: {finalized_path}")
    artifact = finalized.get("artifact") if isinstance(finalized, dict) else None
    if (
        not isinstance(artifact, dict)
        or artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft"
        or artifact.get("parentSFTAdapterSHA256") is not None
        or re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("adapterSHA256") or "")) is None
        or artifact.get("adapterManifestSHA256") != artifact.get("adapterSHA256")
    ):
        raise SystemExit(f"Finalized variant manifest lacks adapter lineage: {finalized_path}")
    attestation = config.get("variantAttestation")
    if not isinstance(attestation, dict):
        raise SystemExit(f"Prepared config lacks variant attestation: {config_path}")
    for field in (
        "baseModelRevision",
        "baseModelIndexDigest",
        "baseModelIndexReferencedShardNames",
        "baseModelIndexShardBindingSHA256",
        "baseModelArtifactDigest",
        "baseModelWeightShards",
        "baseModelTokenizerDigest",
        "trainingEnvironmentSHA256",
    ):
        if finalized.get(field) != attestation.get(field):
            raise SystemExit(f"Finalized variant manifest {field} does not match the prepared attestation")
    if (
        finalized.get("baseModelID") != config.get("baseModelID")
        or attestation.get("variant") != config.get("variant")
        or attestation.get("variantManifestSHA256") != config.get("variantManifestSHA256")
        or finalized.get("trainingCorpusSHA256") != attestation.get("trainingCorpusSHA256")
        or finalized.get("trainingConfigSHA256")
        != attestation.get("effectiveTrainingConfigSHA256")
        or not isinstance(finalized.get("trainingEnvironment"), dict)
        or canonical_sha256(finalized["trainingEnvironment"])
        != config.get("trainingEnvironmentSHA256")
        or {
            name: contract.get("sha256")
            for name, contract in sorted((finalized.get("datasets") or {}).items())
            if isinstance(contract, dict) and isinstance(contract.get("sha256"), str)
        }
        != attestation.get("laneHashes")
    ):
        raise SystemExit(f"Finalized variant manifest does not match the prepared attestation: {finalized_path}")
    verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=artifact["adapterSHA256"],
        expected_training_phase="sft",
    )
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        not isinstance(adapter_config, dict)
        or adapter_config.get("base_model_name_or_path") != config.get("base_model_name")
    ):
        raise SystemExit(f"Adapter base model does not match the prepared config: {adapter_dir}")
print("canonical adapter artifact verification passed")
PY

if [[ "$CONVERT_GGUF" == "1" ]]; then
  CONVERTER="${LUMEN_AIO_LORA_CONVERTER:-$HOME/.unsloth/llama.cpp/convert_lora_to_gguf.py}"
  LLAMA_CPP_REVISION="34558825a27f4d74dcfd7a91bfde4464baa2a30a"
  if [[ ! -f "$CONVERTER" ]]; then
    log "LoRA converter not found at $CONVERTER; cloning llama.cpp into run workspace"
    git init "$RUN_ROOT/llama.cpp"
    git -C "$RUN_ROOT/llama.cpp" remote add origin https://github.com/ggml-org/llama.cpp
    git -C "$RUN_ROOT/llama.cpp" fetch --depth 1 origin "$LLAMA_CPP_REVISION"
    git -C "$RUN_ROOT/llama.cpp" checkout --detach FETCH_HEAD
    CONVERTER="$RUN_ROOT/llama.cpp/convert_lora_to_gguf.py"
  fi
  [[ -f "$CONVERTER" ]] || die "missing convert_lora_to_gguf.py"
  [[ "$(git -C "$(dirname "$CONVERTER")" rev-parse HEAD)" == "$LLAMA_CPP_REVISION" ]] || die "llama.cpp converter revision does not match the pinned training environment"

  while IFS= read -r agent; do
    [[ -n "$agent" ]] || continue
    base_model="$("$TRAIN_PY" - "$RUN_ROOT/configs/$agent.json" <<'PY'
import hashlib, json, os, sys
from huggingface_hub import snapshot_download
cfg=json.loads(open(sys.argv[1], encoding="utf-8").read())
snapshot=snapshot_download(repo_id=cfg["base_model_name"], revision=cfg["baseModelRevision"])
def sha256(path):
    digest=hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
for filename, expected in (("model.safetensors.index.json", cfg["baseModelIndexDigest"]), ("tokenizer.json", cfg["baseModelTokenizerDigest"])):
    digest=sha256(f"{snapshot}/{filename}")
    if digest != expected:
        raise SystemExit(f"Pinned base-model artifact digest mismatch during conversion: {filename}")
shards=sorted(cfg["baseModelWeightShards"], key=lambda item: item["filename"])
contract={"schemaVersion":"lumen.base-model-weight-shards/1.0.0","shards":shards}
if hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != cfg["baseModelArtifactDigest"]:
    raise SystemExit("Base-model artifact digest is not bound to the declared weight shards")
index=json.load(open(f"{snapshot}/model.safetensors.index.json", encoding="utf-8"))
referenced=sorted(set(index.get("weight_map", {}).values()))
if referenced != [item["filename"] for item in shards] or referenced != cfg["baseModelIndexReferencedShardNames"]:
    raise SystemExit("Base-model index shard set does not match the declared weight shards")
binding={
    "schemaVersion":"lumen.base-model-index-shard-binding/1.0.0",
    "indexDigest":cfg["baseModelIndexDigest"],
    "referencedShardNames":referenced,
    "shardContractDigest":cfg["baseModelArtifactDigest"],
}
if hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != cfg["baseModelIndexShardBindingSHA256"]:
    raise SystemExit("Base-model index-to-shard binding digest mismatch during conversion")
for item in shards:
    path=f"{snapshot}/{item['filename']}"
    digest=sha256(path)
    if os.path.getsize(path) != item["size"] or digest != item["sha256"]:
        raise SystemExit(f"Pinned base-model weight shard mismatch during conversion: {item['filename']}")
print(snapshot)
PY
)"
    adapter_dir="$RUN_ROOT/models/lora_qwen3_bootstrap/$agent"
    outfile="$RUN_ROOT/models/lora_qwen3_gguf/lumen-$agent-lora.gguf"
    log "converting adapter to GGUF: $agent"
    "$TRAIN_PY" "$CONVERTER" "$adapter_dir" \
      --outfile "$outfile" \
      --base "$base_model" \
      2>&1 | tee "$RUN_ROOT/logs/convert_$agent.log"
  done < <(printf '%s' "$AGENTS_CSV" | tr ',' '\n')
fi

"$TRAIN_PY" - "$RUN_ROOT" "$AGENTS_CSV" "$EXPERIMENT_VARIANT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
agents = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
variant = sys.argv[3]

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
    "variant": variant,
    "agents": {},
}
for agent in agents:
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
    gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
    training_dir = run_root / "training" / agent
    report = training_dir / "training_report.json"
    finalized_manifest = training_dir / "finalized_variant_manifest.json"
    finalized = json.loads(finalized_manifest.read_text(encoding="utf-8"))
    config = json.loads((run_root / "configs" / f"{agent}.json").read_text(encoding="utf-8"))
    summary["agents"][agent] = {
        "adapter_dir": str(adapter_dir),
        "adapter_dir_exists": adapter_dir.exists(),
        "training_report": str(report),
        "training_report_exists": report.exists(),
        "finalized_variant_manifest": str(finalized_manifest),
        "finalized_variant_manifest_sha256": finalized["variantManifestSHA256"],
        "adapter_sha256": finalized["artifact"]["adapterSHA256"],
        "adapter_gguf": str(gguf),
        "adapter_gguf_exists": gguf.exists(),
        "adapter_gguf_sha256": sha(gguf),
        "adapter_gguf_size_bytes": gguf.stat().st_size if gguf.exists() else 0,
        "variant": config["variant"],
        "variantManifestSHA256": config["variantManifestSHA256"],
        "variantAttestation": config["variantAttestation"],
        "dataset_dir": config["dataset_dir"],
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
  log "uploading canonical adapters, finalized manifests, and optional GGUFs to Hugging Face repo: $adapter_repo"
  "$HF_CLI" "${create_args[@]}"
  while IFS= read -r agent; do
    [[ -n "$agent" ]] || continue
    "$HF_CLI" upload "$adapter_repo" "$RUN_ROOT/models/lora_qwen3_bootstrap/$agent" "runs/$RUN_ID/adapters/$agent" --repo-type model
    "$HF_CLI" upload "$adapter_repo" "$RUN_ROOT/training/$agent/finalized_variant_manifest.json" "runs/$RUN_ID/manifests/$agent/variant_manifest.json" --repo-type model
  done < <(printf '%s' "$AGENTS_CSV" | tr ',' '\n')
  if [[ "$CONVERT_GGUF" == "1" ]]; then
    "$HF_CLI" upload "$adapter_repo" "$RUN_ROOT/models/lora_qwen3_gguf" "runs/$RUN_ID/gguf" --repo-type model
  fi
fi

log "done"
log "summary: $RUN_ROOT/aio_summary.json"
