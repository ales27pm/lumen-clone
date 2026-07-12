from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import spaces
from huggingface_hub import HfApi, snapshot_download


APP_ROOT = Path(__file__).resolve().parent
DEFAULTS = json.loads((APP_ROOT / "lumen_zero_gpu_defaults.json").read_text(encoding="utf-8"))
DEFAULT_GPU_SIZE = os.environ.get("LUMEN_ZERO_GPU_SIZE", str(DEFAULTS.get("gpu_size", "large")))
MAX_ZERO_GPU_DURATION = int(os.environ.get("LUMEN_ZERO_GPU_MAX_DURATION_SECONDS", "1200"))
REQUESTED_GPU_DURATION = int(os.environ.get("LUMEN_ZERO_GPU_DURATION_SECONDS", str(DEFAULTS.get("gpu_duration_seconds", 1200))))
DEFAULT_GPU_DURATION = min(REQUESTED_GPU_DURATION, MAX_ZERO_GPU_DURATION)
AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
EXPERIMENT_VARIANTS = (
    "internal_only",
    "internal_plus_public_baseline",
    "internal_plus_public_optimized",
)
DEFAULT_EXPERIMENT_VARIANT = str(
    os.environ.get(
        "LUMEN_ZERO_GPU_EXPERIMENT_VARIANT",
        DEFAULTS.get("experiment_variant", "internal_plus_public_optimized"),
    )
)
REQUIRED_VARIANT_DATASET_FILES = (
    "train_sft.jsonl",
    "val_sft.jsonl",
    "train_dpo.jsonl",
    "val_dpo.jsonl",
)
UNCONTROLLED_CONFIG_FIELDS = {
    "adapterExport",
    "adapter_gguf_output_path",
    "adapter_output_dir",
    "dataset_dir",
    "gguf_output_dir",
    "gguf_repo_id",
    "mergeExport",
    "output_dir",
}
RUNTIME_LINEAGE_CONFIG_FIELDS = {
    "variant",
    "variantAttestation",
    "variantManifestSHA256",
}


def _csv_agents(value: str) -> list[str]:
    agents = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = [agent for agent in agents if agent not in AGENTS]
    if unsupported:
        raise ValueError(f"Unsupported agents: {', '.join(unsupported)}")
    if not agents:
        raise ValueError("Select at least one agent")
    return agents


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _experiment_variant(value: str) -> str:
    variant = value.strip()
    if variant not in EXPERIMENT_VARIANTS:
        raise ValueError(
            f"Unsupported experiment variant: {variant or '<empty>'}. "
            f"Expected one of: {', '.join(EXPERIMENT_VARIANTS)}"
        )
    return variant


DEFAULT_EXPERIMENT_VARIANT = _experiment_variant(DEFAULT_EXPERIMENT_VARIANT)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{lineno}")
        records.append(record)
    return records


def _require_dataset_contract(
    manifest: dict[str, Any],
    *,
    key: str,
    records: list[dict[str, Any]],
    manifest_path: Path,
) -> None:
    datasets = manifest.get("datasets")
    contract = datasets.get(key) if isinstance(datasets, dict) else None
    if not isinstance(contract, dict):
        raise ValueError(f"Experiment variant manifest is missing datasets.{key}: {manifest_path}")
    if type(contract.get("count")) is not int or contract["count"] != len(records):
        raise ValueError(f"Experiment variant dataset count mismatch for datasets.{key}: {manifest_path}")
    if contract.get("sha256") != _canonical_sha256(records):
        raise ValueError(f"Experiment variant dataset hash mismatch for datasets.{key}: {manifest_path}")


def _variant_dataset(agent_root: Path, *, agent: str, variant: str) -> tuple[Path, dict[str, Any]]:
    variant = _experiment_variant(variant)
    variant_root = agent_root / "experiments" / variant
    for filename in REQUIRED_VARIANT_DATASET_FILES:
        path = variant_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated experiment dataset for {agent}/{variant}: {path}")
    manifest_path = variant_root / "variant_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing generated experiment variant manifest for {agent}/{variant}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Experiment variant manifest is not an object: {manifest_path}")
    if manifest.get("agent") != agent or manifest.get("variant") != variant:
        raise ValueError(f"Experiment variant manifest identity mismatch: {manifest_path}")
    expected_sha = manifest.get("variantManifestSHA256")
    unsigned = dict(manifest)
    unsigned.pop("variantManifestSHA256", None)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or _canonical_sha256(unsigned) != expected_sha:
        raise ValueError(f"Experiment variant manifest integrity check failed: {manifest_path}")
    lanes = {
        filename.removesuffix(".jsonl"): _read_jsonl(variant_root / filename)
        for filename in REQUIRED_VARIANT_DATASET_FILES
    }
    _require_dataset_contract(manifest, key="trainSFT", records=lanes["train_sft"], manifest_path=manifest_path)
    _require_dataset_contract(manifest, key="validationSFT", records=lanes["val_sft"], manifest_path=manifest_path)
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    if "trainDPO" in datasets or "validationDPO" in datasets:
        _require_dataset_contract(manifest, key="trainDPO", records=lanes["train_dpo"], manifest_path=manifest_path)
        _require_dataset_contract(manifest, key="validationDPO", records=lanes["val_dpo"], manifest_path=manifest_path)
    else:
        _require_dataset_contract(
            manifest,
            key="dpo",
            records=[*lanes["train_dpo"], *lanes["val_dpo"]],
            manifest_path=manifest_path,
        )
    training_corpus = [
        *lanes["train_sft"],
        *lanes["val_sft"],
        *lanes["train_dpo"],
        *lanes["val_dpo"],
    ]
    if manifest.get("trainingCorpusSHA256") != _canonical_sha256(training_corpus):
        raise ValueError(f"Experiment variant training-corpus hash mismatch: {manifest_path}")
    return variant_root, manifest


def _training_attestation(cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    datasets = manifest["datasets"]
    controlled = manifest["controlledTrainingConfig"]
    effective_controlled = {key: cfg.get(key) for key in controlled}
    unexpected_fields = set(cfg) - set(controlled) - UNCONTROLLED_CONFIG_FIELDS - RUNTIME_LINEAGE_CONFIG_FIELDS
    if effective_controlled != controlled or unexpected_fields:
        raise ValueError("Effective training configuration drifted from the controlled variant")
    return {
        "schema": "lumen.training-variant-attestation/1.0.0",
        "variant": manifest["variant"],
        "variantManifestSHA256": manifest["variantManifestSHA256"],
        "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
        "laneHashes": {
            name: contract["sha256"]
            for name, contract in sorted(datasets.items())
            if isinstance(contract, dict) and isinstance(contract.get("sha256"), str)
        },
        "effectiveTrainingConfigSHA256": _canonical_sha256(effective_controlled),
    }


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _copy_dataset_snapshot(run_root: Path, dataset_repo: str, revision: str, path_in_repo: str, token: str) -> Path:
    allow_pattern = f"{path_in_repo}/**"
    snapshot = Path(
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[allow_pattern],
            token=token,
        )
    )
    source = snapshot / path_in_repo
    if not source.exists():
        raise FileNotFoundError(f"Downloaded dataset snapshot did not contain {path_in_repo}")
    target = run_root / "generated" / "fine_tuning"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def _prepare_configs(
    *,
    source_root: Path,
    run_root: Path,
    agents: list[str],
    base_model_override: str,
    seed: int,
    variant: str,
) -> list[dict[str, Any]]:
    runtime_manifest = json.loads((source_root / "adapter_runtime_manifest.json").read_text(encoding="utf-8"))
    base_by_agent = {
        item["agent"]: item.get("baseModelID") or runtime_manifest.get("sharedBaseModelID") or "Qwen/Qwen3-1.7B"
        for item in runtime_manifest.get("adapters", [])
        if isinstance(item, dict) and item.get("agent")
    }
    prepared: list[dict[str, Any]] = []
    config_root = run_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        agent_dir = source_root / agent
        variant_dir, variant_manifest = _variant_dataset(agent_dir, agent=agent, variant=variant)
        cfg_path = agent_dir / "unsloth_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"Generated training config is not an object: {cfg_path}")
        controlled = variant_manifest.get("controlledTrainingConfig")
        controlled_keys = set(controlled) if isinstance(controlled, dict) else set()
        unexpected_fields = set(cfg) - controlled_keys - UNCONTROLLED_CONFIG_FIELDS
        if (
            not isinstance(controlled, dict)
            or variant_manifest.get("trainingConfigSHA256") != _canonical_sha256(controlled)
            or any(cfg.get(key) != value for key, value in controlled.items())
            or unexpected_fields
        ):
            raise ValueError(f"Generated training config is not bound to the variant manifest: {cfg_path}")
        base = base_model_override.strip() or base_by_agent.get(agent) or cfg.get("base_model_name") or "Qwen/Qwen3-1.7B"
        if variant_manifest.get("baseModelID") != base:
            raise ValueError(f"Base-model override would break the controlled variant for {agent}: {base}")
        if variant_manifest.get("seed") != int(seed):
            raise ValueError(f"Seed override would break the controlled variant for {agent}: {seed}")
        adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
        adapter_gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"

        cfg["base_model_name"] = base
        cfg["baseModelID"] = base
        cfg["dataset_dir"] = str(variant_dir)
        cfg["variant"] = variant
        cfg["variantManifestSHA256"] = variant_manifest["variantManifestSHA256"]
        cfg["output_dir"] = str(adapter_dir)
        cfg["adapter_output_dir"] = str(adapter_dir)
        cfg["adapter_gguf_output_path"] = str(adapter_gguf)
        cfg["seed"] = int(seed)
        cfg["merge_adapters_by_default"] = False
        cfg["release_bake_enabled_by_default"] = False
        for env_name, key in (
            ("LUMEN_ZERO_GPU_MAX_TRAIN_RECORDS", "max_train_records"),
            ("LUMEN_ZERO_GPU_MAX_VAL_RECORDS", "max_val_records"),
            ("LUMEN_ZERO_GPU_MAX_SEQ_LENGTH", "max_seq_length"),
            ("LUMEN_ZERO_GPU_NUM_TRAIN_EPOCHS", "num_train_epochs"),
        ):
            override = _optional_int_env(env_name)
            if override is not None and override != cfg.get(key):
                raise ValueError(f"{env_name} would break the controlled variant for {agent}: {override}")
        cfg.setdefault("adapterExport", {})
        cfg["adapterExport"]["trainBaseModelWeights"] = False
        cfg["adapterExport"]["mergeAdaptersByDefault"] = False
        cfg["adapterExport"]["adapterArtifact"] = str(adapter_dir)
        cfg["adapterExport"]["adapterDirectory"] = str(adapter_dir)
        cfg["adapterExport"]["adapterGGUFArtifact"] = str(adapter_gguf)
        attestation = _training_attestation(cfg, variant_manifest)
        cfg["variantAttestation"] = attestation

        out = config_root / f"{agent}.json"
        out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prepared.append(
            {
                "agent": agent,
                "config": str(out),
                "dataset_dir": str(variant_dir),
                "variant": variant,
                "variantManifestSHA256": variant_manifest["variantManifestSHA256"],
                "variantAttestation": attestation,
                "base_model_name": base,
                "adapter_dir": str(adapter_dir),
                "adapter_gguf": str(adapter_gguf),
            }
        )
    return prepared


def _validate_nonempty_assistant_outputs(source_root: Path, agents: list[str], variant: str) -> None:
    bad: list[str] = []
    for agent in agents:
        variant_root, _ = _variant_dataset(source_root / agent, agent=agent, variant=variant)
        for split in ("train_sft.jsonl", "val_sft.jsonl"):
            path = variant_root / split
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = record.get("messages") or []
                assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
                if not str(assistant).strip() or str(assistant).strip().lower() in {"null", "none"}:
                    bad.append(f"{path}:{lineno}")
    if bad:
        raise RuntimeError("Refusing to train on empty/null assistant outputs:\n" + "\n".join(bad[:20]))


def _run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        rc = process.wait()
    if rc != 0:
        tail = ""
        if log_path.exists():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"Command failed with exit {rc}: {' '.join(command)}. See {log_path}\n{tail}")


def _convert_lora_to_gguf(run_root: Path, prepared: list[dict[str, Any]], token: str) -> None:
    converter = Path(os.environ.get("LUMEN_LORA_CONVERTER", str(Path.home() / ".unsloth/llama.cpp/convert_lora_to_gguf.py")))
    if not converter.exists():
        clone_dir = run_root / "llama.cpp"
        _run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(clone_dir)], cwd=run_root, log_path=run_root / "logs" / "clone_llama_cpp.log")
        converter = clone_dir / "convert_lora_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Missing convert_lora_to_gguf.py: {converter}")

    for item in prepared:
        agent = item["agent"]
        outfile = Path(item["adapter_gguf"])
        outfile.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                str(converter),
                item["adapter_dir"],
                "--outfile",
                str(outfile),
                "--base-model-id",
                item["base_model_name"],
            ],
            cwd=run_root,
            log_path=run_root / "logs" / f"convert_{agent}.log",
        )


def _upload_outputs(run_root: Path, prepared: list[dict[str, Any]], adapter_repo: str, run_id: str, token: str, include_gguf: bool) -> dict[str, Any]:
    api = HfApi(token=token)
    private = os.environ.get("LUMEN_ZERO_GPU_PRIVATE_ADAPTERS", "1") == "1"
    api.create_repo(repo_id=adapter_repo, repo_type="model", private=private, exist_ok=True, token=token)

    uploaded: dict[str, Any] = {}
    for item in prepared:
        agent = item["agent"]
        adapter_dir = Path(item["adapter_dir"])
        adapter_path = f"runs/{run_id}/adapters/{agent}"
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=adapter_repo,
            repo_type="model",
            path_in_repo=adapter_path,
            commit_message=f"Upload Lumen {agent} adapter {run_id}",
            token=token,
        )
        entry: dict[str, Any] = {
            "adapter_dir": str(adapter_dir),
            "adapter_repo": adapter_repo,
            "adapter_path_in_repo": adapter_path,
            "variant": item["variant"],
            "variantManifestSHA256": item["variantManifestSHA256"],
        }
        gguf = Path(item["adapter_gguf"])
        if include_gguf and gguf.exists():
            gguf_path = f"runs/{run_id}/lora_gguf/{gguf.name}"
            api.upload_file(
                path_or_fileobj=str(gguf),
                repo_id=adapter_repo,
                repo_type="model",
                path_in_repo=gguf_path,
                commit_message=f"Upload Lumen {agent} LoRA GGUF {run_id}",
                token=token,
            )
            entry["adapter_gguf_path_in_repo"] = gguf_path
            entry["adapter_gguf_sha256"] = _sha256(gguf)
            entry["adapter_gguf_size_bytes"] = gguf.stat().st_size
        uploaded[agent] = entry
    return uploaded


@spaces.GPU(size=DEFAULT_GPU_SIZE, duration=DEFAULT_GPU_DURATION)
def train_lumen_adapters(
    run_id: str,
    agents_csv: str,
    base_model_override: str,
    seed: int,
    assistant_only_loss: bool,
    resume: bool,
    convert_gguf: bool,
    upload_outputs: bool,
    gpu_size: str,
    experiment_variant: str = DEFAULT_EXPERIMENT_VARIANT,
) -> dict[str, Any]:
    try:
        del gpu_size
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN Space secret is required")
        agents = _csv_agents(agents_csv)
        experiment_variant = _experiment_variant(experiment_variant)
        dataset_repo = os.environ.get("LUMEN_ZERO_GPU_DATASET_REPO", str(DEFAULTS["dataset_repo"]))
        dataset_revision = os.environ.get("LUMEN_ZERO_GPU_DATASET_REVISION", str(DEFAULTS.get("dataset_revision", "main")))
        dataset_path = os.environ.get("LUMEN_ZERO_GPU_DATASET_PATH", str(DEFAULTS["dataset_path_in_repo"]))
        adapter_repo = os.environ.get("LUMEN_ZERO_GPU_ADAPTER_REPO", str(DEFAULTS["adapter_repo"]))
        run_id = run_id.strip() or os.environ.get("LUMEN_ZERO_GPU_RUN_ID", str(DEFAULTS["run_id"]))

        run_root = Path(os.environ.get("LUMEN_ZERO_GPU_WORKDIR", "/tmp/lumen_zerogpu_runs")) / run_id
        if run_root.exists() and not resume:
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)

        source_root = _copy_dataset_snapshot(run_root, dataset_repo, dataset_revision, dataset_path, token)
        _validate_nonempty_assistant_outputs(source_root, agents, experiment_variant)
        prepared = _prepare_configs(
            source_root=source_root,
            run_root=run_root,
            agents=agents,
            base_model_override=base_model_override,
            seed=int(seed),
            variant=experiment_variant,
        )
        run_manifest = {
            "schema": "lumen.zerogpu.training_run/1.0.0",
            "run_id": run_id,
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "dataset_path": dataset_path,
            "variant": experiment_variant,
            "agents": prepared,
        }
        (run_root / "lumen_zerogpu_run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for item in prepared:
            agent = item["agent"]
            command = [sys.executable, str(APP_ROOT / "lumen_train_sft.py"), "--config", item["config"], "--seed", str(seed)]
            if assistant_only_loss:
                command.append("--assistant-only-loss")
            if resume:
                command.append("--resume-from-checkpoint")
            _run(command, cwd=APP_ROOT, log_path=run_root / "logs" / f"train_{agent}.log")

        if convert_gguf:
            _convert_lora_to_gguf(run_root, prepared, token)

        for item in prepared:
            gguf = Path(item["adapter_gguf"])
            if gguf.exists():
                item["adapter_gguf_sha256"] = _sha256(gguf)
                item["adapter_gguf_size_bytes"] = gguf.stat().st_size

        uploads = _upload_outputs(run_root, prepared, adapter_repo, run_id, token, convert_gguf) if upload_outputs else {}
        summary = {
            "schema": "lumen.zerogpu.training_summary/1.0.0",
            "ok": True,
            "run_id": run_id,
            "run_root": str(run_root),
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "dataset_path": dataset_path,
            "adapter_repo": adapter_repo,
            "variant": experiment_variant,
            "run_manifest": str(run_root / "lumen_zerogpu_run_manifest.json"),
            "agents": prepared,
            "uploads": uploads,
            "fresh_run": not resume,
            "resume": bool(resume),
            "assistant_only_loss": bool(assistant_only_loss),
            "convert_gguf": bool(convert_gguf),
        }
        (run_root / "lumen_zerogpu_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    except Exception as exc:
        return {
            "schema": "lumen.zerogpu.training_summary/1.0.0",
            "ok": False,
            "run_id": run_id,
            "variant": experiment_variant,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }


with gr.Blocks() as demo:
    gr.Markdown("# Lumen ZeroGPU Adapter Trainer")
    with gr.Row():
        run_id = gr.Textbox(value=str(DEFAULTS.get("run_id", "")), label="Run ID")
        agents = gr.Textbox(value=",".join(DEFAULTS.get("agents", AGENTS)), label="Agents")
    with gr.Row():
        base_model = gr.Textbox(value=str(DEFAULTS.get("base_model_override", "")), label="Base model override")
        seed = gr.Number(value=42, precision=0, label="Seed")
        gpu_size = gr.Dropdown(choices=["large", "xlarge"], value=DEFAULT_GPU_SIZE, label="ZeroGPU size")
        experiment_variant = gr.Dropdown(
            choices=list(EXPERIMENT_VARIANTS),
            value=DEFAULT_EXPERIMENT_VARIANT,
            label="Experiment variant",
        )
    with gr.Row():
        assistant_loss = gr.Checkbox(value=True, label="Assistant-only loss")
        resume = gr.Checkbox(value=False, label="Resume")
        convert = gr.Checkbox(value=True, label="Convert LoRA to GGUF")
        upload = gr.Checkbox(value=True, label="Upload outputs")
    output = gr.JSON(label="Training summary")
    run = gr.Button("Train adapters", variant="primary")
    run.click(
        fn=train_lumen_adapters,
        inputs=[run_id, agents, base_model, seed, assistant_loss, resume, convert, upload, gpu_size, experiment_variant],
        outputs=output,
        api_name="train_lumen_adapters",
    )


if __name__ == "__main__":
    demo.queue().launch()
