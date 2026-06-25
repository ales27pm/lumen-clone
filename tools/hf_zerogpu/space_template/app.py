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
) -> list[dict[str, str]]:
    runtime_manifest = json.loads((source_root / "adapter_runtime_manifest.json").read_text(encoding="utf-8"))
    base_by_agent = {
        item["agent"]: item.get("baseModelID") or runtime_manifest.get("sharedBaseModelID") or "Qwen/Qwen3-1.7B"
        for item in runtime_manifest.get("adapters", [])
        if isinstance(item, dict) and item.get("agent")
    }
    prepared: list[dict[str, str]] = []
    config_root = run_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        agent_dir = source_root / agent
        cfg_path = agent_dir / "unsloth_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base = base_model_override.strip() or base_by_agent.get(agent) or cfg.get("base_model_name") or "Qwen/Qwen3-1.7B"
        adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
        adapter_gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"

        cfg["base_model_name"] = base
        cfg["baseModelID"] = base
        cfg["dataset_dir"] = str(agent_dir)
        cfg["output_dir"] = str(adapter_dir)
        cfg["adapter_output_dir"] = str(adapter_dir)
        cfg["adapter_gguf_output_path"] = str(adapter_gguf)
        cfg["seed"] = int(seed)
        cfg["merge_adapters_by_default"] = False
        cfg["release_bake_enabled_by_default"] = False
        cfg.setdefault("adapterExport", {})
        cfg["adapterExport"]["trainBaseModelWeights"] = False
        cfg["adapterExport"]["mergeAdaptersByDefault"] = False
        cfg["adapterExport"]["adapterArtifact"] = str(adapter_dir)
        cfg["adapterExport"]["adapterDirectory"] = str(adapter_dir)
        cfg["adapterExport"]["adapterGGUFArtifact"] = str(adapter_gguf)

        out = config_root / f"{agent}.json"
        out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prepared.append(
            {
                "agent": agent,
                "config": str(out),
                "dataset_dir": str(agent_dir),
                "base_model_name": base,
                "adapter_dir": str(adapter_dir),
                "adapter_gguf": str(adapter_gguf),
            }
        )
    return prepared


def _validate_nonempty_assistant_outputs(source_root: Path, agents: list[str]) -> None:
    bad: list[str] = []
    for agent in agents:
        for split in ("train_sft.jsonl", "val_sft.jsonl"):
            path = source_root / agent / split
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
            handle.write(line)
            handle.flush()
        rc = process.wait()
    if rc != 0:
        tail = ""
        if log_path.exists():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"Command failed with exit {rc}: {' '.join(command)}. See {log_path}\n{tail}")


def _convert_lora_to_gguf(run_root: Path, prepared: list[dict[str, str]], token: str) -> None:
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


def _upload_outputs(run_root: Path, prepared: list[dict[str, str]], adapter_repo: str, run_id: str, token: str, include_gguf: bool) -> dict[str, Any]:
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
) -> dict[str, Any]:
    try:
        del gpu_size
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN Space secret is required")
        agents = _csv_agents(agents_csv)
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
        _validate_nonempty_assistant_outputs(source_root, agents)
        prepared = _prepare_configs(source_root=source_root, run_root=run_root, agents=agents, base_model_override=base_model_override, seed=int(seed))

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
    with gr.Row():
        assistant_loss = gr.Checkbox(value=True, label="Assistant-only loss")
        resume = gr.Checkbox(value=False, label="Resume")
        convert = gr.Checkbox(value=True, label="Convert LoRA to GGUF")
        upload = gr.Checkbox(value=True, label="Upload outputs")
    output = gr.JSON(label="Training summary")
    run = gr.Button("Train adapters", variant="primary")
    run.click(
        fn=train_lumen_adapters,
        inputs=[run_id, agents, base_model, seed, assistant_loss, resume, convert, upload, gpu_size],
        outputs=output,
        api_name="train_lumen_adapters",
    )


if __name__ == "__main__":
    demo.queue().launch()
