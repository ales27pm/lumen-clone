#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
SPACE_TEMPLATE = Path(__file__).resolve().parent / "space_template"


@dataclass(frozen=True)
class SpaceBuild:
    run_id: str
    run_root: Path
    space_dir: Path
    dataset_dir: Path
    dataset_path_in_repo: str
    defaults_path: Path


def parse_agents(value: str) -> list[str]:
    agents = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = [agent for agent in agents if agent not in AGENTS]
    if unsupported:
        raise ValueError(f"Unsupported agents: {', '.join(unsupported)}. Expected subset of: {', '.join(AGENTS)}")
    if not agents:
        raise ValueError("At least one agent must be selected")
    return agents


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def require_dataset_source(path: Path, agents: Sequence[str]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset source: {path}")
    if not (path / "adapter_runtime_manifest.json").exists():
        raise FileNotFoundError(f"Missing adapter_runtime_manifest.json in {path}")
    for agent in agents:
        agent_dir = path / agent
        for filename in ("train_sft.jsonl", "val_sft.jsonl", "unsloth_config.json"):
            required = agent_dir / filename
            if not required.exists():
                raise FileNotFoundError(f"Missing required fine-tuning file: {required}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_template_tree(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def write_space_bundle(
    *,
    root: Path,
    run_id: str,
    run_root: Path,
    dataset_source: Path,
    space_repo: str,
    dataset_repo: str,
    adapter_repo: str,
    agents: Sequence[str],
    base_model: str,
    gpu_size: str,
    gpu_duration_seconds: int,
) -> SpaceBuild:
    run_root.mkdir(parents=True, exist_ok=True)
    space_dir = run_root / "space"
    dataset_dir = run_root / "dataset_snapshot" / "fine_tuning"
    reset_dir(space_dir)
    reset_dir(dataset_dir)

    shutil.copytree(dataset_source, dataset_dir, dirs_exist_ok=True)
    copy_template_tree(SPACE_TEMPLATE, space_dir)
    shutil.copy2(root / "tools/fine_tuning/unsloth/train_sft.py", space_dir / "lumen_train_sft.py")

    dataset_path_in_repo = f"runs/{run_id}/fine_tuning"
    defaults = {
        "schema": "lumen.zerogpu.defaults/1.0.0",
        "run_id": run_id,
        "space_repo": space_repo,
        "dataset_repo": dataset_repo,
        "dataset_revision": "main",
        "dataset_path_in_repo": dataset_path_in_repo,
        "adapter_repo": adapter_repo,
        "agents": list(agents),
        "base_model_override": base_model,
        "gpu_size": gpu_size,
        "gpu_duration_seconds": gpu_duration_seconds,
        "fresh_run": True,
        "resume_default": False,
        "adapter_first": True,
    }
    defaults_path = space_dir / "lumen_zero_gpu_defaults.json"
    defaults_path.write_text(json.dumps(defaults, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = (space_dir / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("{{SPACE_REPO}}", space_repo)
    readme = readme.replace("{{DATASET_REPO}}", dataset_repo)
    readme = readme.replace("{{ADAPTER_REPO}}", adapter_repo)
    readme = readme.replace("{{GPU_SIZE}}", gpu_size)
    readme = readme.replace("{{GPU_DURATION_SECONDS}}", str(gpu_duration_seconds))
    (space_dir / "README.md").write_text(readme, encoding="utf-8")

    return SpaceBuild(
        run_id=run_id,
        run_root=run_root,
        space_dir=space_dir,
        dataset_dir=dataset_dir,
        dataset_path_in_repo=dataset_path_in_repo,
        defaults_path=defaults_path,
    )


def import_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Missing huggingface_hub. Install with: pip install -U huggingface_hub") from exc
    return HfApi


def add_space_value(api: Any, *, repo_id: str, key: str, value: str, secret: bool, token: str | None, dry_run: bool) -> None:
    label = "secret" if secret else "variable"
    print(f"Set Space {label}: {key}")
    if dry_run:
        return
    method_name = "add_space_secret" if secret else "add_space_variable"
    method = getattr(api, method_name, None)
    if method is None:
        print(f"warning: installed huggingface_hub has no {method_name}; set {key} manually in Space settings", file=sys.stderr)
        return
    method(repo_id=repo_id, key=key, value=value, token=token)


def request_zerogpu_hardware(api: Any, *, repo_id: str, hardware: str, token: str | None, dry_run: bool) -> None:
    print(f"Request Space hardware: {hardware}")
    if dry_run:
        return
    method = getattr(api, "request_space_hardware", None)
    if method is None:
        print("warning: installed huggingface_hub has no request_space_hardware; select ZeroGPU in Space settings", file=sys.stderr)
        return
    try:
        method(repo_id=repo_id, hardware=hardware, token=token)
    except Exception as exc:
        print(f"warning: could not request hardware '{hardware}': {exc}", file=sys.stderr)
        print("warning: select ZeroGPU manually in the Space settings if the Hub API rejected this hardware id", file=sys.stderr)


def wait_for_space_revision(api: Any, *, repo_id: str, token: str | None, timeout_seconds: int, dry_run: bool) -> None:
    if dry_run:
        return
    info = api.space_info(repo_id, files_metadata=False, token=token)
    target_sha = getattr(info, "sha", None)
    if not target_sha:
        print("warning: could not determine Space target revision before trigger", file=sys.stderr)
        return
    print(f"Wait for Space runtime revision: {target_sha}")
    started = time.monotonic()
    last_status: dict[str, Any] = {}
    while time.monotonic() - started < timeout_seconds:
        runtime = api.get_space_runtime(repo_id)
        raw = getattr(runtime, "raw", {}) or {}
        runtime_sha = raw.get("sha")
        stage = getattr(runtime, "stage", None)
        last_status = {"stage": stage, "runtime_sha": runtime_sha, "target_sha": target_sha}
        print(json.dumps(last_status, sort_keys=True))
        if stage == "RUNNING" and runtime_sha == target_sha:
            return
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for Space runtime revision: {last_status}")


def upload_to_hub(
    *,
    build: SpaceBuild,
    space_repo: str,
    dataset_repo: str,
    adapter_repo: str,
    private_space: bool,
    private_dataset: bool,
    private_adapters: bool,
    zero_gpu_hardware: str,
    token: str | None,
    dry_run: bool,
) -> None:
    HfApi = import_hf_api()
    api = HfApi(token=token)

    print(f"Create/update dataset repo: {dataset_repo}")
    print(f"Create/update adapter repo: {adapter_repo}")
    print(f"Create/update Space repo: {space_repo}")
    if not dry_run:
        api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=private_dataset, exist_ok=True, token=token)
        api.create_repo(repo_id=adapter_repo, repo_type="model", private=private_adapters, exist_ok=True, token=token)
        try:
            api.create_repo(
                repo_id=space_repo,
                repo_type="space",
                space_sdk="gradio",
                private=private_space,
                exist_ok=True,
                token=token,
            )
        except TypeError:
            api.create_repo(
                repo_id=space_repo,
                repo_type="space",
                private=private_space,
                exist_ok=True,
                token=token,
            )

    print(f"Upload dataset snapshot: {build.dataset_dir} -> {dataset_repo}/{build.dataset_path_in_repo}")
    if not dry_run:
        api.upload_folder(
            folder_path=str(build.dataset_dir),
            repo_id=dataset_repo,
            repo_type="dataset",
            path_in_repo=build.dataset_path_in_repo,
            commit_message=f"Upload Lumen fine-tuning dataset snapshot {build.run_id}",
            token=token,
        )

    print(f"Upload Space bundle: {build.space_dir} -> {space_repo}")
    if not dry_run:
        api.upload_folder(
            folder_path=str(build.space_dir),
            repo_id=space_repo,
            repo_type="space",
            commit_message=f"Update Lumen ZeroGPU trainer {build.run_id}",
            token=token,
        )

    if token:
        add_space_value(api, repo_id=space_repo, key="HF_TOKEN", value=token, secret=True, token=token, dry_run=dry_run)
    else:
        print("warning: HF_TOKEN is not set locally; add it as a Space secret before triggering training", file=sys.stderr)

    defaults = read_json(build.defaults_path)
    gpu_duration_seconds = str(defaults.get("gpu_duration_seconds", 1200))
    variables = {
        "LUMEN_ZERO_GPU_DATASET_REPO": dataset_repo,
        "LUMEN_ZERO_GPU_DATASET_REVISION": "main",
        "LUMEN_ZERO_GPU_DATASET_PATH": build.dataset_path_in_repo,
        "LUMEN_ZERO_GPU_ADAPTER_REPO": adapter_repo,
        "LUMEN_ZERO_GPU_RUN_ID": build.run_id,
        "LUMEN_ZERO_GPU_SIZE": str(defaults.get("gpu_size", "large")),
        "LUMEN_ZERO_GPU_DURATION_SECONDS": gpu_duration_seconds,
        "LUMEN_ZERO_GPU_MAX_DURATION_SECONDS": gpu_duration_seconds,
    }
    for key, value in variables.items():
        add_space_value(api, repo_id=space_repo, key=key, value=value, secret=False, token=token, dry_run=dry_run)

    request_zerogpu_hardware(api, repo_id=space_repo, hardware=zero_gpu_hardware, token=token, dry_run=dry_run)


def trigger_space_training(
    *,
    space_repo: str,
    run_id: str,
    agents: Sequence[str],
    base_model: str,
    seed: int,
    gpu_size: str,
    token: str | None,
    timeout_seconds: int,
    dry_run: bool,
) -> None:
    print(f"Trigger Space training via Gradio API: {space_repo}")
    if dry_run:
        return
    started = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - started < timeout_seconds:
        try:
            _trigger_space_training_via_gradio_api(
                space_repo=space_repo,
                run_id=run_id,
                agents=agents,
                base_model=base_model,
                seed=seed,
                gpu_size=gpu_size,
                token=token,
            )
            return
        except Exception as exc:
            last_error = exc
            print(f"Space not ready yet or trigger failed transiently: {exc}", file=sys.stderr)
            if _is_terminal_space_trigger_error(exc):
                raise
            time.sleep(20)
    raise RuntimeError(f"Timed out waiting for Space trigger readiness after {timeout_seconds}s: {last_error}")


def _is_terminal_space_trigger_error(exc: Exception) -> bool:
    message = str(exc).lower()
    terminal_fragments = (
        "zerogpu quota exceeded",
        "quota exceeded",
        "zerogpu illegal duration",
        "gpu task aborted",
        "zerogpu worker error",
        "requested gpu duration",
        '"ok": false',
        "'ok': false",
    )
    return any(fragment in message for fragment in terminal_fragments)


def _trigger_space_training_via_gradio_api(
    *,
    space_repo: str,
    run_id: str,
    agents: Sequence[str],
    base_model: str,
    seed: int,
    gpu_size: str,
    token: str | None,
) -> None:
    import httpx

    space_name = space_repo.replace("/", "-")
    base_url = f"https://{space_name}.hf.space"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {
        "data": [
            run_id,
            ",".join(agents),
            base_model,
            seed,
            True,
            False,
            True,
            True,
            gpu_size,
        ]
    }
    with httpx.Client(timeout=None) as client:
        response = client.post(f"{base_url}/gradio_api/call/train_lumen_adapters", headers=headers, json=payload)
        response.raise_for_status()
        event_id = str(response.json()["event_id"])
        print(f"Triggered Space training event: {event_id}")
        event_name = ""
        data_lines: list[str] = []
        with client.stream("GET", f"{base_url}/gradio_api/call/train_lumen_adapters/{event_id}", headers=headers) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                if not line:
                    continue
                print(line, flush=True)
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    data = line.split(":", 1)[1].strip()
                    data_lines.append(data)
                    if event_name == "error":
                        raise RuntimeError(f"Space training failed: {data}")
                    if event_name in {"complete", "done"}:
                        _raise_if_space_payload_failed(data)
                        return


def _raise_if_space_payload_failed(data: str) -> None:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and optionally trigger a Hugging Face ZeroGPU Space for Lumen adapter training.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier used in HF dataset/adaptor paths.")
    parser.add_argument("--run-root", type=Path, required=True, help="Local ignored workspace for generated Space files.")
    parser.add_argument("--dataset-source", type=Path, required=True, help="Local generated fine_tuning directory.")
    parser.add_argument("--space-repo", required=True, help="HF Space repo id, e.g. user/lumen-zerogpu-adapter-trainer.")
    parser.add_argument("--dataset-repo", required=True, help="HF dataset repo id used for the run snapshot.")
    parser.add_argument("--adapter-repo", required=True, help="HF model repo id used for trained adapters.")
    parser.add_argument("--agents", default=",".join(AGENTS), help="Comma-separated agent slots to train.")
    parser.add_argument("--base-model", default="", help="Optional base model override. Empty keeps generated per-agent config values.")
    parser.add_argument("--gpu-size", choices=("large", "xlarge"), default="large", help="ZeroGPU decorator size.")
    parser.add_argument("--gpu-duration-seconds", type=int, default=1200, help="ZeroGPU function duration budget.")
    parser.add_argument("--zero-gpu-hardware", default=os.environ.get("LUMEN_ZERO_GPU_HARDWARE", "zero-a10g"), help="HF hardware id requested for the Space.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--trigger", action="store_true", help="Trigger Space training after upload.")
    parser.add_argument("--trigger-timeout-seconds", type=int, default=900, help="Time to wait for Space readiness before triggering.")
    parser.add_argument("--private-space", action="store_true", help="Create/update Space as private.")
    parser.add_argument("--private-dataset", action="store_true", help="Create/update dataset repo as private.")
    parser.add_argument("--private-adapters", action="store_true", help="Create/update adapter model repo as private.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and print actions without calling Hugging Face.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    run_root = args.run_root.resolve()
    dataset_source = args.dataset_source.resolve()
    agents = parse_agents(args.agents)
    require_dataset_source(dataset_source, agents)
    read_json(dataset_source / "adapter_runtime_manifest.json")

    build = write_space_bundle(
        root=root,
        run_id=args.run_id,
        run_root=run_root,
        dataset_source=dataset_source,
        space_repo=args.space_repo,
        dataset_repo=args.dataset_repo,
        adapter_repo=args.adapter_repo,
        agents=agents,
        base_model=args.base_model,
        gpu_size=args.gpu_size,
        gpu_duration_seconds=args.gpu_duration_seconds,
    )
    print(f"Wrote Space bundle: {build.space_dir}")
    print(f"Wrote dataset snapshot: {build.dataset_dir}")
    print(f"Wrote defaults: {build.defaults_path}")

    token = os.environ.get("HF_TOKEN")
    upload_to_hub(
        build=build,
        space_repo=args.space_repo,
        dataset_repo=args.dataset_repo,
        adapter_repo=args.adapter_repo,
        private_space=args.private_space,
        private_dataset=args.private_dataset,
        private_adapters=args.private_adapters,
        zero_gpu_hardware=args.zero_gpu_hardware,
        token=token,
        dry_run=args.dry_run,
    )

    if args.trigger:
        HfApi = import_hf_api()
        wait_for_space_revision(
            HfApi(token=token),
            repo_id=args.space_repo,
            token=token,
            timeout_seconds=args.trigger_timeout_seconds,
            dry_run=args.dry_run,
        )
        trigger_space_training(
            space_repo=args.space_repo,
            run_id=args.run_id,
            agents=agents,
            base_model=args.base_model,
            seed=args.seed,
            gpu_size=args.gpu_size,
            token=token,
            timeout_seconds=args.trigger_timeout_seconds,
            dry_run=args.dry_run,
        )
    else:
        print("Trigger skipped. Set LUMEN_ZERO_GPU_TRIGGER=1 or pass --trigger to start training through the Space API.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
