from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import Any

import pytest

from tools.hf_zerogpu import build_lumen_zerogpu_space as builder
from tools.hf_zerogpu.build_lumen_zerogpu_space import (
    SpaceBuild,
    delete_space_secret_if_present,
    parse_agents,
    parse_experiment_variant,
    require_dataset_source,
    write_space_bundle,
)


def _write_agent_fixture(root: Path, agent: str) -> None:
    agent_dir = root / agent
    agent_dir.mkdir(parents=True)
    (agent_dir / "train_sft.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}) + "\n",
        encoding="utf-8",
    )
    (agent_dir / "val_sft.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}) + "\n",
        encoding="utf-8",
    )
    (agent_dir / "unsloth_config.json").write_text(
        json.dumps(
            {
                "agent": agent,
                "base_model_name": "Qwen/Qwen3-1.7B",
                "max_seq_length": 128,
                "load_in_4bit": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0,
                "learning_rate": 0.0002,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_train_epochs": 1,
                "warmup_steps": 0,
                "output_dir": f"models/lora_qwen3_bootstrap/{agent}",
                "dataset_dir": f"generated/fine_tuning/{agent}",
            }
        ),
        encoding="utf-8",
    )
    variant_dir = agent_dir / "experiments" / "internal_plus_public_optimized"
    variant_dir.mkdir(parents=True)
    for filename in ("train_sft.jsonl", "val_sft.jsonl"):
        (variant_dir / filename).write_text(
            json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}) + "\n",
            encoding="utf-8",
        )
    for filename in ("train_dpo.jsonl", "val_dpo.jsonl"):
        (variant_dir / filename).write_text("", encoding="utf-8")
    (variant_dir / "variant_manifest.json").write_text("{}\n", encoding="utf-8")


def test_parse_agents_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError):
        parse_agents("executor,unknown")


def test_experiment_variant_is_strict_and_defaults_to_optimized() -> None:
    assert parse_experiment_variant("internal_plus_public_optimized") == "internal_plus_public_optimized"
    with pytest.raises(ValueError):
        parse_experiment_variant("optimized")


def test_require_dataset_source_requires_selected_variant_files(tmp_path: Path) -> None:
    dataset = tmp_path / "fine_tuning"
    dataset.mkdir()
    (dataset / "adapter_runtime_manifest.json").write_text("{}\n", encoding="utf-8")
    _write_agent_fixture(dataset, "executor")
    require_dataset_source(dataset, ["executor"])
    (dataset / "executor" / "experiments" / "internal_plus_public_optimized" / "variant_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        require_dataset_source(dataset, ["executor"])


def test_long_lived_space_stream_disconnect_is_terminal() -> None:
    assert builder._is_terminal_space_trigger_error(
        RuntimeError("peer closed connection without sending complete message body (incomplete chunked read)")
    )


def test_gradio_trigger_carries_selected_experiment_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[dict[str, Any]] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"event_id": "event"}

        def iter_lines(self) -> list[str]:
            return ["event: complete", 'data: {"ok": true}']

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def post(self, _url: str, **kwargs: Any) -> FakeResponse:
            posts.append(kwargs["json"])
            return FakeResponse()

        def stream(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
            return FakeResponse()

    httpx = ModuleType("httpx")
    httpx.Client = FakeClient
    monkeypatch.setitem(sys.modules, "httpx", httpx)

    builder._trigger_space_training_via_gradio_api(
        space_repo="user/space",
        run_id="run",
        agents=["executor"],
        base_model="",
        seed=42,
        gpu_size="large",
        experiment_variant="internal_only",
        token="token",
    )

    assert posts[0]["data"][-1] == "internal_only"


def test_write_space_bundle_copies_dataset_and_writes_defaults(tmp_path: Path) -> None:
    dataset = tmp_path / "fine_tuning"
    dataset.mkdir()
    (dataset / "adapter_runtime_manifest.json").write_text(
        json.dumps(
            {
                "sharedBaseModelID": "Qwen/Qwen3-1.7B",
                "adapterRepoID": "user/adapters",
                "adapters": [{"agent": "executor", "baseModelID": "Qwen/Qwen3-1.7B"}],
            }
        ),
        encoding="utf-8",
    )
    _write_agent_fixture(dataset, "executor")

    build = write_space_bundle(
        root=Path(__file__).resolve().parents[3],
        run_id="test-run",
        run_root=tmp_path / "run",
        dataset_source=dataset,
        space_repo="user/space",
        dataset_repo="user/datasets",
        adapter_repo="user/adapters",
        agents=["executor"],
        base_model="",
        gpu_size="large",
        gpu_duration_seconds=3600,
    )

    defaults = json.loads(build.defaults_path.read_text(encoding="utf-8"))
    assert defaults["fresh_run"] is True
    assert defaults["resume_default"] is False
    assert defaults["adapter_first"] is True
    assert defaults["experiment_variant"] == "internal_plus_public_optimized"
    assert defaults["dataset_path_in_repo"] == "runs/test-run/fine_tuning"
    assert (build.space_dir / "app.py").exists()
    assert (build.space_dir / "lumen_train_sft.py").exists()
    assert (build.dataset_dir / "executor" / "train_sft.jsonl").exists()


def test_delete_space_secret_if_present_tolerates_not_found() -> None:
    calls: list[tuple[str, str, str | None]] = []

    class NotFoundError(Exception):
        response = SimpleNamespace(status_code=404)

    class FakeApi:
        def delete_space_secret(self, *, repo_id: str, key: str, token: str | None) -> None:
            calls.append((repo_id, key, token))
            raise NotFoundError

    delete_space_secret_if_present(
        FakeApi(),
        repo_id="user/space",
        key="LUMEN_ZERO_GPU_DURATION_SECONDS",
        token="token",
        dry_run=False,
    )

    assert calls == [("user/space", "LUMEN_ZERO_GPU_DURATION_SECONDS", "token")]


def test_delete_space_secret_if_present_propagates_other_failures() -> None:
    class ForbiddenError(Exception):
        response = SimpleNamespace(status_code=403)

    class FakeApi:
        def delete_space_secret(self, *, repo_id: str, key: str, token: str | None) -> None:
            raise ForbiddenError

    with pytest.raises(ForbiddenError):
        delete_space_secret_if_present(
            FakeApi(),
            repo_id="user/space",
            key="LUMEN_ZERO_GPU_DURATION_SECONDS",
            token="token",
            dry_run=False,
        )


def test_upload_to_hub_removes_legacy_duration_secrets_and_restarts_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeHfApi:
        def __init__(self, *, token: str | None) -> None:
            calls.append(("init", {"token": token}))

        def create_repo(self, **kwargs: Any) -> None:
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs: Any) -> None:
            calls.append(("upload_folder", kwargs))

        def add_space_secret(self, **kwargs: Any) -> None:
            calls.append(("add_space_secret", kwargs))

        def delete_space_secret(self, **kwargs: Any) -> None:
            calls.append(("delete_space_secret", kwargs))

        def add_space_variable(self, **kwargs: Any) -> None:
            calls.append(("add_space_variable", kwargs))

        def request_space_hardware(self, **kwargs: Any) -> None:
            calls.append(("request_space_hardware", kwargs))

        def restart_space(self, **kwargs: Any) -> None:
            calls.append(("restart_space", kwargs))

    dataset_dir = tmp_path / "dataset"
    space_dir = tmp_path / "space"
    dataset_dir.mkdir()
    space_dir.mkdir()
    defaults_path = space_dir / "lumen_zero_gpu_defaults.json"
    defaults_path.write_text(
        json.dumps({"gpu_duration_seconds": 3600, "gpu_size": "large"}),
        encoding="utf-8",
    )
    build = SpaceBuild(
        run_id="test-run",
        run_root=tmp_path,
        space_dir=space_dir,
        dataset_dir=dataset_dir,
        dataset_path_in_repo="runs/test-run/fine_tuning",
        defaults_path=defaults_path,
    )
    monkeypatch.setattr(builder, "import_hf_api", lambda: FakeHfApi)

    builder.upload_to_hub(
        build=build,
        space_repo="user/space",
        dataset_repo="user/dataset",
        adapter_repo="user/adapters",
        private_space=True,
        private_dataset=True,
        private_adapters=True,
        zero_gpu_hardware="zero-a10g",
        token="token",
        dry_run=False,
    )

    delete_calls = [details for name, details in calls if name == "delete_space_secret"]
    assert [details["key"] for details in delete_calls] == list(builder.LEGACY_DURATION_SECRET_KEYS)
    variable_calls = {
        details["key"]: details["value"]
        for name, details in calls
        if name == "add_space_variable"
    }
    assert {
        key: variable_calls[key]
        for key in builder.OPTIONAL_TRAINING_VARIABLE_KEYS
    } == {key: "0" for key in builder.OPTIONAL_TRAINING_VARIABLE_KEYS}
    call_names = [name for name, _ in calls]
    assert max(index for index, name in enumerate(call_names) if name == "delete_space_secret") < min(
        index for index, name in enumerate(call_names) if name == "add_space_variable"
    )
    assert call_names[-2:] == ["request_space_hardware", "restart_space"]
    assert call_names.count("restart_space") == 1
