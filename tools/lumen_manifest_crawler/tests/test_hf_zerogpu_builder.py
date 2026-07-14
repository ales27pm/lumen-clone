from __future__ import annotations

import json
import os
import subprocess
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


def _build_space_fixture(tmp_path: Path) -> SpaceBuild:
    dataset = tmp_path / "fine_tuning"
    dataset.mkdir()
    (dataset / "adapter_runtime_manifest.json").write_text(
        json.dumps(
            {
                "sharedBaseModelID": "Qwen/Qwen3-1.7B",
                "adapterRepoID": "user/adapters",
                "adapters": [
                    {"agent": "executor", "baseModelID": "Qwen/Qwen3-1.7B"}
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_agent_fixture(dataset, "executor")
    return write_space_bundle(
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
        experiment_variant="internal_plus_public_optimized",
        container_image_digest="sha256:" + "c" * 64,
    )


def test_parse_agents_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError):
        parse_agents("executor,unknown")


def test_experiment_variant_is_strict() -> None:
    assert parse_experiment_variant("internal_plus_public_optimized") == "internal_plus_public_optimized"
    with pytest.raises(ValueError):
        parse_experiment_variant("optimized")


def test_require_dataset_source_requires_selected_variant_files(tmp_path: Path) -> None:
    dataset = tmp_path / "fine_tuning"
    dataset.mkdir()
    (dataset / "adapter_runtime_manifest.json").write_text("{}\n", encoding="utf-8")
    _write_agent_fixture(dataset, "executor")
    require_dataset_source(dataset, ["executor"], "internal_plus_public_optimized")
    (dataset / "executor" / "experiments" / "internal_plus_public_optimized" / "variant_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        require_dataset_source(dataset, ["executor"], "internal_plus_public_optimized")


def test_long_lived_space_stream_disconnect_is_terminal() -> None:
    assert builder._is_terminal_space_trigger_error(
        RuntimeError("peer closed connection without sending complete message body (incomplete chunked read)")
    )


def test_gradio_trigger_carries_selected_experiment_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[dict[str, Any]] = []
    headers: list[dict[str, str]] = []

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
            headers.append(kwargs["headers"])
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
        admin_token="Lumen-Admin-Token-0123456789-ABCDEF",
    )

    assert posts[0]["data"][-3:] == ["internal_only", True, False]
    assert posts[0]["data"][5] is False
    assert headers[0]["X-Lumen-Admin-Token"] == "Lumen-Admin-Token-0123456789-ABCDEF"


def test_write_space_bundle_copies_dataset_and_writes_defaults(tmp_path: Path) -> None:
    build = _build_space_fixture(tmp_path)

    defaults = json.loads(build.defaults_path.read_text(encoding="utf-8"))
    assert defaults["fresh_run"] is True
    assert defaults["resume_default"] is False
    assert defaults["adapter_first"] is True
    assert defaults["requested_experiment_variant"] == "internal_plus_public_optimized"
    assert defaults["container_image_digest"] == "sha256:" + "c" * 64
    assert defaults["container_image_digest_source"] == "operator_declared"
    assert defaults["runtime_image_binding_status"] == "manual_validation_required"
    assert defaults["runtime_image_binding_verified"] is False
    assert defaults["dataset_path_in_repo"] == "runs/test-run/fine_tuning"
    assert defaults["dataset_revision"] == "pending_dataset_upload"
    assert defaults["trainingCodeManifest"]["phase"] == "sft"
    assert len(defaults["trainingCodeSHA256"]) == 64
    assert len(defaults["trainingDependencyLockSHA256"]) == 64
    assert len(defaults["requirementsSHA256"]) == 64
    assert defaults["spaceConfiguration"] == {
        "schemaVersion": "lumen.zerogpu.space-configuration/1.0.0",
        "sdk": "gradio",
        "appFile": "app.py",
        "pythonVersion": "3.10",
        "suggestedHardware": None,
        "spaceConfigurationSHA256": defaults["spaceConfigurationSHA256"],
    }
    assert len(defaults["spaceConfigurationSHA256"]) == 64
    assert "suggested_hardware" not in (
        build.space_dir / "README.md"
    ).read_text(encoding="utf-8")
    assert (build.space_dir / "app.py").exists()
    assert (build.space_dir / "lumen_training" / "__init__.py").exists()
    assert (build.space_dir / "lumen_training" / "train_sft.py").exists()
    assert (build.space_dir / "lumen_training" / "train_dpo.py").exists()
    assert (build.space_dir / "lumen_training" / "adapter_artifact.py").exists()
    assert (
        build.space_dir
        / "lumen_manifest_crawler"
        / "dataset"
        / "adapter_evaluation.py"
    ).exists()
    assert (build.dataset_dir / "executor" / "train_sft.jsonl").exists()


def test_built_space_training_package_imports_and_exposes_entrypoints(
    tmp_path: Path,
) -> None:
    build = _build_space_fixture(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(build.space_dir)
    repository_root = str(Path(__file__).resolve().parents[3])

    for module in ("lumen_training.train_sft", "lumen_training.train_dpo"):
        help_result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=build.space_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert help_result.returncode == 0, help_result.stderr

    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import lumen_training.train_sft as sft; "
                "import lumen_training.train_dpo as dpo; "
                "from lumen_manifest_crawler.dataset import adapter_evaluation; "
                "assert dpo._training_runtime_lineage is sft._training_runtime_lineage; "
                "assert adapter_evaluation.default_training_code_manifest(); "
                "print(json.dumps(sys.path))"
            ),
        ],
        cwd=build.space_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr
    assert repository_root not in json.loads(import_result.stdout)


def test_built_space_post_training_adapter_verification_succeeds(
    tmp_path: Path,
) -> None:
    build = _build_space_fixture(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(build.space_dir)
    repository_root = str(Path(__file__).resolve().parents[3])
    script = f"""
import json
import sys
from pathlib import Path
from types import ModuleType

class DummyComponent:
    def __init__(self, *_args, **_kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return None
    def click(self, **_kwargs):
        return None
    def queue(self):
        return self
    def launch(self):
        return None

gradio = ModuleType("gradio")
for name in ("Blocks", "Row", "Textbox", "Number", "Dropdown", "Checkbox", "JSON", "Button"):
    setattr(gradio, name, DummyComponent)
gradio.Markdown = lambda *_args, **_kwargs: None
spaces = ModuleType("spaces")
spaces.GPU = lambda **_kwargs: (lambda function: function)
hub = ModuleType("huggingface_hub")
hub.HfApi = object
hub.snapshot_download = lambda **_kwargs: ""
sys.modules.update({{"gradio": gradio, "spaces": spaces, "huggingface_hub": hub}})

assert Path("adapter_artifact.py").exists() is False
assert Path({repository_root!r}) not in [Path(value or ".").resolve() for value in sys.path]
import app
from lumen_training.adapter_artifact import write_adapter_artifact_manifest

adapter_dir = Path("synthetic_adapter")
adapter_dir.mkdir()
(adapter_dir / "adapter_config.json").write_text(json.dumps({{
    "peft_type": "LORA",
    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
    "target_modules": ["q_proj"],
}}), encoding="utf-8")
header = json.dumps({{
    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {{
        "dtype": "F32",
        "shape": [1],
        "data_offsets": [0, 4],
    }}
}}, separators=(",", ":")).encode("utf-8")
header += b" " * (-len(header) % 8)
(adapter_dir / "adapter_model.safetensors").write_bytes(
    len(header).to_bytes(8, "little") + header + b"\\x00\\x00\\x00\\x00"
)
(adapter_dir / "tokenizer.json").write_text("{{}}", encoding="utf-8")
(adapter_dir / "tokenizer_config.json").write_text("{{}}", encoding="utf-8")
artifact = write_adapter_artifact_manifest(adapter_dir, training_phase="sft")

runtime = {{
    "runtimeSourceKind": "huggingface_space",
    "runtimeSourceRevision": "4" * 40,
    "expectedRuntimeSourceRevision": "4" * 40,
    "observedRepositoryRevision": "4" * 40,
    "observedRuntimeRevision": None,
    "runtimeSourceBindingStatus": "operator_declared_unverified",
    "runtimeSourceBindingMethod": "huggingface_repository_head_supplemental",
}}
training_environment = {{
    "schemaVersion": "lumen.adapter-training-environment/1.0.0",
    "runtimeImageBindingStatus": "manual_validation_required",
    "runtimeImageBindingVerified": False,
}}
training_environment_sha = app._canonical_sha256(training_environment)
lane_hashes = {{"trainSFT": "a" * 64, "validationSFT": "b" * 64}}
item = {{
    "agent": "executor",
    "variant": "internal_plus_public_optimized",
    "variantManifestSHA256": "1" * 64,
    "base_model_name": "Qwen/Qwen3-1.7B",
    "baseModelRevision": "2" * 40,
    "baseModelIndexDigest": "3" * 64,
    "baseModelIndexReferencedShardNames": ["model-00001-of-00001.safetensors"],
    "baseModelIndexShardBindingSHA256": "4" * 64,
    "baseModelArtifactDigest": "5" * 64,
    "baseModelWeightShards": [{{"filename": "model-00001-of-00001.safetensors", "sha256": "6" * 64}}],
    "baseModelTokenizerDigest": "7" * 64,
    "trainingEnvironmentSHA256": training_environment_sha,
    "trainingCodeSHA256": "8" * 64,
    "trainingDependencyLockSHA256": "9" * 64,
    "requirementsSHA256": "a" * 64,
    "spaceConfigurationSHA256": app.DEFAULTS["spaceConfigurationSHA256"],
    "runtimeImageBindingStatus": "manual_validation_required",
    "runtimeImageBindingVerified": False,
    "variantAttestation": {{
        "variant": "internal_plus_public_optimized",
        "variantManifestSHA256": "1" * 64,
        "trainingEnvironmentSHA256": training_environment_sha,
        "trainingCorpusSHA256": "b" * 64,
        "effectiveTrainingConfigSHA256": "c" * 64,
        "laneHashes": lane_hashes,
    }},
    "adapter_dir": str(adapter_dir),
    "finalized_variant_manifest": "finalized_variant_manifest.json",
    **runtime,
}}
finalized = {{
    "agent": item["agent"],
    "variant": item["variant"],
    "sourceVariantManifestSHA256": item["variantManifestSHA256"],
    "baseModelID": item["base_model_name"],
    "baseModelRevision": item["baseModelRevision"],
    "baseModelIndexDigest": item["baseModelIndexDigest"],
    "baseModelIndexReferencedShardNames": item["baseModelIndexReferencedShardNames"],
    "baseModelIndexShardBindingSHA256": item["baseModelIndexShardBindingSHA256"],
    "baseModelArtifactDigest": item["baseModelArtifactDigest"],
    "baseModelWeightShards": item["baseModelWeightShards"],
    "baseModelTokenizerDigest": item["baseModelTokenizerDigest"],
    "trainingEnvironmentSHA256": training_environment_sha,
    "trainingEnvironment": training_environment,
    "trainingCodeSHA256": item["trainingCodeSHA256"],
    "trainingDependencyLockSHA256": item["trainingDependencyLockSHA256"],
    "requirementsSHA256": item["requirementsSHA256"],
    "spaceConfigurationSHA256": item["spaceConfigurationSHA256"],
    "trainingCorpusSHA256": item["variantAttestation"]["trainingCorpusSHA256"],
    "trainingConfigSHA256": item["variantAttestation"]["effectiveTrainingConfigSHA256"],
    "datasets": {{name: {{"sha256": digest}} for name, digest in lane_hashes.items()}},
    "artifact": {{
        "status": "trained",
        "trainingPhase": "sft",
        "parentSFTAdapterSHA256": None,
        "adapterSHA256": artifact["adapterSHA256"],
        "adapterManifestSHA256": artifact["adapterSHA256"],
    }},
    **runtime,
}}
finalized["variantManifestSHA256"] = app._canonical_sha256(finalized)
Path(item["finalized_variant_manifest"]).write_text(
    json.dumps(finalized), encoding="utf-8"
)

verified_dir, verified_manifest = app._verify_trained_adapter(item)
assert verified_dir == adapter_dir
assert verified_manifest["artifact"]["adapterSHA256"] == artifact["adapterSHA256"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=build.space_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "logical_path",
    [
        "unlisted_helper.py",
        "lumen_manifest_crawler/output/__init__.py",
        "lumen_manifest_crawler/output/hashing.py",
        "lumen_manifest_crawler/dataset/public_adapter_eval_sources.json",
        "lumen_manifest_crawler/dataset/public_adapter_eval_fingerprints.json",
        "lumen_training/train_sft.py",
        "lumen_training/train_dpo.py",
        "requirements.txt",
    ],
)
def test_built_space_closure_rejects_behavior_file_drift(
    tmp_path: Path,
    logical_path: str,
) -> None:
    build = _build_space_fixture(tmp_path)
    defaults = json.loads(build.defaults_path.read_text(encoding="utf-8"))
    lineage = builder._load_training_lineage_module(
        Path(__file__).resolve().parents[3]
    )
    target = build.space_dir / logical_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        target.read_text(encoding="utf-8") + "\n# controlled drift\n"
        if target.exists()
        else "# unexpected behavior file\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        lineage.verify_training_code_manifest(
            defaults["trainingCodeManifest"],
            root=build.space_dir,
        )


def test_volatile_run_files_do_not_change_controlled_code_digest(
    tmp_path: Path,
) -> None:
    build = _build_space_fixture(tmp_path)
    defaults = json.loads(build.defaults_path.read_text(encoding="utf-8"))
    lineage = builder._load_training_lineage_module(
        Path(__file__).resolve().parents[3]
    )
    manifest = defaults["trainingCodeManifest"]
    expected = lineage.verify_training_code_manifest(
        manifest,
        root=build.space_dir,
    )

    build.defaults_path.write_text("{\"volatile\": true}\n", encoding="utf-8")
    (build.space_dir / "lumen_zero_gpu_run_manifest.json").write_text(
        "{\"volatile\": true}\n",
        encoding="utf-8",
    )
    assert (
        lineage.verify_training_code_manifest(manifest, root=build.space_dir)
        == expected
    )


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
        visibility = {
            ("dataset", "user/dataset"): False,
            ("model", "user/adapters"): False,
            ("space", "user/space"): False,
        }

        def __init__(self, *, token: str | None) -> None:
            calls.append(("init", {"token": token}))

        def create_repo(self, **kwargs: Any) -> None:
            calls.append(("create_repo", kwargs))

        def update_repo_settings(self, **kwargs: Any) -> None:
            calls.append(("update_repo_settings", kwargs))
            self.visibility[(kwargs["repo_type"], kwargs["repo_id"])] = kwargs["private"]

        def dataset_info(self, **kwargs: Any) -> Any:
            calls.append(("dataset_info", kwargs))
            return SimpleNamespace(private=self.visibility[("dataset", kwargs["repo_id"])])

        def model_info(self, **kwargs: Any) -> Any:
            calls.append(("model_info", kwargs))
            return SimpleNamespace(private=self.visibility[("model", kwargs["repo_id"])])

        def space_info(self, **kwargs: Any) -> Any:
            calls.append(("space_info", kwargs))
            return SimpleNamespace(private=self.visibility[("space", kwargs["repo_id"])])

        def upload_folder(self, **kwargs: Any) -> Any:
            calls.append(("upload_folder", kwargs))
            revision = "d" * 40 if kwargs["repo_type"] == "dataset" else "e" * 40
            return SimpleNamespace(oid=revision)

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
        admin_token="Lumen-Admin-Token-0123456789-ABCDEF",
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
    assert variable_calls["LUMEN_ZERO_GPU_DATASET_REVISION"] == "d" * 40
    assert (
        variable_calls["LUMEN_ZERO_GPU_EXPECTED_RUNTIME_SOURCE_REVISION"]
        == "e" * 40
    )
    assert variable_calls["LUMEN_ZERO_GPU_RUNTIME_SOURCE_REVISION"] == "e" * 40
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    assert defaults["dataset_revision"] == "d" * 40
    uploads = [details for name, details in calls if name == "upload_folder"]
    assert [details["repo_type"] for details in uploads] == ["dataset", "space"]
    first_upload = call_names.index("upload_folder")
    assert call_names[:first_upload].count("update_repo_settings") == 3
    assert {"dataset_info", "model_info", "space_info"}.issubset(
        set(call_names[:first_upload])
    )
    secret_keys = {
        details["key"] for name, details in calls if name == "add_space_secret"
    }
    assert secret_keys == {
        "LUMEN_ZERO_GPU_HUB_TOKEN",
        "LUMEN_ZERO_GPU_ADMIN_TOKEN",
    }


@pytest.mark.parametrize(
    ("initial_private", "requested_private"),
    [(False, True), (True, False)],
)
def test_ensure_repository_visibility_updates_existing_repository_and_verifies_readback(
    initial_private: bool,
    requested_private: bool,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeApi:
        private = initial_private

        def create_repo(self, **kwargs: Any) -> None:
            calls.append(("create_repo", kwargs))
            # Existing-repository behavior: create_repo does not change visibility.

        def update_repo_settings(self, **kwargs: Any) -> None:
            calls.append(("update_repo_settings", kwargs))
            self.private = kwargs["private"]

        def model_info(self, **kwargs: Any) -> Any:
            calls.append(("model_info", kwargs))
            return SimpleNamespace(private=self.private)

    builder.ensure_repository_visibility(
        FakeApi(),
        repo_id="user/adapters",
        repo_type="model",
        private=requested_private,
        token="token",
    )

    assert [name for name, _ in calls] == [
        "create_repo",
        "update_repo_settings",
        "model_info",
    ]
    assert calls[1][1]["private"] is requested_private
    assert calls[2][1]["token"] == "token"


@pytest.mark.parametrize("failure", ["update", "readback_mismatch"])
def test_upload_to_hub_does_not_upload_before_visibility_postcondition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    calls: list[str] = []

    class FakeHfApi:
        def __init__(self, *, token: str | None) -> None:
            calls.append("init")

        def create_repo(self, **_kwargs: Any) -> None:
            calls.append("create_repo")

        def update_repo_settings(self, **kwargs: Any) -> None:
            calls.append("update_repo_settings")
            if failure == "update":
                raise RuntimeError("settings update failed")

        def dataset_info(self, **_kwargs: Any) -> Any:
            calls.append("dataset_info")
            return SimpleNamespace(private=False)

        def upload_folder(self, **_kwargs: Any) -> None:
            calls.append("upload_folder")
            pytest.fail("upload must not run before visibility is verified")

    dataset_dir = tmp_path / "dataset"
    space_dir = tmp_path / "space"
    dataset_dir.mkdir()
    space_dir.mkdir()
    defaults_path = space_dir / "lumen_zero_gpu_defaults.json"
    defaults_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(builder, "import_hf_api", lambda: FakeHfApi)

    expected = (
        "settings update failed"
        if failure == "update"
        else "Repository visibility postcondition failed"
    )
    with pytest.raises(RuntimeError, match=expected):
        builder.upload_to_hub(
            build=SpaceBuild(
                run_id="test-run",
                run_root=tmp_path,
                space_dir=space_dir,
                dataset_dir=dataset_dir,
                dataset_path_in_repo="runs/test-run/fine_tuning",
                defaults_path=defaults_path,
            ),
            space_repo="user/space",
            dataset_repo="user/dataset",
            adapter_repo="user/adapters",
            private_space=True,
            private_dataset=True,
            private_adapters=True,
            zero_gpu_hardware="zero-a10g",
            token="token",
            admin_token="Lumen-Admin-Token-0123456789-ABCDEF",
            dry_run=False,
        )

    assert "upload_folder" not in calls


def test_upload_to_hub_requires_dedicated_repository_token_before_hf_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builder,
        "import_hf_api",
        lambda: pytest.fail("missing credentials must fail before HfApi is imported"),
    )
    space_dir = tmp_path / "space"
    dataset_dir = tmp_path / "dataset"
    space_dir.mkdir()
    dataset_dir.mkdir()
    defaults_path = space_dir / "lumen_zero_gpu_defaults.json"
    defaults_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="LUMEN_ZERO_GPU_HUB_TOKEN"):
        builder.upload_to_hub(
            build=SpaceBuild(
                run_id="test-run",
                run_root=tmp_path,
                space_dir=space_dir,
                dataset_dir=dataset_dir,
                dataset_path_in_repo="runs/test-run/fine_tuning",
                defaults_path=defaults_path,
            ),
            space_repo="user/space",
            dataset_repo="user/dataset",
            adapter_repo="user/adapters",
            private_space=True,
            private_dataset=True,
            private_adapters=True,
            zero_gpu_hardware="zero-a10g",
            token=None,
            admin_token="Lumen-Admin-Token-0123456789-ABCDEF",
            dry_run=False,
        )


def test_main_rejects_legacy_hf_token_for_real_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_ADMIN_TOKEN",
        "Lumen-Admin-Token-0123456789-ABCDEF",
    )
    monkeypatch.delenv("LUMEN_ZERO_GPU_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "legacy-broad-token")
    monkeypatch.setattr(
        builder,
        "import_hf_api",
        lambda: pytest.fail("legacy credentials must fail before HfApi is imported"),
    )

    with pytest.raises(ValueError, match="LUMEN_ZERO_GPU_HUB_TOKEN"):
        builder.main(
            [
                "--run-id", "run",
                "--run-root", str(tmp_path / "run"),
                "--dataset-source", str(tmp_path / "missing-dataset"),
                "--space-repo", "user/space",
                "--dataset-repo", "user/dataset",
                "--adapter-repo", "user/adapters",
                "--experiment-variant", "internal_only",
                "--container-image-digest", "sha256:" + "a" * 64,
            ]
        )


def _required_builder_args() -> list[str]:
    return [
        "--run-id", "run",
        "--run-root", "run-root",
        "--dataset-source", "datasets",
        "--space-repo", "user/space",
        "--dataset-repo", "user/dataset",
        "--adapter-repo", "user/adapters",
        "--experiment-variant", "internal_only",
        "--container-image-digest", "sha256:" + "a" * 64,
    ]


def test_repository_visibility_is_private_by_default() -> None:
    defaults = builder.parse_args(
        _required_builder_args()
    )
    assert defaults.public_space is False
    assert defaults.public_dataset is False
    assert defaults.public_adapters is False


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--public-space", (True, False, False)),
        ("--public-dataset", (False, True, False)),
        ("--public-adapters", (False, False, True)),
    ],
)
def test_public_repository_overrides_are_explicit_and_isolated(
    flag: str,
    expected: tuple[bool, bool, bool],
) -> None:
    parsed = builder.parse_args([*_required_builder_args(), flag])
    assert (
        parsed.public_space,
        parsed.public_dataset,
        parsed.public_adapters,
    ) == expected


@pytest.mark.parametrize(
    ("flags", "expected_private"),
    [
        ([], (True, True, True)),
        (["--public-space"], (False, True, True)),
        (["--public-dataset"], (True, False, True)),
        (["--public-adapters"], (True, True, False)),
    ],
)
def test_main_passes_private_by_default_visibility_to_hub_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    expected_private: tuple[bool, bool, bool],
) -> None:
    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_ADMIN_TOKEN",
        "Lumen-Admin-Token-0123456789-ABCDEF",
    )
    monkeypatch.delenv("LUMEN_ZERO_GPU_HUB_TOKEN", raising=False)
    monkeypatch.setattr(builder, "require_dataset_source", lambda *_args: None)
    monkeypatch.setattr(builder, "read_json", lambda *_args: {})
    build = SpaceBuild(
        run_id="run-internal_only",
        run_root=tmp_path / "run",
        space_dir=tmp_path / "space",
        dataset_dir=tmp_path / "dataset",
        dataset_path_in_repo="runs/run/fine_tuning",
        defaults_path=tmp_path / "space" / "lumen_zero_gpu_defaults.json",
    )
    monkeypatch.setattr(builder, "write_space_bundle", lambda **_kwargs: build)
    captured: dict[str, Any] = {}

    def fake_upload(**kwargs: Any) -> builder.HubUpload:
        captured.update(kwargs)
        return builder.HubUpload(
            dataset_revision="dry_run_not_uploaded",
            runtime_source_revision="dry_run_not_uploaded",
        )

    monkeypatch.setattr(builder, "upload_to_hub", fake_upload)

    assert builder.main(
        [
            *_required_builder_args(),
            "--root", str(tmp_path),
            "--run-root", str(tmp_path / "run"),
            "--dataset-source", str(tmp_path / "dataset-source"),
            "--dry-run",
            *flags,
        ]
    ) == 0
    assert (
        captured["private_space"],
        captured["private_dataset"],
        captured["private_adapters"],
    ) == expected_private


def test_one_click_launcher_is_private_unless_public_overrides_are_set() -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/hf_zerogpu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")
    assert 'PUBLIC_SPACE="${LUMEN_ZERO_GPU_PUBLIC_SPACE:-0}"' in script
    assert 'PUBLIC_DATASET="${LUMEN_ZERO_GPU_PUBLIC_DATASET:-0}"' in script
    assert 'PUBLIC_ADAPTERS="${LUMEN_ZERO_GPU_PUBLIC_ADAPTERS:-0}"' in script
    assert "args+=(--public-space)" in script
    assert "args+=(--public-dataset)" in script
    assert "args+=(--public-adapters)" in script
    assert "args+=(--private-dataset)" not in script
    assert "args+=(--private-adapters)" not in script


def _resume_launcher_environment(*, python: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LUMEN_ZERO_GPU_EXPERIMENT_VARIANT": "internal_only",
            "LUMEN_ZERO_GPU_CONTAINER_IMAGE_DIGEST": "sha256:" + "a" * 64,
            "LUMEN_ZERO_GPU_ADMIN_TOKEN": "Lumen-Admin-Token-0123456789-ABCDEF",
            "LUMEN_ZERO_GPU_HUB_TOKEN": "hf_fine_grained_repository_token",
            "LUMEN_ZERO_GPU_AGENTS": "cortex,executor,mouth",
            "LUMEN_ZERO_GPU_AGENT_BATCH_SIZE": "1",
            "LUMEN_ZERO_GPU_RESUME": "1",
            "LUMEN_ZERO_GPU_SKIP_INSTALL": "1",
            "LUMEN_ZERO_GPU_USE_ACTIVE_PYTHON": "1",
            "LUMEN_ZERO_GPU_PYTHON": str(python),
            "LUMEN_ZERO_GPU_TRIGGER": "0",
        }
    )
    return env


def test_one_click_multi_batch_resume_requires_explicit_batch() -> None:
    script = Path(__file__).resolve().parents[3] / "scripts/hf_zerogpu_train_lumen_adapters_aio.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=script.parents[1],
        env=_resume_launcher_environment(python=Path("/usr/bin/false")),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "LUMEN_ZERO_GPU_RESUME_BATCH must select the explicit batch" in result.stderr


def test_one_click_multi_batch_resume_runs_only_selected_batch(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts/hf_zerogpu_train_lumen_adapters_aio.sh"
    capture = tmp_path / "builder-arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >> \"$LUMEN_TEST_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = _resume_launcher_environment(python=fake_python)
    env["LUMEN_ZERO_GPU_RESUME_BATCH"] = "2"
    env["LUMEN_TEST_CAPTURE"] = str(capture)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=script.parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments.count("--agents") == 1
    assert arguments[arguments.index("--agents") + 1] == "executor"
    assert arguments[arguments.index("--run-id") + 1].endswith("-b02-executor")


def test_resume_trigger_does_not_rebuild_or_upload_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_ADMIN_TOKEN",
        "Lumen-Admin-Token-0123456789-ABCDEF",
    )
    monkeypatch.setenv("LUMEN_ZERO_GPU_HUB_TOKEN", "fine-grained-token")
    monkeypatch.setattr(
        builder,
        "write_space_bundle",
        lambda **_kwargs: pytest.fail("resume must not rebuild the Space"),
    )
    monkeypatch.setattr(
        builder,
        "upload_to_hub",
        lambda **_kwargs: pytest.fail("resume must not upload a new snapshot"),
    )
    monkeypatch.setattr(
        builder,
        "import_hf_api",
        lambda: (lambda **_kwargs: object()),
    )

    assert builder.main(
        [
            "--run-id", "resume-run",
            "--run-root", str(tmp_path / "missing-run-root"),
            "--dataset-source", str(tmp_path / "missing-dataset"),
            "--space-repo", "user/space",
            "--dataset-repo", "user/dataset",
            "--adapter-repo", "user/adapters",
            "--experiment-variant", "internal_only",
            "--container-image-digest", "sha256:" + "a" * 64,
            "--resume",
            "--trigger",
            "--dry-run",
        ]
    ) == 0
