from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.fine_tuning.unsloth import ubuntu_pipeline


REPO_ROOT = Path(__file__).resolve().parents[4]
FAKE_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def test_docker_context_includes_the_dependency_lineage_build_preflight() -> None:
    docker_root = REPO_ROOT / "tools/fine_tuning/unsloth"
    dockerfile = (docker_root / "Dockerfile.ubuntu-cu128").read_text(
        encoding="utf-8"
    )
    dockerignore = set(
        (docker_root / "Dockerfile.ubuntu-cu128.dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert (
        "COPY tools/fine_tuning/unsloth/training_lineage.py "
        "/tmp/lumen-training-lineage.py"
    ) in dockerfile
    assert "lineage.verify_training_dependency_lock(" in dockerfile
    assert "setpriv --reuid=nobody --regid=nogroup --init-groups python" in dockerfile
    assert "lineage.build_resolved_training_environment_snapshot()" in dockerfile
    assert "lineage.verify_resolved_training_environment(environment)" in dockerfile
    assert {
        "!tools/",
        "!tools/fine_tuning/",
        "!tools/fine_tuning/unsloth/",
        "!tools/fine_tuning/unsloth/training_lineage.py",
    } <= dockerignore


def test_ubuntu_image_maps_the_invoking_non_root_identity() -> None:
    dockerfile = (
        REPO_ROOT / "tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128"
    ).read_text(encoding="utf-8")
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    assert "ARG LUMEN_RUNTIME_UID=1000" in dockerfile
    assert "ARG LUMEN_RUNTIME_GID=1000" in dockerfile
    assert 'getent passwd "${LUMEN_RUNTIME_UID}"' in dockerfile
    assert 'getent group "${LUMEN_RUNTIME_GID}"' in dockerfile
    assert "pwd.getpwuid(uid).pw_uid == uid" in dockerfile
    assert "grp.getgrgid(gid).gr_gid == gid" in dockerfile
    assert "USER ${LUMEN_RUNTIME_UID}:${LUMEN_RUNTIME_GID}" in dockerfile
    assert '--build-arg "LUMEN_RUNTIME_UID=$RUNTIME_UID"' in launcher
    assert '--build-arg "LUMEN_RUNTIME_GID=$RUNTIME_GID"' in launcher
    assert "pwd.getpwuid(uid).pw_uid == uid" in launcher
    assert 'HOME=$RUNTIME_HOME' in launcher


def test_ubuntu_launcher_rejects_an_image_without_the_runtime_identity(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")
    bash_major = int(
        subprocess.check_output(
            ["bash", "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'],
            text=True,
        )
    )
    if bash_major < 4:
        pytest.skip("Ubuntu launcher requires Bash 4 associative arrays")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    fake_commands = {
        "uname": """#!/bin/sh
printf 'Linux\\n'
""",
        "id": """#!/bin/sh
case "${1:-}" in
  -u) printf '12345\\n' ;;
  -g) printf '23456\\n' ;;
  *) exit 2 ;;
esac
""",
        "nvidia-smi": """#!/bin/sh
exit 0
""",
        "docker": f"""#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "${{1:-}}" = image ] && [ "${{2:-}}" = inspect ]; then
  printf '{FAKE_IMAGE_DIGEST}\\n'
  exit 0
fi
case "$*" in
  *'/opt/lumen-venv/bin/python'*) exit 23 ;;
esac
exit 0
""",
    }
    for name, source in fake_commands.items():
        path = fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            "--prepare-only",
            "--no-pull",
            "--run-id",
            "identity-probe",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--hf-cache",
            str(tmp_path / "hf-cache"),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "training image lacks the invoking user's passwd/group mapping" in (
        result.stderr
    )
    logged = docker_log.read_text(encoding="utf-8")
    assert "--build-arg LUMEN_RUNTIME_UID=12345" in logged
    assert "--build-arg LUMEN_RUNTIME_GID=23456" in logged
    assert "--user 12345:23456" in logged
    assert "/opt/lumen-venv/bin/python" in logged
    assert "--gpus all" not in logged


def test_unsloth_import_guard_fails_closed_if_transformers_loaded_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.fine_tuning.unsloth.train_sft import (
        _require_unsloth_before_transformers,
    )

    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    with pytest.raises(RuntimeError, match="before Transformers"):
        _require_unsloth_before_transformers()

    monkeypatch.setitem(sys.modules, "unsloth", ModuleType("unsloth"))
    _require_unsloth_before_transformers()


def test_ubuntu_trainers_import_unsloth_before_transformers_seeding() -> None:
    for filename in ("train_sft.py", "train_dpo.py"):
        source = (
            REPO_ROOT / "tools/fine_tuning/unsloth" / filename
        ).read_text(encoding="utf-8")
        main_source = source[source.index("def main() -> None:") :]
        assert main_source.index("from unsloth import FastLanguageModel") < (
            main_source.index("_seed_everything(seed)")
        )

    evaluator = (
        REPO_ROOT / "tools/fine_tuning/unsloth/evaluate_adapter.py"
    ).read_text(encoding="utf-8")
    loader = evaluator[evaluator.index("def load_inference_model(") :]
    assert loader.index("from unsloth import FastLanguageModel") < loader.index(
        "_seed_everything(int(cfg[\"seed\"]))"
    )

    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF" not in launcher


def test_agent_and_run_root_validation_fails_closed(tmp_path: Path) -> None:
    assert ubuntu_pipeline.parse_agents("cortex,executor") == ("cortex", "executor")
    with pytest.raises(RuntimeError, match="duplicates"):
        ubuntu_pipeline.parse_agents("cortex,cortex")
    with pytest.raises(RuntimeError, match="Unsupported agents"):
        ubuntu_pipeline.parse_agents("cortex,unknown")
    with pytest.raises(RuntimeError, match="must be a child"):
        ubuntu_pipeline.validate_run_root(tmp_path, allowed_parent=tmp_path)
    child = tmp_path / "run-one"
    assert ubuntu_pipeline.validate_run_root(child, allowed_parent=tmp_path) == child


def test_current_optimized_artifacts_pass_static_preflight(tmp_path: Path) -> None:
    result = ubuntu_pipeline.static_preflight(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        agents=ubuntu_pipeline.AGENTS,
        variant="internal_plus_public_optimized",
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
        run_root=tmp_path / "run-one",
        allowed_parent=tmp_path,
    )

    assert result["status"] == "static_ready"
    assert result["trainingReady"] is False
    assert [entry["agent"] for entry in result["agents"]] == list(
        ubuntu_pipeline.AGENTS
    )
    assert not (tmp_path / "run-one").exists()


def test_shell_static_preflight_has_no_run_side_effects(tmp_path: Path) -> None:
    run_id = "static-run"
    variant = "internal_plus_public_optimized"
    run_root = tmp_path / f"{run_id}-{variant}"
    environment = {
        **os.environ,
        "LUMEN_AIO_EXPERIMENT_VARIANT": variant,
        "LUMEN_AIO_CONTAINER_IMAGE_DIGEST": FAKE_IMAGE_DIGEST,
        "LUMEN_AIO_RUN_ID": run_id,
        "LUMEN_AIO_RUN_ROOT": str(run_root),
        "LUMEN_AIO_ALLOWED_RUN_PARENT": str(tmp_path),
        "LUMEN_AIO_STATIC_PREFLIGHT": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/ubuntu_train_lumen_adapters_aio.sh"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "static_ready"' in result.stdout
    assert not run_root.exists()


def test_prepare_binds_the_same_resolved_environment_into_config_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.fine_tuning.unsloth import train_sft

    environment_sha = "e" * 64
    lineage = {
        "resolvedTrainingEnvironment": {
            "schemaVersion": "lumen.resolved-training-environment/1.0.0"
        },
        "resolvedTrainingEnvironmentSHA256": "r" * 64,
        "resolvedTrainingEnvironmentScanAudit": {"distributionCount": 1},
        "spaceConfigurationSHA256": None,
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "observedAccelerator": {"backend": "cuda", "deviceCount": 1},
    }
    environment = {"trainingEnvironmentSHA256": environment_sha}
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_runtime_lineage",
        lambda **_kwargs: (lineage, environment),
    )
    monkeypatch.setattr(
        train_sft,
        "_training_environment",
        lambda *_args, **_kwargs: environment,
    )

    run_root = tmp_path / "prepared"
    ubuntu_pipeline.prepare_run(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        run_root=run_root,
        agents=("cortex",),
        variant="internal_plus_public_optimized",
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
    )
    config = json.loads(
        (run_root / "configs" / "cortex.json").read_text(encoding="utf-8")
    )

    assert config["trainingEnvironmentSHA256"] == environment_sha
    assert config["variantAttestation"]["trainingEnvironmentSHA256"] == environment_sha
    assert config["resolvedTrainingEnvironment"] == lineage[
        "resolvedTrainingEnvironment"
    ]
    assert (
        run_root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    ).is_file()


def test_final_config_switches_to_verified_preference_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = {
        "agent": "cortex",
        "preference_trainer": "dpo",
        "trainingCodeManifestsByPhase": {"dpo": {"phase": "dpo"}},
        "trainingCodeSHA256ByPhase": {"dpo": "d" * 64},
        "variantAttestation": {
            "trainingEnvironmentSHA256": "a" * 64,
        },
        "adapterExport": {
            "adapterArtifact": "old-sft",
            "adapterDirectory": "old-sft",
            "adapterGGUFArtifact": "old-gguf",
        },
    }
    (config_dir / "cortex.json").write_text(json.dumps(config), encoding="utf-8")
    behavior = (
        tmp_path
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    behavior.parent.mkdir(parents=True)
    behavior.write_text("{}\n", encoding="utf-8")
    run_manifest = {
        "behaviorManifestFileSHA256": ubuntu_pipeline.file_sha256(behavior),
    }
    run_manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(
        run_manifest
    )
    ubuntu_pipeline.write_object(tmp_path / "aio_run_manifest.json", run_manifest)
    finalized = {
        "trainingEnvironmentSHA256": "e" * 64,
        "resolvedTrainingEnvironment": {"schemaVersion": "test"},
        "resolvedTrainingEnvironmentSHA256": "f" * 64,
        "observedAccelerator": {"backend": "cuda"},
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "artifact": {"adapterSHA256": "b" * 64},
        **{field: "c" * 40 for field in ubuntu_pipeline.RUNTIME_SOURCE_FIELDS},
    }
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: {
            "adapterSHA256": "b" * 64,
            "parentSFTAdapterSHA256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_manifest_integrity",
        lambda *_args: finalized,
    )

    result = ubuntu_pipeline.write_final_config(tmp_path, "cortex")
    written = json.loads(Path(result["config"]).read_text(encoding="utf-8"))

    assert written["adapter_training_phase"] == "sft_dpo"
    assert written["parent_sft_adapter_sha256"] == "a" * 64
    assert written["adapter_output_dir"].endswith("models/lora_qwen3_dpo/cortex")
    assert written["trainingCodeSHA256"] == "d" * 64
    assert written["variantAttestation"]["trainingEnvironmentSHA256"] == "e" * 64
    assert written["adapterExport"]["adapterArtifact"].endswith(
        "models/lora_qwen3_dpo/cortex"
    )


def test_summary_rejects_failed_full_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_sft",
        lambda *_args: {"adapterSHA256": "a" * 64},
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: {
            "adapterSHA256": "b" * 64,
            "finalizedVariantManifestSHA256": "f" * 64,
        },
    )
    evaluation_dir = tmp_path / "evaluation" / "cortex"
    evaluation_dir.mkdir(parents=True)
    candidate = evaluation_dir / "candidate_outputs.jsonl"
    candidate.write_text('{"candidate":"failed"}\n', encoding="utf-8")
    report = {
        "candidateOutputsSHA256": "c" * 64,
    }
    report["reportSHA256"] = ubuntu_pipeline.canonical_sha256(report)
    report_path = evaluation_dir / "evaluation_report.json"
    ubuntu_pipeline.write_object(report_path, report)
    run_manifest = {
        "status": "quality_gate_failed",
        "agent": "cortex",
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": "f" * 64,
        "candidateOutputsFileSHA256": ubuntu_pipeline.file_sha256(candidate),
        "candidateOutputsSHA256": "c" * 64,
        "evaluationReportFileSHA256": ubuntu_pipeline.file_sha256(report_path),
        "evaluationReportSHA256": report["reportSHA256"],
        "completeEvaluation": True,
        "qualityGatePassed": False,
    }
    run_manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(
        run_manifest
    )
    ubuntu_pipeline.write_object(
        evaluation_dir / "evaluation_run_manifest.json",
        run_manifest,
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        ubuntu_pipeline.write_summary(
            run_root=tmp_path,
            agents=("cortex",),
            variant="internal_plus_public_optimized",
            preference=True,
            require_gguf=False,
            require_evaluation=True,
        )


def test_resume_reuses_only_a_gguf_bound_to_the_completed_summary(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "models" / "lora_qwen3_gguf" / "lumen-cortex-lora.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF-adapter")
    summary = {
        "agents": {
            "cortex": {
                "adapterGGUFSHA256": ubuntu_pipeline.file_sha256(gguf),
                "adapterGGUFSizeBytes": gguf.stat().st_size,
            }
        }
    }
    summary["summarySHA256"] = ubuntu_pipeline.canonical_sha256(summary)
    ubuntu_pipeline.write_object(tmp_path / "aio_summary.json", summary)

    verified = ubuntu_pipeline.verify_gguf(tmp_path, "cortex")
    assert verified["adapterGGUFSHA256"] == ubuntu_pipeline.file_sha256(gguf)

    gguf.write_bytes(b"GGUF-tampered")
    with pytest.raises(RuntimeError, match="does not match"):
        ubuntu_pipeline.verify_gguf(tmp_path, "cortex")


def test_upload_cli_is_private_unless_public_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ubuntu_pipeline.py",
            "upload",
            "--run-root",
            str(tmp_path),
            "--agents",
            "cortex",
            "--run-id",
            "run-one",
            "--token-file",
            str(tmp_path / "token"),
        ],
    )
    assert ubuntu_pipeline.parse_args().public is False

    sys.argv.append("--public")
    assert ubuntu_pipeline.parse_args().public is True


def test_trainers_save_only_the_unified_fast_tokenizer_format() -> None:
    for filename in ("train_sft.py", "train_dpo.py"):
        source = (
            REPO_ROOT / "tools" / "fine_tuning" / "unsloth" / filename
        ).read_text(encoding="utf-8")
        assert "tokenizer.save_pretrained" in source
        assert "legacy_format=False" in source


def test_controlled_trainers_evaluate_and_save_each_epoch() -> None:
    sft_source = (REPO_ROOT / "tools/fine_tuning/unsloth/train_sft.py").read_text(
        encoding="utf-8"
    )
    dpo_source = (REPO_ROOT / "tools/fine_tuning/unsloth/train_dpo.py").read_text(
        encoding="utf-8"
    )
    for source in (sft_source, dpo_source):
        assert 'eval_strategy="epoch"' in source or '"eval_strategy": "epoch"' in source
        assert 'save_strategy="epoch"' in source or '"save_strategy": "epoch"' in source
        assert "trainer.evaluate()" in source
