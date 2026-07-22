from __future__ import annotations

import copy
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tools.fine_tuning.unsloth import ubuntu_pipeline


REPO_ROOT = Path(__file__).resolve().parents[4]


def _launcher_function(name: str) -> str:
    lines = (
        REPO_ROOT / "scripts" / "ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == f"{name}() {{"
    )
    end = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].strip() == "}"
    )
    return "\n".join(lines[start : end + 1])


def test_incomplete_preparation_accepts_only_private_snapshot_copy_state(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    training = run_root / "training"
    training.mkdir(parents=True)
    run_root.chmod(0o700)

    tokenizer = training / "global_tokenizer_snapshot"
    tokenizer.mkdir(mode=0o700)
    partial_tokenizer = tokenizer / "tokenizer.json"
    partial_tokenizer.write_bytes(b"partial")
    partial_tokenizer.chmod(0o600)

    runtime_staging = training / ".base_model_runtime_snapshot.a1b2c3d4"
    runtime_staging.mkdir(mode=0o700)
    partial_shard = runtime_staging / "model-00001-of-00002.safetensors"
    partial_shard.write_bytes(b"partial")
    partial_shard.chmod(0o644)

    ubuntu_pipeline._assert_incomplete_preparation_has_no_progress(
        run_root,
        agents=("cortex",),
    )

    checkpoint = training / "cortex" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"progress")
    with pytest.raises(RuntimeError, match="training progress"):
        ubuntu_pipeline._assert_incomplete_preparation_has_no_progress(
            run_root,
            agents=("cortex",),
        )


def test_phase_runtime_evidence_is_reconstructed_and_rejects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_snapshot = tmp_path / "runtime"
    runtime_snapshot.mkdir()
    ubuntu_pipeline.write_object(
        runtime_snapshot / "config.json",
        {"max_position_embeddings": 128},
    )
    ubuntu_pipeline.write_object(
        runtime_snapshot / "generation_config.json",
        {"max_length": 20},
    )
    tokenizer_snapshot = tmp_path / "tokenizer"
    tokenizer_snapshot.mkdir()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    tokenizer_files: list[dict[str, Any]] = []
    for name, payload in (
        ("tokenizer.json", b"tokenizer"),
        ("tokenizer_config.json", b"tokenizer config"),
    ):
        (adapter_dir / name).write_bytes(payload)
        tokenizer_files.append(
            {
                "path": name,
                "sizeBytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    tokenizer_verification = {
        "snapshotPath": str(tokenizer_snapshot),
        "snapshotVerificationSHA256": "1" * 64,
    }
    runtime_verification = {
        "snapshotPath": str(runtime_snapshot),
        "snapshotVerificationSHA256": "2" * 64,
    }
    generation_file = {
        "path": "generation_config.json",
        "sizeBytes": 1,
        "sha256": "3" * 64,
        "huggingFaceBlobID": "4" * 40,
    }
    config = {
        "baseModelID": "Qwen/Qwen3-1.7B",
        "baseModelRevision": "5" * 40,
        "baseModelIndexDigest": "6" * 64,
        "baseModelIndexShardBindingSHA256": "7" * 64,
        "baseModelArtifactDigest": "8" * 64,
        "baseModelTokenizerDigest": tokenizer_files[0]["sha256"],
        "baseModelTokenizerFiles": tokenizer_files,
        "baseModelTokenizerClosureSHA256": "9" * 64,
        "baseModelGenerationConfigFile": generation_file,
        "baseModelTokenizerSnapshotPath": str(tokenizer_snapshot),
        "baseModelTokenizerSnapshotVerification": tokenizer_verification,
        "baseModelRuntimeSnapshotPath": str(runtime_snapshot),
        "baseModelRuntimeSnapshotVerification": runtime_verification,
        "max_seq_length": 64,
    }
    source_generation_payload = {"max_length": 20, "do_sample": False}
    runtime_generation_payload = {
        **source_generation_payload,
        "max_length": 128,
    }

    class FakeGenerationConfig:
        @classmethod
        def from_pretrained(
            cls,
            path: str,
            *,
            local_files_only: bool,
        ) -> "FakeGenerationConfig":
            assert path == str(runtime_snapshot)
            assert local_files_only is True
            return cls()

        def to_dict(self) -> dict[str, Any]:
            return dict(source_generation_payload)

    transformers = ModuleType("transformers")
    transformers.GenerationConfig = FakeGenerationConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model_unsigned = {
        "schemaVersion": "lumen.runtime-model-binding/1.3.0",
        "baseModelID": config["baseModelID"],
        "baseModelRevision": config["baseModelRevision"],
        "baseModelIndexDigest": config["baseModelIndexDigest"],
        "baseModelIndexShardBindingSHA256": config[
            "baseModelIndexShardBindingSHA256"
        ],
        "baseModelArtifactDigest": config["baseModelArtifactDigest"],
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelGenerationConfigFile": generation_file,
        "runtimeSnapshotVerificationSHA256": runtime_verification[
            "snapshotVerificationSHA256"
        ],
        "runtimeSnapshotPath": str(runtime_snapshot),
        "modelConfigSHA256": "a" * 64,
        "modelConfigVerificationStatus": (
            "attested_runtime_observation_not_independently_reconstructed"
        ),
        "sourceGenerationConfigSHA256": ubuntu_pipeline.canonical_sha256(
            source_generation_payload
        ),
        "generationConfigSHA256": ubuntu_pipeline.canonical_sha256(
            runtime_generation_payload
        ),
        "generationConfigSource": "verified_private_generation_config_file",
        "allowedGenerationConfigTransformations": {
            "maxLength": {
                "source": "verified_runtime_model.config.max_position_embeddings",
                "sourceValue": 128,
                "originalValue": 20,
                "runtimeValue": 128,
            }
        },
        "runtimeLoadMaterialization": {"fixture": "verified"},
        "localFilesOnly": True,
    }
    runtime_model_binding = {
        **model_unsigned,
        "runtimeModelBindingSHA256": ubuntu_pipeline.canonical_sha256(
            model_unsigned
        ),
    }
    tokenizer_unsigned = {
        "schemaVersion": "lumen.runtime-tokenizer-binding/1.1.0",
        "baseModelID": config["baseModelID"],
        "baseModelRevision": config["baseModelRevision"],
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "runtimeSnapshotVerificationSHA256": runtime_verification[
            "snapshotVerificationSHA256"
        ],
        "runtimeSnapshotPath": str(runtime_snapshot),
        "backendContractSHA256": "c" * 64,
        "allowedRuntimeTransformations": {
            "modelMaxLength": 128,
            "paddingSide": "left",
            "truncationSide": "right",
        },
    }
    runtime_tokenizer_binding = {
        **tokenizer_unsigned,
        "runtimeTokenizerBindingSHA256": ubuntu_pipeline.canonical_sha256(
            tokenizer_unsigned
        ),
    }
    peft_unsigned = {
        "schemaVersion": "lumen.peft-base-model-identity/1.0.0",
        "baseModelID": config["baseModelID"],
        "baseModelRevision": config["baseModelRevision"],
        "adapterNames": ["default"],
        "privateRuntimePathPersisted": False,
    }
    peft_evidence = {
        **peft_unsigned,
        "peftBaseModelIdentitySHA256": ubuntu_pipeline.canonical_sha256(
            peft_unsigned
        ),
    }
    adapter_unsigned = {
        "schemaVersion": "lumen.adapter-base-tokenizer-binding/1.0.0",
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "runtimeSnapshotVerificationSHA256": runtime_verification[
            "snapshotVerificationSHA256"
        ],
        "files": tokenizer_files,
        "transformation": "exact_byte_subset_no_derived_tokenizer",
    }
    adapter_evidence = {
        **adapter_unsigned,
        "adapterTokenizerBindingSHA256": ubuntu_pipeline.canonical_sha256(
            adapter_unsigned
        ),
    }
    report = {
        "baseModelTokenizerDigest": config["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": tokenizer_files,
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelGenerationConfigFile": generation_file,
        "baseModelTokenizerSnapshotPath": str(tokenizer_snapshot),
        "baseModelTokenizerSnapshotVerification": tokenizer_verification,
        "baseModelRuntimeSnapshotPath": str(runtime_snapshot),
        "baseModelRuntimeSnapshotVerification": runtime_verification,
        "runtimeModelBinding": runtime_model_binding,
        "runtimeTokenizerBinding": runtime_tokenizer_binding,
        "peftBaseModelIdentity": peft_evidence,
        "adapterTokenizerBinding": adapter_evidence,
    }
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_private_tokenizer_snapshot_binding",
        lambda _config: tokenizer_verification,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_private_base_model_runtime_snapshot_binding",
        lambda _config: runtime_verification,
    )
    from tools.fine_tuning.unsloth import runtime_binding_smoke_gate

    monkeypatch.setattr(
        runtime_binding_smoke_gate,
        "verify_runtime_load_materialization_evidence",
        lambda value, _config, _bound_config_payload=None: dict(value),
    )

    ubuntu_pipeline._verify_phase_runtime_evidence(
        config=config,
        report=report,
        adapter_dir=adapter_dir,
    )

    mutated = copy.deepcopy(report)
    mutated_transformations = mutated["runtimeTokenizerBinding"][
        "allowedRuntimeTransformations"
    ]
    mutated_transformations["modelMaxLength"] = 64
    mutated_unsigned = dict(mutated["runtimeTokenizerBinding"])
    mutated_unsigned.pop("runtimeTokenizerBindingSHA256")
    mutated["runtimeTokenizerBinding"]["runtimeTokenizerBindingSHA256"] = (
        ubuntu_pipeline.canonical_sha256(mutated_unsigned)
    )
    with pytest.raises(RuntimeError, match="Runtime tokenizer binding drifted"):
        ubuntu_pipeline._verify_phase_runtime_evidence(
            config=config,
            report=mutated,
            adapter_dir=adapter_dir,
        )

    mutated = copy.deepcopy(report)
    mutated["peftBaseModelIdentity"]["adapterNames"] = ["other"]
    mutated_unsigned = dict(mutated["peftBaseModelIdentity"])
    mutated_unsigned.pop("peftBaseModelIdentitySHA256")
    mutated["peftBaseModelIdentity"]["peftBaseModelIdentitySHA256"] = (
        ubuntu_pipeline.canonical_sha256(mutated_unsigned)
    )
    with pytest.raises(RuntimeError, match="PEFT base-model identity"):
        ubuntu_pipeline._verify_phase_runtime_evidence(
            config=config,
            report=mutated,
            adapter_dir=adapter_dir,
        )


def test_verify_evaluation_replays_against_verified_preference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_phase = {
        "phase": "dpo",
        "adapterSHA256": "a" * 64,
        "finalizedVariantManifestSHA256": "b" * 64,
    }
    expected = {"status": "quality_gate_passed"}
    calls: list[tuple[Path, str, dict[str, Any]]] = []
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda run_root, agent: final_phase,
    )

    def fake_verify(
        run_root: Path,
        agent: str,
        *,
        final_phase: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append((run_root, agent, final_phase))
        return expected

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_evaluation_outputs",
        fake_verify,
    )

    assert ubuntu_pipeline.verify_evaluation(tmp_path, "cortex") == expected
    assert calls == [(tmp_path, "cortex", final_phase)]


def _mock_gguf_install_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_prepared_gguf_agents",
        lambda _run_root: ({"agents": [{"agent": "cortex"}]}, ("cortex",)),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_reject_managed_symlinks",
        lambda _run_root: None,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_gguf_inventory",
        lambda _run_root, _agents, *, require_all: {},
    )
    def fake_receipt(
        run_root: Path,
        agent: str,
        *,
        artifact_path: Path,
        receipt_path: Path,
    ) -> dict[str, Any]:
        del receipt_path
        final_path, _ = ubuntu_pipeline._gguf_owned_paths(run_root, agent)
        return {
            "adapterGGUF": str(final_path),
            "adapterGGUFSHA256": "c" * 64,
            "adapterGGUFSizeBytes": artifact_path.stat().st_size,
            **{
                field: f"semantic-{field}"
                for field in ubuntu_pipeline.ADAPTER_GGUF_SEMANTIC_FIELDS
            },
            "conversionReceiptSHA256": "d" * 64,
            "qualification": ubuntu_pipeline.GGUF_CONVERSION_QUALIFICATION,
            "tensorEquivalenceStatus": (
                ubuntu_pipeline.GGUF_TENSOR_EQUIVALENCE_STATUS
            ),
            "runtimeModelBindingSHA256": "e" * 64,
            "runtimeTokenizerBindingSHA256": "f" * 64,
        }

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_gguf_conversion_receipt",
        fake_receipt,
    )

    def fake_verify(run_root: Path, path: Path) -> dict[str, Any]:
        agent = "cortex"
        _, receipt_path = ubuntu_pipeline._gguf_owned_paths(run_root, agent)
        return ubuntu_pipeline._gguf_verification_evidence(
            fake_receipt(
                run_root,
                agent,
                artifact_path=path,
                receipt_path=receipt_path,
            ),
            receipt_path=receipt_path,
        )

    monkeypatch.setattr(ubuntu_pipeline, "verify_gguf_file", fake_verify)


def test_install_gguf_file_verifies_then_atomically_promotes_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_gguf_install_lineage(monkeypatch)
    run_root = tmp_path / "run"
    final_dir = run_root / "models" / "lora_qwen3_gguf"
    receipt_dir = run_root / "models" / "lora_qwen3_gguf_receipts"
    staging_dir = run_root / ".gguf-staging" / "cortex"
    final_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True)
    staging_dir.mkdir(parents=True)
    (run_root / ".gguf-staging").chmod(0o700)
    staging_dir.chmod(0o700)
    staging_path = staging_dir / "lumen-cortex-lora.gguf"
    staging_path.write_bytes(b"verified staged payload")
    staging_path.chmod(0o600)
    staged_receipt = staging_dir / "conversion_receipt.json"
    staged_receipt.write_text("{}\n", encoding="utf-8")
    staged_receipt.chmod(0o600)

    result = ubuntu_pipeline.install_gguf_file(
        run_root,
        "cortex",
        staging_path,
    )

    final_path = final_dir / staging_path.name
    assert final_path.read_bytes() == b"verified staged payload"
    final_receipt = receipt_dir / "lumen-cortex-lora.conversion.json"
    assert final_receipt.read_text(encoding="utf-8") == "{}\n"
    assert not (run_root / ".gguf-staging").exists()
    assert result == {
        "agent": "cortex",
        "adapterGGUF": str(final_path),
        "adapterGGUFSHA256": "c" * 64,
        "adapterGGUFSizeBytes": len(b"verified staged payload"),
        **{
            field: f"semantic-{field}"
            for field in ubuntu_pipeline.ADAPTER_GGUF_SEMANTIC_FIELDS
        },
        "adapterGGUFConversionReceipt": str(final_receipt),
        "adapterGGUFConversionReceiptSHA256": "d" * 64,
        "adapterGGUFConversionQualification": (
            ubuntu_pipeline.GGUF_CONVERSION_QUALIFICATION
        ),
        "adapterGGUFTensorEquivalenceStatus": (
            ubuntu_pipeline.GGUF_TENSOR_EQUIVALENCE_STATUS
        ),
        "adapterGGUFRuntimeModelBindingSHA256": "e" * 64,
        "adapterGGUFRuntimeTokenizerBindingSHA256": "f" * 64,
    }


def test_install_gguf_file_rejects_a_symlink_staging_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_gguf_install_lineage(monkeypatch)
    run_root = tmp_path / "run"
    (run_root / "models" / "lora_qwen3_gguf").mkdir(parents=True)
    (run_root / "models" / "lora_qwen3_gguf_receipts").mkdir(parents=True)
    staging_dir = run_root / ".gguf-staging" / "cortex"
    staging_dir.mkdir(parents=True)
    (run_root / ".gguf-staging").chmod(0o700)
    staging_dir.chmod(0o700)
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"outside")
    staging_path = staging_dir / "lumen-cortex-lora.gguf"
    staging_path.symlink_to(outside)
    receipt = staging_dir / "conversion_receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    receipt.chmod(0o600)

    with pytest.raises(RuntimeError, match="unsafe or unexpected entries"):
        ubuntu_pipeline.install_gguf_file(
            run_root,
            "cortex",
            staging_path,
        )
    assert outside.read_bytes() == b"outside"


def test_launcher_resumes_evaluation_and_gguf_without_completed_summary() -> None:
    launcher = (
        REPO_ROOT / "scripts" / "ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")

    assert "verify-evaluation" in launcher
    assert 'log "verified existing frozen evaluation: $agent"' in launcher
    assert 'clean_agent_evaluation "$agent"' in launcher
    assert launcher.count('clean_agent_posttraining_artifacts "$agent"') == 2
    assert "--verify-checkpoint-only" in launcher
    assert "preserving verified interrupted evaluation checkpoint" in launcher
    assert "classify-completed-evaluation" in launcher
    assert "preserving verified terminal evaluation evidence" in launcher
    assert "preserving it instead of deleting progress" in launcher
    evaluate_function = _launcher_function("evaluate_agent")
    completed_classifier = evaluate_function.index(
        "classify-completed-evaluation"
    )
    checkpoint_verifier = evaluate_function.index("--verify-checkpoint-only")
    normal_evaluation = evaluate_function.rindex(
        "tools.fine_tuning.unsloth.evaluate_adapter"
    )
    assert completed_classifier < checkpoint_verifier < normal_evaluation
    assert 'clean_agent_evaluation "$agent"' not in evaluate_function
    assert '>> "$classification_log" 2>&1' in evaluate_function
    assert '>> "$checkpoint_log" 2>&1' in evaluate_function
    assert "verify-gguf-file" in launcher
    assert "install-gguf-file" in launcher
    assert "write-gguf-conversion-receipt" in launcher
    assert "ubuntu_pipeline verify-gguf \\" not in launcher
    assert '--outfile "$staged_outfile"' in launcher
    assert "os.replace(staging, destination)" in launcher
    assert 'git init "$CONVERTER_STAGING"' in launcher
    assert 'git init "$CONVERTER_REPO"' not in launcher
    assert "discarding an incomplete staged llama.cpp checkout" in launcher
    assert "recovering the verified staged llama.cpp checkout" in launcher
    assert "evaluation attempt started:" in launcher
    assert 'tee -a "$RUN_ROOT/logs/evaluate_$agent.log"' in launcher
    assert "GGUF conversion attempt started:" in launcher
    assert 'tee -a "$RUN_ROOT/logs/convert_$agent.log"' in launcher


@pytest.mark.parametrize(
    ("classifier_exit", "checkpoint_exit", "expected_exit", "resumes"),
    (
        (0, 0, 99, False),
        (70, 70, 99, False),
        (137, 137, 99, False),
        (1, 70, 99, False),
        (1, 137, 99, False),
        (1, 0, 0, True),
    ),
)
def test_launcher_preserves_evaluation_for_terminal_operational_and_signal_states(
    tmp_path: Path,
    classifier_exit: int,
    checkpoint_exit: int,
    expected_exit: int,
    resumes: bool,
) -> None:
    run_root = tmp_path / "run"
    evaluation_dir = run_root / "evaluation" / "cortex"
    evaluation_dir.mkdir(parents=True)
    (run_root / "logs").mkdir()
    sentinel = evaluation_dir / "sentinel"
    sentinel.write_bytes(b"irreplaceable evaluation progress")
    call_log = tmp_path / "calls.log"
    verify_count = tmp_path / "verify-count"
    verify_count.write_text("0\n", encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -u
printf '%s\\n' "$*" >> "$CALL_LOG"
case " $* " in
  *" classify-completed-evaluation "*) exit "$CLASSIFIER_EXIT" ;;
  *" verify-evaluation "*)
    count="$(cat "$VERIFY_COUNT_FILE")"
    printf '%s\\n' "$((count + 1))" > "$VERIFY_COUNT_FILE"
    if [[ "$count" == "0" ]]; then exit 1; fi
    exit 0
    ;;
  *" --verify-checkpoint-only "*) exit "$CHECKPOINT_EXIT" ;;
  *" tools.fine_tuning.unsloth.evaluate_adapter "*)
    printf '%s\\n' normal-evaluation >> "$CALL_LOG"
    exit 0
    ;;
  *) exit 0 ;;
esac
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    harness = "\n".join(
        (
            "set -euo pipefail",
            f"ROOT={shlex.quote(str(REPO_ROOT))}",
            f"RUN_ROOT={shlex.quote(str(run_root))}",
            f"TRAIN_PY={shlex.quote(str(fake_python))}",
            "RESUME=1",
            "EVAL_MAX_EXAMPLES=",
            "log() { printf '%s\\n' \"$*\"; }",
            "die() { printf '%s\\n' \"$*\" >&2; exit 99; }",
            (
                "clean_agent_evaluation() { printf '%s\\n' cleanup "
                '>> "$CALL_LOG"; rm -rf -- "$RUN_ROOT/evaluation/$1"; }'
            ),
            _launcher_function("verify_evaluation"),
            _launcher_function("evaluate_agent"),
            "evaluate_agent cortex",
        )
    )
    env = {
        **os.environ,
        "CALL_LOG": str(call_log),
        "VERIFY_COUNT_FILE": str(verify_count),
        "CLASSIFIER_EXIT": str(classifier_exit),
        "CHECKPOINT_EXIT": str(checkpoint_exit),
    }

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == expected_exit, result.stderr
    assert sentinel.read_bytes() == b"irreplaceable evaluation progress"
    calls = call_log.read_text(encoding="utf-8")
    assert "cleanup" not in calls
    assert ("normal-evaluation" in calls) is resumes


def test_launcher_qualifies_each_agent_before_starting_the_next() -> None:
    launcher = (
        REPO_ROOT / "scripts" / "ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")

    assert launcher.count('for agent in "${AGENTS[@]}"; do') == 1
    loop = launcher.split(
        '# Deliberately qualify and export one agent before spending GPU time on the\n'
        '# next. A quality failure therefore stops the fleet at the earliest boundary.\n',
        1,
    )[1].split("\ndone\n", 1)[0]
    ordered_markers = (
        'train_sft \\\n',
        'train_dpo \\\n',
        'evaluate_agent "$agent"',
        'convert_agent_gguf "$agent"',
    )
    positions = [loop.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)


def _conversion_receipt(agent: str, marker: str) -> dict[str, Any]:
    receipt = {
        field: None
        for field in ubuntu_pipeline.GGUF_CONVERSION_RECEIPT_FIELDS
        if field != "conversionReceiptSHA256"
    }
    receipt.update(
        {
            "schema": ubuntu_pipeline.GGUF_CONVERSION_RECEIPT_SCHEMA_VERSION,
            "agent": agent,
            "qualification": ubuntu_pipeline.GGUF_CONVERSION_QUALIFICATION,
            "tensorEquivalenceStatus": (
                ubuntu_pipeline.GGUF_TENSOR_EQUIVALENCE_STATUS
            ),
            "adapterGGUFSHA256": marker * 64,
        }
    )
    receipt["conversionReceiptSHA256"] = ubuntu_pipeline.canonical_sha256(
        receipt
    )
    return receipt


def test_conversion_receipt_rejects_a_renamed_cross_agent_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "lumen-cortex-lora.gguf"
    artifact.write_bytes(b"cross-agent output")
    receipt_path = tmp_path / "lumen-cortex-lora.conversion.json"
    ubuntu_pipeline.write_object(
        receipt_path,
        _conversion_receipt("executor", "e"),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_gguf_conversion_receipt_payload",
        lambda _run_root, _agent, _artifact_path: _conversion_receipt(
            "cortex", "c"
        ),
    )

    with pytest.raises(RuntimeError, match="integrity checks"):
        ubuntu_pipeline._verified_gguf_conversion_receipt(
            tmp_path,
            "cortex",
            artifact_path=artifact,
            receipt_path=receipt_path,
        )


def test_conversion_receipt_rejects_self_consistent_stale_adapter_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "lumen-cortex-lora.gguf"
    artifact.write_bytes(b"stale output")
    receipt_path = tmp_path / "lumen-cortex-lora.conversion.json"
    stale = _conversion_receipt("cortex", "a")
    current = _conversion_receipt("cortex", "b")
    ubuntu_pipeline.write_object(receipt_path, stale)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_gguf_conversion_receipt_payload",
        lambda _run_root, _agent, _artifact_path, **_kwargs: current,
    )

    with pytest.raises(RuntimeError, match="current lineage"):
        ubuntu_pipeline._verified_gguf_conversion_receipt(
            tmp_path,
            "cortex",
            artifact_path=artifact,
            receipt_path=receipt_path,
        )


def test_conversion_receipt_is_written_durably_inside_private_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "models" / "lora_qwen3_gguf").mkdir(parents=True)
    (run_root / "models" / "lora_qwen3_gguf_receipts").mkdir(parents=True)
    staging_dir = run_root / ".gguf-staging" / "cortex"
    staging_dir.mkdir(parents=True)
    (run_root / ".gguf-staging").chmod(0o700)
    staging_dir.chmod(0o700)
    artifact = staging_dir / "lumen-cortex-lora.gguf"
    artifact.write_bytes(b"converted")
    artifact.chmod(0o600)
    snapshot_proof = (
        staging_dir / ubuntu_pipeline.GGUF_BASE_SNAPSHOT_VERIFICATION_FILENAME
    )
    ubuntu_pipeline.write_object(snapshot_proof, {})
    snapshot_proof.chmod(0o600)
    expected = _conversion_receipt("cortex", "c")
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_prepared_gguf_agents",
        lambda _run_root: ({"agents": [{"agent": "cortex"}]}, ("cortex",)),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_reject_managed_symlinks",
        lambda _run_root: None,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_gguf_inventory",
        lambda _run_root, _agents, *, require_all: {},
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_gguf_conversion_receipt_payload",
        lambda _run_root, _agent, _artifact_path, **_kwargs: expected,
    )

    result = ubuntu_pipeline.write_gguf_conversion_receipt(
        run_root,
        "cortex",
        artifact,
    )

    receipt = staging_dir / "conversion_receipt.json"
    assert ubuntu_pipeline.read_object(receipt) == expected
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert result["adapterGGUFConversionReceipt"] == str(receipt)
    assert result["adapterGGUFConversionReceiptSHA256"] == expected[
        "conversionReceiptSHA256"
    ]


def test_clean_agent_gguf_staging_removes_the_empty_root_exactly_once(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    staging_root = run_root / ".gguf-staging"
    staging_dir = staging_root / "cortex"
    staging_dir.mkdir(parents=True)
    staging_root.chmod(0o700)
    staging_dir.chmod(0o700)
    (staging_dir / "partial.gguf").write_bytes(b"partial")
    function = _launcher_function("clean_agent_gguf_staging")
    script = "\n".join(
        (
            "set -Eeuo pipefail",
            f"RUN_ROOT={shlex.quote(str(run_root))}",
            "AGENTS_CSV=cortex",
            f"GGUF_STAGING_ROOT={shlex.quote(str(staging_root))}",
            'die() { printf "%s\\n" "$*" >&2; exit 1; }',
            function,
            'clean_agent_gguf_staging "cortex"',
            '[[ ! -e "$GGUF_STAGING_ROOT" ]]',
        )
    )

    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not staging_root.exists()


def test_invalid_gguf_cleanup_removes_the_artifact_and_its_receipt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    gguf_root = run_root / "models" / "lora_qwen3_gguf"
    receipt_root = run_root / "models" / "lora_qwen3_gguf_receipts"
    gguf_root.mkdir(parents=True)
    receipt_root.mkdir(parents=True)
    gguf = gguf_root / "lumen-cortex-lora.gguf"
    receipt = receipt_root / "lumen-cortex-lora.conversion.json"
    gguf.write_bytes(b"invalid")
    receipt.write_text("{}\n", encoding="utf-8")
    function = _launcher_function("remove_invalid_agent_gguf")
    script = "\n".join(
        (
            "set -Eeuo pipefail",
            f"RUN_ROOT={shlex.quote(str(run_root))}",
            "AGENTS_CSV=cortex",
            'die() { printf "%s\\n" "$*" >&2; exit 1; }',
            function,
            'remove_invalid_agent_gguf "cortex"',
            f"[[ ! -e {shlex.quote(str(gguf))} ]]",
            f"[[ ! -e {shlex.quote(str(receipt))} ]]",
        )
    )

    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not gguf.exists()
    assert not receipt.exists()


def test_converter_cleanup_removes_only_the_private_owned_staging_directory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    final_checkout = run_root / "llama.cpp"
    staging_checkout = run_root / ".llama.cpp.staging"
    staging_checkout.mkdir(mode=0o700)
    (staging_checkout / "partial-fetch").write_bytes(b"partial")
    function = _launcher_function("remove_derived_converter_checkout")
    script = "\n".join(
        (
            "set -Eeuo pipefail",
            f"TRAIN_PY={shlex.quote(sys.executable)}",
            f"CONVERTER_REPO={shlex.quote(str(final_checkout))}",
            f"CONVERTER_STAGING={shlex.quote(str(staging_checkout))}",
            'die() { printf "%s\\n" "$*" >&2; exit 1; }',
            function,
            'remove_derived_converter_checkout "$CONVERTER_STAGING"',
            '[[ ! -e "$CONVERTER_STAGING" ]]',
        )
    )

    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not staging_checkout.exists()


def test_converter_cleanup_rejects_a_symlink_staging_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"keep")
    final_checkout = run_root / "llama.cpp"
    staging_checkout = run_root / ".llama.cpp.staging"
    staging_checkout.symlink_to(outside, target_is_directory=True)
    function = _launcher_function("remove_derived_converter_checkout")
    script = "\n".join(
        (
            "set -Eeuo pipefail",
            f"TRAIN_PY={shlex.quote(sys.executable)}",
            f"CONVERTER_REPO={shlex.quote(str(final_checkout))}",
            f"CONVERTER_STAGING={shlex.quote(str(staging_checkout))}",
            'die() { printf "%s\\n" "$*" >&2; exit 1; }',
            function,
            'remove_derived_converter_checkout "$CONVERTER_STAGING"',
        )
    )

    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "symlink converter path" in result.stderr
    assert (outside / "keep").read_bytes() == b"keep"
