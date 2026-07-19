from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import ubuntu_pipeline


def _state_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run"
    (run_root / "training").mkdir(parents=True)
    (run_root / "configs").mkdir()
    (run_root / "configs" / "cortex.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    snapshot = (
        run_root
        / "training"
        / ubuntu_pipeline.GLOBAL_TOKENIZER_SNAPSHOT_DIRNAME
    )
    audit = (
        run_root
        / "training"
        / ubuntu_pipeline.GLOBAL_TOKENIZER_PREFLIGHT_FILENAME
    )
    return run_root, snapshot, audit


def test_tokenizer_resume_state_without_snapshot_or_audit_is_recoverable(
    tmp_path: Path,
) -> None:
    run_root, _, _ = _state_paths(tmp_path)

    assert ubuntu_pipeline._validated_global_tokenizer_resume_state(
        run_root=run_root,
        agents=("cortex",),
    ) == "not_started"


def test_tokenizer_resume_state_verified_snapshot_without_audit_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, snapshot, _ = _state_paths(tmp_path)
    snapshot.mkdir()
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_global_tokenizer_snapshot_contract",
        lambda **_: {"snapshot": "verified"},
    )

    assert ubuntu_pipeline._validated_global_tokenizer_resume_state(
        run_root=run_root,
        agents=("cortex",),
    ) == "verified_snapshot_audit_pending"


def test_tokenizer_resume_state_rejects_audit_without_snapshot(
    tmp_path: Path,
) -> None:
    run_root, _, audit = _state_paths(tmp_path)
    audit.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exists without its verified snapshot"):
        ubuntu_pipeline._validated_global_tokenizer_resume_state(
            run_root=run_root,
            agents=("cortex",),
        )


def test_tokenizer_resume_state_requires_exact_snapshot_and_audit_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, snapshot, audit = _state_paths(tmp_path)
    snapshot.mkdir()
    phase = {"schemaVersion": "fixture"}
    audit_record = {
        "tokenizerClosure": {"snapshot": "verified"},
        "agents": [
            {
                "agent": "cortex",
                "sftDatasetFileSHA256": {"train": "a" * 64},
                "preferenceDatasetFileSHA256": {"train": "b" * 64},
                "sftTrainingCodeSHA256": "c" * 64,
                "preferenceTrainingCodeSHA256": "d" * 64,
                "sft": phase,
                "preference": phase,
            }
        ],
    }
    audit.write_text(json.dumps(audit_record), encoding="utf-8")
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_global_tokenizer_snapshot_contract",
        lambda **_: {"snapshot": "verified"},
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_validated_global_tokenizer_closure_record",
        lambda *_args, **_kwargs: {"snapshot": "verified"},
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_validated_base_model_tokenizer_closure",
        lambda *_args, **_kwargs: {"closure": "same"},
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_load_verified_global_tokenizer_snapshot",
        lambda **_kwargs: (object(), {"snapshot": "verified"}),
    )
    verified_phases: list[str] = []
    def verify_global(**kwargs: object) -> dict[str, object]:
        verified_phases.append(str(kwargs["phase"]))
        return audit_record

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_global_tokenizer_preflight",
        verify_global,
    )

    assert ubuntu_pipeline._validated_global_tokenizer_resume_state(
        run_root=run_root,
        agents=("cortex",),
    ) == "verified_snapshot_and_audit"
    assert verified_phases == ["sft", "preference"]

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_validated_global_tokenizer_closure_record",
        lambda *_args, **_kwargs: {"snapshot": "different"},
    )
    with pytest.raises(RuntimeError, match="drifted from its verified snapshot"):
        ubuntu_pipeline._validated_global_tokenizer_resume_state(
            run_root=run_root,
            agents=("cortex",),
        )
