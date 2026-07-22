from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.fine_tuning.unsloth import ubuntu_pipeline


def _prepared_tree(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "prepared-run"
    nested = run_root / "generated" / "fine_tuning" / "cortex"
    nested.mkdir(parents=True)
    payload = nested / "train_sft.jsonl"
    payload.write_bytes(b'{"messages":[]}\n')
    (run_root / "aio_run_manifest.json").write_text(
        '{"schema":"test"}\n',
        encoding="utf-8",
    )
    return run_root, payload


def test_prepare_input_closure_accepts_unchanged_tree(tmp_path: Path) -> None:
    run_root, _ = _prepared_tree(tmp_path)
    closure = ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)
    try:
        closure.verify_unchanged()
    finally:
        closure.close()


def test_prepare_input_closure_rejects_in_place_mutation_restored_bytes(
    tmp_path: Path,
) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    original = payload.read_bytes()
    original_stat = payload.stat()
    closure = ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)
    try:
        payload.write_bytes(b"x" * len(original))
        payload.write_bytes(original)
        os.utime(
            payload,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        with pytest.raises(RuntimeError, match="input file changed"):
            closure.verify_unchanged()
    finally:
        closure.close()


def test_prepare_input_closure_rejects_atomic_replacement_with_same_bytes(
    tmp_path: Path,
) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    replacement = tmp_path / "same-bytes-replacement"
    replacement.write_bytes(payload.read_bytes())
    replacement.chmod(payload.stat().st_mode & 0o777)
    closure = ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)
    try:
        os.replace(replacement, payload)
        with pytest.raises(RuntimeError, match="input (directory|inventory)"):
            closure.verify_unchanged()
    finally:
        closure.close()


def test_prepare_input_closure_rejects_transient_add_then_remove(
    tmp_path: Path,
) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    closure = ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)
    try:
        transient = payload.parent / "transient.json"
        transient.write_text("{}\n", encoding="utf-8")
        transient.unlink()
        with pytest.raises(RuntimeError, match="input directory changed"):
            closure.verify_unchanged()
    finally:
        closure.close()


def test_prepare_input_closure_rejects_inventory_path_change(
    tmp_path: Path,
) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    closure = ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)
    try:
        payload.rename(payload.with_name("renamed.jsonl"))
        with pytest.raises(RuntimeError, match="input directory changed"):
            closure.verify_unchanged()
    finally:
        closure.close()


def test_prepare_input_closure_rejects_nested_directory_replacement(
    tmp_path: Path,
) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    original_directory = payload.parent
    replacement = tmp_path / "replacement-cortex"
    replacement.mkdir()
    (replacement / payload.name).write_bytes(payload.read_bytes())
    closure = ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)
    displaced = tmp_path / "original-cortex"
    try:
        original_directory.rename(displaced)
        replacement.rename(original_directory)
        with pytest.raises(RuntimeError, match="input directory changed"):
            closure.verify_unchanged()
    finally:
        closure.close()


def test_prepare_input_closure_rejects_mutation_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    original = ubuntu_pipeline._descriptor_sha256
    mutated = False

    def mutate_after_hash(descriptor: int) -> str:
        nonlocal mutated
        digest = original(descriptor)
        if not mutated and os.readlink(f"/proc/self/fd/{descriptor}") == str(payload):
            payload.write_bytes(b"mutated during hash\n")
            mutated = True
        return digest

    monkeypatch.setattr(ubuntu_pipeline, "_descriptor_sha256", mutate_after_hash)
    with pytest.raises(RuntimeError, match="changed while it was hashed"):
        ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)


def test_prepare_input_closure_rejects_symbolic_links(tmp_path: Path) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    (run_root / "linked-input").symlink_to(payload)

    with pytest.raises(RuntimeError, match="contains a symbolic link"):
        ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)


def test_prepare_input_closure_partial_failure_does_not_leak_descriptors(
    tmp_path: Path,
) -> None:
    run_root, payload = _prepared_tree(tmp_path)
    (run_root / "linked-input").symlink_to(payload)
    before = ubuntu_pipeline._open_descriptor_count()

    with pytest.raises(RuntimeError, match="contains a symbolic link"):
        ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)

    assert ubuntu_pipeline._open_descriptor_count() == before


def test_production_prepare_input_closure_requires_exact_readonly_mount(
    tmp_path: Path,
) -> None:
    run_root, _ = _prepared_tree(tmp_path)

    with pytest.raises(RuntimeError, match="must be an exact mount point"):
        ubuntu_pipeline._acquire_prepared_input_closure(run_root)


def test_exact_prepare_mount_identity_requires_read_only_unique_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount_point = str(tmp_path.resolve())

    def identity(*, options: tuple[str, ...]) -> ubuntu_pipeline._MountIdentity:
        return ubuntu_pipeline._MountIdentity(
            mount_id=10,
            parent_id=1,
            device="0:1",
            root="/",
            mount_point=mount_point,
            mount_options=options,
            filesystem_type="ext4",
            mount_source="/dev/test",
            super_options=("rw",),
        )

    readonly = identity(options=("nodev", "ro"))
    monkeypatch.setattr(ubuntu_pipeline, "_mountinfo_records", lambda: (readonly,))
    assert ubuntu_pipeline._exact_readonly_mount_identity(tmp_path) == readonly

    writable = identity(options=("nodev", "rw"))
    monkeypatch.setattr(ubuntu_pipeline, "_mountinfo_records", lambda: (writable,))
    with pytest.raises(RuntimeError, match="mounted read-only"):
        ubuntu_pipeline._exact_readonly_mount_identity(tmp_path)

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_mountinfo_records",
        lambda: (readonly, readonly),
    )
    with pytest.raises(RuntimeError, match="exact mount point"):
        ubuntu_pipeline._exact_readonly_mount_identity(tmp_path)


def test_prepare_input_closure_rejects_nested_mount_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _ = _prepared_tree(tmp_path)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_mounted_descendants",
        lambda path: [path / "generated"],
    )

    with pytest.raises(RuntimeError, match="contains nested mounts"):
        ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)


def test_prepare_input_closure_rejects_special_file(tmp_path: Path) -> None:
    run_root, _ = _prepared_tree(tmp_path)
    fifo = run_root / "unexpected.fifo"
    os.mkfifo(fifo)

    with pytest.raises(RuntimeError, match="contains a special file"):
        ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)


def test_prepare_input_closure_rejects_non_utf8_path(tmp_path: Path) -> None:
    run_root, _ = _prepared_tree(tmp_path)
    root_bytes = os.fsencode(run_root)
    descriptor = os.open(
        root_bytes + b"/invalid-\xff.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    os.close(descriptor)

    with pytest.raises(RuntimeError, match="non-UTF-8 path"):
        ubuntu_pipeline._acquire_prepared_input_closure_test_only(run_root)


def test_prepare_input_closure_rejects_insufficient_fd_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ubuntu_pipeline.resource,
        "getrlimit",
        lambda _: (40, 40),
    )
    monkeypatch.setattr(ubuntu_pipeline, "_open_descriptor_count", lambda: 10)

    with pytest.raises(RuntimeError, match="lacks file-descriptor headroom"):
        ubuntu_pipeline._require_prepared_input_fd_headroom(31)


def test_prepare_only_postcondition_holds_closure_across_all_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_root, _ = _prepared_tree(tmp_path)
    events: list[str] = []

    class FakeClosure:
        inventory = ({"path": ".", "kind": "directory"},)
        inventory_sha256 = "e" * 64
        mount_identity_sha256 = "f" * 64

        def verify_unchanged(self) -> None:
            events.append("verify-closure")

        def close(self) -> None:
            events.append("close-closure")

    def acquire(path: Path) -> FakeClosure:
        assert path == run_root.resolve()
        events.append("acquire-closure")
        return FakeClosure()

    def validate(**_: object) -> dict[str, str]:
        events.append("validate-runtime")
        return {
            "trainingEnvironmentSHA256": "a" * 64,
            "observedAccelerator": "test",
        }

    def retokenize(**_: object) -> dict[str, object]:
        events.append("retokenize")
        return {
            "globalPreflightSHA256": "b" * 64,
            "tokenizerClosure": {
                "tokenizerClosureSHA256": "c" * 64,
            },
        }

    def verify_runtime_binding_smoke(
        _run_root: Path,
        _agents: tuple[str, ...],
    ) -> dict[str, str]:
        events.append("verify-runtime-binding-smoke")
        return {"runtimeBindingSmokeGateSHA256": "d" * 64}

    monkeypatch.setattr(
        ubuntu_pipeline,
        "parse_args",
        lambda: SimpleNamespace(
            command="verify-container-postcondition",
            root=tmp_path,
            run_root=run_root,
            agents="cortex",
            variant="internal_plus_public_optimized",
            container_digest="sha256:" + ("d" * 64),
            evaluation_scope="full",
            evaluation_max_examples=None,
            gguf_requested=True,
            prepare_only=True,
        ),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_acquire_prepared_input_closure",
        acquire,
    )
    monkeypatch.setattr(ubuntu_pipeline, "validate_prepared_runtime", validate)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_prepared_global_tokenizer_preflight",
        retokenize,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        verify_runtime_binding_smoke,
    )

    ubuntu_pipeline.main()

    assert events == [
        "acquire-closure",
        "validate-runtime",
        "retokenize",
        "verify-runtime-binding-smoke",
        "verify-closure",
        "close-closure",
    ]
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "prepared_postcondition_verified"
    assert output["prepareInputMountStatus"] == "exact_readonly_mount_verified"
    assert output["prepareInputClosureEntryCount"] == 1
    assert output["runtimeBindingSmokeGateSHA256"] == "d" * 64


def test_prepare_only_postcondition_closes_closure_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _ = _prepared_tree(tmp_path)
    events: list[str] = []

    class FakeClosure:
        def close(self) -> None:
            events.append("close-closure")

    def acquire(path: Path) -> FakeClosure:
        assert path == run_root.resolve()
        events.append("acquire-closure")
        return FakeClosure()

    monkeypatch.setattr(
        ubuntu_pipeline,
        "parse_args",
        lambda: SimpleNamespace(
            command="verify-container-postcondition",
            root=tmp_path,
            run_root=run_root,
            agents="cortex",
            variant="internal_plus_public_optimized",
            container_digest="sha256:" + ("d" * 64),
            evaluation_scope="full",
            evaluation_max_examples=None,
            gguf_requested=True,
            prepare_only=True,
        ),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_acquire_prepared_input_closure",
        acquire,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "validate_prepared_runtime",
        lambda **_: (_ for _ in ()).throw(RuntimeError("validation failed")),
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        ubuntu_pipeline.main()

    assert events == ["acquire-closure", "close-closure"]
