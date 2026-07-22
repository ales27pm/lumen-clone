from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import ubuntu_source_integrity


REPO_ROOT = Path(__file__).resolve().parents[4]


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _write_minimal_closure(root: Path) -> None:
    paths = set(ubuntu_source_integrity.REQUIRED_ORCHESTRATION_PATHS)
    paths.update(
        f"{prefix}fixture.py"
        for prefix in ubuntu_source_integrity.ORCHESTRATION_PREFIXES
    )
    for relative in sorted(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "tools/fine_tuning/unsloth/ignored_startup.py\n",
        encoding="utf-8",
    )


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Lumen Test")
    _git(root, "config", "user.email", "lumen@example.invalid")
    _write_minimal_closure(root)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root


def test_clean_repository_attestation_binds_worktree_and_orchestration(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)

    record = ubuntu_source_integrity.attest_repository(root)

    assert record["baseCommit"] == _git(root, "rev-parse", "HEAD")
    assert record["dirtyState"] is False
    assert len(record["workingTreeDigest"]) == 64
    assert len(record["ubuntuOrchestrationCodeSHA256"]) == 64
    orchestration_paths = {
        item["path"] for item in record["orchestrationManifest"]["files"]
    }
    assert "lumen_manifest_crawler/__init__.py" in orchestration_paths
    assert "lumen_manifest_crawler/__main__.py" not in orchestration_paths
    assert ubuntu_source_integrity.validate_attestation_record(record) == record
    assert ubuntu_source_integrity.verify_snapshot_attestation(root, record) == record


def test_git_repository_and_index_environment_overrides_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    expected = _git(root, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "attacker-worktree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "attacker-objects"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.filemode")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")

    record = ubuntu_source_integrity.attest_repository(root)
    assert record["baseCommit"] == expected


@pytest.mark.parametrize("state", ("staged", "unstaged", "untracked"))
def test_repository_attestation_rejects_every_git_dirty_state(
    tmp_path: Path,
    state: str,
) -> None:
    root = _repository(tmp_path)
    target = root / "scripts" / "ubuntu_train_lumen_full_pipeline.sh"
    if state == "staged":
        target.write_text("staged\n", encoding="utf-8")
        _git(root, "add", str(target.relative_to(root)))
    elif state == "unstaged":
        target.write_text("unstaged\n", encoding="utf-8")
    else:
        (root / "untracked.py").write_text("raise SystemExit\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean checkout"):
        ubuntu_source_integrity.attest_repository(root)


def test_repository_attestation_rejects_ignored_code_in_execution_closure(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    ignored = root / "tools/fine_tuning/unsloth/ignored_startup.py"
    ignored.write_text("raise SystemExit\n", encoding="utf-8")
    assert _git(root, "status", "--porcelain=v1") == ""

    with pytest.raises(RuntimeError, match="untracked or ignored"):
        ubuntu_source_integrity.attest_repository(root)


@pytest.mark.parametrize(
    ("flag", "message"),
    (
        ("--assume-unchanged", "assume-unchanged"),
        ("--skip-worktree", "skip-worktree"),
    ),
)
def test_repository_attestation_rejects_hidden_index_flags(
    tmp_path: Path,
    flag: str,
    message: str,
) -> None:
    root = _repository(tmp_path)
    relative = "tools/fine_tuning/unsloth/ubuntu_pipeline.py"
    _git(root, "update-index", flag, relative)
    (root / relative).write_text("hidden dirty bytes\n", encoding="utf-8")
    assert _git(root, "status", "--porcelain=v1") == ""

    with pytest.raises(RuntimeError, match=message):
        ubuntu_source_integrity.attest_repository(root)


def test_orchestration_bytes_are_independently_compared_to_git_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    relative = "tools/fine_tuning/unsloth/ubuntu_pipeline.py"
    _git(root, "update-index", "--assume-unchanged", relative)
    (root / relative).write_text("hidden dirty bytes\n", encoding="utf-8")
    monkeypatch.setattr(
        ubuntu_source_integrity,
        "_reject_hidden_index_state",
        lambda _root: None,
    )

    with pytest.raises(RuntimeError, match="staged Git blob"):
        ubuntu_source_integrity.attest_repository(root)


def test_repository_attestation_rejects_dirty_submodule(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    _git(child, "init")
    _git(child, "config", "user.name", "Lumen Test")
    _git(child, "config", "user.email", "lumen@example.invalid")
    (child / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(child, "add", ".")
    _git(child, "commit", "-m", "child")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "vendor/dependency",
    )
    _git(root, "commit", "-am", "add submodule")
    ubuntu_source_integrity.attest_repository(root)

    (root / "vendor/dependency/untracked.txt").write_text(
        "dirty\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="clean checkout"):
        ubuntu_source_integrity.attest_repository(root)


def test_image_attestation_rejects_orchestration_tamper(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    host = ubuntu_source_integrity.attest_repository(root)
    image = ubuntu_source_integrity.build_image_attestation(
        root,
        base_commit=host["baseCommit"],
        working_tree_digest=host["workingTreeDigest"],
        expected_orchestration_digest=host["ubuntuOrchestrationCodeSHA256"],
    )
    assert image == host

    target = root / "tools/fine_tuning/unsloth/ubuntu_pipeline.py"
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no longer matches"):
        ubuntu_source_integrity.verify_snapshot_attestation(root, image)


@pytest.mark.parametrize("mutation", ("tampered", "missing"))
def test_image_attestation_rejects_repo_root_crawler_shim_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _repository(tmp_path)
    host = ubuntu_source_integrity.attest_repository(root)
    shim = root / "lumen_manifest_crawler/__init__.py"
    if mutation == "tampered":
        shim.write_text("tampered crawler shim\n", encoding="utf-8")
    else:
        shim.unlink()

    with pytest.raises(RuntimeError):
        ubuntu_source_integrity.verify_snapshot_attestation(root, host)


def test_isolated_image_layout_imports_only_the_baked_crawler_shim(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    shim = source_root / "lumen_manifest_crawler/__init__.py"
    shim.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "lumen_manifest_crawler/__init__.py", shim)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(
        REPO_ROOT / "tools/lumen_manifest_crawler/lumen_manifest_crawler",
        source_root / "tools/lumen_manifest_crawler/lumen_manifest_crawler",
        ignore=ignore,
    )
    shutil.copytree(
        REPO_ROOT / "tools/fine_tuning/unsloth",
        source_root / "tools/fine_tuning/unsloth",
        ignore=ignore,
    )
    code = """
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source_root))
import lumen_manifest_crawler
from lumen_manifest_crawler.dataset import chat_template_contract
from tools.fine_tuning.unsloth import ubuntu_pipeline

assert Path(lumen_manifest_crawler.__file__).resolve() == source_root / "lumen_manifest_crawler/__init__.py"
assert Path(chat_template_contract.__file__).resolve().is_relative_to(
    source_root / "tools/lumen_manifest_crawler/lumen_manifest_crawler"
)
assert Path(ubuntu_pipeline.__file__).resolve().is_relative_to(source_root)
"""
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source_root)],
        cwd=Path("/"),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_rehashed_record_cannot_omit_required_orchestration_source(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    record = json.loads(
        json.dumps(ubuntu_source_integrity.attest_repository(root))
    )
    omitted = "tools/fine_tuning/unsloth/ubuntu_uploader.py"
    record["orchestrationManifest"]["files"] = [
        item
        for item in record["orchestrationManifest"]["files"]
        if item["path"] != omitted
    ]
    record["ubuntuOrchestrationCodeSHA256"] = (
        ubuntu_source_integrity.canonical_sha256(
            record["orchestrationManifest"]
        )
    )
    record.pop("sourceIntegritySHA256")
    record["sourceIntegritySHA256"] = ubuntu_source_integrity.canonical_sha256(
        record
    )

    with pytest.raises(RuntimeError, match="omits required"):
        ubuntu_source_integrity.validate_attestation_record(record)


def test_isolated_uploader_bootstrap_loads_the_exact_sibling_verifier() -> None:
    uploader = Path(__file__).resolve().parents[1] / "ubuntu_uploader.py"
    result = subprocess.run(
        [sys.executable, "-I", str(uploader), "upload"],
        cwd=Path("/"),
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "LUMEN_UBUNTU_SOURCE_ATTESTATION_PATH"}
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing regular Ubuntu source-integrity record" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
