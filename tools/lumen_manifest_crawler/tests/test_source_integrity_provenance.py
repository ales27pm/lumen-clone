from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lumen_manifest_crawler.crawler import _repository_working_tree_provenance
from lumen_manifest_crawler.manifest import SourceIntegrity


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "tests@lumen.invalid")
    _git(root, "config", "user.name", "Lumen Tests")
    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "source.py")
    _git(root, "commit", "--quiet", "-m", "initial")
    return root


def test_source_integrity_migrates_legacy_commit_without_reemitting_it():
    integrity = SourceIntegrity.model_validate({"commit": "a" * 40, "files": []})

    assert integrity.commit == "a" * 40
    assert integrity.baseCommit == "a" * 40
    assert integrity.model_dump() == {
        "baseCommit": "a" * 40,
        "workingTreeDigest": None,
        "dirtyState": None,
        "files": [],
    }
    assert integrity.lineage_dict() == {
        "baseCommit": "a" * 40,
        "workingTreeDigest": None,
        "dirtyState": None,
    }


def test_working_tree_provenance_changes_for_source_edits(tmp_path: Path):
    root = _repository(tmp_path)
    clean_digest, clean_dirty = _repository_working_tree_provenance(root)

    (root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed_digest, changed_dirty = _repository_working_tree_provenance(root)

    assert clean_digest is not None
    assert clean_dirty is False
    assert changed_digest is not None
    assert changed_digest != clean_digest
    assert changed_dirty is True


def test_generated_outputs_do_not_make_source_provenance_self_referential(tmp_path: Path):
    root = _repository(tmp_path)
    clean_digest, clean_dirty = _repository_working_tree_provenance(root)
    generated = root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    generated.parent.mkdir(parents=True)
    generated.write_text('{"generated":true}\n', encoding="utf-8")

    regenerated_digest, regenerated_dirty = _repository_working_tree_provenance(root)

    assert regenerated_digest == clean_digest
    assert clean_dirty is False
    assert regenerated_dirty is False


def test_untracked_source_is_bound_and_marked_dirty(tmp_path: Path):
    root = _repository(tmp_path)
    clean_digest, _ = _repository_working_tree_provenance(root)
    (root / "new_helper.py").write_text("VALUE = 3\n", encoding="utf-8")

    changed_digest, dirty = _repository_working_tree_provenance(root)

    assert changed_digest != clean_digest
    assert dirty is True


def test_hidden_source_directory_is_not_mistaken_for_generated_output(tmp_path: Path):
    root = _repository(tmp_path)
    clean_digest, _ = _repository_working_tree_provenance(root)
    hidden_source = root / ".generated" / "source.py"
    hidden_source.parent.mkdir()
    hidden_source.write_text("VALUE = 4\n", encoding="utf-8")

    changed_digest, dirty = _repository_working_tree_provenance(root)

    assert changed_digest != clean_digest
    assert dirty is True


def test_xcode_attestation_cli_matches_crawler_contract(tmp_path: Path):
    root = _repository(tmp_path)
    expected_digest, expected_dirty = _repository_working_tree_provenance(root)
    script = (
        Path(__file__).resolve().parents[1]
        / "lumen_manifest_crawler"
        / "source_integrity.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(root)],
        text=True,
        capture_output=True,
        check=True,
    )

    digest, dirty = completed.stdout.strip().split()
    assert digest == expected_digest
    assert dirty == ("true" if expected_dirty else "false")
