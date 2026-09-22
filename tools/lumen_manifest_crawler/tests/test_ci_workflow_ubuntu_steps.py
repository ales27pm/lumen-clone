"""Regression tests for the Ubuntu launcher/test steps added to the CI workflow.

These tests deliberately avoid a YAML parsing dependency (PyYAML is not part of
the crawler's declared dependencies, and CI installs only ``pydantic``,
``typer``, ``rich``, and ``pytest`` via ``uv run --with``). Instead they parse
the well-known, consistently indented GitHub Actions step structure with
regular expressions, mirroring how ``scripts/check-generated-jsonl-artifacts.py``
inspects generated text without extra dependencies.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

WORKFLOW_RELATIVE_PATH = ".github/workflows/lumen-integration.yml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _workflow_text() -> str:
    return (_repo_root() / WORKFLOW_RELATIVE_PATH).read_text(encoding="utf-8")


def _job_section(text: str, job_name: str) -> str:
    """Return the raw text belonging to a top-level job block."""
    job_pattern = re.compile(rf"^  {re.escape(job_name)}:\s*$", re.MULTILINE)
    match = job_pattern.search(text)
    assert match, f"job {job_name!r} was not found in {WORKFLOW_RELATIVE_PATH}"
    start = match.end()
    next_job = re.search(r"^  [A-Za-z0-9_-]+:\s*$", text[start:], re.MULTILINE)
    end = start + next_job.start() if next_job else len(text)
    return text[start:end]


def _step_blocks(job_text: str) -> list[tuple[str, str]]:
    """Return ``(step_name, raw_block_text)`` pairs in the order they appear."""
    starts = list(re.finditer(r"^ {6}- name: (.+)$", job_text, re.MULTILINE))
    assert starts, "expected at least one named step in the job"
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(starts):
        name = match.group(1).strip()
        block_start = match.start()
        block_end = starts[index + 1].start() if index + 1 < len(starts) else len(job_text)
        blocks.append((name, job_text[block_start:block_end]))
    return blocks


def _block_by_name(blocks: list[tuple[str, str]], name: str) -> str:
    for step_name, block in blocks:
        if step_name == name:
            return block
    raise AssertionError(f"no step named {name!r} found; available: {[n for n, _ in blocks]}")


@pytest.fixture(scope="module")
def static_job_text() -> str:
    return _job_section(_workflow_text(), "static-and-python")


@pytest.fixture(scope="module")
def static_job_steps(static_job_text: str) -> list[tuple[str, str]]:
    return _step_blocks(static_job_text)


class TestWorkflowStructure:
    def test_workflow_file_exists(self) -> None:
        assert (_repo_root() / WORKFLOW_RELATIVE_PATH).is_file()

    def test_static_job_contains_expected_step_names_in_order(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        names = [name for name, _ in static_job_steps]
        assert names == [
            "Set up Python",
            "Set up uv",
            "Python syntax compile",
            "Generated artifact validation",
            "Adapter runtime invariants",
            "Ubuntu launcher syntax",
            "Python tests",
            "Ubuntu source, lineage, evaluation, and launcher tests",
            "Git diff whitespace check",
        ]

    def test_static_job_step_names_are_unique(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        names = [name for name, _ in static_job_steps]
        assert len(names) == len(set(names))


class TestUbuntuLauncherSyntaxStep:
    def test_step_checks_both_launcher_scripts(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        block = _block_by_name(static_job_steps, "Ubuntu launcher syntax")
        assert "bash -n scripts/ubuntu_train_lumen_full_pipeline.sh" in block
        assert "bash -n scripts/ubuntu_train_lumen_adapters_aio.sh" in block

    def test_step_runs_before_python_tests_and_after_adapter_invariants(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        names = [name for name, _ in static_job_steps]
        assert names.index("Adapter runtime invariants") < names.index(
            "Ubuntu launcher syntax"
        )
        assert names.index("Ubuntu launcher syntax") < names.index("Python tests")

    @pytest.mark.parametrize(
        "script",
        [
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            "scripts/ubuntu_train_lumen_adapters_aio.sh",
        ],
    )
    def test_referenced_launcher_script_exists(self, script: str) -> None:
        assert (_repo_root() / script).is_file(), f"missing launcher script {script}"

    @pytest.mark.parametrize(
        "script",
        [
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            "scripts/ubuntu_train_lumen_adapters_aio.sh",
        ],
    )
    def test_referenced_launcher_script_has_valid_bash_syntax(self, script: str) -> None:
        result = subprocess.run(
            ["bash", "-n", str(_repo_root() / script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestUbuntuSourceLineageEvaluationLauncherTestsStep:
    STEP_NAME = "Ubuntu source, lineage, evaluation, and launcher tests"

    def test_step_disables_bytecode_writing(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        block = _block_by_name(static_job_steps, self.STEP_NAME)
        assert re.search(r"PYTHONDONTWRITEBYTECODE:\s*'1'", block)

    def test_step_runs_pytest_against_the_unsloth_test_directory(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        block = _block_by_name(static_job_steps, self.STEP_NAME)
        assert "python -m pytest tools/fine_tuning/unsloth/tests" in block

    def test_step_uses_the_pinned_uv_python_and_editable_crawler_install(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        block = _block_by_name(static_job_steps, self.STEP_NAME)
        assert "uv run --python 3.12" in block
        assert "--with-editable ./tools/lumen_manifest_crawler" in block

    def test_step_runs_after_the_general_python_tests_and_before_whitespace_check(
        self, static_job_steps: list[tuple[str, str]]
    ) -> None:
        names = [name for name, _ in static_job_steps]
        assert names.index("Python tests") < names.index(self.STEP_NAME)
        assert names.index(self.STEP_NAME) < names.index("Git diff whitespace check")

    def test_referenced_unsloth_test_directory_exists_with_tests(self) -> None:
        test_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "tests"
        assert test_dir.is_dir()
        assert any(test_dir.glob("test_*.py")), "expected at least one test_*.py file"

    def test_referenced_unsloth_test_directory_is_not_in_root_pytest_testpaths(self) -> None:
        # This directory is intentionally excluded from the default pytest.ini
        # testpaths (which only lists tools/lumen_manifest_crawler/tests and
        # tools/pipeline/tests). The dedicated CI step above is what wires it in.
        pytest_ini = (_repo_root() / "pytest.ini").read_text(encoding="utf-8")
        assert "tools/fine_tuning/unsloth/tests" not in pytest_ini