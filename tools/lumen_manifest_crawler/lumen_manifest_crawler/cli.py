from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.fine_tuning import compile_agent_fine_tuning_datasets
from lumen_manifest_crawler.dataset.runtime_ingest import load_runtime_audit_reports
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts, generate_manifest_markdown
from lumen_manifest_crawler.improvement_loop import AgentImprovementLoopConfig, run_agent_improvement_loop
from lumen_manifest_crawler.output.writer import write_outputs
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets, validate_manifest

logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)
generate_app = typer.Typer(help="Generate AgentBehaviorManifest.json and grounded datasets.", invoke_without_command=True)
app.add_typer(generate_app, name="generate")
console = Console()


def _split_command(value: Optional[str]) -> tuple[str, ...]:
    if not value or not value.strip():
        return ()
    return tuple(shlex.split(value))


@generate_app.callback()
def generate(
    root: Path = typer.Option(Path("."), "--root", help="Repository root to scan."),
    output: Path = typer.Option(Path("generated/agent_manifest"), "--output", help="Output directory."),
    pretty: bool = typer.Option(False, "--pretty", help="Also write pretty formatted manifest."),
    runtime_audit: Annotated[Optional[list[Path]], typer.Option("--runtime-audit", help="RuntimeManifestAuditor JSON report file or directory. Can be passed multiple times.")] = None,
    deterministic: bool = typer.Option(True, "--deterministic/--non-deterministic", help="Use deterministic timestamps and splits for CI-stable generated files."),
    generate_system_prompts: bool = typer.Option(False, "--generate-system-prompts", help="Generate fleet_system_prompts.json, AgentBehaviorManifest.md, and cross-model training artifacts."),
    export_md: bool = typer.Option(False, "--export-md", help="Generate only AgentBehaviorManifest.md, unless full fleet artifact generation is also requested."),
    cross_model_train_dir: Optional[Path] = typer.Option(None, "--cross-model-train-dir", help="Directory for cross_model_training.jsonl. Defaults to <output>/cross_model_training."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate into a temporary directory and print a machine-readable diff without changing output."),
    diff: bool = typer.Option(False, "--diff", help="Alias for --dry-run."),
    incremental: bool = typer.Option(False, "--incremental", help="Skip generation when the current manifest fingerprint matches the previous output and no runtime audit is provided."),
    strict: bool = typer.Option(False, "--strict", help="Promote selected validation warnings to hard failures."),
    fail_on_change: bool = typer.Option(False, "--fail-on-change", help="Exit non-zero if generated outputs leave tracked or untracked git changes."),
    fail_on_validation: bool = typer.Option(True, "--fail-on-validation/--no-fail-on-validation", help="Exit non-zero on hard validation failures."),
    generate_agent_fine_tuning: bool = typer.Option(False, "--generate-agent-fine-tuning", help="Generate per-agent fine-tuning datasets."),
    fine_tuning_output: Optional[Path] = typer.Option(None, "--fine-tuning-output", help="Output directory for per-agent fine-tuning datasets."),
    unsloth_output: Optional[Path] = typer.Option(None, "--unsloth-output", help="Alias for --fine-tuning-output."),
    agent_filter: Optional[str] = typer.Option(None, "--agent-filter", help="Comma-separated agent names to output."),
) -> None:
    """Generate AgentBehaviorManifest.json and state-of-the-art grounded datasets."""
    root = root.resolve()
    output = output.resolve()
    dry_run = dry_run or diff

    manifest = generate_manifest(root)
    manifest_fingerprint = _manifest_fingerprint(manifest)

    if incremental and not runtime_audit and not dry_run and _is_incremental_hit(output, manifest_fingerprint):
        console.print(f"[green]Incremental generation skipped; manifest fingerprint unchanged for {output}[/green]")
        return

    runtime_audit_reports = load_runtime_audit_reports(runtime_audit)
    datasets = generate_all_datasets(manifest, runtime_audit_paths=runtime_audit, deterministic=deterministic)
    report = validate_manifest(manifest, datasets, strict=strict)
    should_generate_full_fleet_artifacts = generate_system_prompts or cross_model_train_dir is not None
    fleet_artifacts = generate_fleet_artifacts(manifest) if should_generate_full_fleet_artifacts else None
    manifest_markdown = None if fleet_artifacts else (generate_manifest_markdown(manifest) if export_md else None)

    fine_tuning_datasets = None
    if generate_agent_fine_tuning:
        fine_tuning_datasets = compile_agent_fine_tuning_datasets(
            manifest,
            datasets,
            fleet_artifacts=fleet_artifacts,
            runtime_audit_reports=runtime_audit_reports,
        )
        if agent_filter:
            allowed = {agent.strip() for agent in agent_filter.split(",") if agent.strip()}
            fine_tuning_datasets = {key: value for key, value in fine_tuning_datasets.items() if key in allowed}
        ft_failures = validate_agent_fine_tuning_datasets(
            manifest,
            fine_tuning_datasets,
            runtime_audit_reports=runtime_audit_reports,
        )
        for failure in ft_failures:
            report.failures.append(failure)

    target_output = output
    target_cross_dir = cross_model_train_dir.resolve() if cross_model_train_dir else None
    if dry_run:
        temp_root = Path(tempfile.mkdtemp(prefix="lumen-manifest-dry-run-"))
        target_output = temp_root / "agent_manifest"
        target_cross_dir = (temp_root / "cross_model_training") if cross_model_train_dir else None

    write_outputs(
        target_output,
        manifest,
        report,
        datasets,
        pretty=pretty,
        fleet_artifacts=fleet_artifacts,
        manifest_markdown=manifest_markdown,
        cross_model_train_dir=target_cross_dir,
        incremental_fingerprint=manifest_fingerprint,
        fine_tuning_datasets=fine_tuning_datasets,
        fine_tuning_output_dir=(unsloth_output or fine_tuning_output),
    )

    if dry_run:
        diff_report = _diff_directories(output, target_output)
        if cross_model_train_dir and target_cross_dir:
            diff_report["cross_model_training"] = _diff_directories(cross_model_train_dir.resolve(), target_cross_dir)
        console.print(json.dumps(diff_report, ensure_ascii=False, indent=2, sort_keys=True))
        if fail_on_change and diff_report.get("changed"):
            raise typer.Exit(code=1)
        return

    compiled_count = sum(len(records) for name, records in datasets.items() if name != "dataset_manifest")
    families_count = sum(1 for name in datasets if name != "dataset_manifest")
    console.print(f"[bold]Tools:[/bold] {len(manifest.tools)}")
    console.print(f"[bold]Intents:[/bold] {len(manifest.intents)}")
    console.print(f"[bold]Model slots:[/bold] {len(manifest.fleet.slots)}")
    console.print(f"[bold]Datasets:[/bold] {compiled_count} records across {families_count} families")
    if fleet_artifacts:
        console.print(f"[bold]Fleet self-knowledge:[/bold] {len(fleet_artifacts.system_prompts)} prompts and {len(fleet_artifacts.cross_model_training)} cross-model records")
    elif manifest_markdown:
        console.print("[bold]Fleet markdown:[/bold] wrote AgentBehaviorManifest.md")
    if runtime_audit:
        console.print(f"[bold]Runtime audit inputs:[/bold] {len(runtime_audit)} path(s)")
    if strict:
        console.print("[bold]Strict validation:[/bold] enabled")
    if report.failures:
        console.print(f"[red]Validation failed with {len(report.failures)} hard failure(s).[/red]")
        for failure in report.failures:
            console.print(f"  [red]- {failure.code}:[/red] {failure.message}")
        if fail_on_validation:
            raise typer.Exit(code=1)
    if report.warnings:
        console.print(f"[yellow]Warnings:[/yellow] {len(report.warnings)}")
        for warning in report.warnings[:20]:
            console.print(f"  [yellow]- {warning.code}:[/yellow] {warning.message}")
    if fail_on_change and _has_git_changes(root):
        console.print("[red]Generated outputs differ from the git working tree, or git status could not be verified. Commit regenerated artifacts or fix the git status check.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Wrote manifest and dataset outputs to {output}[/green]")


@app.command("improve-loop")
def improve_loop(
    root: Path = typer.Option(Path("."), "--root", help="Repository root to scan."),
    output: Path = typer.Option(Path("generated/agent_manifest"), "--output", help="Manifest and dataset output directory."),
    loop_output: Path = typer.Option(Path("generated/agent_improvement_loop"), "--loop-output", help="Loop state, gap report, and next-action prompt output directory."),
    runtime_audit: Annotated[Optional[list[Path]], typer.Option("--runtime-audit", help="In-app dataset package JSON file or directory. Can be passed multiple times.")] = None,
    deterministic: bool = typer.Option(True, "--deterministic/--non-deterministic", help="Use deterministic timestamps and splits."),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Write pretty manifest output."),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Promote selected validation warnings to gaps."),
    generate_system_prompts: bool = typer.Option(True, "--generate-system-prompts/--no-generate-system-prompts", help="Generate fleet prompts and cross-model artifacts."),
    generate_agent_fine_tuning: bool = typer.Option(True, "--generate-agent-fine-tuning/--no-generate-agent-fine-tuning", help="Generate per-agent fine-tuning datasets."),
    fine_tuning_output: Optional[Path] = typer.Option(None, "--fine-tuning-output", help="Output directory for per-agent fine-tuning datasets."),
    cross_model_train_dir: Optional[Path] = typer.Option(None, "--cross-model-train-dir", help="Directory for cross-model training artifacts."),
    build_command: Optional[str] = typer.Option(None, "--build-command", help="Optional build/TestFlight archive command to run after generation. Space-separated."),
    test_command: Optional[str] = typer.Option(None, "--test-command", help="Optional local validation command to run before generation. Space-separated."),
    train_command: Optional[str] = typer.Option(None, "--train-command", help="Optional training command to run after generation. Space-separated."),
    dry_run_commands: bool = typer.Option(False, "--dry-run-commands", help="Record build/test/train commands without executing them."),
    app_run_mode: str = typer.Option("testflight", "--app-run-mode", help="Live app runtime mode. Default: testflight."),
    testflight_build_label: Optional[str] = typer.Option(None, "--testflight-build-label", help="Human-readable TestFlight build/version label for the runbook."),
    require_testflight_runtime_audit: bool = typer.Option(False, "--require-testflight-runtime-audit", help="Treat missing TestFlight in-app audit JSON as an error gap."),
    testflight_scenario_limit: int = typer.Option(120, "--testflight-scenario-limit", min=1, help="Maximum scenarios to write for TestFlight replay."),
    fail_on_validation: bool = typer.Option(False, "--fail-on-validation/--no-fail-on-validation", help="Exit non-zero if the loop finds critical/error gaps."),
) -> None:
    """Run one closed improvement-loop cycle and emit TestFlight runtime handoff artifacts."""
    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=root,
            output=output,
            loop_output=loop_output,
            runtime_audit_paths=tuple(runtime_audit or []),
            deterministic=deterministic,
            pretty=pretty,
            strict=strict,
            generate_system_prompts=generate_system_prompts,
            generate_agent_fine_tuning=generate_agent_fine_tuning,
            fine_tuning_output=fine_tuning_output,
            cross_model_train_dir=cross_model_train_dir,
            build_command=_split_command(build_command),
            test_command=_split_command(test_command),
            train_command=_split_command(train_command),
            fail_on_validation=False,
            dry_run_commands=dry_run_commands,
            app_run_mode=app_run_mode,
            testflight_build_label=testflight_build_label,
            require_testflight_runtime_audit=require_testflight_runtime_audit,
            testflight_scenario_limit=testflight_scenario_limit,
        )
    )
    console.print(f"[bold]Loop passed:[/bold] {result.passed}")
    console.print(f"[bold]Gaps:[/bold] {len(result.gaps)}")
    console.print(f"[bold]TestFlight scenarios:[/bold] {len(result.testflight_scenarios)}")
    console.print(f"[bold]Next-action prompts:[/bold] {len(result.next_prompts)}")
    console.print(f"[green]Wrote loop outputs to {loop_output.resolve()}[/green]")
    if fail_on_validation and not result.passed:
        raise typer.Exit(code=1)


def _manifest_fingerprint(manifest: Any) -> str:
    payload = json.dumps(_canonicalize(manifest.output_dict()), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        canonical_items = [_canonicalize(item) for item in value]
        return canonical_items
    return value


def _is_incremental_hit(output: Path, manifest_fingerprint: str) -> bool:
    existing_hash_path = output / "AgentBehaviorManifest.incremental.sha256"
    dataset_manifest_path = output / "dataset_manifest.json"
    if not existing_hash_path.exists() or not dataset_manifest_path.exists():
        return False
    existing = existing_hash_path.read_text(encoding="utf-8").strip()
    return existing == manifest_fingerprint


def _diff_directories(existing_dir: Path, generated_dir: Path) -> dict[str, Any]:
    existing_files = _file_hashes(existing_dir)
    generated_files = _file_hashes(generated_dir)
    existing_paths = set(existing_files)
    generated_paths = set(generated_files)
    added = sorted(generated_paths - existing_paths)
    removed = sorted(existing_paths - generated_paths)
    modified = sorted(path for path in existing_paths.intersection(generated_paths) if existing_files[path] != generated_files[path])
    return {
        "existingDir": str(existing_dir),
        "generatedDir": str(generated_dir),
        "changed": bool(added or removed or modified),
        "added": added,
        "removed": removed,
        "modified": modified,
    }


def _file_hashes(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        rel = path.relative_to(directory).as_posix()
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _has_git_changes(root: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(completed.stdout.strip())
    except Exception as e:
        logger.exception("Failed to verify git working tree changes with `git status --porcelain`: %s", e)
        console.print(f"[red]Failed to verify git working tree changes: {e}[/red]")
        return True


if __name__ == "__main__":
    app()
