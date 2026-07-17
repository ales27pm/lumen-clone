from pathlib import Path

from lumen_manifest_crawler.cli import _should_generate_full_fleet_artifacts


def test_agent_fine_tuning_requires_complete_fleet_artifacts() -> None:
    assert _should_generate_full_fleet_artifacts(
        generate_system_prompts=False,
        cross_model_train_dir=None,
        generate_agent_fine_tuning=True,
    )


def test_fleet_artifacts_are_skipped_when_no_output_consumes_them() -> None:
    assert not _should_generate_full_fleet_artifacts(
        generate_system_prompts=False,
        cross_model_train_dir=None,
        generate_agent_fine_tuning=False,
    )


def test_explicit_fleet_outputs_still_require_complete_artifacts() -> None:
    assert _should_generate_full_fleet_artifacts(
        generate_system_prompts=True,
        cross_model_train_dir=None,
        generate_agent_fine_tuning=False,
    )
    assert _should_generate_full_fleet_artifacts(
        generate_system_prompts=False,
        cross_model_train_dir=Path("cross-model"),
        generate_agent_fine_tuning=False,
    )
