from __future__ import annotations

from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import train_sft


def _config(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "executor-finetune"
    return {
        "agent": "executor",
        "output_dir": str(root / "work"),
        "adapter_output_dir": str(root / "adapter"),
    }


def test_sft_adapter_path_requires_agent_and_finetune_tokens(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["adapter_output_dir"] = str(tmp_path / "dataset")

    with pytest.raises(ValueError, match="adapter_output_dir must include slot token"):
        train_sft.validate_artifact_path_config(config)


@pytest.mark.parametrize("nested_path", ["work", "adapter", "equal"])
def test_sft_work_and_adapter_paths_reject_both_containment_directions(
    tmp_path: Path,
    nested_path: str,
) -> None:
    config = _config(tmp_path)
    root = tmp_path / "executor-finetune"
    if nested_path == "work":
        config["adapter_output_dir"] = str(root)
    elif nested_path == "adapter":
        config["output_dir"] = str(root)
    else:
        config["adapter_output_dir"] = config["output_dir"]

    with pytest.raises(ValueError, match="must be separate"):
        train_sft.validate_artifact_path_config(config)


def test_sft_work_and_adapter_sibling_paths_are_valid(tmp_path: Path) -> None:
    config = _config(tmp_path)

    output_dir, adapter_output_dir = train_sft.validate_sft_artifact_paths(config)

    assert output_dir == Path(config["output_dir"]).resolve()
    assert adapter_output_dir == Path(config["adapter_output_dir"]).resolve()
