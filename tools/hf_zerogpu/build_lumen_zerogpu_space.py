#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from lumen_manifest_crawler.dataset.optimization_policy import (
        EXPERIMENT_VARIANT_SCHEMA_VERSION,
        NON_TRAINING_CONFIG_FIELDS,
        effective_variant_training_config,
        invariant_training_config,
    )
except ImportError:  # Direct execution uses the repository checkout.
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.optimization_policy import (
        EXPERIMENT_VARIANT_SCHEMA_VERSION,
        NON_TRAINING_CONFIG_FIELDS,
        effective_variant_training_config,
        invariant_training_config,
    )


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
EXPERIMENT_VARIANTS = (
    "internal_only",
    "internal_plus_public_baseline",
    "internal_plus_public_optimized",
)
CONTAINER_IMAGE_DIGEST_SOURCE = "operator_declared"
RUNTIME_IMAGE_BINDING_STATUS = "manual_validation_required"
SPACE_TEMPLATE = Path(__file__).resolve().parent / "space_template"
LEGACY_DURATION_SECRET_KEYS = (
    "LUMEN_ZERO_GPU_DURATION_SECONDS",
    "LUMEN_ZERO_GPU_MAX_DURATION_SECONDS",
)
OPTIONAL_TRAINING_VARIABLE_KEYS = (
    "LUMEN_ZERO_GPU_MAX_TRAIN_RECORDS",
    "LUMEN_ZERO_GPU_MAX_VAL_RECORDS",
    "LUMEN_ZERO_GPU_MAX_SEQ_LENGTH",
    "LUMEN_ZERO_GPU_NUM_TRAIN_EPOCHS",
)
_DATASET_BUILD_NONCONTROLLED_CONFIG_FIELDS = set(
    NON_TRAINING_CONFIG_FIELDS
)


@dataclass(frozen=True)
class SpaceBuild:
    run_id: str
    run_root: Path
    space_dir: Path
    dataset_dir: Path
    dataset_path_in_repo: str
    defaults_path: Path


@dataclass(frozen=True)
class HubUpload:
    dataset_revision: str
    runtime_source_revision: str


IMMUTABLE_HUB_REVISION = re.compile(r"[0-9a-f]{40}")
MIN_ADMIN_TOKEN_LENGTH = 32


def parse_agents(value: str) -> list[str]:
    agents = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = [agent for agent in agents if agent not in AGENTS]
    if unsupported:
        raise ValueError(f"Unsupported agents: {', '.join(unsupported)}. Expected subset of: {', '.join(AGENTS)}")
    if not agents:
        raise ValueError("At least one agent must be selected")
    return agents


def parse_experiment_variant(value: str) -> str:
    variant = value.strip()
    if variant not in EXPERIMENT_VARIANTS:
        raise ValueError(f"Unsupported experiment variant: {variant or '<empty>'}. Expected one of: {', '.join(EXPERIMENT_VARIANTS)}")
    return variant


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def validate_admin_token(value: str | None) -> str:
    token = value or ""
    if (
        len(token) < MIN_ADMIN_TOKEN_LENGTH
        or any(character.isspace() for character in token)
        or len(set(token)) < 12
    ):
        raise ValueError(
            "LUMEN_ZERO_GPU_ADMIN_TOKEN must be at least 32 non-whitespace "
            "characters with sufficient entropy"
        )
    return token


def _immutable_hub_revision(value: Any, *, label: str) -> str:
    revision = str(value or "").strip().lower()
    if IMMUTABLE_HUB_REVISION.fullmatch(revision) is None:
        raise RuntimeError(f"{label} did not return a full immutable Hub commit SHA")
    return revision


def _commit_sha(result: Any, *, api: Any, repo_id: str, repo_type: str) -> str:
    for attribute in ("oid", "commit_id", "sha"):
        value = getattr(result, attribute, None)
        if value:
            return _immutable_hub_revision(value, label=f"{repo_type} upload")
    info_method = getattr(api, f"{repo_type}_info", None)
    if info_method is None:
        raise RuntimeError(f"Unable to resolve immutable {repo_type} revision after upload")
    info = info_method(repo_id=repo_id, files_metadata=False)
    return _immutable_hub_revision(
        getattr(info, "sha", None),
        label=f"{repo_type} repository",
    )


def _write_defaults(path: Path, defaults: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(defaults, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_training_lineage_module(root: Path) -> Any:
    path = root / "tools/fine_tuning/unsloth/training_lineage.py"
    spec = importlib.util.spec_from_file_location("lumen_training_lineage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training-lineage helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_dataset_source(
    path: Path,
    agents: Sequence[str],
    experiment_variant: str,
) -> None:
    experiment_variant = parse_experiment_variant(experiment_variant)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset source: {path}")
    if not (path / "adapter_runtime_manifest.json").exists():
        raise FileNotFoundError(f"Missing adapter_runtime_manifest.json in {path}")
    for agent in agents:
        agent_dir = path / agent
        for filename in ("train_sft.jsonl", "val_sft.jsonl", "unsloth_config.json"):
            required = agent_dir / filename
            if not required.exists():
                raise FileNotFoundError(f"Missing required fine-tuning file: {required}")
        variant_dir = agent_dir / "experiments" / experiment_variant
        for filename in ("train_sft.jsonl", "val_sft.jsonl", "train_dpo.jsonl", "val_dpo.jsonl", "variant_manifest.json"):
            required = variant_dir / filename
            if not required.is_file():
                raise FileNotFoundError(f"Missing required experiment variant file: {required}")
        config = read_json(agent_dir / "unsloth_config.json")
        manifest_path = variant_dir / "variant_manifest.json"
        manifest = read_json(manifest_path)
        unsigned = dict(manifest)
        declared_manifest_sha256 = unsigned.pop("variantManifestSHA256", None)
        if (
            manifest.get("schemaVersion") != EXPERIMENT_VARIANT_SCHEMA_VERSION
            or manifest.get("agent") != agent
            or manifest.get("variant") != experiment_variant
            or declared_manifest_sha256 != _canonical_sha256(unsigned)
        ):
            raise ValueError(f"Invalid experiment variant manifest: {manifest_path}")
        lanes = {
            name: _read_jsonl(variant_dir / filename)
            for name, filename in (
                ("trainSFT", "train_sft.jsonl"),
                ("validationSFT", "val_sft.jsonl"),
                ("trainDPO", "train_dpo.jsonl"),
                ("validationDPO", "val_dpo.jsonl"),
            )
        }
        datasets = manifest.get("datasets")
        if not isinstance(datasets, dict) or any(
            not isinstance(datasets.get(name), dict)
            or datasets[name].get("count") != len(records)
            or datasets[name].get("sha256") != _canonical_sha256(records)
            for name, records in lanes.items()
        ):
            raise ValueError(f"Variant dataset lineage drifted: {manifest_path}")
        controlled = manifest.get("controlledTrainingConfig")
        if (
            not isinstance(controlled, dict)
            or manifest.get("trainingConfigSHA256")
            != _canonical_sha256(controlled)
        ):
            raise ValueError(f"Variant training config drifted: {manifest_path}")
        if (
            type(manifest.get("seed")) is not int
            or type(controlled.get("seed")) is not int
            or manifest.get("seed") != controlled.get("seed")
        ):
            raise ValueError(f"Variant seed contract drifted: {manifest_path}")
        effective_variant_training_config(
            agent=agent,
            base_config=config,
            controlled_config=controlled,
            noncontrolled_fields=_DATASET_BUILD_NONCONTROLLED_CONFIG_FIELDS,
            sft_train_record_count=len(lanes["trainSFT"]),
            dpo_train_record_count=len(lanes["trainDPO"]),
        )
        invariant = invariant_training_config(
            controlled,
            agent=agent,
            sft_train_record_count=len(lanes["trainSFT"]),
            dpo_train_record_count=len(lanes["trainDPO"]),
        )
        if manifest.get("trainingConfigInvariantSHA256") != _canonical_sha256(
            invariant
        ):
            raise ValueError(
                f"Variant invariant training config drifted: {manifest_path}"
            )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_template_tree(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def write_space_bundle(
    *,
    root: Path,
    run_id: str,
    run_root: Path,
    dataset_source: Path,
    space_repo: str,
    dataset_repo: str,
    adapter_repo: str,
    agents: Sequence[str],
    base_model: str,
    gpu_size: str,
    gpu_duration_seconds: int,
    experiment_variant: str,
    container_image_digest: str,
) -> SpaceBuild:
    experiment_variant = parse_experiment_variant(experiment_variant)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", container_image_digest):
        raise ValueError("container_image_digest must be an operator-declared sha256:<digest>")
    run_root.mkdir(parents=True, exist_ok=True)
    space_dir = run_root / "space"
    dataset_dir = run_root / "dataset_snapshot" / "fine_tuning"
    reset_dir(space_dir)
    reset_dir(dataset_dir)

    shutil.copytree(dataset_source, dataset_dir, dirs_exist_ok=True)
    copy_template_tree(SPACE_TEMPLATE, space_dir)
    training_source = root / "tools/fine_tuning/unsloth"
    training_package = space_dir / "lumen_training"
    training_package.mkdir()
    for filename in (
        "adapter_artifact.py",
        "train_dpo.py",
        "train_sft.py",
        "training_lineage.py",
    ):
        shutil.copy2(training_source / filename, training_package / filename)
    shutil.copy2(
        training_source / "lumen_training/__init__.py",
        training_package / "__init__.py",
    )
    shutil.copytree(
        root / "tools/lumen_manifest_crawler/lumen_manifest_crawler",
        space_dir / "lumen_manifest_crawler",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    lineage = _load_training_lineage_module(root)
    training_code_bundle = lineage.repository_training_code_bundle(root)
    training_code_bundle_sha256 = lineage.verify_training_code_bundle(
        training_code_bundle,
        deployed_root=space_dir,
    )
    training_code_manifests_by_phase = training_code_bundle["phases"]
    training_code_sha256_by_phase = {
        phase: manifest["trainingCodeSHA256"]
        for phase, manifest in sorted(training_code_manifests_by_phase.items())
    }
    training_code_manifest = training_code_manifests_by_phase["sft"]
    training_code_sha256 = training_code_sha256_by_phase["sft"]
    dependency_lock = lineage.build_training_dependency_lock(space_dir / "requirements.txt")
    dependency_lock_sha256 = lineage.verify_training_dependency_lock(
        dependency_lock,
        requirements_path=space_dir / "requirements.txt",
    )

    readme_path = space_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace("{{SPACE_REPO}}", space_repo)
    readme = readme.replace("{{DATASET_REPO}}", dataset_repo)
    readme = readme.replace("{{ADAPTER_REPO}}", adapter_repo)
    readme = readme.replace("{{GPU_SIZE}}", gpu_size)
    readme = readme.replace("{{GPU_DURATION_SECONDS}}", str(gpu_duration_seconds))
    readme_path.write_text(readme, encoding="utf-8")
    space_configuration = lineage.build_space_configuration(readme_path)
    space_configuration_sha256 = lineage.verify_space_configuration(
        space_configuration,
        readme_path=readme_path,
    )

    dataset_path_in_repo = f"runs/{run_id}/fine_tuning"
    defaults = {
        "schema": "lumen.zerogpu.defaults/1.0.0",
        "run_id": run_id,
        "space_repo": space_repo,
        "dataset_repo": dataset_repo,
        # Replaced with the immutable commit returned by the dataset upload
        # before the Space bundle is uploaded or configured.
        "dataset_revision": "pending_dataset_upload",
        "dataset_path_in_repo": dataset_path_in_repo,
        "adapter_repo": adapter_repo,
        "agents": list(agents),
        "base_model_override": base_model,
        "gpu_size": gpu_size,
        "gpu_duration_seconds": gpu_duration_seconds,
        "requested_experiment_variant": experiment_variant,
        "container_image_digest": container_image_digest,
        "container_image_digest_source": CONTAINER_IMAGE_DIGEST_SOURCE,
        "runtime_image_binding_status": RUNTIME_IMAGE_BINDING_STATUS,
        "runtime_image_binding_verified": False,
        "trainingCodeManifest": training_code_manifest,
        "trainingCodeSHA256": training_code_sha256,
        "trainingCodeManifestsByPhase": training_code_manifests_by_phase,
        "trainingCodeSHA256ByPhase": training_code_sha256_by_phase,
        "trainingCodeBundleSHA256": training_code_bundle_sha256,
        "trainingDependencyLock": dependency_lock,
        "trainingDependencyLockSHA256": dependency_lock_sha256,
        "requirementsSHA256": dependency_lock["requirementsSHA256"],
        "spaceConfiguration": space_configuration,
        "spaceConfigurationSHA256": space_configuration_sha256,
        "runtimeSourceKind": "huggingface_space",
        "fresh_run": True,
        "resume_default": False,
        "adapter_first": True,
    }
    defaults_path = space_dir / "lumen_zero_gpu_defaults.json"
    _write_defaults(defaults_path, defaults)

    return SpaceBuild(
        run_id=run_id,
        run_root=run_root,
        space_dir=space_dir,
        dataset_dir=dataset_dir,
        dataset_path_in_repo=dataset_path_in_repo,
        defaults_path=defaults_path,
    )


def import_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Missing pinned huggingface_hub dependency; install the project automation lock") from exc
    return HfApi


def add_space_value(api: Any, *, repo_id: str, key: str, value: str, secret: bool, token: str | None, dry_run: bool) -> None:
    label = "secret" if secret else "variable"
    print(f"Set Space {label}: {key}")
    if dry_run:
        return
    method_name = "add_space_secret" if secret else "add_space_variable"
    method = getattr(api, method_name, None)
    if method is None:
        print(f"warning: installed huggingface_hub has no {method_name}; set {key} manually in Space settings", file=sys.stderr)
        return
    method(repo_id=repo_id, key=key, value=value, token=token)


def delete_space_secret_if_present(
    api: Any,
    *,
    repo_id: str,
    key: str,
    token: str | None,
    dry_run: bool,
) -> None:
    print(f"Delete legacy Space secret if present: {key}")
    if dry_run:
        return
    method = getattr(api, "delete_space_secret", None)
    if method is None:
        raise RuntimeError(f"installed huggingface_hub has no delete_space_secret; remove {key} manually in Space settings")
    try:
        method(repo_id=repo_id, key=key, token=token)
    except Exception as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 404:
            print(f"Legacy Space secret already absent: {key}")
            return
        raise


def restart_space_after_configuration(
    api: Any,
    *,
    repo_id: str,
    token: str | None,
    dry_run: bool,
) -> None:
    print("Restart Space after final configuration")
    if dry_run:
        return
    method = getattr(api, "restart_space", None)
    if method is None:
        raise RuntimeError("installed huggingface_hub has no restart_space; restart the Space manually before triggering training")
    method(repo_id=repo_id, token=token)


def request_zerogpu_hardware(api: Any, *, repo_id: str, hardware: str, token: str | None, dry_run: bool) -> None:
    print(f"Request Space hardware: {hardware}")
    if dry_run:
        return
    method = getattr(api, "request_space_hardware", None)
    if method is None:
        print("warning: installed huggingface_hub has no request_space_hardware; select ZeroGPU in Space settings", file=sys.stderr)
        return
    try:
        method(repo_id=repo_id, hardware=hardware, token=token)
    except Exception as exc:
        print(f"warning: could not request hardware '{hardware}': {exc}", file=sys.stderr)
        print("warning: select ZeroGPU manually in the Space settings if the Hub API rejected this hardware id", file=sys.stderr)


def wait_for_space_revision(api: Any, *, repo_id: str, token: str | None, timeout_seconds: int, dry_run: bool) -> None:
    if dry_run:
        return
    info = api.space_info(repo_id, files_metadata=False, token=token)
    target_sha = getattr(info, "sha", None)
    if not target_sha:
        print("warning: could not determine Space target revision before trigger", file=sys.stderr)
        return
    print(f"Wait for Space runtime revision: {target_sha}")
    started = time.monotonic()
    last_status: dict[str, Any] = {}
    while time.monotonic() - started < timeout_seconds:
        runtime = api.get_space_runtime(repo_id)
        raw = getattr(runtime, "raw", {}) or {}
        runtime_sha = raw.get("sha")
        stage = getattr(runtime, "stage", None)
        last_status = {"stage": stage, "runtime_sha": runtime_sha, "target_sha": target_sha}
        print(json.dumps(last_status, sort_keys=True))
        if stage == "RUNNING" and runtime_sha == target_sha:
            return
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for Space runtime revision: {last_status}")


def ensure_repository_visibility(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    private: bool,
    token: str,
    space_sdk: str | None = None,
    confirm_visibility_change: bool = False,
) -> None:
    info_method = getattr(api, f"{repo_type}_info", None)
    if info_method is None:
        raise RuntimeError(
            f"Installed huggingface_hub cannot verify {repo_type} repository visibility"
        )

    existing_private: bool | None = None
    try:
        existing_info = info_method(
            repo_id=repo_id,
            files_metadata=False,
            token=token,
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) != 404:
            raise
    else:
        existing_private = getattr(existing_info, "private", None)
        if not isinstance(existing_private, bool):
            raise RuntimeError(
                f"Unable to determine existing {repo_type} repository visibility "
                f"for {repo_id}"
            )

    if existing_private is not None:
        if existing_private is not private:
            if not confirm_visibility_change:
                current = "private" if existing_private else "public"
                requested = "private" if private else "public"
                raise RuntimeError(
                    f"Existing {repo_type} repository {repo_id} is {current}, but "
                    f"the requested visibility is {requested}; explicitly confirm "
                    "this repository visibility migration before deployment"
                )
            api.update_repo_settings(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                token=token,
            )
    else:
        create_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "private": private,
            "exist_ok": False,
            "token": token,
        }
        if space_sdk is not None:
            create_kwargs["space_sdk"] = space_sdk
        try:
            api.create_repo(**create_kwargs)
        except TypeError:
            if space_sdk is None:
                raise
            create_kwargs.pop("space_sdk")
            api.create_repo(**create_kwargs)

    info = info_method(
        repo_id=repo_id,
        files_metadata=False,
        token=token,
    )
    actual_private = getattr(info, "private", None)
    if not isinstance(actual_private, bool) or actual_private is not private:
        expected = "private" if private else "public"
        actual = (
            "private"
            if actual_private is True
            else "public"
            if actual_private is False
            else "unknown"
        )
        raise RuntimeError(
            f"Repository visibility postcondition failed for {repo_type} "
            f"repository {repo_id}: expected {expected}, observed {actual}"
        )


def upload_to_hub(
    *,
    build: SpaceBuild,
    space_repo: str,
    dataset_repo: str,
    adapter_repo: str,
    private_space: bool,
    private_dataset: bool,
    private_adapters: bool,
    confirm_space_visibility_change: bool = False,
    confirm_dataset_visibility_change: bool = False,
    confirm_adapter_visibility_change: bool = False,
    zero_gpu_hardware: str,
    token: str | None,
    admin_token: str,
    dry_run: bool,
) -> HubUpload:
    admin_token = validate_admin_token(admin_token)
    if not dry_run and not token:
        raise ValueError(
            "A fine-grained LUMEN_ZERO_GPU_HUB_TOKEN scoped to the required repositories is required"
        )
    HfApi = import_hf_api()
    api = HfApi(token=token)

    print(
        f"Create/update dataset repo: {dataset_repo} "
        f"({'private' if private_dataset else 'public'})"
    )
    print(
        f"Create/update adapter repo: {adapter_repo} "
        f"({'private' if private_adapters else 'public'})"
    )
    print(
        f"Create/update Space repo: {space_repo} "
        f"({'private' if private_space else 'public'})"
    )
    if not dry_run:
        ensure_repository_visibility(
            api,
            repo_id=dataset_repo,
            repo_type="dataset",
            private=private_dataset,
            token=token,
            confirm_visibility_change=confirm_dataset_visibility_change,
        )
        ensure_repository_visibility(
            api,
            repo_id=adapter_repo,
            repo_type="model",
            private=private_adapters,
            token=token,
            confirm_visibility_change=confirm_adapter_visibility_change,
        )
        ensure_repository_visibility(
            api,
            repo_id=space_repo,
            repo_type="space",
            private=private_space,
            token=token,
            space_sdk="gradio",
            confirm_visibility_change=confirm_space_visibility_change,
        )

    print(f"Upload dataset snapshot: {build.dataset_dir} -> {dataset_repo}/{build.dataset_path_in_repo}")
    dataset_revision = "dry_run_not_uploaded"
    if not dry_run:
        dataset_upload = api.upload_folder(
            folder_path=str(build.dataset_dir),
            repo_id=dataset_repo,
            repo_type="dataset",
            path_in_repo=build.dataset_path_in_repo,
            commit_message=f"Upload Lumen fine-tuning dataset snapshot {build.run_id}",
            token=token,
        )
        dataset_revision = _commit_sha(
            dataset_upload,
            api=api,
            repo_id=dataset_repo,
            repo_type="dataset",
        )
    defaults = read_json(build.defaults_path)
    defaults["dataset_revision"] = dataset_revision
    _write_defaults(build.defaults_path, defaults)

    print(f"Upload Space bundle: {build.space_dir} -> {space_repo}")
    runtime_source_revision = "dry_run_not_uploaded"
    if not dry_run:
        space_upload = api.upload_folder(
            folder_path=str(build.space_dir),
            repo_id=space_repo,
            repo_type="space",
            commit_message=f"Update Lumen ZeroGPU trainer {build.run_id}",
            token=token,
        )
        runtime_source_revision = _commit_sha(
            space_upload,
            api=api,
            repo_id=space_repo,
            repo_type="space",
        )

    if token:
        add_space_value(
            api,
            repo_id=space_repo,
            key="LUMEN_ZERO_GPU_HUB_TOKEN",
            value=token,
            secret=True,
            token=token,
            dry_run=dry_run,
        )
    add_space_value(
        api,
        repo_id=space_repo,
        key="LUMEN_ZERO_GPU_ADMIN_TOKEN",
        value=admin_token,
        secret=True,
        token=token,
        dry_run=dry_run,
    )

    gpu_duration_seconds = str(defaults.get("gpu_duration_seconds", 1200))
    for key in LEGACY_DURATION_SECRET_KEYS:
        delete_space_secret_if_present(api, repo_id=space_repo, key=key, token=token, dry_run=dry_run)

    variables = {
        "LUMEN_ZERO_GPU_DATASET_REPO": dataset_repo,
        "LUMEN_ZERO_GPU_DATASET_REVISION": dataset_revision,
        "LUMEN_ZERO_GPU_DATASET_PATH": build.dataset_path_in_repo,
        "LUMEN_ZERO_GPU_ADAPTER_REPO": adapter_repo,
        "LUMEN_ZERO_GPU_PRIVATE_ADAPTERS": "1" if private_adapters else "0",
        "LUMEN_ZERO_GPU_RUN_ID": build.run_id,
        "LUMEN_ZERO_GPU_RUNTIME_SOURCE_KIND": "huggingface_space",
        "LUMEN_ZERO_GPU_EXPECTED_RUNTIME_SOURCE_REVISION": runtime_source_revision,
        "LUMEN_ZERO_GPU_RUNTIME_SOURCE_REVISION": runtime_source_revision,
        "LUMEN_ZERO_GPU_SIZE": str(defaults.get("gpu_size", "large")),
        "LUMEN_ZERO_GPU_DURATION_SECONDS": gpu_duration_seconds,
        "LUMEN_ZERO_GPU_MAX_DURATION_SECONDS": gpu_duration_seconds,
    }
    # Always overwrite optional knobs. A neutral zero makes the Space retain
    # each generated per-agent config value and clears stale caps from an older run.
    for key in OPTIONAL_TRAINING_VARIABLE_KEYS:
        variables[key] = os.environ.get(key, "0")
    for key, value in variables.items():
        add_space_value(api, repo_id=space_repo, key=key, value=value, secret=False, token=token, dry_run=dry_run)

    request_zerogpu_hardware(api, repo_id=space_repo, hardware=zero_gpu_hardware, token=token, dry_run=dry_run)
    restart_space_after_configuration(api, repo_id=space_repo, token=token, dry_run=dry_run)
    return HubUpload(
        dataset_revision=dataset_revision,
        runtime_source_revision=runtime_source_revision,
    )


def trigger_space_training(
    *,
    space_repo: str,
    run_id: str,
    agents: Sequence[str],
    base_model: str,
    seed: int,
    gpu_size: str,
    token: str | None,
    admin_token: str,
    timeout_seconds: int,
    dry_run: bool,
    experiment_variant: str,
    destructive_reset: bool,
    resume: bool,
) -> None:
    experiment_variant = parse_experiment_variant(experiment_variant)
    print(f"Trigger Space training via Gradio API: {space_repo}")
    if dry_run:
        return
    started = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - started < timeout_seconds:
        try:
            _trigger_space_training_via_gradio_api(
                space_repo=space_repo,
                run_id=run_id,
                agents=agents,
                base_model=base_model,
                seed=seed,
                gpu_size=gpu_size,
                experiment_variant=experiment_variant,
                token=token,
                admin_token=admin_token,
                destructive_reset=destructive_reset,
                resume=resume,
                deadline=started + timeout_seconds,
            )
            return
        except Exception as exc:
            last_error = exc
            print(f"Space not ready yet or trigger failed transiently: {exc}", file=sys.stderr)
            if _is_terminal_space_trigger_error(exc):
                raise
            time.sleep(20)
    raise RuntimeError(f"Timed out waiting for Space trigger readiness after {timeout_seconds}s: {last_error}")


def _is_terminal_space_trigger_error(exc: Exception) -> bool:
    message = str(exc).lower()
    terminal_fragments = (
        "zerogpu quota exceeded",
        "quota exceeded",
        "zerogpu illegal duration",
        "gpu task aborted",
        "zerogpu worker error",
        "requested gpu duration",
        "incomplete chunked read",
        "peer closed connection without sending complete message body",
        '"ok": false',
        "'ok': false",
    )
    return any(fragment in message for fragment in terminal_fragments)


def _trigger_space_training_via_gradio_api(
    *,
    space_repo: str,
    run_id: str,
    agents: Sequence[str],
    base_model: str,
    seed: int,
    gpu_size: str,
    experiment_variant: str,
    token: str | None,
    admin_token: str,
    destructive_reset: bool = False,
    resume: bool = False,
    deadline: float | None = None,
) -> None:
    import httpx

    space_name = space_repo.replace("/", "-")
    base_url = f"https://{space_name}.hf.space"
    headers = {"X-Lumen-Admin-Token": validate_admin_token(admin_token)}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "data": [
            run_id,
            ",".join(agents),
            base_model,
            seed,
            True,
            bool(resume),
            True,
            True,
            gpu_size,
            experiment_variant,
            True,
            bool(destructive_reset),
        ]
    }
    with httpx.Client(timeout=None) as client:
        response = client.post(f"{base_url}/gradio_api/call/train_lumen_adapters", headers=headers, json=payload)
        response.raise_for_status()
        event_id = str(response.json()["event_id"])
        print(f"Triggered Space training event: {event_id}")
        event_name = ""
        data_lines: list[str] = []
        with client.stream("GET", f"{base_url}/gradio_api/call/train_lumen_adapters/{event_id}", headers=headers) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Timed out while waiting for Space training event stream to complete")
                if not line:
                    continue
                print(line, flush=True)
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    data = line.split(":", 1)[1].strip()
                    data_lines.append(data)
                    if event_name == "error":
                        raise RuntimeError(f"Space training failed: {data}")
                    if event_name in {"complete", "done"}:
                        _raise_if_space_payload_failed(data)
                        return


def _raise_if_space_payload_failed(data: str) -> None:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and optionally trigger a Hugging Face ZeroGPU Space for Lumen adapter training.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier used in HF dataset/adaptor paths.")
    parser.add_argument("--run-root", type=Path, required=True, help="Local ignored workspace for generated Space files.")
    parser.add_argument("--dataset-source", type=Path, required=True, help="Local generated fine_tuning directory.")
    parser.add_argument("--space-repo", required=True, help="HF Space repo id, e.g. user/lumen-zerogpu-adapter-trainer.")
    parser.add_argument("--dataset-repo", required=True, help="HF dataset repo id used for the run snapshot.")
    parser.add_argument("--adapter-repo", required=True, help="HF model repo id used for trained adapters.")
    parser.add_argument("--agents", default=",".join(AGENTS), help="Comma-separated agent slots to train.")
    parser.add_argument("--base-model", default="", help="Optional base model override. Empty keeps generated per-agent config values.")
    parser.add_argument("--gpu-size", choices=("large", "xlarge"), default="large", help="ZeroGPU decorator size.")
    parser.add_argument("--gpu-duration-seconds", type=int, default=1200, help="ZeroGPU function duration budget.")
    parser.add_argument("--zero-gpu-hardware", default=os.environ.get("LUMEN_ZERO_GPU_HARDWARE", "zero-a10g"), help="HF hardware id requested for the Space.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--experiment-variant",
        choices=EXPERIMENT_VARIANTS,
        required=True,
        help="Controlled dataset variant to train.",
    )
    parser.add_argument(
        "--container-image-digest",
        required=True,
        help=(
            "Operator-declared sha256:<digest> for the intended training image. "
            "Gradio ZeroGPU does not expose trusted runtime-image binding, so promotion "
            "still requires separate manual verification."
        ),
    )
    parser.add_argument("--trigger", action="store_true", help="Trigger Space training after upload.")
    parser.add_argument("--trigger-timeout-seconds", type=int, default=900, help="Time to wait for Space readiness before triggering.")
    space_visibility = parser.add_mutually_exclusive_group()
    space_visibility.add_argument(
        "--public-space",
        action="store_true",
        help="Explicitly create/update the Space as public. Application-level authorization remains required.",
    )
    space_visibility.add_argument(
        "--private-space",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    dataset_visibility = parser.add_mutually_exclusive_group()
    dataset_visibility.add_argument(
        "--public-dataset",
        action="store_true",
        help="Explicitly create/update the dataset repository as public.",
    )
    dataset_visibility.add_argument(
        "--private-dataset",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    adapter_visibility = parser.add_mutually_exclusive_group()
    adapter_visibility.add_argument(
        "--public-adapters",
        action="store_true",
        help="Explicitly create/update the adapter model repository as public.",
    )
    adapter_visibility.add_argument(
        "--private-adapters",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--confirm-space-visibility-change",
        action="store_true",
        help="Confirm changing the visibility of an existing Space repository.",
    )
    parser.add_argument(
        "--confirm-dataset-visibility-change",
        action="store_true",
        help="Confirm changing the visibility of an existing dataset repository.",
    )
    parser.add_argument(
        "--confirm-adapter-visibility-change",
        action="store_true",
        help="Confirm changing the visibility of an existing adapter/model repository.",
    )
    parser.add_argument(
        "--destructive-reset",
        action="store_true",
        help="When triggering a fresh run, explicitly replace an existing run workspace.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after the Space validates the existing immutable run and checkpoint lineage.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare and print actions without calling Hugging Face.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume and args.destructive_reset:
        raise ValueError("--resume and --destructive-reset are mutually exclusive")
    agents = parse_agents(args.agents)
    run_id = args.run_id
    variant_marker = f"-{args.experiment_variant}"
    if not run_id.endswith(variant_marker):
        run_id += variant_marker

    token = os.environ.get("LUMEN_ZERO_GPU_HUB_TOKEN")
    if not args.dry_run and not token:
        raise ValueError(
            "Set a fine-grained LUMEN_ZERO_GPU_HUB_TOKEN scoped to the required repositories"
        )
    admin_token = validate_admin_token(os.environ.get("LUMEN_ZERO_GPU_ADMIN_TOKEN"))

    if args.resume:
        if not args.trigger:
            raise ValueError("--resume requires --trigger and never rebuilds or uploads the Space bundle")
        print("Resume mode: preserve the deployed Space, dataset revision, and local run workspace")
        HfApi = import_hf_api()
        wait_for_space_revision(
            HfApi(token=token),
            repo_id=args.space_repo,
            token=token,
            timeout_seconds=args.trigger_timeout_seconds,
            dry_run=args.dry_run,
        )
        trigger_space_training(
            space_repo=args.space_repo,
            run_id=run_id,
            agents=agents,
            base_model=args.base_model,
            seed=args.seed,
            gpu_size=args.gpu_size,
            token=token,
            admin_token=admin_token,
            timeout_seconds=args.trigger_timeout_seconds,
            dry_run=args.dry_run,
            experiment_variant=args.experiment_variant,
            destructive_reset=False,
            resume=True,
        )
        return 0

    root = args.root.resolve()
    run_root = args.run_root.resolve()
    dataset_source = args.dataset_source.resolve()
    require_dataset_source(dataset_source, agents, args.experiment_variant)
    read_json(dataset_source / "adapter_runtime_manifest.json")

    build = write_space_bundle(
        root=root,
        run_id=run_id,
        run_root=run_root,
        dataset_source=dataset_source,
        space_repo=args.space_repo,
        dataset_repo=args.dataset_repo,
        adapter_repo=args.adapter_repo,
        agents=agents,
        base_model=args.base_model,
        gpu_size=args.gpu_size,
        gpu_duration_seconds=args.gpu_duration_seconds,
        experiment_variant=args.experiment_variant,
        container_image_digest=args.container_image_digest,
    )
    print(f"Wrote Space bundle: {build.space_dir}")
    print(f"Wrote dataset snapshot: {build.dataset_dir}")
    print(f"Wrote defaults: {build.defaults_path}")

    upload = upload_to_hub(
        build=build,
        space_repo=args.space_repo,
        dataset_repo=args.dataset_repo,
        adapter_repo=args.adapter_repo,
        private_space=not args.public_space,
        private_dataset=not args.public_dataset,
        private_adapters=not args.public_adapters,
        confirm_space_visibility_change=args.confirm_space_visibility_change,
        confirm_dataset_visibility_change=args.confirm_dataset_visibility_change,
        confirm_adapter_visibility_change=args.confirm_adapter_visibility_change,
        zero_gpu_hardware=args.zero_gpu_hardware,
        token=token,
        admin_token=admin_token,
        dry_run=args.dry_run,
    )
    print(f"Pinned dataset revision: {upload.dataset_revision}")
    print(f"Uploaded Space revision: {upload.runtime_source_revision}")

    if args.trigger:
        HfApi = import_hf_api()
        wait_for_space_revision(
            HfApi(token=token),
            repo_id=args.space_repo,
            token=token,
            timeout_seconds=args.trigger_timeout_seconds,
            dry_run=args.dry_run,
        )
        trigger_space_training(
            space_repo=args.space_repo,
            run_id=run_id,
            agents=agents,
            base_model=args.base_model,
            seed=args.seed,
            gpu_size=args.gpu_size,
            token=token,
            admin_token=admin_token,
            timeout_seconds=args.trigger_timeout_seconds,
            dry_run=args.dry_run,
            experiment_variant=args.experiment_variant,
            destructive_reset=args.destructive_reset,
            resume=args.resume,
        )
    else:
        print("Trigger skipped. Set LUMEN_ZERO_GPU_TRIGGER=1 or pass --trigger to start training through the Space API.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
