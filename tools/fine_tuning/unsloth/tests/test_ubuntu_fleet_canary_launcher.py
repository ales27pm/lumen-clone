from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER = REPO_ROOT / "scripts/ubuntu_run_fleet_canary.sh"
OLLAMA_ID = "b" * 64


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _harness(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    attestor_dir = repository / "tools/fine_tuning/unsloth"
    fake_bin = tmp_path / "bin"
    workspace = tmp_path / "workspace"
    docker_root = tmp_path / "docker-root"
    state = tmp_path / "state"
    for path in (scripts, attestor_dir, fake_bin, workspace, docker_root, state):
        path.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o700)
    state.chmod(0o700)

    host_state = state / "host-state"
    wrapper_source = WRAPPER.read_text(encoding="utf-8")
    test_replacements = {
        'HOST_STATE_ROOT="/var/tmp/lumen-fleet-canary-state"': (
            f'HOST_STATE_ROOT="{host_state}"'
        ),
        "MIN_DOCKER_FREE_BYTES=42949672960": "MIN_DOCKER_FREE_BYTES=0",
        "MIN_WORKSPACE_FREE_BYTES=53687091200": "MIN_WORKSPACE_FREE_BYTES=0",
    }
    for production_value, test_value in test_replacements.items():
        assert wrapper_source.count(production_value) == 1
        wrapper_source = wrapper_source.replace(production_value, test_value)
    _write_executable(scripts / WRAPPER.name, wrapper_source)
    _write_executable(
        scripts / "ubuntu_train_lumen_full_pipeline.sh",
        f"""#!/usr/bin/env bash
set -eu
[[ -e /dev/fd/8 ]]
python3 -I - \
  "{host_state}/ollama-restore-required" \
  "$FAKE_OLLAMA_ID" \
  "$LUMEN_UBUNTU_EXPECTED_SOURCE_COMMIT" \
  "$LUMEN_UBUNTU_RUN_ID" \
  "$LUMEN_UBUNTU_OUTPUT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["schemaVersion"] == "lumen.ollama-restore-required/1.0.0"
assert value["containerName"] == "mongars-ollama-1"
assert value["containerID"] == sys.argv[2]
assert value["restartPolicy"] == "unless-stopped"
assert value["sourceCommit"] == sys.argv[3]
assert value["runID"] == sys.argv[4]
assert value["expectedRunRoot"] == (
    f"{{sys.argv[5]}}/{{sys.argv[4]}}-internal_plus_public_optimized"
)
assert re.fullmatch(
    r"[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z",
    value["createdAtUTC"],
)
PY
: >"$FAKE_STATE_DIR/pipeline-lock-inherited"
printf 'pipeline\n' >>"$FAKE_STATE_DIR/events"
printf '%s\n' "$@" >"$FAKE_STATE_DIR/pipeline-args"
env | LC_ALL=C sort | grep '^LUMEN_UBUNTU_' >"$FAKE_STATE_DIR/pipeline-env"
: >"$FAKE_STATE_DIR/pipeline-called"
exit "${{FAKE_PIPELINE_EXIT:-0}}"
""",
    )
    (attestor_dir / "ubuntu_source_integrity.py").write_text(
        """from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--root") + 1])
commit = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"],
    text=True,
).strip()
print(json.dumps({"baseCommit": commit}))
""",
        encoding="utf-8",
    )

    _write_executable(
        fake_bin / "uname",
        """#!/usr/bin/env bash
printf 'Linux\n'
""",
    )
    _write_executable(
        fake_bin / "flock",
        """#!/usr/bin/env bash
printf '%s\n' "${LUMEN_FLEET_CANARY_ACTIVE_LOCK_PATH:-}" >"$FAKE_STATE_DIR/lock-path"
exit "${FAKE_FLOCK_EXIT:-0}"
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
set -eu
state="$(cat "$FAKE_STATE_DIR/ollama-state")"
case "$*" in
  *--query-gpu=memory.total,memory.free*)
    if [[ "$state" == "running" ]]; then
      printf '8192, 3868\n'
    else
      printf '8192, 7800\n'
      if [[ "${FAKE_MEMORY_QUERY_FAIL_WHEN_STOPPED:-0}" == "1" ]]; then
        exit 1
      fi
    fi
    ;;
  *--query-compute-apps=process_name*)
    if [[ "${FAKE_COMPUTE_QUERY_EXIT:-0}" != "0" ]]; then
      printf 'compute query failed\n' >&2
      exit "$FAKE_COMPUTE_QUERY_EXIT"
    fi
    if [[ "$state" == "running" ]]; then
      printf '/usr/lib/ollama/llama-server\n'
    elif [[ -n "${FAKE_SURVIVING_OLLAMA_PROCESS:-}" ]]; then
      printf '%s\n' "$FAKE_SURVIVING_OLLAMA_PROCESS"
    fi
    ;;
  *)
    printf 'unexpected nvidia-smi arguments: %s\n' "$*" >&2
    exit 90
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
command_name="${1:-}"
shift || true
case "$command_name" in
  info)
    if [[ "${1:-}" == "--format" ]]; then
      printf '%s\n' "$FAKE_DOCKER_ROOT"
    fi
    ;;
  ps)
    if [[ "${FAKE_RUNNING_TRAINING_AFTER_PIPELINE:-0}" == "1" \
      && -f "$FAKE_STATE_DIR/pipeline-called" ]]; then
      printf '%064d\n' 9
    fi
    ;;
  inspect)
    format=""
    target=""
    while (($#)); do
      case "$1" in
        --format)
          format="$2"
          shift 2
          ;;
        *)
          target="$1"
          shift
          ;;
      esac
    done
    state="$(cat "$FAKE_STATE_DIR/ollama-state")"
    status=exited
    running=false
    health=none
    if [[ "$state" == "running" ]]; then
      status=running
      running=true
      health=healthy
      if [[ -f "$FAKE_STATE_DIR/started" ]]; then
        health="${FAKE_RESTORE_HEALTH:-healthy}"
      fi
    fi
    case "$format" in
      *'{{.Id}}|'*)
        printf '%s|%s|%s|%s\n' \
          "$FAKE_OLLAMA_ID" "$status" "$health" "${FAKE_RESTART_POLICY:-unless-stopped}"
        ;;
      '{{.Id}}')
        printf '%s\n' "${FAKE_CURRENT_OLLAMA_ID:-$FAKE_OLLAMA_ID}"
        ;;
      '{{.HostConfig.RestartPolicy.Name}}')
        printf '%s\n' "${FAKE_RESTART_POLICY:-unless-stopped}"
        ;;
      '{{.State.Running}}')
        printf '%s\n' "$running"
        ;;
      *'{{.State.Running}}|'*)
        printf '%s|%s\n' "$running" "$health"
        ;;
      *)
        printf 'unexpected docker inspect format: %s target=%s\n' "$format" "$target" >&2
        exit 91
        ;;
    esac
    ;;
  stop)
    printf 'stop\n' >>"$FAKE_STATE_DIR/events"
    if [[ "${FAKE_STOP_EXIT:-0}" != "0" ]]; then
      exit "$FAKE_STOP_EXIT"
    fi
    printf 'stopped\n' >"$FAKE_STATE_DIR/ollama-state"
    ;;
  start)
    printf 'start\n' >>"$FAKE_STATE_DIR/events"
    if [[ "${FAKE_START_EXIT:-0}" != "0" ]]; then
      exit "$FAKE_START_EXIT"
    fi
    : >"$FAKE_STATE_DIR/started"
    printf 'running\n' >"$FAKE_STATE_DIR/ollama-state"
    printf '%s\n' "$FAKE_OLLAMA_ID"
    ;;
  *)
    printf 'unexpected docker command: %s %s\n' "$command_name" "$*" >&2
    exit 92
    ;;
esac
""",
    )
    (state / "ollama-state").write_text("running\n", encoding="utf-8")
    (state / "events").write_text("", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    _git(repository, "config", "user.name", "Lumen Test")
    _git(repository, "config", "user.email", "lumen@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "fixture")
    commit = _git(repository, "rev-parse", "HEAD")

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_STATE_DIR": str(state),
        "FAKE_DOCKER_ROOT": str(docker_root),
        "FAKE_OLLAMA_ID": OLLAMA_ID,
        "LUMEN_FLEET_CANARY_GPU_WAIT_SECONDS": "0",
        "LUMEN_FLEET_CANARY_HEALTH_WAIT_SECONDS": "0",
        # These inherited values must not weaken the wrapper's fixed contract.
        "LUMEN_UBUNTU_BUILD_IMAGE": "0",
        "LUMEN_UBUNTU_RESUME": "1",
        "LUMEN_UBUNTU_UPLOAD": "1",
        "LUMEN_UBUNTU_EVAL_MAX_EXAMPLES": "2",
    }
    return {
        "repository": repository,
        "wrapper": scripts / WRAPPER.name,
        "workspace": workspace,
        "state": state,
        "host_state": host_state,
        "commit": commit,
        "environment": environment,
    }


def _run(
    harness: Mapping[str, object],
    *,
    extra_arguments: tuple[str, ...] = (),
    environment: Mapping[str, str] | None = None,
    confirm: bool = True,
) -> subprocess.CompletedProcess[str]:
    arguments = [
        "bash",
        str(harness["wrapper"]),
        "--workspace-root",
        str(harness["workspace"]),
        "--expected-commit",
        str(harness["commit"]),
        "--run-id",
        "fleetcanary-test",
        *extra_arguments,
    ]
    if confirm:
        arguments.append("--confirm-stop-ollama")
    return subprocess.run(
        arguments,
        cwd=harness["repository"],
        env=dict(environment or harness["environment"]),
        text=True,
        capture_output=True,
        check=False,
    )


def _events(harness: Mapping[str, object]) -> list[str]:
    state = Path(str(harness["state"]))
    return (state / "events").read_text(encoding="utf-8").splitlines()


def test_help_has_no_host_side_effects() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--confirm-stop-ollama" in result.stdout
    assert "optimized variant" in result.stdout
    assert "no upload" in result.stdout


def test_production_contract_fixes_resource_floors_and_host_state() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "MIN_DOCKER_FREE_BYTES=42949672960" in source
    assert "MIN_WORKSPACE_FREE_BYTES=53687091200" in source
    assert "MIN_FREE_VRAM_MIB=7168" in source
    assert "LUMEN_FLEET_CANARY_MIN_DOCKER_FREE_BYTES" not in source
    assert "LUMEN_FLEET_CANARY_MIN_WORKSPACE_FREE_BYTES" not in source
    assert "LUMEN_FLEET_CANARY_MIN_FREE_VRAM_MIB" not in source
    assert 'HOST_STATE_ROOT="/var/tmp/lumen-fleet-canary-state"' in source
    assert "$WORKSPACE_ROOT/operator-state" not in source
    assert 'exec 8<>"$HOST_LOCK_FILE"' in source
    assert "flock -n 8" in source


def test_strict_contract_stops_launches_and_restores(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    result = _run(harness)

    assert result.returncode == 0, result.stderr
    assert _events(harness) == ["stop", "pipeline", "start"]
    state = Path(str(harness["state"]))
    arguments = (state / "pipeline-args").read_text(encoding="utf-8").splitlines()
    output_root = str(Path(str(harness["workspace"])) / "output")
    hf_cache = str(Path(str(harness["workspace"])) / "hf-cache")
    assert arguments == [
        "--variant",
        "optimized",
        "--agents",
        "fleet",
        "--no-gguf",
        "--run-id",
        "fleetcanary-test",
        "--output-dir",
        output_root,
        "--hf-cache",
        hf_cache,
        "--expected-source-commit",
        str(harness["commit"]),
    ]
    for forbidden in (
        "--resume",
        "--eval-smoke",
        "--no-evaluate",
        "--upload",
        "--public",
        "--overwrite",
        "--no-build",
    ):
        assert forbidden not in arguments
    launch_environment = dict(
        line.split("=", 1)
        for line in (state / "pipeline-env")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert launch_environment["LUMEN_UBUNTU_BUILD_IMAGE"] == "1"
    assert launch_environment["LUMEN_UBUNTU_PULL_BASE"] == "1"
    assert launch_environment["LUMEN_UBUNTU_RESUME"] == "0"
    assert launch_environment["LUMEN_UBUNTU_UPLOAD"] == "0"
    assert launch_environment["LUMEN_UBUNTU_OVERWRITE"] == "0"
    assert launch_environment["LUMEN_UBUNTU_PREPARE_ONLY"] == "0"
    assert launch_environment["LUMEN_UBUNTU_EVALUATE"] == "1"
    assert launch_environment["LUMEN_UBUNTU_EVAL_MAX_EXAMPLES"] == ""
    assert launch_environment["LUMEN_UBUNTU_CONVERT_GGUF"] == "0"
    assert launch_environment["LUMEN_UBUNTU_AGENTS"] == "fleet"
    assert launch_environment["LUMEN_UBUNTU_VARIANT"] == "optimized"
    assert (
        launch_environment["LUMEN_UBUNTU_EXPECTED_SOURCE_COMMIT"]
        == harness["commit"]
    )
    assert launch_environment["LUMEN_UBUNTU_HOST_RESERVATION_FD"] == "8"
    assert (state / "ollama-state").read_text(encoding="utf-8") == "running\n"
    assert not (
        Path(str(harness["host_state"])) / "ollama-restore-required"
    ).exists()
    assert (state / "lock-path").read_text(encoding="utf-8").strip() == str(
        Path(str(harness["host_state"])).resolve() / "fleet-canary.lock"
    )
    assert (state / "pipeline-lock-inherited").is_file()


def test_pipeline_failure_preserves_exit_and_restores_ollama(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_PIPELINE_EXIT": "3",
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 3
    assert _events(harness) == ["stop", "pipeline", "start"]
    assert "canonical pipeline exited 3" in result.stderr
    assert (
        Path(str(harness["state"])) / "ollama-state"
    ).read_text(encoding="utf-8") == "running\n"


def test_preflight_only_never_stops_or_launches(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    result = _run(
        harness,
        extra_arguments=("--preflight-only",),
        confirm=False,
    )

    assert result.returncode == 0, result.stderr
    assert _events(harness) == []
    assert "Ollama was not stopped" in result.stdout
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()


def test_actual_launch_requires_explicit_stop_confirmation(tmp_path: Path) -> None:
    harness = _harness(tmp_path)

    result = _run(harness, confirm=False)

    assert result.returncode == 1
    assert _events(harness) == []
    assert "requires --confirm-stop-ollama" in result.stderr
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()


def test_stop_failure_never_launches_training(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_STOP_EXIT": "1",
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 1
    assert _events(harness) == ["stop"]
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()
    assert not (
        Path(str(harness["host_state"])) / "ollama-restore-required"
    ).exists()


def test_compute_process_query_failure_restores_without_launching(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_COMPUTE_QUERY_EXIT": "1",
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 1
    assert _events(harness) == ["stop", "start"]
    assert "unable to inspect GPU compute processes" in result.stderr
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()
    assert (
        Path(str(harness["state"])) / "ollama-state"
    ).read_text(encoding="utf-8") == "running\n"


def test_ollama_llama_server_process_restores_without_launching(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_SURVIVING_OLLAMA_PROCESS": (
            "/usr/lib/ollama/runners/cuda_v12/ollama_llama_server"
        ),
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 1
    assert _events(harness) == ["stop", "start"]
    assert "an Ollama GPU process remained" in result.stderr
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()
    assert (
        Path(str(harness["state"])) / "ollama-state"
    ).read_text(encoding="utf-8") == "running\n"


def test_partial_gpu_snapshot_failure_restores_without_launching(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_MEMORY_QUERY_FAIL_WHEN_STOPPED": "1",
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 1
    assert _events(harness) == ["stop", "start"]
    assert "unable to read GPU memory after Ollama stop" in result.stderr
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()
    assert (
        Path(str(harness["state"])) / "ollama-state"
    ).read_text(encoding="utf-8") == "running\n"


def test_restore_health_failure_retains_marker_and_fails_handoff(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_RESTORE_HEALTH": "unhealthy",
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 70
    assert _events(harness) == ["stop", "pipeline", "start"]
    assert "manual recovery is required" in result.stderr
    assert (
        Path(str(harness["host_state"])) / "ollama-restore-required"
    ).is_file()


def test_running_durable_training_container_prevents_ollama_restart(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_PIPELINE_EXIT": "129",
        "FAKE_RUNNING_TRAINING_AFTER_PIPELINE": "1",
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 71
    assert _events(harness) == ["stop", "pipeline"]
    assert "still running" in result.stderr
    assert (
        Path(str(harness["state"])) / "ollama-state"
    ).read_text(encoding="utf-8") == "stopped\n"
    assert (
        Path(str(harness["host_state"])) / "ollama-restore-required"
    ).is_file()


def test_host_global_lock_contention_fails_before_docker_or_training(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    environment = {
        **dict(harness["environment"]),
        "FAKE_FLOCK_EXIT": "1",
    }

    result = _run(harness, environment=environment)

    assert result.returncode == 1
    assert "another strict Fleet canary wrapper holds" in result.stderr
    assert _events(harness) == []
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()
    lock_path = (Path(str(harness["state"])) / "lock-path").read_text(
        encoding="utf-8"
    ).strip()
    assert lock_path == str(
        Path(str(harness["host_state"])).resolve() / "fleet-canary.lock"
    )
    assert not lock_path.startswith(str(Path(str(harness["workspace"]))))


def test_global_restore_marker_blocks_a_different_workspace(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    host_state = Path(str(harness["host_state"]))
    host_state.mkdir(mode=0o700)
    marker = host_state / "ollama-restore-required"
    marker.write_text("{}\n", encoding="utf-8")
    marker.chmod(0o600)
    alternate_workspace = tmp_path / "alternate-workspace"

    result = subprocess.run(
        [
            "bash",
            str(harness["wrapper"]),
            "--workspace-root",
            str(alternate_workspace),
            "--expected-commit",
            str(harness["commit"]),
            "--confirm-stop-ollama",
        ],
        cwd=harness["repository"],
        env=dict(harness["environment"]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "a prior Ollama restore marker requires manual recovery" in result.stderr
    assert not alternate_workspace.exists()
    assert _events(harness) == []
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()


def test_source_overlapping_workspace_fails_without_dirtying_checkout(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    repository = Path(str(harness["repository"]))
    overlapping_workspace = repository / "managed-workspace"

    result = subprocess.run(
        [
            "bash",
            str(harness["wrapper"]),
            "--workspace-root",
            str(overlapping_workspace),
            "--expected-commit",
            str(harness["commit"]),
            "--confirm-stop-ollama",
        ],
        cwd=repository,
        env=dict(harness["environment"]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "workspace root overlaps the source checkout" in result.stderr
    assert not overlapping_workspace.exists()
    assert _git(repository, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert _events(harness) == []


def test_commit_mismatch_fails_before_docker_or_training(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    wrong_commit = "f" * 40

    result = subprocess.run(
        [
            "bash",
            str(harness["wrapper"]),
            "--workspace-root",
            str(harness["workspace"]),
            "--expected-commit",
            wrong_commit,
            "--confirm-stop-ollama",
        ],
        cwd=harness["repository"],
        env=dict(harness["environment"]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "does not match --expected-commit" in result.stderr
    assert _events(harness) == []
    assert not (Path(str(harness["state"])) / "pipeline-called").exists()
