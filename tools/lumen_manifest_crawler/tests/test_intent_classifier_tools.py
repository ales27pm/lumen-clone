import json
import subprocess
import sys
from pathlib import Path
import importlib.util


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, cwd=repo_root())


def test_dataset_generation_is_deterministic_and_schema_aligned(tmp_path: Path):
    out1 = tmp_path / "a.jsonl"
    out2 = tmp_path / "b.jsonl"
    cmd = [sys.executable, "tools/intent_classifier/generate_intent_dataset.py", "--out", str(out1), "--repeat", "2", "--seed", "27"]
    r1 = run_cmd(cmd)
    assert r1.returncode == 0, r1.stderr
    cmd[3] = str(out2)
    r2 = run_cmd(cmd)
    assert r2.returncode == 0, r2.stderr

    text1 = out1.read_text(encoding="utf-8")
    text2 = out2.read_text(encoding="utf-8")
    assert text1 == text2

    schema = json.loads((repo_root() / "tools/intent_classifier/intents.schema.json").read_text(encoding="utf-8"))
    allowed = set(schema["properties"]["intent"]["enum"])
    for line in text1.splitlines():
        row = json.loads(line)
        assert row["intent"] in allowed
        assert row["text"]


def test_dataset_repeat_zero_fails(tmp_path: Path):
    out = tmp_path / "bad.jsonl"
    r = run_cmd([sys.executable, "tools/intent_classifier/generate_intent_dataset.py", "--out", str(out), "--repeat", "0"])
    assert r.returncode != 0
    assert "--repeat must be >= 1" in (r.stderr + r.stdout)


def test_training_fails_when_dataset_missing(tmp_path: Path):
    missing = tmp_path / "missing.jsonl"
    out = tmp_path / "intent.pkl"
    r = run_cmd([sys.executable, "tools/intent_classifier/train_intent_classifier.py", "--dataset", str(missing), "--model-out", str(out)])
    assert r.returncode != 0
    assert "Dataset file not found" in (r.stderr + r.stdout)


def test_improve_loop_parser_defaults_and_enable_flag():
    module_path = repo_root() / "tools/lumen_terminal_improve_loop.py"
    spec = importlib.util.spec_from_file_location("lumen_terminal_improve_loop", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["lumen_terminal_improve_loop"] = module
    spec.loader.exec_module(module)

    defaults = module.parse_args([])
    assert defaults.train_intent_classifier is False

    enabled = module.parse_args(["--train-intent-classifier"])
    assert enabled.train_intent_classifier is True
