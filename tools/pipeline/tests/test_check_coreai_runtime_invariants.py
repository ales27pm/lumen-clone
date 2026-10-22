"""Negative checks for immutable dependency and local-only Core AI wiring."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("coreai_gate", ROOT / "tools/check_coreai_runtime_invariants.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
FILES = (
    "ios/Lumen.xcodeproj/project.pbxproj",
    "ios/Lumen/Services/LLM/CoreAI/CoreAIGemmaService.swift",
    "ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift",
    "ios/Lumen/Views/ChatView.swift",
    "ios/Lumen/Assistant/AgentKernelContracts.swift",
    "ios/Lumen/Services/IntentClassification/IntentClassifierService.swift",
)


@pytest.fixture
def source(tmp_path):
    for relative in FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text())
    return tmp_path


def test_current_wiring_satisfies_static_contract(source):
    assert GATE.check(source) == []


@pytest.mark.parametrize("file,before,after,expected", (
    (FILES[0], "27.1;", "18.0;", "deployment"),
    (FILES[0], "9db747612f972001279f02adb04d3ef982310d0e", "0" * 40, "immutable"),
    (FILES[0], "productRef = C0A100000000000000000011", "productRef = DEADBEEFDEADBEEFDEADBEEF", "linked"),
    (FILES[1], "if #available(iOS 27.1, *)", "if #available(iOS 27.2, *)", "lifecycle"),
    (FILES[1], "AutoTokenizer.from(modelFolder:", "AutoTokenizer.from(pretrained:", "entirely local"),
    (FILES[2], "if generation.preventsToolExecution {", "if false {", "runtime errors"),
    (FILES[3], "let kernel = AssistantKernel.shared", "let kernel = AppLlamaService.shared.stream(request)", "Plain chat"),
    (FILES[4], "let allowsToolExecution: Bool", "let ignoresToolExecution: Bool", "tools disabled"),
    (FILES[5], "if CoreAIGemmaModelStore.isEnabled {", "if false {", "auxiliary learned"),
))
def test_unsafe_drift_is_rejected(source, file, before, after, expected):
    target = source / file
    text = target.read_text()
    assert before in text
    target.write_text(text.replace(before, after))
    assert any(expected in failure for failure in GATE.check(source))
