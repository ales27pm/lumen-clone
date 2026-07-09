import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "check_agent_kernel_boundary.py"


def load_guard_module():
    spec = importlib.util.spec_from_file_location("check_agent_kernel_boundary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scan_file_marks_debug_only_legacy_calls(tmp_path):
    guard = load_guard_module()
    source = tmp_path / "Probe.swift"
    source.write_text(
        """
#if DEBUG
AgentService.shared.run(request, options: options)
#endif
""".strip(),
        encoding="utf-8",
    )

    findings = guard.scan_file(source)

    assert findings == [
        (2, "AgentService.shared.run", "AgentService.shared.run(request, options: options)", True)
    ]


def test_scan_file_treats_debug_else_branch_as_release(tmp_path):
    guard = load_guard_module()
    source = tmp_path / "Probe.swift"
    source.write_text(
        """
#if DEBUG
SlotAgentService.shared.run(request, options: options)
#else
AgentService.shared.run(request, options: options)
#endif
""".strip(),
        encoding="utf-8",
    )

    findings = guard.scan_file(source)

    assert findings == [
        (2, "SlotAgentService.shared.run", "SlotAgentService.shared.run(request, options: options)", True),
        (4, "AgentService.shared.run", "AgentService.shared.run(request, options: options)", False),
    ]


def test_structured_agent_executor_is_not_allowlisted():
    guard = load_guard_module()

    assert "ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift" not in guard.DOCUMENTED_COMPATIBILITY_BRIDGES
