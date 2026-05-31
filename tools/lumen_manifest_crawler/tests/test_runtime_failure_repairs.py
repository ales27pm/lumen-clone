from lumen_manifest_crawler.dataset.compiler import _repair_for_runtime_failure


def test_trace_parse_error_adds_rem_and_regression_eval() -> None:
    repair = _repair_for_runtime_failure(
        {"type": "trace_parse_error", "actual": "noJSONObject", "scenario": "scenario"},
        known_tools=["calendar.create"],
    )
    assert repair["action"] == "add_strict_trace_json_format_samples"
    assert repair["alsoAdd"] == ["rem_repair_sample", "trace_parse_regression_eval"]
    assert repair["failure"] == "noJSONObject"


def test_trace_tool_without_allowed_set_adds_rem_and_regression_eval() -> None:
    repair = _repair_for_runtime_failure(
        {"type": "trace_tool_without_allowed_set", "actual": "camera.capture", "scenario": "scenario"},
        known_tools=["camera.capture"],
    )
    assert repair["action"] == "add_tool_allowed_set_trace_repairs"
    assert repair["alsoAdd"] == ["rem_repair_sample", "trace_allowed_set_regression_eval"]
    assert repair["knownToolIDs"] == ["camera.capture"]


def test_approval_sensitive_tool_selected_expands_coverage() -> None:
    repair = _repair_for_runtime_failure(
        {"type": "approval_sensitive_tool_selected", "scenario": "calendar.create"},
        known_tools=["calendar.create"],
    )
    assert repair["action"] == "regenerate_approval_boundary_samples"
    assert repair["alsoAdd"] == ["approval_boundary_dpo_pairs", "approval_confirmation_ui_regression_eval"]
