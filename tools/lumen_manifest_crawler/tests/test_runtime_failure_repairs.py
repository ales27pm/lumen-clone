from lumen_manifest_crawler.dataset.compiler import _repair_for_runtime_failure


def test_trace_parse_error_adds_rem_and_regression_eval() -> None:
    repair = _repair_for_runtime_failure(
        {"type": "trace_parse_error", "actual": "noJSONObject", "scenario": "scenario"},
        known_tools=["calendar.create"],
    )
    assert repair["action"] == "add_strict_trace_json_format_samples"
    assert repair["alsoAdd"] == ["rem_repair_sample", "trace_parse_regression_eval"]
    assert repair["failure"] == "noJSONObject"


def test_prompt_budget_overflow_adds_agent_json_budget_eval() -> None:
    repair = _repair_for_runtime_failure(
        {"type": "prompt_budget_overflow", "actual": "contextWindowExceeded", "scenario": "scenario"},
        known_tools=["weather"],
    )
    assert repair["action"] == "compact_agent_json_prompt_budget"
    assert repair["alsoAdd"] == ["agent_json_context_budget_regression_eval", "rem_repair_sample"]
    assert repair["failure"] == "contextWindowExceeded"


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


def test_dynamic_local_lookup_tool_violation_teaches_plan_execute_evaluate() -> None:
    repair = _repair_for_runtime_failure(
        {
            "type": "tool_not_allowed_by_runtime_router",
            "actual": "maps.search",
            "scenario": "Where is the nearest free tax clinic tomorrow?",
            "problem": "Cortex selected a map-only tool for a time-sensitive local public lookup.",
        },
        known_tools=["location.current", "maps.search", "web.search"],
    )
    assert repair["action"] == "add_plan_gather_execute_evaluate_samples"
    assert repair["focusToolID"] == "web.search"
    assert repair["rejectedToolID"] == "maps.search"
    assert "mouth_grounded_answer_eval" in repair["alsoAdd"]
    assert repair["expectedPlan"] == [
        "classify as dynamic local public lookup",
        "gather current location when available",
        "run web.search for fresh public schedule/hours/event evidence",
        "evaluate whether the observation answers the user's time-sensitive question before finalizing",
    ]
