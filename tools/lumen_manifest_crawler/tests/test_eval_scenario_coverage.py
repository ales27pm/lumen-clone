import json
from pathlib import Path

import pytest

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset.adapter_evaluation import upgrade_evaluation_record
from lumen_manifest_crawler.dataset.compiler import (
    TOOL_SCENARIO_ARGUMENT_COVERAGE,
    TOOL_SCENARIO_PROMPTS,
    DatasetCompilerConfig,
    _build_eval_records,
    _build_tool_schema_records,
)
from lumen_manifest_crawler.dataset.fine_tuning import _build_agent_eval_records
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolArgumentManifest, ToolManifest
from lumen_manifest_crawler.validators import validate_manifest


@pytest.fixture(scope="module")
def repo_compiled_evals() -> tuple[AgentBehaviorManifest, list[dict]]:
    repo_root = Path(__file__).resolve().parents[3]
    manifest = generate_manifest(repo_root)
    return manifest, _build_eval_records(manifest, DatasetCompilerConfig())


def _manifest() -> AgentBehaviorManifest:
    return AgentBehaviorManifest(
        tools=[
            ToolManifest(id="plain.ping", displayName="Ping", description="Check connectivity"),
            ToolManifest(
                id="notes.create",
                displayName="Create Note",
                description="Create a note",
                requiresApproval=True,
                permissionKey="notes",
                arguments=[ToolArgumentManifest(name="title", type="string", required=True)],
            ),
            ToolManifest(
                id="files.read",
                displayName="Read File",
                description="Read a previously imported local document by name",
                arguments=[ToolArgumentManifest(name="name", type="string", required=True)],
            ),
            ToolManifest(id="mail.send", displayName="Send Mail", description="Send email", requiresApproval=True),
            ToolManifest(id="location.current", displayName="Current Location", description="Read current location", permissionKey="location"),
        ]
    )


def _runtime_scenario(records: list[dict], tool_id: str, prompt: str) -> dict:
    return next(
        record
        for record in records
        if record.get("taskType") == "tool_runtime_scenario_selection"
        and record["expected"]["selectedToolID"] == tool_id
        and record["messages"][-1]["content"] == prompt
    )


def _assert_actionable(record: dict, coverage: list[str]) -> None:
    assert record["metadata"]["argumentCoverage"] == coverage
    assert record["expected"]["mustPersistActionStep"] is True
    assert "status" not in record["expected"]


def _assert_needs_clarification(record: dict, coverage: list[str]) -> None:
    assert record["metadata"]["argumentCoverage"] == coverage
    assert record["expected"]["status"] == "needs_clarification"
    assert "mustPersistActionStep" not in record["expected"]
    assert "requiresApproval" not in record["expected"]
    assert "permissionKey" not in record["expected"]


def test_runtime_eval_scenarios_have_required_coverage():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    runtime = [r for r in evals if r["taskType"] == "tool_runtime_scenario_selection"]
    by_tool = {
        tool.id: [r for r in runtime if r["expected"]["selectedToolID"] == tool.id]
        for tool in manifest.tools
    }

    for tool in manifest.tools:
        items = by_tool[tool.id]
        assert len(items) >= 5
        natural = [r for r in items if r["metadata"].get("scenarioKind") == "natural_intent"]
        assert len(natural) >= 2
        assert any(r["metadata"].get("scenarioKind") == "explicit_tool_schema" for r in items)
        for record in natural:
            prompt = "\n".join(m["content"] for m in record["messages"])
            assert tool.id not in prompt
            assert record["metadata"]["toolIDVisibleInPrompt"] is False

    note_coverage = set()
    for record in by_tool["notes.create"]:
        note_coverage.update(record["metadata"].get("argumentCoverage") or [])
    assert "title" in note_coverage
    assert any(r["metadata"].get("approvalCoverage") for r in by_tool["mail.send"])
    assert any(r["metadata"].get("permissionCoverage") for r in by_tool["location.current"])


def test_runtime_selection_contract_is_conditional_on_argument_coverage():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    runtime = [record for record in evals if record["taskType"] == "tool_runtime_scenario_selection"]
    tools_by_id = {tool.id: tool for tool in manifest.tools}

    assert runtime
    for record in runtime:
        expected = record["expected"]
        tool = tools_by_id[expected["selectedToolID"]]
        required_arguments = [
            argument.name for argument in tool.arguments if argument.required
        ]
        covered_arguments = set(record["metadata"].get("argumentCoverage") or [])
        missing_arguments = [
            argument
            for argument in required_arguments
            if argument not in covered_arguments
        ]

        assert "tool" not in expected
        assert "requiredArguments" not in expected
        if missing_arguments:
            assert expected["status"] == "needs_clarification"
            assert expected["missingArguments"] == missing_arguments
            assert "mustPersistActionStep" not in expected
            assert "requiresApproval" not in expected
            assert "permissionKey" not in expected
        else:
            assert "status" not in expected
            assert expected["mustPersistActionStep"] is True
            assert expected["requiresApproval"] is tool.requiresApproval
            assert expected["permissionKey"] == tool.permissionKey

    note_argument_scenario = next(
        record
        for record in runtime
        if record["expected"]["selectedToolID"] == "notes.create"
        and record["metadata"].get("scenarioKind") == "argument_completion"
    )
    assert note_argument_scenario["metadata"]["argumentCoverage"] == ["title"]
    assert note_argument_scenario["expected"]["mustPersistActionStep"] is True
    assert note_argument_scenario["expected"]["requiresApproval"] is True


def test_optional_only_argument_completion_does_not_fabricate_values_or_coverage():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="weather.optional",
                displayName="Current Weather",
                description="Get current weather",
                arguments=[
                    ToolArgumentManifest(name="location", type="string", required=False),
                    ToolArgumentManifest(name="city", type="string", required=False),
                ],
            )
        ]
    )
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    scenario = next(
        record
        for record in evals
        if record["taskType"] == "tool_runtime_scenario_selection"
        and record["metadata"].get("scenarioKind") == "argument_completion"
    )

    assert scenario["messages"][-1]["content"] == "Help me with current weather."
    assert scenario["metadata"]["argumentCoverage"] == []
    assert scenario["expected"]["mustPersistActionStep"] is True
    assert "location" not in scenario["messages"][-1]["content"].casefold()
    assert "city" not in scenario["messages"][-1]["content"].casefold()


def test_partial_argument_clarifications_bind_only_still_missing_fields(
    repo_compiled_evals: tuple[AgentBehaviorManifest, list[dict]],
) -> None:
    manifest, evals = repo_compiled_evals
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    partial = [
        record
        for record in evals
        if record["taskType"] == "tool_runtime_scenario_selection"
        and record["expected"].get("status") == "needs_clarification"
        and record["metadata"].get("argumentCoverage")
    ]

    assert len(partial) == 17
    for record in partial:
        tool = tools_by_id[record["expected"]["selectedToolID"]]
        covered = set(record["metadata"]["argumentCoverage"])
        expected_missing = [
            argument.name
            for argument in tool.arguments
            if argument.required and argument.name not in covered
        ]
        assert expected_missing
        assert record["expected"]["missingArguments"] == expected_missing

    routed = _build_agent_eval_records(
        manifest,
        {"eval_scenarios": evals},
        set(tools_by_id),
    )["cortex"]
    metrics_by_prompt = {
        record["messages"][-1]["content"]: next(
            metric
            for metric in record["metrics"]
            if metric.get("type") == "cortex_route_contract"
        )
        for record in routed
        if record["metadata"].get("evalType")
        == "tool_runtime_scenario_selection"
    }
    for record in partial:
        metric = metrics_by_prompt[record["messages"][-1]["content"]]
        assert metric["mode"] == "clarification"
        assert metric["requiredArguments"] == record["expected"][
            "missingArguments"
        ]


def test_curated_required_argument_coverage_is_exhaustive_and_manifest_valid(
    repo_compiled_evals: tuple[AgentBehaviorManifest, list[dict]],
) -> None:
    manifest, _ = repo_compiled_evals
    tools_by_id = {tool.id: tool for tool in manifest.tools}

    assert set(TOOL_SCENARIO_PROMPTS).issubset(tools_by_id)
    for tool_id, prompts in TOOL_SCENARIO_PROMPTS.items():
        assert prompts
        assert len(prompts) == len(set(prompts))
        required_arguments = {
            argument.name
            for argument in tools_by_id[tool_id].arguments
            if argument.required
        }
        audited_coverage = TOOL_SCENARIO_ARGUMENT_COVERAGE.get(tool_id)
        if not required_arguments:
            assert audited_coverage is None
            continue

        assert audited_coverage is not None
        assert set(audited_coverage) == set(prompts)
        for covered_arguments in audited_coverage.values():
            assert isinstance(covered_arguments, tuple)
            assert len(covered_arguments) == len(set(covered_arguments))
            assert set(covered_arguments).issubset(required_arguments)

    required_curated_tools = {
        tool_id
        for tool_id in TOOL_SCENARIO_PROMPTS
        if any(argument.required for argument in tools_by_id[tool_id].arguments)
    }
    assert set(TOOL_SCENARIO_ARGUMENT_COVERAGE) == required_curated_tools


def test_curated_natural_prompts_distinguish_actionable_partial_and_ambiguous_values(
    repo_compiled_evals: tuple[AgentBehaviorManifest, list[dict]],
) -> None:
    _, evals = repo_compiled_evals

    actionable_cases = [
        ("maps.search", "Find a hardware store nearby.", ["query"]),
        ("maps.search", "Search maps for coffee near me.", ["query"]),
        ("web.search", "Search the web for Core ML conversion tips.", ["query"]),
        ("contacts.search", "Find Antoine in my contacts.", ["query"]),
        ("calendar.create", "Create a calendar event for a meeting in 10 minutes.", ["title", "startsInMinutes"]),
        ("calendar.create", "Add a dentist appointment tomorrow at 2 PM.", ["title", "startsInMinutes"]),
        ("reminders.create", "Remind me to charge the scooter battery.", ["title"]),
        ("files.read", "Read the imported project notes file.", ["name"]),
        ("outlook.message.read", "Read the latest email.", ["messageId"]),
        ("mail.draft", "Draft an email to Antoine about the show.", ["to", "body"]),
        ("memory.recall", "Search stored memory for the Aurora rollback checklist.", ["query"]),
        (
            "memory.save",
            "Store this as a preference: lead with observed error codes before suggesting fixes.",
            ["content", "kind"],
        ),
        ("reminders.list", "Fetch all unresolved reminder entries.", []),
    ]
    clarification_cases = [
        ("maps.search", "Show me on map.", []),
        ("calendar.create", "Create a calendar entry named Ridgeview permit review.", ["title"]),
        ("alarm.countdown", "Start a 10 minute countdown alarm.", ["durationSeconds"]),
        ("reminders.create", "Add a reminder for tomorrow morning.", []),
        ("files.read", "Open the document I imported.", []),
        ("messages.draft", "Draft a message to Sylvie.", ["to"]),
        ("outlook.message.move", "Move this email to the project folder.", ["destination"]),
        ("web.fetch", "Open and read this URL.", []),
    ]

    for tool_id, prompt, coverage in actionable_cases:
        _assert_actionable(_runtime_scenario(evals, tool_id, prompt), coverage)
    for tool_id, prompt, coverage in clarification_cases:
        _assert_needs_clarification(_runtime_scenario(evals, tool_id, prompt), coverage)


def test_curated_runtime_contract_survives_cortex_routing_without_executor_leakage(
    repo_compiled_evals: tuple[AgentBehaviorManifest, list[dict]],
) -> None:
    manifest, evals = repo_compiled_evals
    routed = _build_agent_eval_records(
        manifest,
        {"eval_scenarios": evals},
        {tool.id for tool in manifest.tools},
    )
    routed_cortex_by_prompt = {
        record["messages"][-1]["content"]: record
        for record in routed["cortex"]
        if record["metadata"].get("evalType") == "tool_runtime_scenario_selection"
    }

    for tool_id, prompt in (
        ("maps.search", "Find a hardware store nearby."),
        ("maps.search", "Show me on map."),
        ("calendar.create", "Create a calendar event for a meeting in 10 minutes."),
        ("alarm.countdown", "Start a 10 minute countdown alarm."),
    ):
        compiled = _runtime_scenario(evals, tool_id, prompt)
        routed_record = routed_cortex_by_prompt[prompt]
        assert routed_record["expected"] == compiled["expected"]
        assert routed_record["metadata"]["agent"] == "cortex"

    assert not any(
        record["metadata"].get("evalType") == "tool_runtime_scenario_selection"
        for record in routed["executor"]
    )


def test_latest_outlook_message_reference_is_actionable_and_routes_as_message_id(
    repo_compiled_evals: tuple[AgentBehaviorManifest, list[dict]],
) -> None:
    manifest, evals = repo_compiled_evals
    prompt = "Read the latest email."
    compiled = _runtime_scenario(evals, "outlook.message.read", prompt)

    _assert_actionable(compiled, ["messageId"])
    assert compiled["expected"]["selectedToolID"] == "outlook.message.read"
    assert "missingArguments" not in compiled["expected"]

    routed = _build_agent_eval_records(
        manifest,
        {"eval_scenarios": evals},
        {tool.id for tool in manifest.tools},
    )
    routed_record = next(
        record
        for record in routed["cortex"]
        if record["metadata"].get("evalType")
        == "tool_runtime_scenario_selection"
        and record["messages"][-1]["content"] == prompt
    )
    route_metric = next(
        metric
        for metric in routed_record["metrics"]
        if metric.get("type") == "cortex_route_contract"
    )

    assert routed_record["expected"] == compiled["expected"]
    assert route_metric["mode"] == "actionable"
    assert route_metric["expectedToolID"] == "outlook.message.read"
    assert "requiredArguments" not in route_metric


def test_files_read_missing_name_clarifies_while_supplied_name_is_actionable():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    file_scenarios = [
        record
        for record in evals
        if record["taskType"] == "tool_runtime_scenario_selection"
        and record["expected"]["selectedToolID"] == "files.read"
    ]
    missing_name_scenarios = [
        record
        for record in file_scenarios
        if record["metadata"].get("argumentCoverage") == []
    ]
    supplied_name_scenarios = [
        record
        for record in file_scenarios
        if record["metadata"].get("argumentCoverage") == ["name"]
    ]

    assert missing_name_scenarios
    assert supplied_name_scenarios
    assert any("Open the document I imported." in record["messages"][-1]["content"] for record in missing_name_scenarios)
    assert any("named build-plan" in record["messages"][-1]["content"] for record in supplied_name_scenarios)
    for record in missing_name_scenarios:
        assert "tool" not in record["expected"]
        assert "requiredArguments" not in record["expected"]
        assert record["expected"]["status"] == "needs_clarification"
        assert "mustPersistActionStep" not in record["expected"]
        assert "requiresApproval" not in record["expected"]

    for record in supplied_name_scenarios:
        if record["metadata"]["scenarioKind"] in {"explicit_tool_schema", "argument_completion"}:
            assert '"name": "example name"' in record["messages"][-1]["content"]
        assert record["expected"]["mustPersistActionStep"] is True
        assert record["expected"]["requiresApproval"] is False
        assert "status" not in record["expected"]


def test_boundary_and_schema_prompts_supply_deterministic_required_values():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    note_runtime = [
        record
        for record in evals
        if record["taskType"] == "tool_runtime_scenario_selection"
        and record["expected"]["selectedToolID"] == "notes.create"
    ]

    for scenario_kind in ("explicit_tool_schema", "argument_completion", "approval_boundary", "permission_boundary"):
        record = next(item for item in note_runtime if item["metadata"]["scenarioKind"] == scenario_kind)
        assert record["metadata"]["argumentCoverage"] == ["title"]
        assert '"title": "example title"' in record["messages"][-1]["content"]
        assert record["expected"]["mustPersistActionStep"] is True
        assert record["expected"]["requiresApproval"] is True

    schema_eval = next(
        record
        for record in evals
        if record["taskType"] == "tool_schema_adherence"
        and record["expected"]["tool"] == "notes.create"
    )
    assert schema_eval["expected"] == {
        "tool": "notes.create",
        "arguments": {"title": "example title"},
    }
    assert '"title": "example title"' in schema_eval["messages"][-1]["content"]
    assert "do not add any other arguments" in schema_eval["messages"][-1]["content"]
    assert "requiredArguments" not in schema_eval["expected"]
    assert "requiresApproval" not in schema_eval["expected"]
    assert "permissionKey" not in schema_eval["expected"]

    schema_cards = _build_tool_schema_records(manifest, DatasetCompilerConfig())
    required_card = next(
        record
        for record in schema_cards
        if record["toolID"] == "notes.create"
        and record["metadata"].get("scenarioKind") == "required_argument_coverage"
    )
    assert '"title": "example title"' in required_card["messages"][-2]["content"]
    assert json.loads(required_card["messages"][-1]["content"])["arguments"] == {
        "title": "example title",
    }


def test_repo_tool_schema_evals_score_exact_arguments_without_boundary_demands(
    repo_compiled_evals: tuple[AgentBehaviorManifest, list[dict]],
):
    manifest, evals = repo_compiled_evals
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    schema_evals = [
        record
        for record in evals
        if record["taskType"] == "tool_schema_adherence"
    ]

    assert len(schema_evals) == len(tools_by_id)
    for record in schema_evals:
        expected = record["expected"]
        tool = tools_by_id[expected["tool"]]
        assert set(expected) == {"tool", "arguments"}
        assert set(expected["arguments"]) == {
            argument.name for argument in tool.arguments if argument.required
        }
        assert "arguments object exactly equal to" in record["messages"][-1]["content"]
        assert "do not add any other arguments" in record["messages"][-1]["content"]

        upgraded = upgrade_evaluation_record(
            {
                **record,
                "metadata": {**record["metadata"], "agent": "executor"},
            }
        )
        assert [metric["type"] for metric in upgraded["metrics"]] == [
            "manifest_tool_call",
            "json_field_equals",
        ]
        assert upgraded["metrics"][1]["expected"] == expected["arguments"]


def test_frozen_eval_records_route_to_their_owning_adapter_contracts():
    manifest = _manifest()
    manifest.sentinels.forbiddenInUserOutput = ["<internal_state>"]
    compiled = _build_eval_records(manifest, DatasetCompilerConfig())
    routed = _build_agent_eval_records(
        manifest,
        {"eval_scenarios": compiled},
        {tool.id for tool in manifest.tools},
    )

    cortex_types = {record["metadata"]["evalType"] for record in routed["cortex"]}
    executor_types = {record["metadata"]["evalType"] for record in routed["executor"]}
    mouth_types = {record["metadata"]["evalType"] for record in routed["mouth"]}

    assert "tool_runtime_scenario_selection" in cortex_types
    assert "tool_schema_adherence" not in cortex_types
    assert "user_output_safety" not in cortex_types
    assert "tool_schema_adherence" in executor_types
    assert "tool_runtime_scenario_selection" not in executor_types
    assert "user_output_safety" in mouth_types

    missing_file = next(
        record
        for record in routed["cortex"]
        if record["metadata"]["evalType"] == "clarification_missing_args"
    )
    assert missing_file["expected"] == {
        "selectedToolID": "files.read",
        "status": "needs_clarification",
        "missingArguments": ["name"],
    }
    for record in routed["cortex"]:
        assert "requiredArguments" not in record["expected"]
        assert "arguments" not in record["expected"]


def test_validator_fails_for_natural_tool_id_leak():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    for record in evals:
        if (
            record.get("taskType") == "tool_runtime_scenario_selection"
            and record["expected"]["selectedToolID"] == "plain.ping"
        ):
            if record["metadata"].get("scenarioKind") == "natural_intent":
                record["messages"][1]["content"] = "Use plain.ping now"
                break
    report = validate_manifest(manifest, {"eval_scenarios": evals})
    assert any(f.code == "tool_id_leak_in_natural_eval" for f in report.failures)


def test_tool_schema_cards_include_permission_kind_and_confirmation_mode():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="calendar.create",
                displayName="Create Event",
                description="Create calendar event",
                requiresApproval=True,
                permissionKey="NSCalendarsFullAccessUsageDescription",
                permissionKind="calendar",
                confirmationMode="userApproval",
                arguments=[ToolArgumentManifest(name="title", type="string", required=True)],
            )
        ]
    )

    records = _build_tool_schema_records(manifest, DatasetCompilerConfig())
    schema_record = next(record for record in records if record["id"].startswith("schema-") and record["toolID"] == "calendar.create")
    payload = json.loads(schema_record["messages"][-1]["content"])

    assert payload["permissionKind"] == "calendar"
    assert payload["confirmationMode"] == "userApproval"
    assert schema_record["metadata"]["permissionKind"] == "calendar"
    assert schema_record["metadata"]["confirmationMode"] == "userApproval"


def test_validator_fails_for_missing_required_argument_coverage():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    for record in evals:
        if (
            record.get("taskType") == "tool_runtime_scenario_selection"
            and record["expected"]["selectedToolID"] == "notes.create"
        ):
            metadata = record.get("metadata") or {}
            metadata["argumentCoverage"] = []
    report = validate_manifest(manifest, {"eval_scenarios": evals})
    assert any(f.code == "missing_argument_eval_coverage" for f in report.failures)


def test_validator_does_not_treat_natural_language_weather_word_as_tool_id_leak():
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(id="weather", displayName="Current Weather", description="Get weather", permissionKey="location"),
        ]
    )
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    report = validate_manifest(manifest, {"eval_scenarios": evals})
    assert report.passed is True
