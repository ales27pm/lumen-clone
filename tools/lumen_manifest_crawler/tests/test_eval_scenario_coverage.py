from lumen_manifest_crawler.dataset.compiler import DatasetCompilerConfig, _build_eval_records
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolArgumentManifest, ToolManifest
from lumen_manifest_crawler.validators import validate_manifest


def _manifest() -> AgentBehaviorManifest:
    return AgentBehaviorManifest(
        tools=[
            ToolManifest(id="plain.ping", displayName="Ping", description="Check connectivity"),
            ToolManifest(
                id="notes.create",
                displayName="Create Note",
                description="Create a note",
                arguments=[ToolArgumentManifest(name="title", type="string", required=True)],
            ),
            ToolManifest(id="mail.send", displayName="Send Mail", description="Send email", requiresApproval=True),
            ToolManifest(id="location.current", displayName="Current Location", description="Read current location", permissionKey="location"),
        ]
    )


def test_runtime_eval_scenarios_have_required_coverage():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    runtime = [r for r in evals if r["taskType"] == "tool_runtime_scenario_selection"]
    by_tool = {tool.id: [r for r in runtime if r["expected"]["tool"] == tool.id] for tool in manifest.tools}

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


def test_validator_fails_for_natural_tool_id_leak():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    for record in evals:
        if record.get("taskType") == "tool_runtime_scenario_selection" and record["expected"]["tool"] == "plain.ping":
            if record["metadata"].get("scenarioKind") == "natural_intent":
                record["messages"][1]["content"] = "Use plain.ping now"
                break
    report = validate_manifest(manifest, {"eval_scenarios": evals})
    assert any(f.code == "tool_id_leak_in_natural_eval" for f in report.failures)


def test_validator_fails_for_missing_required_argument_coverage():
    manifest = _manifest()
    evals = _build_eval_records(manifest, DatasetCompilerConfig())
    for record in evals:
        if record.get("taskType") == "tool_runtime_scenario_selection" and record["expected"]["tool"] == "notes.create":
            metadata = record.get("metadata") or {}
            if metadata.get("scenarioKind") == "argument_completion":
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
