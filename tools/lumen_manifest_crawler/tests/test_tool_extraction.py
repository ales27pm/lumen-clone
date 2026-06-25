from pathlib import Path

from lumen_manifest_crawler.manifest import AgentBehaviorManifest
from lumen_manifest_crawler.swift_extractors.base import SwiftFile, argument_value, clean_swift_string
from lumen_manifest_crawler.swift_extractors.tool_definition import ToolDefinitionExtractor


def test_tool_definition_extraction():
    text = '''
    struct ToolRegistry {
      static let all: [ToolDefinition] = [
        ToolDefinition(
          id: "calendar.create",
          displayName: "Create Calendar Event",
          description: "Creates a calendar event.",
          requiresApproval: true,
          permissionKey: "NSCalendarsFullAccessUsageDescription",
          arguments: [
            ToolArgument(name: "title", type: .string, required: true),
            ToolArgument(name: "startsInMinutes", type: .double, required: true)
          ]
        )
      ]
    }
    '''
    manifest = AgentBehaviorManifest()
    ToolDefinitionExtractor().extract(SwiftFile(Path("ToolDefinition.swift"), "ToolDefinition.swift", text), manifest)
    assert len(manifest.tools) == 1
    tool = manifest.tools[0]
    assert tool.id == "calendar.create"
    assert tool.requiresApproval is True
    assert tool.permissionKey == "NSCalendarsFullAccessUsageDescription"
    assert tool.permissionKind == "calendar"
    assert tool.confirmationMode == "userApproval"
    assert {arg.name for arg in tool.arguments} == {"title", "startsInMinutes"}


def test_argument_value_keeps_quoted_commas():
    block = 'ToolDefinition(id: "maps.search", description: "Find places. Args: query. Use coffee, pharmacy, hardware.", requiresApproval: false)'
    description = clean_swift_string(argument_value(block, "description"))
    assert description == "Find places. Args: query. Use coffee, pharmacy, hardware."


def test_nil_permission_key_normalizes_to_none():
    block = 'ToolDefinition(id: "mail.draft", permissionKey: nil, requiresApproval: true)'
    assert clean_swift_string(argument_value(block, "permissionKey")) is None


def test_args_contract_derives_arguments_from_description():
    text = '''
    enum ToolRegistry {
      static let all: [ToolDefinition] = [
        ToolDefinition(
          id: "trigger.create",
          name: "Schedule Agent Run",
          category: .productivity,
          description: "Schedule a background agent run. Args: title, prompt, schedule, optional inMinutes/atTime/intervalSeconds/beforeMinutes.",
          icon: "alarm",
          tint: "orange",
          requiresApproval: true,
          permissionKey: nil
        )
      ]
    }
    '''
    manifest = AgentBehaviorManifest()
    ToolDefinitionExtractor().extract(SwiftFile(Path("ToolDefinition.swift"), "ToolDefinition.swift", text), manifest)
    tool = manifest.tools[0]
    assert tool.permissionKey is None
    assert tool.permissionKind == "notifications"
    assert tool.confirmationMode == "userApproval"
    assert tool.description == "Schedule a background agent run. Args: title, prompt, schedule, optional inMinutes/atTime/intervalSeconds/beforeMinutes."
    assert [arg.name for arg in tool.arguments] == ["title", "prompt", "schedule", "inMinutes", "atTime", "intervalSeconds", "beforeMinutes"]
    assert {arg.name: arg.required for arg in tool.arguments} == {
        "title": True,
        "prompt": True,
        "schedule": True,
        "inMinutes": False,
        "atTime": False,
        "intervalSeconds": False,
        "beforeMinutes": False,
    }
    assert {arg.name: arg.type for arg in tool.arguments} == {
        "title": "string",
        "prompt": "string",
        "schedule": "string",
        "inMinutes": "number",
        "atTime": "string",
        "intervalSeconds": "number",
        "beforeMinutes": "number",
    }


def test_args_contract_handles_optional_group_and_type_hints():
    text = '''
    enum ToolRegistry {
      static let all: [ToolDefinition] = [
        ToolDefinition(
          id: "alarm.schedule",
          name: "Schedule Alarm",
          description: "Schedule an AlarmKit alarm. Args: title, inMinutes, optional timestamp/repeats/snoozeMinutes.",
          requiresApproval: true,
          permissionKey: "NSAlarmKitUsageDescription"
        ),
        ToolDefinition(
          id: "alarm.cancel",
          name: "Cancel Alarm",
          description: "Cancel a scheduled alarm. Args: id UUID.",
          requiresApproval: true,
          permissionKey: "NSAlarmKitUsageDescription"
        )
      ]
    }
    '''
    manifest = AgentBehaviorManifest()
    ToolDefinitionExtractor().extract(SwiftFile(Path("ToolDefinition.swift"), "ToolDefinition.swift", text), manifest)

    alarm_schedule = next(tool for tool in manifest.tools if tool.id == "alarm.schedule")
    assert [(arg.name, arg.type, arg.required) for arg in alarm_schedule.arguments] == [
        ("title", "string", True),
        ("inMinutes", "number", True),
        ("timestamp", "string", False),
        ("repeats", "bool", False),
        ("snoozeMinutes", "number", False),
    ]

    alarm_cancel = next(tool for tool in manifest.tools if tool.id == "alarm.cancel")
    assert [(arg.name, arg.type) for arg in alarm_cancel.arguments] == [
        ("id", "string"),
    ]


def test_args_contract_ignores_boolean_value_hint_alias():
    text = '''
    enum ToolRegistry {
      static let all: [ToolDefinition] = [
        ToolDefinition(
          id: "outlook.folders.list",
          name: "List Outlook Folders",
          description: "List folders. Args: optional includeHidden true/false.",
          requiresApproval: false,
          permissionKey: nil
        )
      ]
    }
    '''
    manifest = AgentBehaviorManifest()
    ToolDefinitionExtractor().extract(SwiftFile(Path("ToolDefinition.swift"), "ToolDefinition.swift", text), manifest)
    tool = manifest.tools[0]
    assert [(arg.name, arg.type, arg.required) for arg in tool.arguments] == [
        ("includeHidden", "string", False),
    ]


def test_tool_argument_contract_catalog_overrides_description_inference():
    text = '''
    private nonisolated enum ToolArgumentContractCatalog {
      static func arguments(for toolID: String) -> [ToolArgumentDefinition] {
        switch ToolRouteGuard.canonicalToolID(toolID) {
        case "outlook.folders.list":
          return [.init("includeHidden", type: .bool, required: false)]
        case "outlook.messages.list":
          return [
            .init("folder", required: false),
            .init("folderId", required: false),
            .init("limit", type: .number, required: false),
            .init("unreadOnly", type: .bool, required: false)
          ]
        default:
          return []
        }
      }
    }

    enum ToolRegistry {
      static let all: [ToolDefinition] = [
        ToolDefinition(
          id: "outlook.folders.list",
          name: "List Outlook Folders",
          description: "List folders. Args: optional includeHidden true/false.",
          requiresApproval: false,
          permissionKey: nil
        ),
        ToolDefinition(
          id: "outlook.messages.list",
          name: "List Outlook Messages",
          description: "List recent Outlook messages. Args: optional folder or folderId, limit, unreadOnly.",
          requiresApproval: false,
          permissionKey: nil
        )
      ]
    }
    '''
    manifest = AgentBehaviorManifest()
    ToolDefinitionExtractor().extract(SwiftFile(Path("ToolDefinition.swift"), "ToolDefinition.swift", text), manifest)

    folders = next(tool for tool in manifest.tools if tool.id == "outlook.folders.list")
    assert folders.confirmationMode == "none"
    assert [(arg.name, arg.type, arg.required) for arg in folders.arguments] == [
        ("includeHidden", "bool", False),
    ]

    messages = next(tool for tool in manifest.tools if tool.id == "outlook.messages.list")
    assert [(arg.name, arg.type, arg.required) for arg in messages.arguments] == [
        ("folder", "string", False),
        ("folderId", "string", False),
        ("limit", "number", False),
        ("unreadOnly", "bool", False),
    ]
    assert "false" not in {arg.name for tool in manifest.tools for arg in tool.arguments}
    assert all(not (arg.description or "").startswith("Inferred") for tool in manifest.tools for arg in tool.arguments)
