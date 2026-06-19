from pathlib import Path
from textwrap import dedent

from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolManifest
from lumen_manifest_crawler.swift_extractors.base import SwiftFile
from lumen_manifest_crawler.swift_extractors.intent_router import IntentRouterExtractor


def test_intent_router_extracts_intents_and_tools():
    text = dedent('''
    enum UserIntent {
      case localSearch
      case webLookup
    }
    struct IntentRouter {
      func tools(for intent: UserIntent) -> [String] {
        switch intent {
        case .localSearch: return ["maps.search"]
        case .webLookup: return ["web.search"]
        }
      }
    }
    ''')
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="maps.search"), ToolManifest(id="web.search")])
    IntentRouterExtractor().extract(SwiftFile(Path("IntentRouter.swift"), "IntentRouter.swift", text), manifest)
    intents = {intent.id: set(intent.allowedToolIDs) for intent in manifest.intents}
    assert intents["localSearch"] == {"maps.search"}
    assert intents["webLookup"] == {"web.search"}
    assert len(manifest.routingMatrix) == 2


def test_intent_router_resolves_tool_id_collection_references():
    text = dedent('''
    enum UserIntent {
      case reminder
      case maps
      case chat
    }
    struct IntentRouter {
      private static let reminderToolIDs: Set<String> = ["reminders.create", "reminders.list"]
      private static let mapsToolIDs: Set<String> = ["maps.search", "maps.directions", "location.current"]

      static func allowedToolIDs(for intent: UserIntent) -> Set<String> {
        switch intent {
        case .reminder: return reminderToolIDs
        case .maps: return mapsToolIDs
        case .chat: return []
        }
      }
    }
    ''')
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(id="reminders.create"),
            ToolManifest(id="reminders.list"),
            ToolManifest(id="maps.search"),
            ToolManifest(id="maps.directions"),
            ToolManifest(id="location.current"),
        ]
    )
    IntentRouterExtractor().extract(SwiftFile(Path("IntentRouter.swift"), "IntentRouter.swift", text), manifest)
    intents = {intent.id: set(intent.allowedToolIDs) for intent in manifest.intents}
    assert intents["reminder"] == {"reminders.create", "reminders.list"}
    assert intents["maps"] == {"maps.search", "maps.directions", "location.current"}
    assert intents["chat"] == set()
