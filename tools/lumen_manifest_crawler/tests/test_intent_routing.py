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
