from lumen_manifest_crawler.swift_extractors.agent_json_value import AgentJSONValueExtractor
from lumen_manifest_crawler.swift_extractors.intent_router import IntentRouterExtractor
from lumen_manifest_crawler.swift_extractors.memory import MemoryExtractor
from lumen_manifest_crawler.swift_extractors.mimicry import MimicryExtractor
from lumen_manifest_crawler.swift_extractors.model_fleet import ModelFleetExtractor
from lumen_manifest_crawler.swift_extractors.rem_cycle import RemCycleExtractor
from lumen_manifest_crawler.swift_extractors.sentinels import SentinelExtractor
from lumen_manifest_crawler.swift_extractors.tool_definition import ToolDefinitionExtractor

ALL_EXTRACTORS = [
    ToolDefinitionExtractor(),
    IntentRouterExtractor(),
    ModelFleetExtractor(),
    AgentJSONValueExtractor(),
    MemoryExtractor(),
    MimicryExtractor(),
    RemCycleExtractor(),
    SentinelExtractor(),
]
