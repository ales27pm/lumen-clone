from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from lumen_manifest_crawler.manifest import (
    AgentBehaviorManifest,
    AppManifestInfo,
    FleetTopologyManifest,
    FleetTopologySlotManifest,
    RoutingMatrixEntry,
    SourceFileHash,
)
from lumen_manifest_crawler.output.hashing import normalized_repo_path, sha256_file
from lumen_manifest_crawler.swift_extractors import ALL_EXTRACTORS
from lumen_manifest_crawler.swift_extractors.base import SwiftFile

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git",
    ".build",
    ".swiftpm",
    "DerivedData",
    "Pods",
    "node_modules",
    ".expo",
    ".next",
    "build",
    "dist",
    "generated",
}
IGNORED_PLIST_DIRS = IGNORED_DIRS | {"Tests", "UITests", "UnitTests", "Fixtures", "fixtures"}
FRESHNESS_DEFAULTS = {
    "volatile": {"ttlSeconds": 300, "durable": False},
    "shortLived": {"ttlSeconds": 3600, "durable": False},
    "short_lived": {"ttlSeconds": 3600, "durable": False},
    "timeless": {"ttlSeconds": None, "durable": True},
}


def generate_manifest(root: Path) -> AgentBehaviorManifest:
    root = _validated_scan_root(root)
    manifest = AgentBehaviorManifest(app=_read_app_info(root))
    manifest.sourceIntegrity.commit = _git_commit(root)

    swift_files = list(_iter_swift_files(root))
    manifest.sourceIntegrity.files = [
        SourceFileHash(path=normalized_repo_path(root, path), sha256=sha256_file(path))
        for path in swift_files
        if _is_source_of_truth_file(path)
    ]

    for path in swift_files:
        rel = normalized_repo_path(root, path)
        swift_file = SwiftFile(path=path, relpath=rel, text=path.read_text(encoding="utf-8", errors="replace"))
        for extractor in ALL_EXTRACTORS:
            if extractor.accepts(swift_file):
                extractor.extract(swift_file, manifest)

    _finalize_defaults(manifest)
    return manifest


def _validated_scan_root(root: Path) -> Path:
    resolved = root.resolve()
    if resolved == resolved.anchor:
        raise ValueError(
            "Refusing to scan the filesystem root. Pass the repository root explicitly, "
            "for example: --root /content/Lumen"
        )
    if not resolved.exists():
        raise ValueError(f"Manifest crawler root does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Manifest crawler root is not a directory: {resolved}")
    return resolved


def _iter_swift_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            if filename.endswith(".swift"):
                yield Path(dirpath) / filename


def _is_source_of_truth_file(path: Path) -> bool:
    return path.name in {
        "ModelFleet.swift",
        "IntentRouter.swift",
        "ToolDefinition.swift",
        "AgentJSONValue.swift",
        "MimicryProfile.swift",
        "RemCycleService.swift",
        "MemoryItem.swift",
        "MemoryStore.swift",
        "MemoryContextItem.swift",
        "ChatView.swift",
        "AgentService.swift",
        "Trigger.swift",
        "AlarmTools.swift",
    }


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() or None
    except subprocess.CalledProcessError as error:
        logger.debug(
            "git rev-parse HEAD failed in _git_commit: returncode=%s stdout=%r stderr=%r",
            error.returncode,
            error.stdout,
            error.stderr,
        )
        return None
    except Exception as error:
        logger.debug("git rev-parse HEAD failed in _git_commit: %s", error, exc_info=True)
        return None


def _read_app_info(root: Path) -> AppManifestInfo:
    selected = _select_info_plist(root)
    if not selected:
        return AppManifestInfo(name="Lumen")
    text = selected.read_text(encoding="utf-8", errors="replace")
    return AppManifestInfo(
        name="Lumen",
        bundleIdentifier=_plist_value_after_key(text, "CFBundleIdentifier"),
        buildVersion=_plist_value_after_key(text, "CFBundleShortVersionString"),
    )


def _select_info_plist(root: Path) -> Path | None:
    preferred: Path | None = None
    fallback: Path | None = None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_PLIST_DIRS)
        if "Info.plist" not in filenames:
            continue
        plist = Path(dirpath) / "Info.plist"
        text = plist.read_text(encoding="utf-8", errors="replace")
        if "CFBundleIdentifier" not in text:
            continue
        if fallback is None:
            fallback = plist
        if preferred is None and any(part in {"Lumen", "ios"} for part in plist.relative_to(root).parts):
            preferred = plist
            break
    return preferred or fallback


def _plist_value_after_key(text: str, key: str) -> str | None:
    marker = f"<key>{key}</key>"
    index = text.find(marker)
    if index == -1:
        return None
    rest = text[index + len(marker): index + len(marker) + 400]
    start = rest.find("<string>")
    end = rest.find("</string>")
    if start == -1 or end == -1 or end <= start:
        return None
    return rest[start + len("<string>"):end].strip()


def _finalize_defaults(manifest: AgentBehaviorManifest) -> None:
    if not manifest.sentinels.forbiddenInUserOutput:
        manifest.sentinels.forbiddenInUserOutput = sorted({
            "<user_final_text>",
            "<private_reasoning>",
            "<tool_json>",
            "<internal_state>",
            "<scratchpad>",
            "<hidden_reasoning>",
        })
    if not manifest.agentProtocols.executorOutput.get("supportedJSONTypes"):
        manifest.agentProtocols.executorOutput["supportedJSONTypes"] = ["string", "double", "int", "bool", "array", "object", "null"]

    _apply_freshness_defaults(manifest)

    known_tools = sorted({tool.id for tool in manifest.tools})
    if manifest.intents and not manifest.routingMatrix:
        manifest.routingMatrix = [
            RoutingMatrixEntry(
                intent=intent.id,
                allowedTools=sorted(intent.allowedToolIDs),
                forbiddenTools=[tool_id for tool_id in known_tools if tool_id not in intent.allowedToolIDs][:25],
            )
            for intent in manifest.intents
        ]
    manifest.fleetTopology = _build_fleet_topology(manifest)


def _apply_freshness_defaults(manifest: AgentBehaviorManifest) -> None:
    for freshness in manifest.memory.freshnessClasses:
        defaults = FRESHNESS_DEFAULTS.get(freshness.id)
        if not defaults:
            continue
        if freshness.ttlSeconds is None and defaults["ttlSeconds"] is not None:
            freshness.ttlSeconds = int(defaults["ttlSeconds"])
        if not freshness.durable:
            freshness.durable = bool(defaults["durable"])


def _build_fleet_topology(manifest: AgentBehaviorManifest) -> FleetTopologyManifest:
    slots = sorted(manifest.fleet.slots, key=lambda slot: slot.id)
    slot_ids = [slot.id for slot in slots]
    topology_slots: dict[str, FleetTopologySlotManifest] = {}
    calls_by_slot = {slot.id: _infer_slot_calls(slot.id, slot_ids) for slot in slots}
    called_by: dict[str, list[str]] = {slot.id: [] for slot in slots}
    for caller, callees in calls_by_slot.items():
        for callee in callees:
            called_by.setdefault(callee, []).append(caller)

    memory_scopes = sorted(manifest.memory.scopes)
    for slot in slots:
        role = slot.role or slot.id
        topology_slots[slot.id] = FleetTopologySlotManifest(
            role=role,
            purpose=_slot_purpose(slot.id, role, slot.responsibilities),
            inputSignature=_slot_input_signature(slot.id, role),
            outputSignature=_slot_output_signature(slot.id, role),
            calls=calls_by_slot.get(slot.id, []),
            calledBy=sorted(called_by.get(slot.id, [])),
            memoryScopes=memory_scopes,
            responsibilities=sorted(slot.responsibilities),
        )
    return FleetTopologyManifest(
        slots=topology_slots,
        externalHandoffTools=_infer_handoff_tools(manifest),
    )


def _infer_slot_calls(slot_id: str, all_slot_ids: list[str]) -> list[str]:
    lowered = slot_id.lower()
    wanted: set[str] = set()
    for candidate in all_slot_ids:
        candidate_lower = candidate.lower()
        if candidate == slot_id:
            continue
        if any(token in lowered for token in ["router", "cortex", "intent", "planner"]):
            wanted.add(candidate)
        elif any(token in lowered for token in ["executor", "tool"]):
            if any(token in candidate_lower for token in ["mouth", "response", "rem", "memory", "reflection"]):
                wanted.add(candidate)
        elif any(token in lowered for token in ["rem", "memory", "reflection"]):
            if any(token in candidate_lower for token in ["cortex", "router", "executor", "tool"]):
                wanted.add(candidate)
        elif any(token in lowered for token in ["mimicry", "style"]):
            if any(token in candidate_lower for token in ["mouth", "response"]):
                wanted.add(candidate)
    return sorted(wanted)


def _infer_handoff_tools(manifest: AgentBehaviorManifest) -> list[str]:
    names = {tool.id for tool in manifest.tools}
    return sorted(tool_id for tool_id in names if any(token in tool_id.lower() for token in ["delegate", "handoff", "route", "intent", "slot", "model"]))


def _slot_purpose(slot_id: str, role: str, responsibilities: list[str]) -> str:
    if responsibilities:
        return responsibilities[0]
    lowered = f"{slot_id} {role}".lower()
    if any(token in lowered for token in ["cortex", "router", "intent", "planner"]):
        return "Classify user goals, select allowed tools or slots, and coordinate the Lumen agent fleet."
    if any(token in lowered for token in ["executor", "tool"]):
        return "Convert approved plans into exact manifest-valid tool calls and structured action results."
    if any(token in lowered for token in ["mouth", "response"]):
        return "Produce the final user-facing response without leaking internal sentinels or tool payloads."
    if any(token in lowered for token in ["mimicry", "style"]):
        return "Adapt user-facing tone, language, and formatting without changing facts or safety boundaries."
    if any(token in lowered for token in ["rem", "memory", "reflection"]):
        return "Reflect on failures, memory policy, runtime drift, and dataset repair actions."
    return f"Perform the {role} role defined by the Lumen model fleet contract."


def _slot_input_signature(slot_id: str, role: str) -> str:
    lowered = f"{slot_id} {role}".lower()
    if any(token in lowered for token in ["cortex", "router", "intent", "planner"]):
        return "User request plus manifest routing matrix, available tools, memory context, and prior agent state."
    if any(token in lowered for token in ["executor", "tool"]):
        return "Approved Cortex plan, exact tool ID, argument candidates, permission state, and approval state."
    if any(token in lowered for token in ["mouth", "response"]):
        return "Tool result, user-visible state, style profile, and safety/sentinel constraints."
    if any(token in lowered for token in ["mimicry", "style"]):
        return "Draft response, user style hints, locale, tone preferences, and safety constraints."
    if any(token in lowered for token in ["rem", "memory", "reflection"]):
        return "Audit failure, conversation summary, memory candidates, freshness policy, and manifest context."
    return "Role-specific input defined by the fleet contract and AgentBehaviorManifest."


def _slot_output_signature(slot_id: str, role: str) -> str:
    lowered = f"{slot_id} {role}".lower()
    if any(token in lowered for token in ["cortex", "router", "intent", "planner"]):
        return "Intent classification, selectedToolID or target slot, approval requirement, and concise routing summary."
    if any(token in lowered for token in ["executor", "tool"]):
        return "Strict JSON tool call, clarification request, approval request, or structured tool result."
    if any(token in lowered for token in ["mouth", "response"]):
        return "Final user-facing text with no private reasoning, raw tool JSON, or forbidden sentinels."
    if any(token in lowered for token in ["mimicry", "style"]):
        return "Style-adjusted user-facing text or style profile that preserves facts and safety boundaries."
    if any(token in lowered for token in ["rem", "memory", "reflection"]):
        return "Repair sample, memory decision, freshness classification, or runtime drift recommendation."
    return "Role-specific output defined by the fleet contract and AgentBehaviorManifest."
