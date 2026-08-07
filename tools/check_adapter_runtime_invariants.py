#!/usr/bin/env python3
"""Static drift guard for Lumen's Qwen3 adapter runtime.

This script intentionally avoids importing app modules. It scans source files for
architecture invariants that must remain true after the Qwen3 shared-base + LoRA
adapter migration.

It is not a replacement for Xcode/device validation. It is a fast guard against
regressing to the slow five-full-GGUF runtime shape.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_FAMILY_SELECTION = ROOT / "ios/Lumen/Services/ModelFamilySelection.swift"
MODEL_ADAPTER_RUNTIME_CONTRACT = ROOT / "ios/Lumen/Services/ModelAdapterRuntimeContract.swift"
MODEL_FLEET = ROOT / "ios/Lumen/Services/ModelFleet.swift"
MODEL_LOADER = ROOT / "ios/Lumen/Services/ModelLoader.swift"
MODEL_RUNTIME_CONTROLLER = ROOT / "ios/Lumen/Services/ModelRuntimeController.swift"
LLAMA_SERVICE = ROOT / "ios/Lumen/Services/LlamaService.swift"
SLOT_COORDINATOR = ROOT / "ios/Lumen/Services/SlotModelRuntimeCoordinator.swift"
MODELS_VIEW = ROOT / "ios/Lumen/Views/ModelsView.swift"
ASSISTANT_RUNTIME_ADAPTERS = ROOT / "ios/Lumen/Assistant/AssistantRuntimeAdapters.swift"
ASSISTANT_RUNTIME_ROUTER = ROOT / "ios/Lumen/Assistant/AssistantRuntimeRouter.swift"
RUNTIME_DASHBOARD_VIEW = ROOT / "ios/Lumen/Views/RuntimeDashboardView.swift"
RUNTIME_DIAGNOSTICS_SNAPSHOT = ROOT / "ios/Lumen/Diagnostics/RuntimeDiagnosticsSnapshot.swift"
DIAGNOSTICS_PROVIDER = ROOT / "ios/Lumen/Diagnostics/DiagnosticsProvider.swift"
MEMORY_STORE = ROOT / "ios/Lumen/Services/MemoryStore.swift"
RAG_STORE = ROOT / "ios/Lumen/Services/RAGStore.swift"
EXPORT_GGUF = ROOT / "tools/fine_tuning/unsloth/export_gguf.py"
DOC = ROOT / "docs/ADAPTER_RUNTIME_IMPROVE_LOOP.md"
TERMINAL_LOOP = ROOT / "tools/lumen_terminal_improve_loop.py"
TRAIN_SFT = ROOT / "tools/fine_tuning/unsloth/train_sft.py"
PBXPROJ = ROOT / "ios/Lumen.xcodeproj/project.pbxproj"
HARDENING_DOC = ROOT / "docs/HARDENING_IOS_LORA_ADAPTER_RUNTIME.md"
QWEN3_CONFIG_DIR = ROOT / "tools/fine_tuning/unsloth/configs_qwen3_bootstrap"

EXPECTED_ADAPTERS = {
    "lumen-cortex-lora.gguf",
    "lumen-executor-lora.gguf",
    "lumen-mouth-lora.gguf",
    "lumen-mimicry-lora.gguf",
    "lumen-rem-lora.gguf",
    "lumen-fleet-lora.gguf",
}
EXPECTED_ZEROGPU_RUN_ID = "20260706T011546Z"

RELEASE_BAKE_DEFAULTS = {
    "lumen-cortex-release-bake-q4_k_m.gguf",
    "lumen-executor-release-bake-q4_k_m.gguf",
    "lumen-mouth-release-bake-q4_k_m.gguf",
    "lumen-mimicry-release-bake-q4_k_m.gguf",
    "lumen-rem-release-bake-q4_k_m.gguf",
}


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_in_order(text: str, markers: list[str], message: str) -> None:
    position = -1
    for marker in markers:
        next_position = text.find(marker, position + 1)
        if next_position < 0:
            fail(f"{message}: missing marker {marker!r}")
        if next_position <= position:
            fail(message)
        position = next_position


def section_after_marker(text: str, marker: str) -> str:
    index = text.find(marker)
    require(index >= 0, f"missing marker: {marker}")
    return text[index:]


def check_catalog() -> None:
    catalog = section_after_marker(read(MODEL_FAMILY_SELECTION), "static var qwen3BootstrapModels")
    contract_source = read(MODEL_ADAPTER_RUNTIME_CONTRACT)
    contract = section_after_marker(contract_source, "static let qwen3AdapterBootstrapContract")

    require(
        "qwen3AdapterBootstrapContract" in catalog
        and "contract.sharedBaseFileName" in catalog
        and contract.count('sharedBaseFileName: "lumen-qwen3-fast-shared-q4_k_m.gguf"') == 1,
        "Qwen3 default catalog must contain exactly one shared chat base filename.",
    )
    for adapter in sorted(EXPECTED_ADAPTERS):
        require(adapter in contract, f"Qwen3 default catalog missing adapter: {adapter}")
        require(
            f'qwen3AdapterSourcePath("{adapter}")' in contract,
            f"Qwen3 adapter contract missing run-scoped source path: {adapter}",
        )
    require(f'private static let qwen3ZeroGPURunID = "{EXPECTED_ZEROGPU_RUN_ID}"' in contract_source, "Qwen3 adapter contract must pin the current ZeroGPU run id.")
    for release_bake in sorted(RELEASE_BAKE_DEFAULTS):
        require(
            release_bake not in catalog and release_bake not in contract,
            f"Qwen3 default catalog must not include release-bake artifact: {release_bake}",
        )
    require(
        'roleID: "fleet"' in contract
        and "lumen-fleet-lora.gguf" in contract
        and "sourcePath: adapter.adapterSourcePath" in catalog
        and "contract.adapterRoles.map" in catalog
        and "role: .roleAdapter" in catalog,
        "Fleet adapter must be represented as a role adapter, not by abusing embedding slot metadata.",
    )
    for expected in (
        "trainRecordCount: 9573",
        "trainRecordCount: 591",
        "trainRecordCount: 4770",
        "validationRecordCount: 1689",
        "validationRecordCount: 104",
        "validationRecordCount: 842",
    ):
        require(expected in contract, f"Qwen3 adapter contract missing current ZeroGPU dataset count: {expected}")


def check_fleet_resolver() -> None:
    text = read(MODEL_FLEET)
    contract = read(MODEL_ADAPTER_RUNTIME_CONTRACT)
    require(
        "LumenModelSelectionPolicy.isChatModelCompatible" in text
        and "else if selectedFamily != .qwen3, let activeText" in text,
        "Qwen3 selection must fail closed instead of relabeling an incompatible active chat model.",
    )
    require(
        text.count("guard let role = contract.adapterRole(for: slot) else { return nil }") >= 2
        and text.count("$0.repoId == role.adapterRepoID && $0.fileName == role.adapterFileName") >= 2,
        "Qwen3 adapters must resolve by exact contract repository and filename in both resolver paths.",
    )
    require(
        "if let exactFile = storedModels.filter" not in text
        and "if text.contains(slotToken) { return (model, 2) }" not in text
        and "if slot == .cortex, text.contains(\"fleet\") { return (model, 1) }" not in text,
        "Qwen3 adapter resolution must not accept filename-only or hint-ranked fallback artifacts.",
    )
    require(
        "configuredRoleAdapterMatchesContract" in text
        and "adapterExpectedSHA256.caseInsensitiveCompare(expected.adapterExpectedSHA256)" in text,
        "Qwen3 assignments must bind adapter repository, filename, size, and SHA-256 to the runtime contract.",
    )
    require(
        "sizeBytes: Int64" in contract
        and "expectedSHA256: String?" in contract
        and "sizeBytes == sharedBaseSizeBytes" in contract
        and "CatalogModel.isValidSHA256(expectedSHA256)" in contract,
        "Qwen3 shared-base identity must include exact size and a valid SHA-256.",
    )
    require(
        "trustedExpectedSHA256" in text
        and "configuredSharedBaseMatchesContract" in text
        and "sharedBaseContract?.sharedBaseSizeBytes" in text
        and "sharedBaseContract?.sharedBaseExpectedSHA256" in text,
        "Qwen3 resolver assignments must derive and bind trusted shared-base integrity metadata.",
    )


def check_runtime() -> None:
    text = read(LLAMA_SERVICE)
    require("private actor AdapterChatRuntime" in text, "AdapterChatRuntime must remain actor-isolated.")
    require(
        "nonisolated static func defaultGenerationSlot(for request: GenerateRequest) -> LumenModelSlot" in text
        and "request.preservesRawStructuredAgentOutput ? .executor : .mouth" in text,
        "Implicit generation must route raw structured output to Executor and plain text to Mouth.",
    )
    require(
        "stream(req, slot: Self.defaultGenerationSlot(for: req))" in text,
        "The no-slot stream overload must use the explicit implicit-generation slot policy.",
    )
    require(
        "func activateRoleAdapter(slot: LumenModelSlot, scale: Float, operationGeneration: UInt64) throws -> Bool" in text,
        "AdapterChatRuntime must own a single activation entry point.",
    )
    runtime_activation = section_after_marker(text, "func activateRoleAdapter(slot: LumenModelSlot, scale: Float, operationGeneration: UInt64) throws -> Bool")
    runtime_activation = runtime_activation.split("func clearAdapters(operationGeneration:", 1)[0]
    require_in_order(
        runtime_activation,
        [
            "guard claimAdapterActivation(generation: operationGeneration)",
            "guard activeAdapterSlot != slot || activeAdapterScale != scale else { return false }",
            "clearAdaptersUnconditionally()",
            "guard let adapter = loadedAdapters[slot]",
            "try context.apply(loraAdapter: adapter, scale: scale)",
        ],
        "AdapterChatRuntime must claim activation ownership before its fast path and requested adapter application.",
    )
    service_activation = section_after_marker(text, "func activateRoleAdapter(slot: LumenModelSlot) async throws")
    service_activation = service_activation.split("func activateRoleAdapterIfNeeded", 1)[0]
    require_in_order(
        service_activation,
        [
            "let activationGeneration = beginAdapterActivation()",
            "let activated = try await runtime.activateRoleAdapter(",
            "operationGeneration: activationGeneration",
            "guard activationGeneration == adapterActivationGeneration",
            "activeAdapterSlot = slot",
        ],
        "AppLlamaService activation fast paths and post-await publication must remain newest-wins.",
    )
    require(
        "await runtime.clearAdapters(operationGeneration: activationGeneration)" in service_activation
        and "lastAdapterFailureReason = error.localizedDescription" in service_activation,
        "Adapter activation failure must clear adapters, reset state, and persist lastAdapterFailureReason.",
    )
    require("let isLast = index == lastIndex" in text, "Prompt decode must mark the final prompt token for logits.")
    require("outputTokenCount: nil" in text, "outputTokenCount must stay nil unless real token counts are threaded through.")
    require(
        "sanitized.split(whereSeparator: { $0.isWhitespace }).count" not in text,
        "Do not report whitespace word count as outputTokenCount.",
    )
    require(
        "func embed(text: String, dimensions: Int = 256) async -> [Double]" not in text
        and "return []\n        }\n    }\n\n    private func loadChatModelSync" not in text,
        "Embedding failures must use the throwing embed(_:) API, not log and return an empty vector.",
    )
    require(
        "adapterApplied" in text and "adapterSlot" in text and "adapterFailureReason" in text,
        "Runtime trace metadata must include adapterApplied, adapterSlot, and adapterFailureReason.",
    )

    shared_load = section_after_marker(text, "func loadSharedChatModel(path: String")
    shared_load = shared_load.split("func loadRoleAdapter(slot:", 1)[0]
    require_in_order(
        shared_load,
        [
            "let operationGeneration = beginSharedChatOperation()",
            "if sharedChatBasePath == path, sharedChatRuntime != nil { return }",
            "let diagnostics = await runtime.runtimeAccelerationDiagnostics()",
            "guard ownsSharedChatOperation(operationGeneration)",
            "sharedChatRuntime = runtime",
        ],
        "Shared-chat same-path fast return and post-await publication must remain newest-wins.",
    )

    role_load = section_after_marker(text, "func loadRoleAdapter(slot: LumenModelSlot, path: String")
    role_load = role_load.split("func loadRoleAdapterIfNeeded", 1)[0]
    require_in_order(
        role_load,
        [
            "let operationGeneration = beginRoleAdapterOperation(slot: slot)",
            "let activationGeneration = beginAdapterActivation()",
            "let loadedNow = try await runtime.loadRoleAdapter(",
            "operationGeneration: operationGeneration",
            "activationGeneration: activationGeneration",
            "guard ownsRoleAdapterOperation(",
            "roleAdapters[slot] = LoadedRoleAdapter",
        ],
        "Role-adapter same-path fast return and post-await publication must remain newest-wins.",
    )

    load_if_needed = section_after_marker(text, "func loadRoleAdapterIfNeeded")
    load_if_needed = load_if_needed.split("func activateRoleAdapter(slot:", 1)[0]
    require(
        "if roleAdapters[slot]?.path == path { return false }" not in load_if_needed
        and "try await loadRoleAdapter(slot: slot, path: path, scale: scale)" in load_if_needed,
        "Role-adapter load-if-needed must delegate same-path ownership registration to loadRoleAdapter.",
    )

    activate_if_needed = section_after_marker(text, "func activateRoleAdapterIfNeeded")
    activate_if_needed = activate_if_needed.split("func clearActiveRoleAdapter", 1)[0]
    require(
        "if activeAdapterSlot == slot { return false }" not in activate_if_needed
        and "try await activateRoleAdapter(slot: slot)" in activate_if_needed,
        "Role-adapter activate-if-needed must delegate same-slot ownership registration to activateRoleAdapter.",
    )

    clear_adapter = section_after_marker(text, "func clearActiveRoleAdapter() async")
    clear_adapter = clear_adapter.split("func unloadRoleAdapter(slot:", 1)[0]
    require_in_order(
        clear_adapter,
        [
            "let activationGeneration = beginAdapterActivation()",
            "await runtime.clearAdapters(operationGeneration: activationGeneration)",
            "guard activationGeneration == adapterActivationGeneration",
            "activeAdapterSlot = nil",
        ],
        "Adapter clear must guard post-await state publication with its activation generation.",
    )

    runtime_role_load = section_after_marker(text, "func loadRoleAdapter(\n        slot: LumenModelSlot,")
    runtime_role_load = runtime_role_load.split("func activateRoleAdapter(slot:", 1)[0]
    require_in_order(
        runtime_role_load,
        [
            "guard claimRoleAdapterOperation(slot: slot, generation: operationGeneration)",
            "guard claimAdapterActivation(generation: activationGeneration)",
            "guard loadedAdapterPaths[slot] != path else { return false }",
            "loadedAdapters[slot] = adapter",
            "loadedAdapterPaths[slot] = path",
        ],
        "AdapterChatRuntime must claim role-load ownership before its same-path fast return and mutation.",
    )

    conditional_adapter_unload = section_after_marker(text, "func unloadRoleAdapter(slot: LumenModelSlot, ifPathEquals")
    conditional_adapter_unload = conditional_adapter_unload.split("func unloadAllRoleAdapters", 1)[0]
    require_in_order(
        conditional_adapter_unload,
        [
            "let operationGeneration = beginRoleAdapterOperation(slot: slot)",
            "let activationGeneration = beginAdapterActivation()",
            "guard currentPath == expectedPath else",
            "await runtime.discardRoleAdapterIfPathDiffers(",
            "await runtime.unloadRoleAdapter(",
            "guard ownsRoleAdapterOperation(",
            "roleAdapters.removeValue(forKey: slot)",
        ],
        "Conditional role-adapter unload must register before its fast return and guard destruction.",
    )

    runtime_conditional_discard = section_after_marker(text, "func discardRoleAdapterIfPathDiffers(")
    runtime_conditional_discard = runtime_conditional_discard.split("func activateRoleAdapter(slot:", 1)[0]
    require_in_order(
        runtime_conditional_discard,
        [
            "guard claimRoleAdapterOperation(slot: slot, generation: operationGeneration)",
            "guard claimAdapterActivation(generation: activationGeneration)",
            "guard loadedAdapterPaths[slot] != expectedPath else { return false }",
            "return removeRoleAdapter(slot: slot)",
        ],
        "Conditional role-adapter no-op must claim both epochs and discard a mismatched hidden handle.",
    )

    conditional_chat_unload = section_after_marker(text, "func unloadAllChat(ifLoadedPathEquals")
    conditional_chat_unload = conditional_chat_unload.split("func unloadEmbed", 1)[0]
    require_in_order(
        conditional_chat_unload,
        [
            "beginSharedChatOperation()",
            "guard loadedChatPath == expectedPath else { return false }",
            "sharedChatRuntime = nil",
        ],
        "Conditional shared-chat unload must register before its mismatch fast return.",
    )


def check_staged_system_adapters() -> None:
    adapters = read(ASSISTANT_RUNTIME_ADAPTERS)
    router = read(ASSISTANT_RUNTIME_ROUTER)
    dashboard = read(RUNTIME_DASHBOARD_VIEW)
    runtime_snapshot = read(RUNTIME_DIAGNOSTICS_SNAPSHOT)
    diagnostics_provider = read(DIAGNOSTICS_PROVIDER)
    require(
        "let supportsGeneration: Bool = false" in adapters
        and "FoundationModels generation is experimental and is excluded from Release routing." in adapters,
        "FoundationModels adapter must remain explicitly experimental until generation is implemented.",
    )
    require(
        "let supportsEmbeddings: Bool = false" in adapters
        and "CoreML embedding runtime is experimental and is excluded from Release routing." in adapters
        and "throw CoreMLRuntimeError.experimentalRuntimeDisabled" in adapters,
        "CoreML embedding adapter must remain explicitly experimental and must throw instead of returning fake embeddings.",
    )
    require(
        "foundation.supportsGeneration, foundation.isAvailable" in router,
        "Runtime router must not select FoundationModels unless generation support is implemented.",
    )
    require(
        "coreML.supportsEmbeddings, coreML.isAvailable" in router,
        "Runtime router must not select CoreML embeddings unless embedding extraction is implemented.",
    )
    require(
        "foundationModelsStatus" in dashboard and "coreMLStatus" in dashboard and "Unavailable" in dashboard,
        "Runtime diagnostics UI must expose staged/unavailable adapter status.",
    )
    require(
        "struct AssistantRuntimeCapabilityMatrix" in adapters
        and "generationSelectable" in adapters
        and "embeddingSelectable" in adapters
        and "runtimeCapabilityRows: [AssistantRuntimeCapabilityRow]" in runtime_snapshot
        and "AssistantRuntimeCapabilityMatrix.current()" in diagnostics_provider
        and "runtimeCapabilityRows: capabilityMatrix.rows" in diagnostics_provider
        and "Runtime Capabilities" in dashboard,
        "Runtime diagnostics must expose a capability matrix for selectable generation and embedding runtimes.",
    )
    require(
        "struct AssistantRuntimeCapabilityMatrix" in adapters
        and "generationSelectable: foundation.supportsGeneration && foundation.isAvailable" in adapters
        and "embeddingSelectable: coreML.supportsEmbeddings && coreML.isAvailable" in adapters,
        "Runtime adapter capability matrix must make experimental generation/embedding non-selectability explicit.",
    )
    require(
        re.search(
            r"allowDiagnosticFallbackSelection:\s*Bool\s*=\s*Self\.defaultAllowDiagnosticFallbackSelection[\s\S]{0,800}"
            r"defaultAllowDiagnosticFallbackSelection[\s\S]{0,240}#if DEBUG[\s\S]{0,120}return true[\s\S]{0,120}#else[\s\S]{0,120}return false",
            router,
        )
        is not None,
        "Runtime router must make deterministic diagnostic fallback non-selectable by default in Release.",
    )
    require(
        "Diagnostic deterministic runtime is excluded from Release routing." in adapters
        and "\"excluded from Release routing\"" in adapters,
        "Deterministic fallback must be an excluded Release runtime, not a production assistant backend.",
    )
    require(
        "runtimeCapabilityRows" in dashboard and "Runtime Capabilities" in dashboard,
        "Runtime diagnostics UI must expose the runtime capability matrix.",
    )


def check_persistence_search_diagnostics() -> None:
    memory = read(MEMORY_STORE)
    rag = read(RAG_STORE)
    require(
        "lexical_fetch_failed:" in memory
        and "fetch_failed:" in memory
        and "combinedDiagnostic(primary:" in memory,
        "Memory recall diagnostics must surface semantic and lexical fetch failures instead of returning empty results silently.",
    )
    require(
        "lexical_fetch_failed:" in rag
        and "semantic_fetch_failed:" in rag
        and "rag_fetch_failed op=resolveVectorCandidates" in rag
        and "rag_fetch_failed op=lexicalSearch" in rag
        and "combinedDiagnostic(primary:" in rag,
        "RAG search diagnostics must surface semantic and lexical fallback fetch failures instead of returning empty results silently.",
    )
    require(
        "guard let availableItems = try? context.fetch(FetchDescriptor<MemoryItem>()) else { return [] }" not in memory,
        "Memory lexical recall must not collapse SwiftData fetch failure to an empty result.",
    )
    require(
        "guard let all = try? context.fetch(descriptor) else { return [] }" not in rag,
        "RAG lexical search must not collapse SwiftData fetch failure to an empty result.",
    )
    require(
        "(try? context.fetch(FetchDescriptor<RAGChunk>())) ?? []" not in rag,
        "RAG semantic candidate resolution must not collapse SwiftData fetch failure to an empty result.",
    )


def check_swift_llama_pin() -> None:
    text = read(PBXPROJ)
    require(
        'repositoryURL = "https://github.com/pgorzelany/swift-llama-cpp.git";' in text,
        "swift-llama-cpp package URL drifted.",
    )
    swift_ref = text[text.find('repositoryURL = "https://github.com/pgorzelany/swift-llama-cpp.git";'):]
    swift_ref = swift_ref.split('/* XCRemoteSwiftPackageReference "microsoft-authentication-library-for-objc" */', 1)[0]
    require("kind = exactVersion;" in swift_ref and "version = 1.2.0;" in swift_ref, "swift-llama-cpp must remain pinned to exactVersion 1.2.0.")


def check_slot_coordinator() -> None:
    text = read(SLOT_COORDINATOR)
    adapter_section_match = re.search(r"private func ensureAdapterRuntimeReady[\s\S]+?private func ensureLegacyRuntimeReady", text)
    require(adapter_section_match is not None, "Missing ensureAdapterRuntimeReady/ensureLegacyRuntimeReady split.")
    adapter_section = adapter_section_match.group(0)
    adapter_switch_section = adapter_section.split("guard let adapterPath", 1)[-1]
    require("unloadAllChat" not in adapter_switch_section, "Qwen3 adapter slot switch must not unload the verified shared base.")
    require(
        "await AppLlamaService.shared.unloadRoleAdapter(slot: slot, ifPathEquals: adapterPath)" in adapter_section
        and adapter_section.count("guard assignmentRemainsOwned(assignment, slot: slot, generation: generation)") >= 4,
        "Failed role adapter activation must conditionally unload only the still-owned failed adapter handle.",
    )
    require(
        "requiresRoleAdapter(assignment: assignment)" in adapter_section
        and 'throw LocalRuntimeError.unavailable("role adapter missing for \\(slot.rawValue): expectedAdapterRepo=\\(expectedRepo); expectedAdapterFile=\\(expectedFile)")' in adapter_section,
        "Qwen3 adapter-required slots must fail hard when their role adapter is missing.",
    )
    require(
        "assignment.expectedRoleAdapterContract != nil" in adapter_section
        and "!assignment.configuredRoleAdapterMatchesContract" in adapter_section
        and "role_adapter_identity_mismatch" in adapter_section,
        "Qwen3 adapter activation must fail closed when contract identity or integrity metadata differs.",
    )
    require(
        "configuredSharedBaseMatchesContract" in adapter_section
        and "shared_base_identity_mismatch" in adapter_section
        and 'role: "chat"' in adapter_section,
        "Qwen3 shared-base activation must fail closed and verify the contract-bound artifact.",
    )
    require(
        "loadedChatFallback" not in text
        and "return assignments[.cortex]" not in text
        and "if slot == .mouth" not in text,
        "An already-loaded or Cortex chat runtime must not satisfy an unassigned role slot.",
    )


def check_model_loader_ownership() -> None:
    text = read(MODEL_LOADER)
    require(
        "let epoch: UInt64" in text
        and "registerChatRequest" in text
        and "registerEmbedRequest" in text
        and "chatLoadEpoch == epoch" in text
        and "embedLoadEpoch == epoch" in text,
        "Chat and embedding load completion must be guarded by monotonic request epochs.",
    )
    chat_section = text.split("private static func ensureFleetChatLoaded", 1)[1].split("private static func completeChatLoad", 1)[0]
    require_in_order(
        chat_section,
        ["if let pending = chatLoadTask", "if await hasLoadedChatRuntime"],
        "Chat must reconcile a mismatched pending load before the already-loaded fast path",
    )
    embed_section = text.split("static func ensureEmbedLoaded", 1)[1].split("private static func finishEmbedLoad", 1)[0]
    require_in_order(
        embed_section,
        ["if let pending = embedLoadTask", "if await hasLoadedEmbeddingRuntime"],
        "Embedding must reconcile a mismatched pending load before the already-loaded fast path",
    )
    chat_completion = text.split("private static func completeChatLoad", 1)[1].split("private static func configureFleetRuntimeIfSelectionIsCurrent", 1)[0]
    embed_completion = text.split("private static func completeEmbeddingLoad", 1)[1].split("private static func embeddingRequestIsCurrent", 1)[0]
    require(
        "unloadAllChat" not in chat_completion and "unloadEmbed" not in embed_completion,
        "A stale load waiter must not globally unload a newer chat or embedding runtime.",
    )


def check_models_view() -> None:
    text = read(MODELS_VIEW)
    require(
        "isAdapter: sm.modelRole == .roleAdapter" in text,
        "Downloaded model rows must know when a row is a role adapter.",
    )
    require(
        "catalog.role == .roleAdapter" in text,
        "Featured model cards must treat role adapters as non-activatable adapter artifacts.",
    )
    require(
        "stored.modelRole != .roleAdapter" in text,
        "Stored role adapters must not be activatable as chat/embedding models.",
    )
    require(
        "LumenModelSelectionPolicy.validatePersistedChatModel" in text
        and "LumenModelSelectionPolicy.isPersistedChatModelCompatible" in text,
        "Model activation and picker surfaces must share the family-aware chat selection policy.",
    )
    require(
        "runtimeController.unloadResolvedModel" in text
        and "cannot be activated while" in text,
        "Model deletion must unload a matching shared runtime and incompatible local imports must explain why they were not activated.",
    )
    controller = read(MODEL_RUNTIME_CONTROLLER)
    require(
        "loadedSharedPath == resolvedPath" in controller
        and "return .allChat" in controller
        and "await AppLlamaService.shared.unloadAllChat()" in controller,
        "Deleting the loaded shared chat base must plan and perform a full chat-runtime unload.",
    )


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    require(function is not None, f"export_gguf.py missing {name}()")
    return function


def _assigned_value(function: ast.FunctionDef, name: str) -> ast.expr | None:
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return node.value
    return None


def check_export_policy() -> None:
    text = read(EXPORT_GGUF)
    require("--release-bake" in text, "export_gguf.py must require explicit --release-bake for merged GGUF export.")
    require("Skipped GGUF release bake by default" in text, "export_gguf.py must skip release-bake by default.")
    require(
        "merge_adapters_by_default" in text and "release_bake_enabled_by_default" in text,
        "export_gguf.py must validate adapter-first config flags.",
    )
    require(
        'config_path = root / f"{agent}.final.json"' in text
        and "The exporter will not fall back" in text,
        "Release bake must require prepared final configs and never fall back to SFT configs.",
    )
    require(
        'PREPARED_RUN_ROOT_ENV = "LUMEN_AIO_RUN_ROOT"' in text
        and "guess a recent run or use checked-in generated configs" in text,
        "Release-bake config discovery must bind an exact prepared run root.",
    )
    tree = ast.parse(text, filename=str(EXPORT_GGUF))
    resolver = _function_node(tree, "_resolve_config_dir")
    gather = _function_node(tree, "gather_configs")
    main = _function_node(tree, "main")

    run_root_value = _assigned_value(resolver, "run_root")
    require(
        run_root_value is not None
        and ast.unparse(run_root_value)
        == "os.environ.get(PREPARED_RUN_ROOT_ENV, '').strip()",
        "Release-bake resolver must read PREPARED_RUN_ROOT_ENV into run_root.",
    )
    require(
        any(
            isinstance(node, ast.Return)
            and node.value is not None
            and ast.unparse(node.value) == "str(Path(run_root) / 'configs')"
            for node in ast.walk(resolver)
        ),
        "Release-bake resolver must return <prepared-run-root>/configs by default.",
    )

    resolved_dir = _assigned_value(main, "config_dir")
    loaded_configs = _assigned_value(main, "configs")
    require(
        isinstance(resolved_dir, ast.Call)
        and ast.unparse(resolved_dir.func) == "_resolve_config_dir",
        "Exporter main must assign config_dir from _resolve_config_dir().",
    )
    require(
        isinstance(loaded_configs, ast.Call)
        and ast.unparse(loaded_configs.func) == "gather_configs"
        and len(loaded_configs.args) >= 2
        and ast.unparse(loaded_configs.args[1]) == "config_dir",
        "Exporter main must pass the resolved config_dir directly to gather_configs().",
    )

    gather_root = _assigned_value(gather, "root")
    release_branch = next(
        (
            node
            for node in ast.walk(gather)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "require_release_bake_lineage"
        ),
        None,
    )
    release_nodes = (
        [nested for statement in release_branch.body for nested in ast.walk(statement)]
        if release_branch is not None
        else []
    )
    require(
        gather_root is not None
        and ast.unparse(gather_root) == "Path(config_dir).resolve()",
        "gather_configs must derive its root from the resolved config_dir.",
    )
    require(
        any(
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "config_path"
            and ast.unparse(node.value) == "root / f'{agent}.final.json'"
            for node in release_nodes
        ),
        "Release-bake gather branch must use <resolved-config-dir>/<agent>.final.json.",
    )


def check_docs() -> None:
    text = read(DOC)
    require("Non-negotiable runtime invariant" in text, "Adapter runtime doctrine doc missing invariant section.")
    require("Default Qwen3 runtime must not" in text, "Adapter runtime doctrine doc must explicitly forbid five full role GGUFs.")
    require("Improve-loop drift checks" in text, "Adapter runtime doctrine doc must include improve-loop drift checks.")


def check_hardening_doc() -> None:
    text = read(HARDENING_DOC)
    required_phrases = [
        "Poison and Antidote",
        "PR 171",
        "Jetsam",
        "single-adapter",
        "swift-llama-cpp",
        "exact version `1.2.0`",
        "Fleet adapter",
        "llama_adapter_lora_init",
        "LlamaContext.removeAllLoraAdapters()",
        "12-step real-device smoke test",
        "adapterApplied=true",
    ]
    for phrase in required_phrases:
        require(phrase in text, f"Hardening guide missing required context: {phrase}")


def check_terminal_loop() -> None:
    text = read(TERMINAL_LOOP)
    require(
        "hf\", \"repos\", \"create\"" in text or '"repos", "create"' in text,
        "Terminal improve-loop must call 'hf repos create' (current Hugging Face CLI), not the legacy 'hf repo create'.",
    )
    require(
        "hf repo create" not in text,
        "Terminal improve-loop must not reference the legacy 'hf repo create' subcommand.",
    )
    require(
        "--base-model-id" in text and "--base" in text,
        "Terminal improve-loop convert stage must thread an explicit --base / --base-model-id to convert_lora_to_gguf.py.",
    )
    require(
        "_resolve_base_for_convert" in text,
        "Terminal improve-loop must validate base model resolution before invoking the LoRA→GGUF converter.",
    )
    require(
        "pipeline_state.json" in text and "--resume" in text and "--state-file" in text,
        "Terminal improve-loop must support resumable pipeline_state.json with --resume / --state-file.",
    )
    require(
        "configs_qwen3_bootstrap" in text,
        "Terminal improve-loop must default to the Qwen3 bootstrap config dir.",
    )
    require(
        "validate_qwen3_configs" in text,
        "Terminal improve-loop must include a strict Qwen3 config validator (no Qwen2.x base in the bootstrap dir).",
    )


def check_qwen3_configs_alignment() -> None:
    require(QWEN3_CONFIG_DIR.exists(), f"missing Qwen3 bootstrap config dir: {QWEN3_CONFIG_DIR.relative_to(ROOT)}")
    forbidden = ("qwen2", "qwen-2")
    for cfg_path in sorted(QWEN3_CONFIG_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"invalid JSON in {cfg_path.relative_to(ROOT)}: {exc}")
        base = str(cfg.get("base_model_name", "")).lower()
        require(
            "qwen3" in base,
            f"{cfg_path.relative_to(ROOT)}: base_model_name must reference Qwen3 (got '{cfg.get('base_model_name')}').",
        )
        for token in forbidden:
            require(
                token not in base,
                f"{cfg_path.relative_to(ROOT)}: base_model_name still references a pre-Qwen3 family ('{cfg.get('base_model_name')}').",
            )
        require(
            cfg.get("merge_adapters_by_default", False) is False,
            f"{cfg_path.relative_to(ROOT)}: merge_adapters_by_default must remain false (adapter-first).",
        )
        require(
            cfg.get("release_bake_enabled_by_default", False) is False,
            f"{cfg_path.relative_to(ROOT)}: release_bake_enabled_by_default must remain false (adapter-first).",
        )


def check_train_sft_reproducibility() -> None:
    text = read(TRAIN_SFT)
    require("--seed" in text, "train_sft.py must accept --seed for reproducibility.")
    require(
        "--resume-from-checkpoint" in text,
        "train_sft.py must accept --resume-from-checkpoint for resumable training.",
    )
    require(
        "--assistant-only-loss" in text or "assistant_only_loss" in text,
        "train_sft.py must support assistant-only loss for instruction-tuning.",
    )
    require(
        "train_manifest.json" in text,
        "train_sft.py must write a train_manifest.json with reproducibility metadata.",
    )


def main() -> int:
    checks = [
        check_catalog,
        check_fleet_resolver,
        check_runtime,
        check_staged_system_adapters,
        check_persistence_search_diagnostics,
        check_swift_llama_pin,
        check_slot_coordinator,
        check_model_loader_ownership,
        check_models_view,
        check_export_policy,
        check_docs,
        check_hardening_doc,
        check_terminal_loop,
        check_qwen3_configs_alignment,
        check_train_sft_reproducibility,
    ]
    failures: list[str] = []
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as exc:  # noqa: BLE001 - command-line checker should report every failure
            failures.append(f"FAIL {check.__name__}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("All Qwen3 adapter runtime invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
