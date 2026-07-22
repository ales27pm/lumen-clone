#!/usr/bin/env python3
"""Static drift guard for Lumen's Qwen3 adapter runtime.

This script intentionally avoids importing app modules. It scans source files for
architecture invariants that must remain true after the Qwen3 shared-base + LoRA
adapter migration.

It is not a replacement for Xcode/device validation. It is a fast guard against
regressing to the slow five-full-GGUF runtime shape.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_FAMILY_SELECTION = ROOT / "ios/Lumen/Services/ModelFamilySelection.swift"
MODEL_ADAPTER_RUNTIME_CONTRACT = ROOT / "ios/Lumen/Services/ModelAdapterRuntimeContract.swift"
MODEL_FLEET = ROOT / "ios/Lumen/Services/ModelFleet.swift"
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
    require(
        "fallbackFamily: LumenModelFamily? = selectedFamily == .qwen3 ? nil : selectedFamily" in text,
        "Qwen3 fallback path must not label non-Qwen3 fallback assignments as Qwen3.",
    )
    require(
        "if text.contains(slotToken) { return (model, 2) }" in text,
        "Adapter ranking must prefer exact role adapter matches.",
    )
    require(
        "if slot == .cortex, text.contains(\"fleet\") { return (model, 1) }" in text,
        "Fleet adapter fallback must rank below exact cortex adapter matches.",
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
        "func activateRoleAdapter(slot: LumenModelSlot, scale: Float) throws" in text,
        "AdapterChatRuntime must own a single activation entry point.",
    )
    runtime_activation = section_after_marker(text, "func activateRoleAdapter(slot: LumenModelSlot, scale: Float) throws")
    runtime_activation = runtime_activation.split("func clearAdapters()", 1)[0]
    require_in_order(
        runtime_activation,
        ["clearAdapters()", "guard let adapter = loadedAdapters[slot]", "try context.apply(loraAdapter: adapter, scale: scale)"],
        "AdapterChatRuntime must clear all LoRA adapters before applying the requested one.",
    )
    service_activation = section_after_marker(text, "func activateRoleAdapter(slot: LumenModelSlot) async throws")
    service_activation = service_activation.split("func activateRoleAdapterIfNeeded", 1)[0]
    require_in_order(
        service_activation,
        ["try await runtime.activateRoleAdapter(slot: loaded.slot, scale: loaded.scale)", "activeAdapterSlot = slot"],
        "AppLlamaService must set activeAdapterSlot only after successful adapter application.",
    )
    require(
        "await runtime.clearAdapters()" in service_activation and "lastAdapterFailureReason = error.localizedDescription" in service_activation,
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
    require("unloadAllChat" not in adapter_section, "Qwen3 adapter slot switch must not call unloadAllChat().")
    require("unloadRoleAdapter(slot: slot)" in adapter_section, "Failed role adapter activation must unload the failed adapter handle.")
    require(
        "requiresRoleAdapter(assignment: assignment)" in adapter_section
        and 'throw LocalRuntimeError.unavailable("role adapter missing for \\(slot.rawValue): expectedAdapterRepo=\\(expectedRepo); expectedAdapterFile=\\(expectedFile)")' in adapter_section,
        "Qwen3 adapter-required slots must fail hard when their role adapter is missing.",
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
