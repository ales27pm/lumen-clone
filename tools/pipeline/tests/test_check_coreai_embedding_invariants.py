"""Negative source mutations for local, explicit, identity-safe embeddings."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("coreai_embedding_gate", ROOT / "tools/check_coreai_embedding_invariants.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


@pytest.fixture
def source(tmp_path):
    for relative in GATE.REQUIRED_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_current_embedding_contract_passes(source):
    assert GATE.check(source) == []


def test_generic_function_body_is_checked_instead_of_default_closures():
    swift = 'func replace<Value: Sendable>(begin: () -> Void = { ignored() }) { checked() }'
    assert GATE.function(swift, "replace").strip() == "checked()"


@pytest.mark.parametrize("file,scope,before,after,expected", (
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "withModelReplacement", "try Task.checkCancellation()", "", "check cancellation before admission and after draining"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "withModelReplacement", "do {\n            try Task.checkCancellation()", "do {", "check cancellation before admission and after draining"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "restoreSelection", "selectionState.snapshot == expected", "true", "compare exact selection revision"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "commit", "revision: UUID()", "revision: revision", "change its revision even for ABA"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingService.swift", "beginModelReplacement", "guard condition() else", "guard true else", "before unloading the current model"),
    ("Services/ModelRuntimeController.swift", "preservingEmbeddingSelection", "guard isCurrent() else { throw error }", "", "only its current IDs"),
    ("Services/ModelRuntimeController.swift", "preservingEmbeddingSelection", "attemptedSelection = try await chooseLegacy()", "attemptedSelection = readSelection()", "own its exact legacy selection revision"),
    ("Services/MemoryStore.swift", "updateContent", "item.embeddingModelIdentifier = nil", "", "source edits must invalidate"),
    ("Services/MemoryStore.swift", "persist", "throw PersistenceFailure.diskWriteDeferred", "return", "write admission denial must fail"),
    ("Services/RAGStore.swift", "stageFileVectors", "catch is CancellationError", "catch is SomeOtherError", "cancellation must stop staged RAG"),
))
def test_deep_failure_boundaries_reject_unsafe_mutations(source, file, scope, before, after, expected):
    path = source / "ios/Lumen" / file
    swift = path.read_text(encoding="utf-8")
    body = GATE.function(swift, scope)
    assert body and before in body, (file, scope, before)
    path.write_text(swift.replace(body, body.replace(before, after, 1), 1), encoding="utf-8")
    failures = GATE.check(source)
    assert any(expected in failure for failure in failures), failures


@pytest.mark.parametrize("file,before,after", (
    ("Views/CoreAITextEmbeddingSettingsSection.swift", "@Environment(AppState.self) private var appState", "@EnvironmentObject private var appState: AppState"),
    ("State/AppState.swift", "@Observable\nfinal class AppState", "final class AppState"),
    ("LumenApp.swift", ".environment(appState)", ".environmentObject(appState)"),
))
def test_embedding_settings_requires_scene_observation_contract(source, file, before, after):
    target = source / "ios/Lumen" / file
    text = target.read_text(encoding="utf-8")
    assert before in text, f"Mutation anchor changed: {file}: {before}"
    target.write_text(text.replace(before, after, 1), encoding="utf-8")
    assert any("AppState Observation" in error for error in GATE.check(source))


@pytest.mark.parametrize("file,before,after", (
    ("Views/ModelsView.swift", "runtimeController.preservingEmbeddingSelection(", "runtimeController.someOtherOperation("),
    ("Services/ModelRuntimeController.swift", "readSelection() == $0", "true"),
    ("Services/ModelRuntimeController.swift", "try await restoreSelection(previousSelection.selected, attemptedSelection)", "try? await restoreSelection(previousSelection.selected, attemptedSelection)"),
))
def test_failed_gguf_load_must_restore_embedding_provider_selection(source, file, before, after):
    target = source / "ios/Lumen" / file
    text = target.read_text(encoding="utf-8")
    assert before in text, f"Mutation anchor changed: {file}: {before}"
    target.write_text(text.replace(before, after, 1), encoding="utf-8")
    assert any("embedding selection transaction" in error for error in GATE.check(source))


def test_missing_service_fails_closed(source):
    (source / GATE.SERVICE).unlink()
    assert any("missing embedding source" in error for error in GATE.check(source))


def test_comments_and_string_literals_do_not_become_runtime_calls(source):
    service = source / GATE.SERVICE
    service.write_text(service.read_text() + '\n// URLSession.shared and CoreAITextEmbeddingService.shared.prepare()\nlet example = "CoreAITextEmbeddingService.shared.prepare()"\n')
    assert GATE.check(source) == []


def test_new_background_prepare_caller_is_rejected(source):
    background = source / "ios/Lumen/Background/UnsafeEmbeddingPreparation.swift"
    background.parent.mkdir(parents=True, exist_ok=True)
    background.write_text("func maintenance() async throws { try await CoreAITextEmbeddingService.shared.prepare() }")
    assert any("outside explicit foreground seams" in error for error in GATE.check(source))


@pytest.mark.parametrize("file,before,after,expected", (
    ("Assistant/AssistantRuntimeRouter.swift", "return .init(runtime: .unavailable, reason: decision.denialReason", "return .init(runtime: .llama, reason: decision.denialReason", "must not silently select another provider"),
    ("Assistant/AssistantRuntimeRouter.swift", "guard coreAIEmbedding.isAvailable else", "guard true else", "compute and availability"),
    ("Services/ModelLoader.swift", "guard !Task.isCancelled, intent == .userChat || intent == .userVoice", "guard !Task.isCancelled, intent == .background || intent == .userVoice", "explicit foreground chat or voice"),
    ("Services/ModelLoader.swift", '            return\n        }\n        guard let selectedID = snapshot.activeEmbeddingModelID', '        }\n        guard let selectedID = snapshot.activeEmbeddingModelID', "must not fall through to GGUF"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingService.swift", "_ = try CoreAITextEmbeddingPolicy.formattedText(request)", "try await prepare()\n        _ = try CoreAITextEmbeddingPolicy.formattedText(request)", "require preparation rather than load"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingService.swift", "guard isLoaded() else", "guard true else", "reject an unprepared runtime"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingService.swift", "import Foundation", "import Foundation\nimport CloudKit", "remain local and non-generative"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "import Foundation", "import Foundation\nfunc unsafeFetch() { _ = URLSession.shared }", "remain local and non-generative"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingService.swift", "for index in 0..<bucket { p[index] = 0 }", "for index in 0..<bucket { p[index] = 1 }", "zero all multimodal"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", GATE.PINNED_FILES["model/embeddinggemma-2-text-mm.aimodel/main.mlirb"][1], "0" * 64, "inventory/hash pins changed"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingPolicy.swift", GATE.REVISION, "0" * 40, "revision must remain immutable"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "payloadPaths == expectedFiles.map(\\.path)", "true", "reject extra multimodal"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "io.inventory(root) == identity.files.map(\\.path)", "true", "inventory must be verified"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "try io.digest(relative: expected.path, descriptor: descriptor) == expected", "true", "hashes must be verified"),
    ("Services/LLM/CoreAITextEmbeddingModelStore.swift", "identity.files == expectedFiles", "true", "identity must match the immutable inventory"),
    ("Services/LLM/CoreAILocalPackageIO.swift", "guard url.isFileURL else", "guard true else", "reject non-file URLs"),
    ("Services/LLM/CoreAILocalPackageIO.swift", "values.isUbiquitousItem != true", "true", "reject cloud-backed resources"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingPolicy.swift", "case .legacyFormatted: throw CoreAITextEmbeddingError.invalidRequest", 'case .legacyFormatted: prefix = ""', "reject already formatted legacy"),
    ("Services/LLM/CoreAI/CoreAITextEmbeddingPolicy.swift", ':mrl-\\(dimensions)', ':mrl-256', "requested projection dimension"),
    ("Services/MemoryStore.swift", "purpose: .retrievalQuery", "", "lost explicit purpose"),
    ("Services/MemoryStore.swift", "SemanticEmbeddingText.queryBody(trimmed)", "SemanticEmbeddingText.query(trimmed)", "raw text with explicit purpose"),
    ("Services/MemoryStore.swift", "embeddingModelIdentifier: embeddingResult.modelIdentifier", "embeddingModelIdentifier: nil", "writes must persist actual provider"),
    ("Services/RAGStore.swift", "formatVersion: result.formatVersion", "formatVersion: SemanticEmbeddingText.formatVersion", "provider's actual embedding format"),
    ("RAG/RAGIndexer.swift", "formatVersion: embeddingResult.formatVersion", "formatVersion: SemanticEmbeddingText.formatVersion", "provider's actual embedding format"),
    ("Services/MemoryStore.swift", "ResourceBudgetGate.diagnosticSnapshot().scenePhase == .active", "true", "requires an explicit foreground operation"),
    ("Services/MemoryStore.swift", "embeddingResult.modelIdentifier == metadata.modelIdentifier", "true", "stop on changed provider identity"),
    ("Models/MemoryItem.swift", "var embeddingModelIdentifier: String? = nil", 'var embeddingModelIdentifier: String = "current-model"', "optional and unknown"),
    ("Services/VectorIndex.swift", "item.embeddingModelIdentifier == metadata.modelIdentifier", "true", "reject incompatible identity/format/dimension"),
    ("Services/VectorIndex.swift", "item.embeddingDimension == metadata.dimension", "true", "reject incompatible identity/format/dimension"),
    ("Services/VectorIndex.swift", "loadedMetadata != metadata", "false", "cache must invalidate"),
    ("Assistant/SemanticEmbeddingText.swift", "CoreAITextEmbeddingService.shared.isLoaded()", "CoreAITextEmbeddingService.shared.prepare()", "discovery must not load or run inference"),
))
def test_unsafe_embedding_mutations_fail(source, file, before, after, expected):
    target = source / "ios/Lumen" / file
    text = target.read_text(encoding="utf-8")
    assert before in text, f"Mutation anchor changed: {file}: {before}"
    target.write_text(text.replace(before, after, 1), encoding="utf-8")
    assert any(expected in error for error in GATE.check(source)), GATE.check(source)


def test_extra_pinned_vision_weight_is_rejected_even_with_a_valid_digest_shape(source):
    target = source / GATE.STORE
    text = target.read_text()
    anchor = "static let pinnedFiles: [FileIdentity] = ["
    assert anchor in text
    extra = '\n        .init(path: "model/vision/main.mlirb", bytes: 1024, sha256: "' + "f" * 64 + '"),'
    target.write_text(text.replace(anchor, anchor + extra, 1))
    assert any("inventory/hash pins changed" in error for error in GATE.check(source))


def test_commented_availability_gate_cannot_satisfy_selected_provider_check(source):
    target = source / "ios/Lumen/Assistant/AssistantRuntimeRouter.swift"
    text = target.read_text()
    anchor = "guard coreAIEmbedding.isAvailable else"
    assert anchor in text
    target.write_text(text.replace(anchor, "/* guard coreAIEmbedding.isAvailable else */ guard true else", 1))
    assert any("compute and availability" in error for error in GATE.check(source))
