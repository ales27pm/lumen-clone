#!/usr/bin/env python3
"""Source policy for optional local text embeddings; not Apple execution evidence.

Checks safety boundaries and immutable artifact identity. This deliberately does
not claim to type-check Swift, evaluate arbitrary control flow, or qualify ANE
execution. Runtime, fresh-index rebuild and device tests remain separate acceptance layers.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = "ios/Lumen/"
SERVICE = BASE + "Services/LLM/CoreAI/CoreAITextEmbeddingService.swift"
STORE = BASE + "Services/LLM/CoreAITextEmbeddingModelStore.swift"
POLICY = BASE + "Services/LLM/CoreAI/CoreAITextEmbeddingPolicy.swift"
SETTINGS = BASE + "Views/CoreAITextEmbeddingSettingsSection.swift"
APP_STATE = BASE + "State/AppState.swift"
APP_SCENE = BASE + "LumenApp.swift"
MODEL_CONTROLLER = BASE + "Services/ModelRuntimeController.swift"
MODELS_VIEW = BASE + "Views/ModelsView.swift"
REQUIRED_FILES = (
    SERVICE, STORE, POLICY, SETTINGS, APP_STATE, APP_SCENE, MODEL_CONTROLLER, MODELS_VIEW,
    BASE + "Services/LLM/CoreAILocalPackageIO.swift",
    BASE + "Assistant/AssistantRuntimeProtocols.swift",
    BASE + "Assistant/AssistantRuntimeRouter.swift",
    BASE + "Assistant/AssistantKernel.swift",
    BASE + "Assistant/CoreAITextEmbeddingRuntimeAdapter.swift",
    BASE + "Assistant/SemanticEmbeddingText.swift",
    BASE + "Models/MemoryItem.swift",
    BASE + "Models/RAGChunk.swift",
    BASE + "Services/MemoryStore.swift",
    BASE + "Services/RAGStore.swift",
    BASE + "Services/VectorIndex.swift",
    BASE + "Services/EmbeddingVectorReset.swift",
    BASE + "Services/EmbeddingIndexLifecycle.swift",
    BASE + "Services/EmbeddingSourceRebuild.swift",
    BASE + "Services/RAGEmbeddingRebuilder.swift",
    BASE + "Services/ModelLoader.swift",
    BASE + "Memory/MemoryCaptureQueue.swift",
    BASE + "RAG/RAGIndexer.swift",
    BASE + "RAG/EmbeddingBatcher.swift",
)

# Published CHECKSUMS.sha256 at the immutable export revision. Changing any
# member requires an intentional new qualification, not accepting local drift.
PINNED_FILES = {
    "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    "NOTICE": (21485, "b9ac552f2eb418f79dc23cc325a7b8c38ae05c1fcc8c1880014bae281a0963a6"),
    "model/MODEL-ATTRIBUTION.json": (5166, "7951556c4559c8609cf67715acc53e58d8c70f7f62242494d7858406b1bbe2f7"),
    "model/config.json": (4643, "0204c8d4532f6930709c3c8eb7f7b25b54e4be42aa7b8c6c397fc17e872fb7bf"),
    "model/embeddinggemma-2-text-mm.aimodel/main.hash": (32, "ea089710e57b22972d6ab4d4b960b385394b81f333ba2f75b4fb4aac1d238e6f"),
    "model/embeddinggemma-2-text-mm.aimodel/main.mlirb": (632206233, "0b51638f2d17794825de21def6ab485d968fdf412853ea209617623a51f1bbde"),
    "model/embeddinggemma-2-text-mm.aimodel/metadata.json": (431, "8984123cfdb32ba8cf526a1f3a41571205475aff40b3f1d65494f476f115f8a6"),
    "model/tokenizer/tokenizer.json": (32170510, "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
    "model/tokenizer/tokenizer_config.json": (1599, "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"),
}
REVISION = "305fc2a96790d9343f38038aeca211dfca56d9a5"

_SWIFT_NONCODE = re.compile(r'/\*.*?\*/|//[^\n]*|\#*""".*?"""\#*|\#*"(?:\\.|[^"\\])*"\#*', re.S)


def code_only(source: str) -> str:
    """Mask comments/literals without changing source offsets for brace scans."""
    return _SWIFT_NONCODE.sub(lambda match: "".join("\n" if c == "\n" else " " for c in match[0]), source)


def paired_end(code: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(code)):
        if code[index] == opening:
            depth += 1
        elif code[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return -1


def function(source: str, name: str, occurrence: int = 0) -> str:
    code = code_only(source)
    matches = list(re.finditer(r"\bfunc\s+" + re.escape(name) + r"\s*(?:<[^<>]*>\s*)?\(", code))
    if not matches:
        return ""
    match = matches[occurrence]
    end_signature = paired_end(code, match.end() - 1, "(", ")")
    start = code.find("{", end_signature)
    end = paired_end(code, start, "{", "}") if start >= 0 else -1
    return source[start + 1:end] if end >= 0 else ""


def branch(source: str, condition: str) -> str:
    code = code_only(source)
    match = re.search(r"\bif\s+" + condition + r"\s*\{", code)
    if not match:
        return ""
    end = paired_end(code, match.end() - 1, "{", "}")
    return source[match.end():end] if end >= 0 else ""


def calls(source: str, name: str) -> list[str]:
    code = code_only(source)
    result = []
    for match in re.finditer(r"\." + re.escape(name) + r"\s*\(", code):
        end = paired_end(code, match.end() - 1, "(", ")")
        if end >= 0:
            result.append(code[match.end():end])
    return result


def check(root: Path) -> list[str]:
    failures: list[str] = []
    sources = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            failures.append(f"missing embedding source: {relative}")
        sources[relative] = path.read_text(encoding="utf-8") if path.is_file() else ""

    def require(source: str, pattern: str, message: str) -> None:
        if not re.search(pattern, source, re.S):
            failures.append(message)

    # This narrow source contract catches mismatched observation mechanisms;
    # it does not type-check Swift or prove that the scene launches.
    require(code_only(sources[APP_STATE]), r"@Observable\s+(?:final\s+)?class\s+AppState\b", "AppState Observation model must remain @Observable")
    require(code_only(sources[APP_SCENE]), r"\.environment\s*\(\s*appState\s*\)", "AppState Observation model must be injected through the scene environment")
    require(code_only(sources[SETTINGS]), r"@Environment\s*\(\s*AppState\.self\s*\)\s+(?:private\s+)?var\s+appState\b", "AppState Observation settings must read the scene's typed environment")

    load = code_only(function(sources[MODELS_VIEW], "load"))
    transaction = re.search(r"runtimeController\.preservingEmbeddingSelection\s*\(", load)
    args_end = paired_end(load, transaction.end() - 1, "(", ")") if transaction else -1
    body_start = load.find("{", args_end + 1) if args_end >= 0 else -1
    transaction_end = paired_end(load, body_start, "{", "}") if body_start >= 0 else -1
    protected_load = load[body_start + 1:transaction_end] if transaction_end >= 0 else ""
    require(protected_load, r"try\s+commitSelection\(sm\).*?try\s+await\s+runtimeController\.load\(sm", "embedding selection transaction must enclose GGUF selection and load")
    selection_transaction = code_only(function(sources[MODEL_CONTROLLER], "preservingEmbeddingSelection"))
    require(selection_transaction, r"if\s+role\s*==\s*\.embedding\s*\{\s*attemptedSelection\s*=\s*try\s+await\s+chooseLegacy\(\)", "embedding selection transaction must own its exact legacy selection revision")
    ownership = r"guard\s+isCurrent\(\),\s*attemptedSelection\.map\(\{\s*readSelection\(\)\s*==\s*\$0\s*\}\)"
    if len(re.findall(ownership, selection_transaction)) != 2:
        failures.append("embedding selection transaction must reject superseded work before and after loading")
    require(selection_transaction, r"catch\s*\{.*?guard\s+try\s+await\s+restoreSelection\(previousSelection\.selected,\s*attemptedSelection\)\s+else\s*\{\s*throw\s+error\s*\}.*?guard\s+isCurrent\(\)\s+else\s*\{\s*throw\s+error\s*\}.*?try\s+rollback\(\)", "embedding selection transaction must conditionally restore its provider and only its current IDs")

    router = function(sources[BASE + "Assistant/AssistantRuntimeRouter.swift"], "embeddingSelection")
    selected = branch(router, r"coreAIEmbedding\s*\.\s*isPreferred")
    require(code_only(selected), r"guard\s+decision\.allowHeavyRuntime.*?guard\s+coreAIEmbedding\.isAvailable", "selected CoreAI embeddings require compute and availability gates")
    require(code_only(selected), r"return\s+\.init\(runtime:\s*\.coreAI", "selected CoreAI embeddings must terminate in their own runtime")
    if re.search(r"runtime:\s*\.(?:llama|coreML|deterministicFallback|foundationModels)\b", code_only(selected)):
        failures.append("selected CoreAI embeddings must not silently select another provider")

    service, store, policy = (sources[path] for path in (SERVICE, STORE, POLICY))
    replacement = code_only(function(store, "withModelReplacement"))
    require(replacement, r"try\s+Task\.checkCancellation\(\).*?try\s+await\s+begin\(\).*?do\s*\{\s*try\s+Task\.checkCancellation\(\).*?try\s+await\s+operation\(\)", "model replacement must check cancellation before admission and after draining")
    require(code_only(function(store, "restoreSelection")), r"beginModelReplacement\(if:\s*\{\s*selectionState\.snapshot\s*==\s*expected", "provider rollback must compare exact selection revision under the replacement gate")
    require(code_only(function(service, "beginModelReplacement")), r"guard\s+condition\(\)\s+else\s*\{\s*return\s+false\s*\}.*?replacementInFlight\s*=\s*true.*?clearModel\(\)", "stale provider rollback must be rejected before unloading the current model")
    require(code_only(function(store, "commit")), r"let\s+committed\s*=\s*SelectionSnapshot\(selected:\s*selected,\s*revision:\s*UUID\(\)\).*?revision\s*=\s*committed\.revision.*?return\s+committed", "each provider selection must change its revision even for ABA choices")
    adapter = sources[BASE + "Assistant/CoreAITextEmbeddingRuntimeAdapter.swift"]
    for label, body in (("service embed", function(service, "embed")), ("service inference", function(service, "runEmbedding")), ("adapter", function(adapter, "embedWithIdentity"))):
        if re.search(r"\b(?:prepare|loadConfiguredModel)\s*\(|\bAIModel\s*\(", code_only(body)):
            failures.append(f"{label} must require preparation rather than load a model")
    require(code_only(function(service, "embed")), r"guard\s+isLoaded\(\)\s+else\s*\{\s*throw", "embedding inference must reject an unprepared runtime")
    require(code_only(function(service, "loadConfiguredModel")), r"ResourceBudgetGate\.allowsForegroundModelLoad", "embedding load requires foreground resource authorization")
    loader = function(sources[BASE + "Services/ModelLoader.swift"], "prepareSelectedMemoryEmbedding")
    require(code_only(loader), r"guard\s+.*?intent\s*==\s*\.userChat\s*\|\|\s*intent\s*==\s*\.userVoice.*?else\s*\{.*?return.*?CoreAITextEmbeddingService\.shared\.prepare\(", "embedding preparation must be restricted to explicit foreground chat or voice")
    loader_selected = branch(loader, r"CoreAITextEmbeddingModelStore\.isSelected")
    require(code_only(loader_selected), r"\breturn\s*$", "selected CoreAI preparation failure must not fall through to GGUF")
    allowed_loaders = {BASE + "Services/ModelLoader.swift", BASE + "Views/CoreAITextEmbeddingSettingsSection.swift"}
    for path in sorted((root / BASE).rglob("*.swift")):
        if path.relative_to(root).as_posix() not in allowed_loaders and re.search(r"CoreAITextEmbeddingService\s*\.\s*shared\s*\.\s*prepare\s*\(", code_only(path.read_text(encoding="utf-8"))):
            failures.append(f"embedding preparation outside explicit foreground seams: {path.relative_to(root)}")

    for relative in (SERVICE, STORE, BASE + "Services/LLM/CoreAILocalPackageIO.swift"):
        code = code_only(sources[relative])
        if re.search(r"\b(?:URLSession|URLRequest|CloudKit|CKContainer|WebSocket|CoreAIRunner|CoreAILanguageModels|UIImage|CGImage|AVAudioFile|AVAudioPCMBuffer)\b", code):
            failures.append(f"text embedding boundary must remain local and non-generative: {relative}")
    # The pinned graph has MM input ports, but the text-only call must zero both.
    inference = function(service, "runEmbedding")
    inference_code = code_only(inference)
    for name in ("multimodalEmbeddings", "multimodalMask"):
        match = re.search(re.escape(name) + r"\.mutableView.*?withUnsafeMutablePointer\s*\{", inference_code, re.S)
        end = paired_end(inference_code, match.end() - 1, "{", "}") if match else -1
        body = inference_code[match.end():end] if match and end >= 0 else ""
        require(body, r"(?:\[[^\]]+\]\s*=\s*0\b|(?:initialize|update)\(repeating:\s*0\s*,)", "text-only embeddings must zero all multimodal input buffers")

    entries = re.findall(r'\.init\(path:\s*"([^"]+)",\s*bytes:\s*([\d_]+),\s*sha256:\s*"([a-f0-9]{64})"\)', store)
    inventory = {name: (int(size.replace("_", "")), digest) for name, size, digest in entries}
    if inventory != PINNED_FILES or len(entries) != len(PINNED_FILES):
        failures.append("immutable text-only embedding package inventory/hash pins changed")
    require(policy, r'static\s+let\s+revision\s*=\s*"' + REVISION + '"', "embedding export revision must remain immutable")
    for field, path in (("graphDigest", "model/embeddinggemma-2-text-mm.aimodel/main.mlirb"), ("tokenizerDigest", "model/tokenizer/tokenizer.json"), ("configurationDigest", "model/config.json")):
        require(policy, r'static\s+let\s+' + field + r'\s*=\s*"' + PINNED_FILES[path][1] + '"', "embedding policy and validated package hashes must agree")
    require(code_only(function(store, "configuredPackage", -1)), r"io\.inventory\(root\)\s*==\s*identity\.files", "installed embedding inventory must be verified before load")
    require(code_only(function(store, "configuredPackage", -1)), r"io\.digest\(.*?\)\s*==\s*expected", "installed embedding hashes must be verified before load")
    require(code_only(function(store, "install")), r"(?:paths|payloadPaths)\s*==\s*expectedFiles.*?else\s*\{\s*throw", "embedding import must reject extra multimodal or unknown assets")
    require(code_only(function(store, "readIdentity")), r"identity\.files\s*==\s*expectedFiles", "installed embedding identity must match the immutable inventory")
    local_io = code_only(sources[BASE + "Services/LLM/CoreAILocalPackageIO.swift"])
    require(local_io, r"guard\s+url\.isFileURL\s+else\s*\{\s*throw", "embedding import must reject non-file URLs")
    require(local_io, r"values\.isUbiquitousItem\s*!=\s*true", "embedding import must reject cloud-backed resources")
    require(code_only(function(policy, "formattedText")), r"case\s+\.legacyFormatted\s*:\s*throw", "CoreAI embeddings must reject already formatted legacy input")
    require(function(policy, "modelIdentifier"), r"baseIdentifier.*?dimensions", "embedding space identity must include the requested projection dimension")

    contracts = sources[BASE + "Assistant/AssistantRuntimeProtocols.swift"]
    for field in (r"let\s+purpose\s*:\s*EmbeddingPurpose", r"let\s+dimensions\s*:\s*Int\?", r"let\s+formatVersion\s*:\s*Int"):
        require(code_only(contracts), field, "embedding contracts must preserve purpose, dimension, and format identity")
    consumer_paths = (BASE + "Services/MemoryStore.swift", BASE + "Services/RAGStore.swift", BASE + "RAG/RAGIndexer.swift", BASE + "RAG/EmbeddingBatcher.swift")
    for relative in consumer_paths:
        source = sources[relative]
        if re.search(r"SemanticEmbeddingText\.(?:query|memoryQuery|document|memoryDocument)\s*\(", code_only(source)):
            failures.append(f"retrieval consumers must pass raw text with explicit purpose: {relative}")
        for arguments in calls(source, "runEmbeddingWithIdentity") + calls(source, "runEmbedding"):
            if not re.search(r"\bpurpose\s*:", arguments):
                failures.append(f"embedding consumer lost explicit purpose: {relative}")
    for relative in (BASE + "Services/RAGStore.swift", BASE + "RAG/RAGIndexer.swift"):
        if re.search(r"formatVersion:\s*SemanticEmbeddingText\.formatVersion", code_only(sources[relative])):
            failures.append(f"RAG must persist the provider's actual embedding format: {relative}")
    memory = sources[BASE + "Services/MemoryStore.swift"]
    edited_memory = code_only(function(memory, "updateContent"))
    require(edited_memory, r"item\.embeddingModelIdentifier\s*=\s*nil.*?MemoryVectorIndex\.shared\.invalidate\(\)", "memory source edits must invalidate both vector identity and cache")
    persisted = code_only(function(memory, "persist"))
    require(persisted, r"guard\s+let\s+reservation\s*=\s*writeBudget\.reserveWrite.*?else\s*\{\s*throw", "memory write admission denial must fail rather than report an unsaved success")
    require(code_only(function(sources[BASE + "Services/RAGStore.swift"], "stageFileVectors")), r"catch\s+is\s+CancellationError\s*\{.*?return\s+IndexResult\(indexedCount:\s*0,\s*mode:\s*\.skipped", "provider cancellation must stop staged RAG replacement")
    require(function(memory, "recallWithDiagnostics"), r"formatVersion:\s*queryEmbedding\.formatVersion.*?modelIdentifier:\s*queryEmbedding\.modelIdentifier.*?dimension:\s*queryVec\.count", "memory query must carry actual provider identity/format/dimension")
    require(function(memory, "rememberResult"), r"embeddingFormatVersion:\s*embeddingResult\.formatVersion.*?embeddingModelIdentifier:\s*embeddingResult\.modelIdentifier.*?embeddingDimension:\s*embedding\.count", "memory writes must persist actual provider identity/format/dimension")
    require(code_only(function(memory, "recallWithDiagnostics")), r"lexicalRecallResult.*?results\.append\(contentsOf:\s*backfill\.items\)", "unknown or stale memories must retain lexical recall")
    rebuilding = code_only(function(memory, "rebuildEmbeddings"))
    inference_call = re.search(r"\bawait\s+embed\s*\(", rebuilding)
    before_inference = rebuilding[:inference_call.start()] if inference_call else ""
    after_inference = rebuilding[inference_call.end():] if inference_call else ""
    before_fetch = rebuilding.split("context.fetch(", 1)[0]
    require(before_fetch, r"scenePhase\s*==\s*\.active", "memory reembedding requires an explicit foreground operation before reading a source page")
    for boundary in (before_inference, after_inference):
        require(boundary, r"scenePhase\s*==\s*\.active", "memory reembedding requires an explicit foreground operation before and after inference")
    for field in ("modelIdentifier", "formatVersion"):
        require(rebuilding, r"embeddingResult\." + field + r"\s*==\s*metadata\." + field, "memory reembedding must stop on changed provider identity")
    require(rebuilding, r"embeddingResult\.vector\.count\s*==\s*metadata\.dimension", "memory reembedding must retain the selected projection dimension")
    page = code_only(function(memory, "rebuildPageDescriptor"))
    require(page, r"fetchLimit\s*=\s*min\(32,\s*max\(1,\s*limit\)\)\s*\+\s*1", "memory rebuild reads must be bounded with one lookahead")
    require(page, r"item\.createdAt\s*>\s*afterDate.*?item\.id\s*>\s*afterID", "memory rebuild must use stable database keyset pagination")
    require(page, r"includePendingChanges\s*=\s*false", "memory rebuild page must exclude unbounded pending inserts")
    if re.search(r"fetchCount\s*\(|\.sorted\s*\{|\.filter\s*\{", rebuilding):
        failures.append("memory rebuild must not materialize or count the whole collection")
    for operation in ("rememberResult", "rebuildEmbeddings"):
        body = code_only(function(memory, operation))
        require(body, r"EmbeddingIndexLifecycle\.captureWrite\(\).*?await\s+embed.*?EmbeddingIndexLifecycle\.permitsWrite", "memory writes must reject results from before a vector reset")
    lifecycle = sources[BASE + "Services/EmbeddingIndexLifecycle.swift"]
    require(code_only(function(lifecycle, "permitsWrite")), r"!resetPending\s*&&\s*captured\s*==\s*generation", "vector writers require current generation and completed reset")
    require(code_only(function(lifecycle, "beginReset")), r"generation\s*=\s*UUID\(\).*?UserDefaults\.standard\.set\(true.*?invalidateCaches\(\)", "reset must invalidate in-flight writers and persist its pending state before clearing")
    reset = code_only(sources[BASE + "Services/EmbeddingVectorReset.swift"])
    for descriptor in ("memoryDescriptor", "ragDescriptor"):
        require(code_only(function(sources[BASE + "Services/EmbeddingVectorReset.swift"], descriptor)), r"fetchLimit\s*=.*?\+\s*1", "vector reset reads require bounded pages with one lookahead")
    if re.search(r"\b(?:FileManager|removeItem|deleteAllData|AIModelCache)\b|context\.delete\s*\(", reset):
        failures.append("vector reset must preserve sources, model files and Apple caches")
    rag_rebuild = code_only(function(sources[BASE + "Services/RAGEmbeddingRebuilder.swift"], "rebuildBatch"))
    require(rag_rebuild, r"EmbeddingIndexLifecycle\.captureWrite\(\).*?await\s+embed.*?EmbeddingIndexLifecycle\.permitsWrite", "RAG source rebuild must reject results from before a reset")
    require(rag_rebuild, r"result\.modelIdentifier\s*==\s*metadata\.modelIdentifier.*?result\.formatVersion\s*==\s*metadata\.formatVersion.*?result\.vector\.count\s*==\s*metadata\.dimension", "RAG source rebuild must retain exact embedding space identity")
    item = sources[BASE + "Models/MemoryItem.swift"]
    for field, kind in (("embeddingFormatVersion", "Int"), ("embeddingModelIdentifier", "String"), ("embeddingDimension", "Int")):
        require(code_only(item), r"var\s+" + field + r"\s*:\s*" + kind + r"\?\s*=\s*nil", "unindexed memory embedding metadata must remain optional and unknown")
    index = sources[BASE + "Services/VectorIndex.swift"]
    matches = code_only(function(index, "matches"))
    for field, value in (("embeddingFormatVersion", "formatVersion"), ("embeddingModelIdentifier", "modelIdentifier"), ("embeddingDimension", "dimension")):
        require(matches, r"item\." + field + r"\s*==\s*metadata\." + value, "memory semantic index must reject incompatible identity/format/dimension")
    require(matches, r"item\.embedding\.count\s*==\s*metadata\.dimension", "memory semantic index must validate actual vector dimensions")
    require(code_only(index), r"loadedMetadata\s*!=\s*metadata", "memory vector cache must invalidate when embedding identity changes")
    require(code_only(index), r"guard\s+EmbeddingIndexLifecycle\.semanticReadsAllowed", "semantic indexes must remain unavailable during a partial reset")
    readiness = function(sources[BASE + "Assistant/SemanticEmbeddingText.swift"], "isLoaded")
    require(code_only(readiness), r"CoreAITextEmbeddingModelStore\.isSelected.*?CoreAITextEmbeddingService\.shared\.isLoaded", "embedding readiness must use the selected provider without a probe")
    if re.search(r"\b(?:prepare|embed|runEmbedding|runEmbeddingWithIdentity)\s*\(", code_only(readiness)):
        failures.append("embedding readiness discovery must not load or run inference")
    return failures


if __name__ == "__main__":
    errors = check(ROOT)
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        raise SystemExit(1)
    print("Core AI text embedding source invariants passed (not compile, native rebuild, or device evidence).")
