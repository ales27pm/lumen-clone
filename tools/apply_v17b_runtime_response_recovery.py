#!/usr/bin/env python3
# Apply V17B runtime-response-recovery edits without using the malformed patch file.
#
# Run from repository root:
#   python3 tools/apply_v17b_runtime_response_recovery.py
#
# Idempotent: if an edit is already present, it keeps going.

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "tools" else Path.cwd()

def path(rel: str) -> Path:
    return ROOT / rel

def read(rel: str) -> str:
    p = path(rel)
    if not p.exists():
        raise SystemExit(f"missing file: {rel}")
    return p.read_text(encoding="utf-8")

def write(rel: str, text: str) -> None:
    path(rel).write_text(text, encoding="utf-8")

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"SKIP already applied: {label}")
        return text
    if old not in text:
        raise SystemExit(f"could not find target block for: {label}")
    print(f"APPLY {label}")
    return text.replace(old, new, 1)

def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            print(f"SKIP already applied: {label}")
            return text
        raise SystemExit(f"could not find target text for: {label}")
    print(f"APPLY {label}")
    return text.replace(old, new, 1)

def patch_agent_behavior_trace() -> None:
    rel = "ios/Lumen/Services/AgentGrounding/AgentBehaviorTrace.swift"
    text = read(rel)

    insert_old = '''nonisolated enum AgentBehaviorTraceRecorder {
    private static let fileName = "agent-behavior-traces.jsonl"
    private static let maxRecentReadBytes = 1_048_576

    static func record(_ trace: AgentBehaviorTrace) {
        do {'''
    insert_new = '''private final class AgentBehaviorTraceMemoryCache: @unchecked Sendable {
    private let lock = NSLock()
    private var traces: [AgentBehaviorTrace] = []
    private let maxTraces = 512

    func remember(_ trace: AgentBehaviorTrace) {
        lock.lock()
        traces.append(trace)
        if traces.count > maxTraces {
            traces.removeFirst(traces.count - maxTraces)
        }
        lock.unlock()
    }

    func recent(limit: Int) -> [AgentBehaviorTrace] {
        lock.lock()
        defer { lock.unlock() }
        return Array(traces.suffix(max(0, limit)))
    }

    func clear() {
        lock.lock()
        traces.removeAll()
        lock.unlock()
    }
}

nonisolated enum AgentBehaviorTraceRecorder {
    private static let fileName = "agent-behavior-traces.jsonl"
    private static let maxRecentReadBytes = 1_048_576
    private static let memoryCache = AgentBehaviorTraceMemoryCache()

    static func record(_ trace: AgentBehaviorTrace) {
        memoryCache.remember(trace)
        do {'''
    text = replace_once(text, insert_old, insert_new, "AgentBehaviorTrace in-memory cache + pre-disk record")

    text = replace_required(
        text,
        '''        let boundedLimit = max(0, limit)
        guard boundedLimit > 0, !Task.isCancelled else { return [] }

        do {''',
        '''        let boundedLimit = max(0, limit)
        guard boundedLimit > 0, !Task.isCancelled else { return [] }
        let inMemory = memoryCache.recent(limit: boundedLimit)

        do {''',
        "AgentBehaviorTrace recent() reads memory cache"
    )

    text = replace_required(
        text,
        '''            guard FileManager.default.fileExists(atPath: path) else { return [] }''',
        '''            guard FileManager.default.fileExists(atPath: path) else { return inMemory }''',
        "AgentBehaviorTrace recent() returns memory when disk file missing"
    )

    text = replace_required(
        text,
        '''            let traces = data.split(separator: 0x0A).compactMap { line -> AgentBehaviorTrace? in''',
        '''            let diskTraces = data.split(separator: 0x0A).compactMap { line -> AgentBehaviorTrace? in''',
        "AgentBehaviorTrace diskTraces naming"
    )

    text = replace_required(
        text,
        '''            return Array(traces.suffix(boundedLimit))
        } catch {
            return []
        }
    }

    private static func completeLineData''',
        '''            return mergedRecentTraces(diskTraces, inMemory, limit: boundedLimit)
        } catch {
            return inMemory
        }
    }

    private static func mergedRecentTraces(_ groups: [AgentBehaviorTrace]..., limit: Int) -> [AgentBehaviorTrace] {
        var byID: [UUID: AgentBehaviorTrace] = [:]
        for trace in groups.flatMap({ $0 }) {
            byID[trace.id] = trace
        }
        let sorted = Array(byID.values)
            .sorted { lhs, rhs in
                if lhs.createdAt == rhs.createdAt { return lhs.id.uuidString < rhs.id.uuidString }
                return lhs.createdAt < rhs.createdAt
            }
        return Array(sorted.suffix(max(0, limit)))
    }

    private static func completeLineData''',
        "AgentBehaviorTrace merges disk + memory traces"
    )

    text = replace_required(
        text,
        '''    static func clear() {
        do {''',
        '''    static func clear() {
        memoryCache.clear()
        do {''',
        "AgentBehaviorTrace clear() clears memory cache"
    )

    write(rel, text)

def patch_planner() -> None:
    rel = "ios/Lumen/Services/DeterministicToolPlanner.swift"
    text = read(rel)
    text = replace_required(
        text,
        '''        case .files:
            if let name = extractFileName(from: prompt) { return action("files.read", ["name": .string(name)]) }
            return nil''',
        '''        case .files:
            if let name = extractFileName(from: prompt) { return action("files.read", ["name": .string(name)]) }
            if containsAny(text, ["attachment", "attached", "this file", "this document", "read file", "read document"]) { return action("files.read") }
            return nil''',
        "DeterministicToolPlanner files.read attachment fallback"
    )
    text = replace_required(
        text,
        '''            if containsAny(text, ["search"]) {
                let query = expandRAGQueryIfNeeded(originalPrompt: prompt)
                return action("rag.search", ["query": .string(query)])
            }''',
        '''            if containsAny(text, ["search", "summarize", "read", "show"]) {
                let query = expandRAGQueryIfNeeded(originalPrompt: prompt)
                return action("rag.search", ["query": .string(query)])
            }''',
        "DeterministicToolPlanner RAG read/summarize/show fallback"
    )
    write(rel, text)

def patch_fallback_texts() -> None:
    rel = "ios/Lumen/Services/LLM/ReasoningAwareStreamParser.swift"
    text = read(rel)
    text = replace_required(
        text,
        '''    static let onlyReasoningFallback = "The model produced only internal reasoning and no final answer. Try again with thinking disabled."''',
        '''    static let onlyReasoningFallback = "I'm ready. Please ask again or tell me what you'd like to do next."''',
        "ReasoningAwareStreamParser visible fallback"
    )
    write(rel, text)

    rel = "ios/Lumen/Services/FinalOutputSanitizer.swift"
    text = read(rel)
    text = replace_required(
        text,
        '''    static let fallback = "I hit an internal response-format issue. Please try again."''',
        '''    static let fallback = "I'm ready. Please ask again or tell me what you'd like to do next."''',
        "FinalOutputSanitizer visible fallback"
    )
    write(rel, text)

    rel = "ios/Lumen/Services/E2ETestRunner.swift"
    text = read(rel)
    text = replace_required(
        text,
        '''            "full local model pipeline is temporarily running in compatibility mode",
            "full agent pipeline",
            "please try again with thinking disabled"
        ]''',
        '''            "full local model pipeline is temporarily running in compatibility mode",
            "full agent pipeline",
            "please try again with thinking disabled",
            "please ask again or tell me what you'd like to do next"
        ]''',
        "E2ETestRunner treats new neutral fallback as quality failure"
    )
    write(rel, text)

def validate() -> None:
    checks = {
        "ios/Lumen/Services/AgentGrounding/AgentBehaviorTrace.swift": [
            "AgentBehaviorTraceMemoryCache",
            "memoryCache.remember(trace)",
            "mergedRecentTraces(diskTraces, inMemory, limit: boundedLimit)",
            "memoryCache.clear()",
        ],
        "ios/Lumen/Services/DeterministicToolPlanner.swift": [
            '"attachment", "attached", "this file", "this document", "read file", "read document"',
            '"search", "summarize", "read", "show"',
        ],
        "ios/Lumen/Services/LLM/ReasoningAwareStreamParser.swift": [
            "I'm ready. Please ask again or tell me what you'd like to do next.",
        ],
        "ios/Lumen/Services/FinalOutputSanitizer.swift": [
            "I'm ready. Please ask again or tell me what you'd like to do next.",
        ],
        "ios/Lumen/Services/E2ETestRunner.swift": [
            "please ask again or tell me what you'd like to do next",
        ],
    }
    for rel, tokens in checks.items():
        text = read(rel)
        missing = [token for token in tokens if token not in text]
        if missing:
            raise SystemExit(f"validation failed for {rel}: missing {missing}")
    print("V17B runtime-response-recovery edits applied and validated.")

def main() -> int:
    patch_agent_behavior_trace()
    patch_planner()
    patch_fallback_texts()
    validate()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
