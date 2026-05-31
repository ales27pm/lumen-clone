import Foundation
import OSLog

nonisolated struct ReasoningAwareStreamParserConfig: Sendable, Equatable {
    static let onlyReasoningFallback = "The model produced only internal reasoning and no final answer. Try again with thinking disabled."

    var captureReasoning: Bool
    var reasoningTraceBudgetCharacters: Int
    var fallbackWhenOnlyReasoning: String

    init(
        captureReasoning: Bool = false,
        reasoningTraceBudgetCharacters: Int = 16_384,
        fallbackWhenOnlyReasoning: String = Self.onlyReasoningFallback
    ) {
        self.captureReasoning = captureReasoning
        self.reasoningTraceBudgetCharacters = max(0, reasoningTraceBudgetCharacters)
        self.fallbackWhenOnlyReasoning = fallbackWhenOnlyReasoning
    }
}

nonisolated struct ReasoningAwareStreamParserDelta: Sendable, Equatable {
    let visibleDelta: String
    let reasoningDelta: String
    let warnings: [String]

    static let empty = ReasoningAwareStreamParserDelta(visibleDelta: "", reasoningDelta: "", warnings: [])
}

nonisolated struct ReasoningAwareStreamParserResult: Codable, Sendable, Equatable {
    let rawModelOutput: String
    let reasoningText: String?
    let visibleAnswer: String
    let parserWarnings: [String]
    let unterminatedReasoningBlock: Bool
    let reasoningWasTruncated: Bool
}

nonisolated struct ReasoningAwareStreamParser: Sendable {
    private enum State: String, Sendable {
        case visible
        case reasoning
    }

    private enum TagKind: Sendable {
        case opening
        case closing
    }

    private struct RecognizedTag {
        let kind: TagKind
        let name: String
        let range: Range<String.Index>
    }

    private static let hiddenTagNames: Set<String> = [
        "think",
        "thinking",
        "thinkresponse",
        "think_response",
        "reasoning",
        "analysis",
        "chain_of_thought"
    ]
    private static let hiddenTagPrefixes: [String] = {
        hiddenTagNames
            .flatMap { tag in
                ["<\(tag)", "</\(tag)"]
            }
            .sorted { $0.count > $1.count }
    }()
    private static let logger = Logger(subsystem: "ai.lumen.llm", category: "reasoning-parser")

    private let config: ReasoningAwareStreamParserConfig
    private var state: State = .visible
    private var reasoningDepth = 0
    private var carry = ""
    private var visibleAnswer = ""
    private var reasoningTrace = ""
    private var rawModelOutput = ""
    private var warnings: [String] = []
    private var emittedWarnings: Set<String> = []
    private var sawReasoningBlock = false
    private var didFinish = false
    private var unterminatedReasoningBlock = false
    private var reasoningWasTruncated = false

    init(config: ReasoningAwareStreamParserConfig = ReasoningAwareStreamParserConfig()) {
        self.config = config
    }

    var result: ReasoningAwareStreamParserResult {
        ReasoningAwareStreamParserResult(
            rawModelOutput: rawModelOutput,
            reasoningText: reasoningTrace.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : reasoningTrace,
            visibleAnswer: visibleAnswer,
            parserWarnings: warnings,
            unterminatedReasoningBlock: unterminatedReasoningBlock,
            reasoningWasTruncated: reasoningWasTruncated
        )
    }

    @discardableResult
    mutating func ingest(_ chunk: String) -> ReasoningAwareStreamParserDelta {
        guard !chunk.isEmpty, didFinish == false else { return .empty }
        rawModelOutput += chunk
        carry += chunk
        return drainCarry(final: false)
    }

    @discardableResult
    mutating func finish() -> ReasoningAwareStreamParserDelta {
        guard didFinish == false else { return .empty }
        didFinish = true

        var delta = drainCarry(final: true)

        if state == .reasoning {
            unterminatedReasoningBlock = true
            addWarning("Unterminated <think> block at end of stream.")
            logTransition("unterminated_reasoning", tag: nil)
        }

        if sawReasoningBlock && visibleAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let fallback = config.fallbackWhenOnlyReasoning
            visibleAnswer = fallback
            delta = ReasoningAwareStreamParserDelta(
                visibleDelta: delta.visibleDelta + fallback,
                reasoningDelta: delta.reasoningDelta,
                warnings: warnings
            )
            addWarning("Model produced reasoning without a final answer.")
        }

        return ReasoningAwareStreamParserDelta(
            visibleDelta: delta.visibleDelta,
            reasoningDelta: delta.reasoningDelta,
            warnings: warnings
        )
    }

    private mutating func drainCarry(final: Bool) -> ReasoningAwareStreamParserDelta {
        var visibleDelta = ""
        var reasoningDelta = ""

        while !carry.isEmpty {
            if let tag = nextRecognizedTag(in: carry) {
                if tag.range.lowerBound > carry.startIndex {
                    route(String(carry[..<tag.range.lowerBound]), visibleDelta: &visibleDelta, reasoningDelta: &reasoningDelta)
                }
                handle(tag)
                carry = String(carry[tag.range.upperBound...])
                continue
            }

            if final {
                if let partialStart = partialHiddenTagStart(in: carry) {
                    if partialStart > carry.startIndex {
                        route(String(carry[..<partialStart]), visibleDelta: &visibleDelta, reasoningDelta: &reasoningDelta)
                    }
                    addWarning("Incomplete reasoning tag at end of stream was discarded.")
                    carry = ""
                    break
                }
                route(carry, visibleDelta: &visibleDelta, reasoningDelta: &reasoningDelta)
                carry = ""
                break
            }

            if let partialStart = partialHiddenTagStart(in: carry) {
                if partialStart > carry.startIndex {
                    route(String(carry[..<partialStart]), visibleDelta: &visibleDelta, reasoningDelta: &reasoningDelta)
                    carry = String(carry[partialStart...])
                }
                break
            }

            route(carry, visibleDelta: &visibleDelta, reasoningDelta: &reasoningDelta)
            carry = ""
            break
        }

        return ReasoningAwareStreamParserDelta(
            visibleDelta: visibleDelta,
            reasoningDelta: reasoningDelta,
            warnings: warnings
        )
    }

    private mutating func route(_ text: String, visibleDelta: inout String, reasoningDelta: inout String) {
        guard !text.isEmpty else { return }

        switch state {
        case .visible:
            visibleAnswer += text
            visibleDelta += text
        case .reasoning:
            guard config.captureReasoning else { return }
            let remaining = max(0, config.reasoningTraceBudgetCharacters - reasoningTrace.count)
            guard remaining > 0 else {
                markReasoningBudgetExceeded()
                return
            }
            let accepted = String(text.prefix(remaining))
            reasoningTrace += accepted
            reasoningDelta += accepted
            if accepted.count < text.count {
                markReasoningBudgetExceeded()
            }
        }
    }

    private mutating func handle(_ tag: RecognizedTag) {
        switch (state, tag.kind) {
        case (.visible, .opening):
            state = .reasoning
            reasoningDepth = 1
            sawReasoningBlock = true
            logTransition("enter_reasoning", tag: tag.name)
        case (.visible, .closing):
            addWarning("Closing </\(tag.name)> appeared without a matching opening tag.")
            logTransition("unmatched_close", tag: tag.name)
        case (.reasoning, .opening):
            reasoningDepth += 1
            sawReasoningBlock = true
            addWarning("Nested <\(tag.name)> block encountered.")
            logTransition("nested_reasoning", tag: tag.name)
        case (.reasoning, .closing):
            if reasoningDepth > 1 {
                reasoningDepth -= 1
                logTransition("exit_nested_reasoning", tag: tag.name)
            } else {
                state = .visible
                reasoningDepth = 0
                logTransition("exit_reasoning", tag: tag.name)
            }
        }
    }

    private func nextRecognizedTag(in text: String) -> RecognizedTag? {
        var searchStart = text.startIndex
        while searchStart < text.endIndex,
              let open = text[searchStart...].firstIndex(of: "<") {
            guard let close = text[open...].firstIndex(of: ">") else { return nil }
            let innerStart = text.index(after: open)
            let inner = String(text[innerStart..<close])
            if let parsed = parseTag(inner) {
                return RecognizedTag(
                    kind: parsed.kind,
                    name: parsed.name,
                    range: open..<text.index(after: close)
                )
            }
            searchStart = text.index(after: close)
        }
        return nil
    }

    private func parseTag(_ inner: String) -> (kind: TagKind, name: String)? {
        var trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else { return nil }

        let kind: TagKind
        if trimmed.hasPrefix("/") {
            kind = .closing
            trimmed = String(trimmed.dropFirst()).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            kind = .opening
        }

        let name = trimmed
            .split { character in
                character.isWhitespace || character == "/" || character == ">"
            }
            .first
            .map(String.init) ?? ""

        guard Self.hiddenTagNames.contains(name) else { return nil }
        return (kind, name)
    }

    private func partialHiddenTagStart(in text: String) -> String.Index? {
        let lower = text.lowercased()
        var candidate = lower.startIndex
        while candidate < lower.endIndex {
            guard let open = lower[candidate...].firstIndex(of: "<") else { return nil }
            let suffix = String(lower[open...])
            if suffix.contains(">") {
                candidate = lower.index(after: open)
                continue
            }
            if isPotentialHiddenTagPrefix(suffix) {
                return text.index(text.startIndex, offsetBy: lower.distance(from: lower.startIndex, to: open))
            }
            candidate = lower.index(after: open)
        }
        return nil
    }

    private func isPotentialHiddenTagPrefix(_ suffix: String) -> Bool {
        guard suffix.hasPrefix("<") else { return false }
        return Self.hiddenTagPrefixes.contains { prefix in
            prefix.hasPrefix(suffix) || suffix.hasPrefix(prefix)
        }
    }

    private mutating func markReasoningBudgetExceeded() {
        reasoningWasTruncated = true
        addWarning("Reasoning exceeded developer trace budget and was truncated.")
    }

    private mutating func addWarning(_ warning: String) {
        guard emittedWarnings.insert(warning).inserted else { return }
        warnings.append(warning)
    }

    private func logTransition(_ transition: String, tag: String?) {
        #if DEBUG
        Self.logger.debug(
            "event=reasoning_parser_transition transition=\(transition, privacy: .public) tag=\(tag ?? "none", privacy: .public) visible_chars=\(visibleAnswer.count, privacy: .public) reasoning_chars=\(reasoningTrace.count, privacy: .public)"
        )
        #endif
    }
}
