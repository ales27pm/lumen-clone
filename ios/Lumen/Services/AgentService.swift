import Foundation
import SwiftData

nonisolated struct AgentRequest: Sendable {
    let systemPrompt: String
    let history: [(role: MessageRole, content: String)]
    let userMessage: String
    let temperature: Double
    let topP: Double
    let repetitionPenalty: Double
    let maxTokens: Int
    let maxSteps: Int
    let availableTools: [ToolDefinition]
    let relevantMemories: [MemoryContextItem]
    let attachments: [ChatAttachment]
    let conversationID: UUID?
    let turnID: UUID?
    let scenarioID: String?
    let e2eRunID: UUID?
    let agentRunID: UUID?

    init(
        systemPrompt: String,
        history: [(role: MessageRole, content: String)],
        userMessage: String,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        maxTokens: Int,
        maxSteps: Int,
        availableTools: [ToolDefinition],
        relevantMemories: [MemoryContextItem],
        attachments: [ChatAttachment] = [],
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil
    ) {
        self.systemPrompt = systemPrompt
        self.history = history
        self.userMessage = userMessage
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.maxSteps = maxSteps
        self.availableTools = availableTools
        self.relevantMemories = relevantMemories
        self.attachments = attachments
        self.conversationID = conversationID
        self.turnID = turnID
        self.scenarioID = scenarioID
        self.e2eRunID = e2eRunID
        self.agentRunID = agentRunID
    }

    init(
        systemPrompt: String,
        history: [(role: MessageRole, content: String)],
        userMessage: String,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        maxTokens: Int,
        maxSteps: Int,
        availableTools: [ToolDefinition],
        legacyRelevantMemories: [String],
        attachments: [ChatAttachment] = [],
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil
    ) {
        self.init(
            systemPrompt: systemPrompt,
            history: history,
            userMessage: userMessage,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            maxSteps: maxSteps,
            availableTools: availableTools,
            relevantMemories: MemoryContextAdapter.fromLegacyStrings(legacyRelevantMemories),
            attachments: attachments,
            conversationID: conversationID,
            turnID: turnID,
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID
        )
    }
}

nonisolated enum AgentEvent: Sendable {
    case step(AgentStep)
    case stepDelta(id: UUID, text: String)
    case finalDelta(String)
    case done(finalText: String, steps: [AgentStep])
    case error(String)
}

// MARK: - Structured turn model

nonisolated struct AgentAction: Sendable, Hashable {
    let tool: String
    let args: AgentJSONArguments

    private struct StructuredOutput: Encodable {
        let action: StructuredAction
    }

    private struct StructuredAction: Encodable {
        let tool: String
        let args: AgentJSONArguments
    }

    var dedupeKey: String {
        let argsStr = args.keys.sorted()
            .map { "\($0)=\(args[$0]?.stringValue ?? "")" }
            .joined(separator: "&")
        return tool + "|" + argsStr
    }

    var displayContent: String {
        if args.isEmpty { return "\(tool)()" }
        let argsStr = args.keys.sorted()
            .map { "\($0)=\(args[$0]?.stringValue ?? "")" }
            .joined(separator: ", ")
        return "\(tool)(\(argsStr))"
    }

    var structuredOutputJSON: String {
        let output = StructuredOutput(action: StructuredAction(tool: tool, args: args))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(output),
              let json = String(data: data, encoding: .utf8) else {
            return #"{"action":{"args":{},"tool":"\#(tool)"}}"#
        }
        return json
    }
}

nonisolated struct AgentTurn: Sendable {
    let thought: String?
    let action: AgentAction?
    let final: String?
    let parseError: AgentTurnParseError?
    let hadNoise: Bool

    var isStructured: Bool { action != nil || (final?.isEmpty == false) }
}

nonisolated enum AgentTurnParseError: String, Error, Sendable, Codable {
    case empty
    case noJSONObject
    case multipleJSONObjects
    case noisyOutput
    case malformedEscapeSequence
    case incompleteJSON
    case invalidJSONObject
    case invalidThoughtType
    case invalidFinalType
    case mixedTurn
    case mixedActionShapes
    case missingActionOrFinal
    case missingActionTool
    case invalidActionType
    case invalidActionArgsType
    case contextWindowExceeded
}

private nonisolated enum AgentThinkBlockSanitizer {
    struct StripResult: Sendable {
        let text: String
        let prefixNoise: String?
        let stripped: Bool
    }

    static func stripLeadingThinkBlocks(from text: String) -> StripResult {
        var remaining = text
        var summaries: [String] = []

        while true {
            let leadingWhitespaceCount = remaining.prefix(while: { $0.isWhitespace }).count
            let afterWhitespace = String(remaining.dropFirst(leadingWhitespaceCount))
            guard let block = leadingThinkBlock(in: afterWhitespace) else { break }

            let redactedChars = block.content.trimmingCharacters(in: .whitespacesAndNewlines).count
            if redactedChars == 0 {
                summaries.append("leading empty <\(block.tag)> block stripped")
            } else {
                summaries.append("leading <\(block.tag)> block stripped (\(redactedChars) chars redacted)")
            }
            remaining = String(afterWhitespace.dropFirst(block.fullLength))
        }

        return StripResult(
            text: remaining,
            prefixNoise: summaries.isEmpty ? nil : summaries.joined(separator: "; "),
            stripped: !summaries.isEmpty
        )
    }

    static func redactedForDiagnostics(_ text: String) -> String {
        let stripped = stripLeadingThinkBlocks(from: text)
        guard stripped.stripped else { return text }
        return [stripped.prefixNoise, stripped.text]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private struct LeadingBlock {
        let tag: String
        let content: String
        let fullLength: Int
    }

    private static func leadingThinkBlock(in text: String) -> LeadingBlock? {
        let lower = text.lowercased()
        let candidates = ["think", "thinking"]
        for tag in candidates {
            let openPrefix = "<\(tag)"
            guard lower.hasPrefix(openPrefix),
                  let openEnd = lower.firstIndex(of: ">") else {
                continue
            }
            let close = "</\(tag)>"
            guard let closeRange = lower.range(of: close, range: openEnd..<lower.endIndex) else {
                continue
            }
            let contentStartOffset = lower.distance(from: lower.startIndex, to: lower.index(after: openEnd))
            let contentEndOffset = lower.distance(from: lower.startIndex, to: closeRange.lowerBound)
            let fullLength = lower.distance(from: lower.startIndex, to: closeRange.upperBound)
            let contentStart = text.index(text.startIndex, offsetBy: contentStartOffset)
            let contentEnd = text.index(text.startIndex, offsetBy: contentEndOffset)
            return LeadingBlock(
                tag: tag,
                content: String(text[contentStart..<contentEnd]),
                fullLength: fullLength
            )
        }
        return nil
    }
}

private nonisolated enum AgentJSONCandidateSelector {
    struct Selection {
        let object: [String: Any]
        let selectedJSON: String
        let prefixNoise: String?
        let suffixNoise: String?
        let hadUnsupportedNoise: Bool
    }

    static func select(from text: String) -> Result<Selection, AgentTurnParseError> {
        let stripped = AgentThinkBlockSanitizer.stripLeadingThinkBlocks(from: text)
        let chars = Array(stripped.text)
        let rangesResult = discoverRanges(in: chars)
        switch rangesResult {
        case .failure(let error):
            if stripped.stripped, stripped.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return .failure(.empty)
            }
            return .failure(error)
        case .success(let ranges):
            guard !ranges.isEmpty else {
                return .failure(stripped.stripped && stripped.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .empty : .noJSONObject)
            }

            var candidates: [(range: (Int, Int), object: [String: Any], score: Int)] = []
            var candidateErrors: [AgentTurnParseError] = []
            for range in ranges {
                switch parseJSONObject(chars: chars, range: range) {
                case .success(let obj):
                    candidates.append((range: range, object: obj, score: scoreCandidate(object: obj)))
                case .failure(let error):
                    candidateErrors.append(error)
                }
            }
            guard !candidates.isEmpty else {
                if ranges.count == 1, let error = candidateErrors.first {
                    return .failure(error)
                }
                return .failure(ranges.count > 1 ? .multipleJSONObjects : .invalidJSONObject)
            }

            candidates.sort { lhs, rhs in
                if lhs.score != rhs.score { return lhs.score > rhs.score }
                return lhs.range.0 > rhs.range.0
            }

            let selected = candidates[0]
            let selectedJSON = String(chars[selected.range.0...selected.range.1])
            let prefix = String(chars[..<selected.range.0])
            let suffixStart = selected.range.1 + 1
            let suffix = suffixStart < chars.count ? String(chars[suffixStart..<chars.count]) : ""
            let prefixNoise = [
                stripped.prefixNoise,
                nonEmpty(stripFenceNoise(prefix))
            ]
                .compactMap { $0 }
                .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                .joined(separator: "; ")
            let suffixNoise = nonEmpty(stripFenceNoise(suffix))
            return .success(
                Selection(
                    object: selected.object,
                    selectedJSON: selectedJSON,
                    prefixNoise: prefixNoise.isEmpty ? nil : prefixNoise,
                    suffixNoise: suffixNoise,
                    hadUnsupportedNoise: !prefixNoise.isEmpty || suffixNoise != nil
                )
            )
        }
    }

    private static func discoverRanges(in chars: [Character]) -> Result<[(Int, Int)], AgentTurnParseError> {
        var ranges: [(Int, Int)] = []
        var depth = 0
        var start: Int?
        var inString = false
        var escape = false
        var i = 0

        while i < chars.count {
            let ch = chars[i]
            if inString {
                if escape {
                    if !isValidEscape(at: i, in: chars) {
                        return .failure(.malformedEscapeSequence)
                    }
                    if ch == "u" { i += 4 }
                    escape = false
                } else if ch == "\\" {
                    escape = true
                } else if ch == "\"" {
                    inString = false
                }
                i += 1
                continue
            }

            if ch == "\"", depth > 0 {
                inString = true
            } else if ch == "{" {
                if depth == 0 { start = i }
                depth += 1
            } else if ch == "}" {
                guard depth > 0 else {
                    i += 1
                    continue
                }
                depth -= 1
                if depth == 0, let s = start {
                    ranges.append((s, i))
                    start = nil
                }
            }
            i += 1
        }

        if inString || depth != 0 { return .failure(.incompleteJSON) }
        return .success(ranges)
    }

    private static func parseJSONObject(chars: [Character], range: (Int, Int)) -> Result<[String: Any], AgentTurnParseError> {
        let jsonStr = String(chars[range.0...range.1])
        let jsonChars = Array(jsonStr)
        if !validateEscapes(in: jsonChars) {
            return .failure(.malformedEscapeSequence)
        }
        guard let data = jsonStr.data(using: .utf8) else { return .failure(.invalidJSONObject) }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .failure(.invalidJSONObject)
        }
        return .success(object)
    }

    private static func scoreCandidate(object: [String: Any]) -> Int {
        var score = 0
        if object["action"] != nil || object["tool"] != nil { score += 4 }
        if object["final"] != nil || object["final_answer"] != nil || object["answer"] != nil { score += 4 }
        if object["thought"] != nil || object["reasoning"] != nil { score += 2 }
        if object["args"] != nil || object["arguments"] != nil || object["input"] != nil { score += 1 }
        return score
    }

    private static func stripFenceNoise(_ text: String) -> String {
        text
            .replacingOccurrences(of: "```json", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "```", with: "")
            .replacingOccurrences(of: "<json>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "</json>", with: "", options: .caseInsensitive)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func nonEmpty(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func isValidEscape(at index: Int, in chars: [Character]) -> Bool {
        let esc = chars[index]
        switch esc {
        case "\"", "\\", "/", "b", "f", "n", "r", "t":
            return true
        case "u":
            guard index + 4 < chars.count else { return false }
            for j in (index + 1)...(index + 4) {
                if chars[j].hexDigitValue == nil { return false }
            }
            return true
        default:
            return false
        }
    }

    private static func validateEscapes(in chars: [Character]) -> Bool {
        var inString = false
        var escape = false
        var i = 0

        while i < chars.count {
            let ch = chars[i]
            if inString {
                if escape {
                    if !isValidEscape(at: i, in: chars) { return false }
                    if ch == "u" { i += 4 }
                    escape = false
                } else if ch == "\\" {
                    escape = true
                } else if ch == "\"" {
                    inString = false
                }
                i += 1
                continue
            }

            if ch == "\"" {
                inString = true
            }
            i += 1
        }

        return true
    }
}

nonisolated enum AgentTurnParser {
    private struct ExtractedJSONObject {
        let object: [String: Any]
        let hadNoise: Bool
    }

    static func parse(_ raw: String) -> AgentTurn {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return AgentTurn(thought: nil, action: nil, final: nil, parseError: .empty, hadNoise: false)
        }

        switch extractSingleJSONObject(from: trimmed) {
        case .success(let extracted):
            return buildTurn(from: extracted.object, hadNoise: extracted.hadNoise)
        case .failure(let error):
            return AgentTurn(thought: nil, action: nil, final: nil, parseError: error, hadNoise: false)
        }
    }

    private static func buildTurn(from obj: [String: Any], hadNoise: Bool) -> AgentTurn {
        if let value = obj["thought"], !(value is String) { return invalid(.invalidThoughtType) }
        if let value = obj["reasoning"], !(value is String) { return invalid(.invalidThoughtType) }
        if let value = obj["final"], !(value is String) { return invalid(.invalidFinalType) }
        if let value = obj["final_answer"], !(value is String) { return invalid(.invalidFinalType) }
        if let value = obj["answer"], !(value is String) { return invalid(.invalidFinalType) }

        let thoughtRaw = (obj["thought"] as? String) ?? (obj["reasoning"] as? String)
        let thought = thoughtRaw?.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanThought = (thought?.isEmpty ?? true) ? nil : thought

        var action: AgentAction?
        let hasNestedAction = obj["action"] != nil
        let hasFlatAction = obj["tool"] != nil || obj["args"] != nil || obj["arguments"] != nil || obj["input"] != nil
        if hasNestedAction && hasFlatAction {
            return invalid(.mixedActionShapes)
        }

        if hasNestedAction {
            guard let act = obj["action"] as? [String: Any] else { return invalid(.invalidActionType) }
            switch parseAction(from: act) {
            case .success(let parsedAction):
                action = parsedAction
            case .failure(let error):
                return invalid(error)
            }
        } else if hasFlatAction {
            switch parseFlatAction(from: obj) {
            case .success(let parsedAction):
                action = parsedAction
            case .failure(let error):
                return invalid(error)
            }
        }

        let finalRaw = (obj["final"] as? String)
            ?? (obj["final_answer"] as? String)
            ?? (obj["answer"] as? String)
        let finalTrimmed = finalRaw?.trimmingCharacters(in: .whitespacesAndNewlines)

        let hasFinal = !(finalTrimmed?.isEmpty ?? true)
        let hasAction = action != nil
        if hasAction && hasFinal { return invalid(.mixedTurn) }
        if !hasAction && !hasFinal { return invalid(.missingActionOrFinal) }

        return AgentTurn(
            thought: cleanThought,
            action: action,
            final: hasFinal ? finalTrimmed : nil,
            parseError: nil,
            hadNoise: hadNoise
        )
    }

    private static func parseAction(from act: [String: Any]) -> Result<AgentAction, AgentTurnParseError> {
        let name = (act["tool"] as? String) ?? (act["name"] as? String) ?? (act["id"] as? String) ?? ""
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .failure(.missingActionTool) }
        guard let args = parseArgs(from: act) else { return .failure(.invalidActionArgsType) }
        return .success(AgentAction(tool: trimmed, args: args))
    }

    private static func parseFlatAction(from obj: [String: Any]) -> Result<AgentAction, AgentTurnParseError> {
        guard let toolName = obj["tool"] as? String else { return .failure(.missingActionTool) }
        let trimmed = toolName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .failure(.missingActionTool) }
        guard let args = parseArgs(from: obj) else { return .failure(.invalidActionArgsType) }
        return .success(AgentAction(tool: trimmed, args: args))
    }

    private static func parseArgs(from obj: [String: Any]) -> AgentJSONArguments? {
        let argsValue = obj["args"] ?? obj["arguments"] ?? obj["input"]
        guard let argsValue else { return [:] }
        if let rawArgs = argsValue as? [String: Any] {
            return normalizeArgs(rawArgs)
        }
        if obj["input"] != nil, let inputText = argsValue as? String {
            let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { return [:] }
            if let data = trimmed.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data),
               let rawArgs = json as? [String: Any] {
                return normalizeArgs(rawArgs)
            }
            return ["query": .string(trimmed)]
        }
        return nil
    }

    private static func normalizeArgs(_ rawArgs: [String: Any]) -> AgentJSONArguments? {
        var args: AgentJSONArguments = [:]
        for (key, value) in rawArgs {
            guard let parsed = AgentJSONValue.parse(value) else { return nil }
            args[key] = parsed
        }
        return args
    }

    private static func extractSingleJSONObject(from text: String) -> Result<ExtractedJSONObject, AgentTurnParseError> {
        switch AgentJSONCandidateSelector.select(from: text) {
        case .success(let selection):
            return .success(
                ExtractedJSONObject(
                    object: selection.object,
                    hadNoise: selection.hadUnsupportedNoise
                )
            )
        case .failure(let error):
            return .failure(error)
        }
    }

    private static func invalid(_ error: AgentTurnParseError) -> AgentTurn {
        AgentTurn(thought: nil, action: nil, final: nil, parseError: error, hadNoise: false)
    }
}

// MARK: - Streaming JSON string extractor

/// Extracts the (possibly partial) string value of specific JSON keys from a growing buffer.
/// Safe on truncated input — returns what has been decoded so far and whether the string is closed.
final class StreamingJSONScanner {
    private var buffer = ""
    private(set) var thought: String = ""
    private(set) var final: String = ""
    private var thoughtDone = false
    private var finalDone = false

    enum Event {
        case thoughtDelta(String)
        case finalDelta(String)
    }

    func feed(_ chunk: String) -> [Event] {
        buffer += chunk
        var events: [Event] = []

        if !thoughtDone, let (value, done) = extractString(key: "thought") {
            if value.count > thought.count {
                let delta = String(value.suffix(value.count - thought.count))
                events.append(.thoughtDelta(delta))
                thought = value
            }
            thoughtDone = done
        }
        if !finalDone, let (value, done) = extractString(key: "final") {
            if value.count > final.count {
                let delta = String(value.suffix(value.count - final.count))
                events.append(.finalDelta(delta))
                final = value
            }
            finalDone = done
        }
        return events
    }

    private func extractString(key: String) -> (String, Bool)? {
        struct JSONContext {
            let isObject: Bool
            var lastToken: Character
        }

        let chars = Array(buffer)
        var stack: [JSONContext] = []
        var i = 0

        func skipWhitespace(from start: Int) -> Int {
            var index = start
            while index < chars.count, chars[index].isWhitespace { index += 1 }
            return index
        }

        func markValueConsumed() {
            guard !stack.isEmpty else { return }
            if stack[stack.count - 1].isObject {
                stack[stack.count - 1].lastToken = "v"
            } else {
                let last = stack[stack.count - 1].lastToken
                if last == "[" || last == "," {
                    stack[stack.count - 1].lastToken = "v"
                }
            }
        }

        func parseJSONString(startingAt quoteIndex: Int) -> (value: String, closed: Bool, nextIndex: Int) {
            var index = quoteIndex + 1
            var output = ""

            while index < chars.count {
                let ch = chars[index]
                if ch == "\"" {
                    return (output, true, index + 1)
                }
                if ch == "\\" {
                    let escIndex = index + 1
                    guard escIndex < chars.count else { return (output, false, chars.count) }
                    let esc = chars[escIndex]
                    switch esc {
                    case "n": output.append("\n")
                    case "t": output.append("\t")
                    case "r": output.append("\r")
                    case "\"": output.append("\"")
                    case "\\": output.append("\\")
                    case "/": output.append("/")
                    case "b": output.append("\u{08}")
                    case "f": output.append("\u{0C}")
                    case "u":
                        let h1 = escIndex + 1
                        let hEnd = h1 + 4
                        guard hEnd <= chars.count else { return (output, false, chars.count) }
                        let hex = String(chars[h1..<hEnd])
                        if let scalar = UInt32(hex, radix: 16), let unicode = Unicode.Scalar(scalar) {
                            output.append(Character(unicode))
                        }
                        index = hEnd
                        continue
                    default:
                        output.append(esc)
                    }
                    index = escIndex + 1
                    continue
                }
                output.append(ch)
                index += 1
            }
            return (output, false, chars.count)
        }

        while i < chars.count {
            let ch = chars[i]

            if ch.isWhitespace {
                i += 1
                continue
            }

            switch ch {
            case "{":
                stack.append(JSONContext(isObject: true, lastToken: "{"))
                i += 1
            case "[":
                stack.append(JSONContext(isObject: false, lastToken: "["))
                i += 1
            case "}":
                if !stack.isEmpty { _ = stack.popLast() }
                markValueConsumed()
                i += 1
            case "]":
                if !stack.isEmpty { _ = stack.popLast() }
                markValueConsumed()
                i += 1
            case ",":
                if !stack.isEmpty { stack[stack.count - 1].lastToken = "," }
                i += 1
            case ":":
                if !stack.isEmpty { stack[stack.count - 1].lastToken = ":" }
                i += 1
            case "\"":
                let parsed = parseJSONString(startingAt: i)
                let isObjectKeyPosition = {
                    guard let context = stack.last, context.isObject else { return false }
                    return context.lastToken == "{" || context.lastToken == ","
                }()

                if isObjectKeyPosition {
                    if !stack.isEmpty { stack[stack.count - 1].lastToken = "k" }
                    if parsed.closed, parsed.value == key {
                        var valueStart = skipWhitespace(from: parsed.nextIndex)
                        guard valueStart < chars.count, chars[valueStart] == ":" else { return nil }
                        valueStart = skipWhitespace(from: valueStart + 1)
                        guard valueStart < chars.count, chars[valueStart] == "\"" else { return nil }
                        let value = parseJSONString(startingAt: valueStart)
                        return (value.value, value.closed)
                    }
                } else {
                    markValueConsumed()
                }
                i = parsed.nextIndex
            default:
                var j = i
                while j < chars.count {
                    let token = chars[j]
                    if token.isWhitespace || token == "," || token == "}" || token == "]" {
                        break
                    }
                    j += 1
                }
                if j > i { markValueConsumed() }
                i = max(j, i + 1)
            }
        }
        return nil
    }
}

// MARK: - Agent parse diagnostics

nonisolated struct AgentParseFailureTrace: Codable, Sendable {
    let id: UUID
    let createdAt: Date
    let parseError: String
    let modelName: String
    let temperature: Double
    let topP: Double
    let maxTokens: Int
    let stepIndex: Int
    let systemPromptPrefix: String
    let userTurnPrefix: String
    let rawOutputPrefix: String
    let streamedThoughtPrefix: String
    let streamedFinalPrefix: String
    let selectedJSONPrefix: String?
    let prefixNoise: String?
    let suffixNoise: String?
}

nonisolated struct AgentParseNoiseTrace: Codable, Sendable {
    let id: UUID
    let createdAt: Date
    let modelName: String
    let temperature: Double
    let topP: Double
    let maxTokens: Int
    let stepIndex: Int
    let systemPromptPrefix: String
    let userTurnPrefix: String
    let rawOutputPrefix: String
    let selectedJSONPrefix: String?
    let prefixNoise: String?
    let suffixNoise: String?
}

nonisolated enum AgentNoiseInspector {
    struct Snapshot: Sendable {
        let selectedJSON: String?
        let prefixNoise: String?
        let suffixNoise: String?
    }

    static func inspect(_ raw: String) -> Snapshot {
        switch AgentJSONCandidateSelector.select(from: raw) {
        case .success(let selection):
            return Snapshot(
                selectedJSON: selection.selectedJSON,
                prefixNoise: selection.prefixNoise,
                suffixNoise: selection.suffixNoise
            )
        case .failure:
            return Snapshot(
                selectedJSON: nil,
                prefixNoise: nonEmpty(AgentThinkBlockSanitizer.redactedForDiagnostics(raw)),
                suffixNoise: nil
            )
        }
    }

    private static func stripFenceNoise(_ text: String) -> String {
        text
            .replacingOccurrences(of: "```json", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "```", with: "")
            .replacingOccurrences(of: "<json>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "</json>", with: "", options: .caseInsensitive)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func nonEmpty(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

nonisolated enum AgentParseFailureRecorder {
    static func record(_ trace: AgentParseFailureTrace) {
        do {
            let directory = try diagnosticsDirectory()
            let url = directory.appendingPathComponent("agent-parse-failures.jsonl", isDirectory: false)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(trace)
            var line = data
            line.append(0x0A)

            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: url, options: [.atomic])
            }
        } catch {
            // Diagnostics must never break chat generation.
        }
    }

    static func diagnosticsDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let directory = base
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent("Agent", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

nonisolated enum AgentParseNoiseRecorder {
    static func record(_ trace: AgentParseNoiseTrace) {
        do {
            let directory = try AgentParseFailureRecorder.diagnosticsDirectory()
            let url = directory.appendingPathComponent("agent-parse-noise.jsonl", isDirectory: false)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(trace)
            var line = data
            line.append(0x0A)

            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: url, options: [.atomic])
            }
        } catch {
            // Diagnostics must never break chat generation.
        }
    }
}

nonisolated struct AgentParseFailureSummaryEntry: Sendable, Hashable {
    let parseError: String
    let prefixSignature: String
    let suffixSignature: String
    let count: Int
}

nonisolated struct AgentParseFailureTrendEntry: Sendable, Hashable {
    let parseError: String
    let prefixSignature: String
    let suffixSignature: String
    let recentCount: Int
    let recentShare: Double
    let baselineShare: Double
    let isRegression: Bool
}

nonisolated struct AgentParseFailureSummary: Sendable {
    let totalLines: Int
    let decodedLines: Int
    let skippedLines: Int
    let topEntries: [AgentParseFailureSummaryEntry]
    let recentLineWindowSize: Int
    let recent24hCount: Int
    let recentLineTopEntries: [AgentParseFailureTrendEntry]
    let recent24hTopEntries: [AgentParseFailureTrendEntry]
}

nonisolated struct AgentParseNoiseSummaryEntry: Sendable, Hashable {
    let modelName: String
    let stepIndex: Int
    let prefixSignature: String
    let suffixSignature: String
    let count: Int
}

nonisolated struct AgentParseNoiseTrendEntry: Sendable, Hashable {
    let modelName: String
    let stepIndex: Int
    let prefixSignature: String
    let suffixSignature: String
    let recentCount: Int
    let recentShare: Double
    let baselineShare: Double
    let isRegression: Bool
}

nonisolated struct AgentParseNoiseSummary: Sendable {
    let totalLines: Int
    let decodedLines: Int
    let skippedLines: Int
    let topEntries: [AgentParseNoiseSummaryEntry]
    let recentLineWindowSize: Int
    let recent24hCount: Int
    let recentLineTopEntries: [AgentParseNoiseTrendEntry]
    let recent24hTopEntries: [AgentParseNoiseTrendEntry]
}

nonisolated enum AgentParseFailureSummaryLoader {
    private static let recentLineWindowSize = 50
    private struct Key: Hashable {
        let parseError: String
        let prefixSignature: String
        let suffixSignature: String
    }

    static func load(topN: Int = 5) -> AgentParseFailureSummary {
        do {
            let directory = try AgentParseFailureRecorder.diagnosticsDirectory()
            let url = directory.appendingPathComponent("agent-parse-failures.jsonl", isDirectory: false)
            guard let data = try? Data(contentsOf: url), !data.isEmpty else {
                return AgentParseFailureSummary(
                    totalLines: 0,
                    decodedLines: 0,
                    skippedLines: 0,
                    topEntries: [],
                    recentLineWindowSize: 0,
                    recent24hCount: 0,
                    recentLineTopEntries: [],
                    recent24hTopEntries: []
                )
            }
            let text = String(decoding: data, as: UTF8.self)
            return load(fromJSONLText: text, topN: topN)
        } catch {
            return AgentParseFailureSummary(
                totalLines: 0,
                decodedLines: 0,
                skippedLines: 0,
                topEntries: [],
                recentLineWindowSize: 0,
                recent24hCount: 0,
                recentLineTopEntries: [],
                recent24hTopEntries: []
            )
        }
    }

    static func load(fromJSONLText text: String, topN: Int = 5) -> AgentParseFailureSummary {
        let lines = text.split(whereSeparator: \.isNewline)
        if lines.isEmpty {
            return AgentParseFailureSummary(
                totalLines: 0,
                decodedLines: 0,
                skippedLines: 0,
                topEntries: [],
                recentLineWindowSize: 0,
                recent24hCount: 0,
                recentLineTopEntries: [],
                recent24hTopEntries: []
            )
        }

        var counts: [Key: Int] = [:]
        var traces: [(key: Key, createdAt: Date)] = []
        var decodedLines = 0
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for line in lines {
            guard let data = String(line).data(using: .utf8),
                  let trace = try? decoder.decode(AgentParseFailureTrace.self, from: data) else {
                continue
            }
            decodedLines += 1
            let key = Key(
                parseError: trace.parseError,
                prefixSignature: noiseSignature(trace.prefixNoise),
                suffixSignature: noiseSignature(trace.suffixNoise)
            )
            counts[key, default: 0] += 1
            traces.append((key: key, createdAt: trace.createdAt))
        }

        let topEntries = counts
            .map {
                AgentParseFailureSummaryEntry(
                    parseError: $0.key.parseError,
                    prefixSignature: $0.key.prefixSignature,
                    suffixSignature: $0.key.suffixSignature,
                    count: $0.value
                )
            }
            .sorted {
                if $0.count != $1.count { return $0.count > $1.count }
                if $0.parseError != $1.parseError { return $0.parseError < $1.parseError }
                if $0.prefixSignature != $1.prefixSignature { return $0.prefixSignature < $1.prefixSignature }
                return $0.suffixSignature < $1.suffixSignature
            }

        let recentLineWindow = Array(traces.suffix(recentLineWindowSize))
        let recent24hWindow: [(key: Key, createdAt: Date)] = {
            guard let newest = traces.map(\.createdAt).max() else { return [] }
            let cutoff = newest.addingTimeInterval(-86_400)
            return traces.filter { $0.createdAt >= cutoff }
        }()

        let recentLineTopEntries = trendEntries(
            baselineCounts: counts,
            recentWindow: recentLineWindow,
            totalBaseline: decodedLines,
            topN: topN
        )
        let recent24hTopEntries = trendEntries(
            baselineCounts: counts,
            recentWindow: recent24hWindow,
            totalBaseline: decodedLines,
            topN: topN
        )

        return AgentParseFailureSummary(
            totalLines: lines.count,
            decodedLines: decodedLines,
            skippedLines: lines.count - decodedLines,
            topEntries: Array(topEntries.prefix(max(0, topN))),
            recentLineWindowSize: recentLineWindow.count,
            recent24hCount: recent24hWindow.count,
            recentLineTopEntries: recentLineTopEntries,
            recent24hTopEntries: recent24hTopEntries
        )
    }

    static func developerText(topN: Int = 5) -> String {
        let summary = load(topN: topN)
        if summary.totalLines == 0 {
            return "• Parse-failure traces: 0"
        }

        var lines: [String] = [
            "• Parse-failure traces: \(summary.decodedLines) loaded (\(summary.skippedLines) skipped)"
        ]
        if summary.topEntries.isEmpty {
            lines.append("• Top signatures: none")
            return lines.joined(separator: "\n")
        }

        lines.append("• Top signatures (all-time):")
        for entry in summary.topEntries {
            lines.append("  - \(entry.count)x \(entry.parseError) | pre=\(entry.prefixSignature) | suf=\(entry.suffixSignature)")
        }
        lines.append("• Recent windows: last \(summary.recentLineWindowSize) lines, last 24h \(summary.recent24hCount) lines")
        lines.append("• Recent top signatures (last \(summary.recentLineWindowSize) lines):")
        if summary.recentLineTopEntries.isEmpty {
            lines.append("  - none")
        } else {
            appendTrendLines(summary.recentLineTopEntries, to: &lines)
        }
        lines.append("• Recent top signatures (last 24h):")
        if summary.recent24hTopEntries.isEmpty {
            lines.append("  - none")
        } else {
            appendTrendLines(summary.recent24hTopEntries, to: &lines)
        }
        return lines.joined(separator: "\n")
    }

    private static func trendEntries(
        baselineCounts: [Key: Int],
        recentWindow: [(key: Key, createdAt: Date)],
        totalBaseline: Int,
        topN: Int
    ) -> [AgentParseFailureTrendEntry] {
        guard !recentWindow.isEmpty, totalBaseline > 0 else { return [] }
        var recentCounts: [Key: Int] = [:]
        for trace in recentWindow {
            recentCounts[trace.key, default: 0] += 1
        }
        let recentTotal = recentWindow.count

        return recentCounts
            .map { key, recentCount in
                let baselineCount = baselineCounts[key, default: 0]
                let recentShare = Double(recentCount) / Double(recentTotal)
                let baselineShare = Double(baselineCount) / Double(totalBaseline)
                return AgentParseFailureTrendEntry(
                    parseError: key.parseError,
                    prefixSignature: key.prefixSignature,
                    suffixSignature: key.suffixSignature,
                    recentCount: recentCount,
                    recentShare: recentShare,
                    baselineShare: baselineShare,
                    isRegression: recentShare > baselineShare
                )
            }
            .sorted {
                if $0.recentCount != $1.recentCount { return $0.recentCount > $1.recentCount }
                if $0.parseError != $1.parseError { return $0.parseError < $1.parseError }
                if $0.prefixSignature != $1.prefixSignature { return $0.prefixSignature < $1.prefixSignature }
                return $0.suffixSignature < $1.suffixSignature
            }
            .prefix(max(0, topN))
            .map { $0 }
    }

    private static func appendTrendLines(_ entries: [AgentParseFailureTrendEntry], to lines: inout [String]) {
        for entry in entries {
            let recentPct = Int((entry.recentShare * 100).rounded())
            let baselinePct = Int((entry.baselineShare * 100).rounded())
            let trend = entry.isRegression ? "↑ regression" : "≈ baseline"
            lines.append(
                "  - \(entry.recentCount)x \(entry.parseError) | pre=\(entry.prefixSignature) | suf=\(entry.suffixSignature) | recent=\(recentPct)% baseline=\(baselinePct)% \(trend)"
            )
        }
    }

    private static func noiseSignature(_ value: String?) -> String {
        let normalized = normalizeNoise(value)
        guard !normalized.isEmpty else { return "none#00" }
        let snippet = String(normalized.prefix(24))
        let bucket = String(format: "%02X", stableHash(normalized) % 64)
        return "\(snippet)#\(bucket)"
    }

    private static func normalizeNoise(_ value: String?) -> String {
        guard let value else { return "" }
        var text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "" }
        text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        return text.lowercased()
    }

    private static func stableHash(_ text: String) -> UInt64 {
        var hash: UInt64 = 5381
        for byte in text.utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt64(byte)
        }
        return hash
    }
}

nonisolated enum AgentParseNoiseSummaryLoader {
    private static let recentLineWindowSize = 50
    private struct Key: Hashable {
        let modelName: String
        let stepIndex: Int
        let prefixSignature: String
        let suffixSignature: String
    }

    static func load(topN: Int = 5) -> AgentParseNoiseSummary {
        do {
            let directory = try AgentParseFailureRecorder.diagnosticsDirectory()
            let url = directory.appendingPathComponent("agent-parse-noise.jsonl", isDirectory: false)
            guard let data = try? Data(contentsOf: url), !data.isEmpty else {
                return AgentParseNoiseSummary(
                    totalLines: 0,
                    decodedLines: 0,
                    skippedLines: 0,
                    topEntries: [],
                    recentLineWindowSize: 0,
                    recent24hCount: 0,
                    recentLineTopEntries: [],
                    recent24hTopEntries: []
                )
            }
            let text = String(decoding: data, as: UTF8.self)
            return load(fromJSONLText: text, topN: topN)
        } catch {
            return AgentParseNoiseSummary(
                totalLines: 0,
                decodedLines: 0,
                skippedLines: 0,
                topEntries: [],
                recentLineWindowSize: 0,
                recent24hCount: 0,
                recentLineTopEntries: [],
                recent24hTopEntries: []
            )
        }
    }

    static func load(fromJSONLText text: String, topN: Int = 5) -> AgentParseNoiseSummary {
        let lines = text.split(whereSeparator: \.isNewline)
        if lines.isEmpty {
            return AgentParseNoiseSummary(
                totalLines: 0,
                decodedLines: 0,
                skippedLines: 0,
                topEntries: [],
                recentLineWindowSize: 0,
                recent24hCount: 0,
                recentLineTopEntries: [],
                recent24hTopEntries: []
            )
        }

        var counts: [Key: Int] = [:]
        var traces: [(key: Key, createdAt: Date)] = []
        var decodedLines = 0
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for line in lines {
            guard let data = String(line).data(using: .utf8),
                  let trace = try? decoder.decode(AgentParseNoiseTrace.self, from: data) else {
                continue
            }
            decodedLines += 1
            let key = Key(
                modelName: trace.modelName,
                stepIndex: trace.stepIndex,
                prefixSignature: noiseSignature(trace.prefixNoise),
                suffixSignature: noiseSignature(trace.suffixNoise)
            )
            counts[key, default: 0] += 1
            traces.append((key: key, createdAt: trace.createdAt))
        }

        let topEntries = counts
            .map {
                AgentParseNoiseSummaryEntry(
                    modelName: $0.key.modelName,
                    stepIndex: $0.key.stepIndex,
                    prefixSignature: $0.key.prefixSignature,
                    suffixSignature: $0.key.suffixSignature,
                    count: $0.value
                )
            }
            .sorted {
                if $0.count != $1.count { return $0.count > $1.count }
                if $0.modelName != $1.modelName { return $0.modelName < $1.modelName }
                if $0.stepIndex != $1.stepIndex { return $0.stepIndex < $1.stepIndex }
                if $0.prefixSignature != $1.prefixSignature { return $0.prefixSignature < $1.prefixSignature }
                return $0.suffixSignature < $1.suffixSignature
            }

        let recentLineWindow = Array(traces.suffix(recentLineWindowSize))
        let recent24hWindow: [(key: Key, createdAt: Date)] = {
            guard let newest = traces.map(\.createdAt).max() else { return [] }
            let cutoff = newest.addingTimeInterval(-86_400)
            return traces.filter { $0.createdAt >= cutoff }
        }()

        let recentLineTopEntries = trendEntries(
            baselineCounts: counts,
            recentWindow: recentLineWindow,
            totalBaseline: decodedLines,
            topN: topN
        )
        let recent24hTopEntries = trendEntries(
            baselineCounts: counts,
            recentWindow: recent24hWindow,
            totalBaseline: decodedLines,
            topN: topN
        )

        return AgentParseNoiseSummary(
            totalLines: lines.count,
            decodedLines: decodedLines,
            skippedLines: lines.count - decodedLines,
            topEntries: Array(topEntries.prefix(max(0, topN))),
            recentLineWindowSize: recentLineWindow.count,
            recent24hCount: recent24hWindow.count,
            recentLineTopEntries: recentLineTopEntries,
            recent24hTopEntries: recent24hTopEntries
        )
    }

    static func developerText(topN: Int = 5) -> String {
        let summary = load(topN: topN)
        if summary.totalLines == 0 {
            return "• Recoverable noise traces: 0"
        }

        var lines: [String] = [
            "• Recoverable noise traces: \(summary.decodedLines) loaded (\(summary.skippedLines) skipped)"
        ]
        if summary.topEntries.isEmpty {
            lines.append("• Top recurring signatures: none")
            return lines.joined(separator: "\n")
        }

        lines.append("• Top recurring signatures (all-time):")
        for entry in summary.topEntries {
            lines.append("  - \(entry.count)x model=\(entry.modelName) step=\(entry.stepIndex) | pre=\(entry.prefixSignature) | suf=\(entry.suffixSignature)")
        }
        lines.append("• Recent windows: last \(summary.recentLineWindowSize) lines, last 24h \(summary.recent24hCount) lines")
        lines.append("• Recent recurring signatures (last 50 lines):")
        if summary.recentLineTopEntries.isEmpty {
            lines.append("  - none")
        } else {
            appendTrendLines(summary.recentLineTopEntries, to: &lines)
        }
        lines.append("• Recent recurring signatures (last 24h):")
        if summary.recent24hTopEntries.isEmpty {
            lines.append("  - none")
        } else {
            appendTrendLines(summary.recent24hTopEntries, to: &lines)
        }
        return lines.joined(separator: "\n")
    }

    private static func trendEntries(
        baselineCounts: [Key: Int],
        recentWindow: [(key: Key, createdAt: Date)],
        totalBaseline: Int,
        topN: Int
    ) -> [AgentParseNoiseTrendEntry] {
        guard !recentWindow.isEmpty, totalBaseline > 0 else { return [] }
        var recentCounts: [Key: Int] = [:]
        for trace in recentWindow {
            recentCounts[trace.key, default: 0] += 1
        }
        let recentTotal = recentWindow.count

        return recentCounts
            .map { key, recentCount in
                let baselineCount = baselineCounts[key, default: 0]
                let recentShare = Double(recentCount) / Double(recentTotal)
                let baselineShare = Double(baselineCount) / Double(totalBaseline)
                return AgentParseNoiseTrendEntry(
                    modelName: key.modelName,
                    stepIndex: key.stepIndex,
                    prefixSignature: key.prefixSignature,
                    suffixSignature: key.suffixSignature,
                    recentCount: recentCount,
                    recentShare: recentShare,
                    baselineShare: baselineShare,
                    isRegression: recentShare > baselineShare
                )
            }
            .sorted {
                if $0.recentCount != $1.recentCount { return $0.recentCount > $1.recentCount }
                if $0.modelName != $1.modelName { return $0.modelName < $1.modelName }
                if $0.stepIndex != $1.stepIndex { return $0.stepIndex < $1.stepIndex }
                if $0.prefixSignature != $1.prefixSignature { return $0.prefixSignature < $1.prefixSignature }
                return $0.suffixSignature < $1.suffixSignature
            }
            .prefix(max(0, topN))
            .map { $0 }
    }

    private static func appendTrendLines(_ entries: [AgentParseNoiseTrendEntry], to lines: inout [String]) {
        for entry in entries {
            let recentPct = Int((entry.recentShare * 100).rounded())
            let baselinePct = Int((entry.baselineShare * 100).rounded())
            let trend = entry.isRegression ? "↑ regression" : "≈ baseline"
            lines.append(
                "  - \(entry.recentCount)x model=\(entry.modelName) step=\(entry.stepIndex) | pre=\(entry.prefixSignature) | suf=\(entry.suffixSignature) | recent=\(recentPct)% baseline=\(baselinePct)% \(trend)"
            )
        }
    }

    private static func noiseSignature(_ value: String?) -> String {
        let normalized = normalizeNoise(value)
        guard !normalized.isEmpty else { return "∅#00" }
        let snippet = String(normalized.prefix(24))
        let bucket = String(format: "%02X", stableHash(normalized) % 64)
        return "\(snippet)#\(bucket)"
    }

    private static func normalizeNoise(_ value: String?) -> String {
        guard let value else { return "" }
        var text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "" }
        text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        return text.lowercased()
    }

    private static func stableHash(_ text: String) -> UInt64 {
        var hash: UInt64 = 5381
        for byte in text.utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt64(byte)
        }
        return hash
    }
}

// MARK: - AgentService

@MainActor
final class AgentService {
    static let shared = AgentService()
    private nonisolated static let structuredTurnMaxTokenCap = 384
    private nonisolated static let structuredTurnMinTokenCap = 128
    private nonisolated static let structuredContextNoteCharCap = 280
    private nonisolated static let structuredUserMessageCharCap = 900
    private nonisolated static let structuredHistoryTurnCharCap = 180
    private nonisolated static let structuredHistoryTotalCharCap = 540
    private nonisolated static let structuredScratchpadCharCap = 900
    private nonisolated static let structuredToolCountCap = 12
    private nonisolated static let structuredToolListCharCap = 780
    private nonisolated static let structuredToolDescriptionCharCap = 88
    private nonisolated static let structuredPromptPreflightSafetyTokens = 128
    private nonisolated static let contextWindowExceededRawOutputPrefix = "Prompt exceeded context window before generation"
    private nonisolated static let structuredAgentModelSlot: LumenModelSlot = .executor
    nonisolated static let structuredAgentResponseSchema = #"{"type":"object","oneOf":[{"required":["action"],"properties":{"thought":{"type":"string"},"action":{"type":"object","required":["tool","args"],"properties":{"tool":{"type":"string"},"args":{"type":"object"}}},"additionalProperties":false},{"required":["final"],"properties":{"thought":{"type":"string"},"final":{"type":"string"}},"additionalProperties":false}]}"#

    private struct StructuredTurnGenerationDiagnostics: Sendable {
        let generationElapsedMs: Int
        let firstTokenLatencyMs: Int?
        let outputTokenCount: Int?
        let estimatedPromptTokenCount: Int?
        let maxTokensRequested: Int
        let maxTokensEffective: Int
        let promptCharCount: Int?
        let emptyOutputReason: String?
        let streamStarted: Bool?
        let selectedRuntime: String?
        let selectedAdapter: String?
        let modelIdentifier: String?
        let modelLoaded: Bool?
        let stopSequences: [String]
        let temperature: Double?
        let topP: Double?
        let cancellationStateBeforeStream: String?
        let firstChunkReceived: Bool?
        let textChunkCount: Int?
        let finalChunkReceived: Bool?
        let streamTerminationReason: String?
    }

    private struct StructuredPromptPreflight: Sendable {
        let request: GenerateRequest
        let contextSize: Int
        let finalPromptChars: Int
        let estimatedPromptTokens: Int
        let fits: Bool

        var totalEstimatedTokens: Int {
            estimatedPromptTokens + request.maxTokens + AgentService.structuredPromptPreflightSafetyTokens
        }
    }

    /// Executes an agent structured turn and streams progress events.
    /// - Parameters:
    ///   - req: The agent request specifying the prompt, conversation context, tools, and generation parameters.
    /// - Returns: An async stream of events representing the agent's thoughts, actions, observations, and final response.
    func run(_ req: AgentRequest) -> AsyncStream<AgentEvent> {
        run(req, options: .default)
    }

    func run(_ req: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentEvent> {
        if options.diagnosticsEnabled, options.allowDeterministicCompatibility {
            return LegacyAgentCompatibilityBridge.runSlotAgentCompatibility(req, options: options)
        }
        if options.allowDeterministicCompatibility,
           SlotAgentService.canCompleteThroughDeterministicCompatibility(req) {
            var compatibilityOptions = options
            compatibilityOptions.groundingMode = .slotAgent
            compatibilityOptions.allowDegradedGrounding = false
            compatibilityOptions.preventDoubleGrounding = true
            return LegacyAgentCompatibilityBridge.runSlotAgentCompatibility(req, options: compatibilityOptions)
        }

        return AsyncStream { continuation in
            let task = Task { @MainActor in
                let effectiveRequest = options.allowDegradedGrounding
                    ? Self.applyLegacyGroundingAssembly(req)
                    : req
                await self.runLoop(effectiveRequest, options: options, continuation: continuation)
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func runLoop(_ req: AgentRequest, options: LegacyAgentRunOptions, continuation: AsyncStream<AgentEvent>.Continuation) async {
        var steps: [AgentStep] = []
        var observations: [(tool: String, result: String)] = []
        var executedActionKeys: Set<String> = []
        var scratchpad = ""
        var finalAnswer = ""
        let memoryCommandPlan = MemoryCommandPlan.saveThenRecall(from: Self.sanitizedStructuredUserMessage(req.userMessage))

        let sys = buildSystemPrompt(req: req)
        let maxSteps = max(1, req.maxSteps)

        stepsLoop: for stepIndex in 0..<maxSteps {
            if Task.isCancelled { break }

            let userTurn = buildAgentUserTurn(req: req, stepIndex: stepIndex, scratchpad: scratchpad)

            var genReq = GenerateRequest(
                systemPrompt: sys,
                history: [],
                userMessage: userTurn,
                temperature: agentTemperature(from: req.temperature),
                topP: agentTopP(from: req.topP),
                repetitionPenalty: max(req.repetitionPenalty, 1.05),
                maxTokens: structuredTurnMaxTokens(from: req.maxTokens, req: req, stepIndex: stepIndex),
                modelName: "agent-json",
                relevantMemories: [],
                attachments: [],
                responseFormat: .constrainedJSON(schema: Self.structuredAgentResponseSchema),
                allowsMemoryPressureContinuation: options.allowsMemoryPressureContinuation
            )

            var scanner = StreamingJSONScanner()
            var raw = ""
            let thoughtStepID = UUID()
            var thoughtStepYielded = false
            var streamedFinalLen = 0
            var generationStartedAt = Date()
            var firstTokenLatencyMs: Int?
            var outputChunks = 0
            var streamStarted = false
            var finalChunkReceived = false
            var completedPayload: CompletedGenerationTracePayload?
            var preflight = await preflightAgentJSONPrompt(genReq)
            var forcedParseError: AgentTurnParseError?
            var runtimePreflightFailureReason: String?

            if !preflight.fits {
                let compactReq = Self.agentJSONContextCompactionRequest(from: genReq)
                let compactPreflight = await preflightAgentJSONPrompt(compactReq)
                if compactPreflight.fits {
                    genReq = compactReq
                    preflight = compactPreflight
                } else {
                    genReq = compactReq
                    preflight = compactPreflight
                    raw = Self.contextWindowExceededRawOutputPrefix
                    forcedParseError = .contextWindowExceeded
                }
            }

            if forcedParseError == nil {
                let runtimePreflight = await ExecutorRuntimePreflight.checkReadiness(
                    allowsLoadedMemoryPressureContinuation: options.allowsMemoryPressureContinuation
                )
                if !runtimePreflight.passed {
                    runtimePreflightFailureReason = runtimePreflight.reason
                    forcedParseError = .empty
                } else {
                    streamStarted = true
                    for await token in await AppLlamaService.shared.stream(genReq, slot: Self.structuredAgentModelSlot) {
                        if Task.isCancelled { break }
                        switch token {
                        case .text(let s):
                            if firstTokenLatencyMs == nil {
                                firstTokenLatencyMs = Int(Date().timeIntervalSince(generationStartedAt) * 1000)
                            }
                            outputChunks += 1
                            raw += s
                            for event in scanner.feed(s) {
                                switch event {
                                case .thoughtDelta:
                                    let current = scanner.thought
                                    if !thoughtStepYielded {
                                        continuation.yield(.step(AgentStep(id: thoughtStepID, kind: .thought, content: current)))
                                        thoughtStepYielded = true
                                    } else {
                                        continuation.yield(.stepDelta(id: thoughtStepID, text: current))
                                    }
                                case .finalDelta(let delta):
                                    streamedFinalLen += delta.count
                                    continuation.yield(.finalDelta(delta))
                                }
                            }
                        case .done:
                            finalChunkReceived = true
                            break
                        }
                    }
                    completedPayload = await AppLlamaService.shared.takeCompletedTracePayload(requestID: genReq.id)
                }
            }

            if !Task.isCancelled,
               forcedParseError == nil,
               Self.runtimeFailureParseError(from: raw) == nil,
               raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let retryGenReq = Self.agentJSONEmptyOutputRetryRequest(from: genReq, userTurn: userTurn)
                let retryPreflight = await preflightAgentJSONPrompt(retryGenReq)
                genReq = retryGenReq
                preflight = retryPreflight
                completedPayload = nil
                if !retryPreflight.fits {
                    raw = Self.contextWindowExceededRawOutputPrefix
                    forcedParseError = .contextWindowExceeded
                } else {
                    scanner = StreamingJSONScanner()
                    raw = ""
                    streamedFinalLen = 0
                    firstTokenLatencyMs = nil
                    outputChunks = 0
                    streamStarted = false
                    finalChunkReceived = false
                    generationStartedAt = Date()

                    streamStarted = true
                    for await token in await AppLlamaService.shared.stream(retryGenReq, slot: Self.structuredAgentModelSlot) {
                        if Task.isCancelled { break }
                        switch token {
                        case .text(let s):
                            if firstTokenLatencyMs == nil {
                                firstTokenLatencyMs = Int(Date().timeIntervalSince(generationStartedAt) * 1000)
                            }
                            outputChunks += 1
                            raw += s
                            for event in scanner.feed(s) {
                                switch event {
                                case .thoughtDelta:
                                    let current = scanner.thought
                                    if !thoughtStepYielded {
                                        continuation.yield(.step(AgentStep(id: thoughtStepID, kind: .thought, content: current)))
                                        thoughtStepYielded = true
                                    } else {
                                        continuation.yield(.stepDelta(id: thoughtStepID, text: current))
                                    }
                                case .finalDelta(let delta):
                                    streamedFinalLen += delta.count
                                    continuation.yield(.finalDelta(delta))
                                }
                            }
                        case .done:
                            finalChunkReceived = true
                            break
                        }
                    }
                    completedPayload = await AppLlamaService.shared.takeCompletedTracePayload(requestID: retryGenReq.id)
                }
            }

            if Task.isCancelled { break }

            let turn = forcedParseError
                .map { AgentTurn(thought: nil, action: nil, final: nil, parseError: $0, hadNoise: false) }
                ?? Self.runtimeFailureParseError(from: raw)
                    .map { AgentTurn(thought: nil, action: nil, final: nil, parseError: $0, hadNoise: false) }
                ?? AgentTurnParser.parse(raw)
            let trimmedRaw = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let localEmptyReason = trimmedRaw.isEmpty
                ? runtimePreflightFailureReason ?? Self.agentJSONEmptyStreamReason(
                    streamStarted: streamStarted,
                    textChunkCount: outputChunks,
                    finalChunkReceived: finalChunkReceived,
                    taskCancelled: Task.isCancelled,
                    maxTokensEffective: genReq.maxTokens
                )
                : nil
            let generationDiagnostics = StructuredTurnGenerationDiagnostics(
                generationElapsedMs: Int(Date().timeIntervalSince(generationStartedAt) * 1000),
                firstTokenLatencyMs: firstTokenLatencyMs,
                outputTokenCount: completedPayload?.outputTokenCount ?? (trimmedRaw.isEmpty ? 0 : nil),
                estimatedPromptTokenCount: completedPayload?.estimatedPromptTokenCount ?? preflight.estimatedPromptTokens,
                maxTokensRequested: completedPayload?.maxTokensRequested ?? req.maxTokens,
                maxTokensEffective: completedPayload?.maxTokensEffective ?? genReq.maxTokens,
                promptCharCount: completedPayload?.promptCharCount ?? preflight.finalPromptChars,
                emptyOutputReason: completedPayload?.emptyOutputReason ?? localEmptyReason,
                streamStarted: completedPayload?.streamStarted ?? streamStarted,
                selectedRuntime: completedPayload?.selectedRuntime,
                selectedAdapter: completedPayload?.selectedAdapter,
                modelIdentifier: completedPayload?.modelIdentifier,
                modelLoaded: completedPayload?.modelLoaded,
                stopSequences: completedPayload?.stopSequences ?? [],
                temperature: completedPayload?.temperature ?? genReq.temperature,
                topP: completedPayload?.topP ?? genReq.topP,
                cancellationStateBeforeStream: completedPayload?.cancellationStateBeforeStream,
                firstChunkReceived: completedPayload?.firstChunkReceived ?? (outputChunks > 0),
                textChunkCount: completedPayload?.textChunkCount ?? outputChunks,
                finalChunkReceived: completedPayload?.finalChunkReceived ?? finalChunkReceived,
                streamTerminationReason: completedPayload?.streamTerminationReason ?? runtimePreflightFailureReason
            )
            recordAgentModelTurnTrace(
                req: req,
                userTurn: userTurn,
                raw: raw,
                turn: turn,
                stepIndex: stepIndex,
                diagnostics: generationDiagnostics
            )

            if turn.hadNoise {
                recordRecoverableNoise(
                    req: req,
                    raw: raw,
                    systemPrompt: sys,
                    userTurn: userTurn,
                    stepIndex: stepIndex
                )
            }

            // Commit thought step with the fully-parsed value (in case streaming extracted less).
            if let thought = turn.thought, !thought.isEmpty {
                let step = AgentStep(id: thoughtStepID, kind: .thought, content: thought)
                if let idx = steps.firstIndex(where: { $0.id == thoughtStepID }) {
                    steps[idx] = step
                } else {
                    steps.append(step)
                }
                if thoughtStepYielded {
                    continuation.yield(.stepDelta(id: thoughtStepID, text: thought))
                } else {
                    continuation.yield(.step(step))
                }
                scratchpad += "\nThought: \(thought)"
            } else if thoughtStepYielded, !scanner.thought.isEmpty {
                // Parser lost the thought but we streamed one — keep what we streamed.
                let partial = scanner.thought
                let step = AgentStep(id: thoughtStepID, kind: .thought, content: partial)
                steps.append(step)
                scratchpad += "\nThought: \(partial)"
            }

            var actionToExecute: AgentAction?
            if let parsedAction = turn.action {
                let repair = Self.repairedMemoryActionIfNeeded(
                    modelAction: parsedAction,
                    memoryPlan: memoryCommandPlan,
                    steps: steps
                )
                if let reflection = repair.reflection {
                    steps.append(reflection)
                    continuation.yield(.step(reflection))
                }
                actionToExecute = repair.action
            } else if turn.parseError == .missingActionTool,
                      let repaired = Self.repairMissingToolActionIfPossible(
                        raw: raw,
                        req: req,
                        observations: observations
                      ) {
                let reflection = AgentStep(
                    kind: .reflection,
                    content: repaired.diagnostic,
                    toolID: repaired.action.tool,
                    toolArgs: repaired.action.args.stringCoerced
                )
                steps.append(reflection)
                continuation.yield(.step(reflection))
                actionToExecute = repaired.action
            } else if turn.final?.isEmpty == false,
                      let requiredMemoryAction = Self.nextRequiredMemoryAction(memoryPlan: memoryCommandPlan, steps: steps) {
                let reflection = AgentStep(
                    kind: .reflection,
                    content: "Memory save-then-recall invariant repaired a premature final before required memory actions completed."
                )
                steps.append(reflection)
                continuation.yield(.step(reflection))
                actionToExecute = requiredMemoryAction
            }

            // Action path
            if let action = actionToExecute {
                let canonicalActionTool = ToolRouteGuard.canonicalToolID(action.tool)
                guard let _ = ToolRegistry.find(id: canonicalActionTool) else {
                    let obs = AgentStep(kind: .observation, content: "Unknown tool: \(action.tool). Emit a final turn instead.", toolID: canonicalActionTool)
                    steps.append(obs)
                    continuation.yield(.step(obs))
                    observations.append((action.tool, obs.content))
                    scratchpad += "\nAction: \(action.displayContent)\nObservation: \(compactScratchpadObservation(obs.content))"
                    if let locationObservation = currentLocationScratchpadContext(from: obs.content) {
                        scratchpad += "\nContext: \(locationObservation)"
                    }
                    continue
                }
                if executedActionKeys.contains(action.dedupeKey) {
                    let reflection = AgentStep(kind: .reflection, content: "Duplicate tool call blocked: \(action.displayContent). Synthesizing answer from observations.")
                    steps.append(reflection)
                    continuation.yield(.step(reflection))
                    finalAnswer = await synthesizeFallback(req: req, observations: observations, reason: .duplicate)
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }
                executedActionKeys.insert(action.dedupeKey)

                if ToolRouteGuard.requiresUserApproval(canonicalActionTool) {
                    let approval = Self.approvalBoundaryFinal(for: canonicalActionTool, action: action)
                    let step = AgentStep(kind: .approvalBoundary, content: approval, toolID: canonicalActionTool, toolArgs: action.args.stringCoerced)
                    steps.append(step)
                    continuation.yield(.step(step))
                    finalAnswer = approval
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }

                let actionStep = AgentStep(kind: .action, content: action.displayContent, toolID: canonicalActionTool, toolArgs: action.args.stringCoerced)
                steps.append(actionStep)
                continuation.yield(.step(actionStep))

                let isEnabled = req.availableTools.contains { ToolRouteGuard.canonicalToolID($0.id) == canonicalActionTool }
                let result: String
                if !isEnabled {
                    result = "Tool \(canonicalActionTool) is disabled. Enable it in Tools."
                } else {
                    result = await SecureToolRegistry.shared.executeLegacyTool(
                        canonicalActionTool,
                        arguments: action.args,
                        approval: .autonomous,
                        conversationID: req.conversationID,
                        turnID: req.turnID
                    )
                }

                let obs = AgentStep(kind: .observation, content: result, toolID: canonicalActionTool)
                steps.append(obs)
                continuation.yield(.step(obs))
                observations.append((canonicalActionTool, result))
                scratchpad += "\nAction: \(action.displayContent)\nObservation: \(compactScratchpadObservation(result))"
                if let locationObservation = currentLocationScratchpadContext(from: result) {
                    scratchpad += "\nContext: \(locationObservation)"
                }

                if let phoneContinuation = Self.phoneCallContinuationAfterContactSearchIfNeeded(
                    actionTool: action.tool,
                    observation: result,
                    prompt: req.userMessage,
                    availableToolIDs: Set(req.availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })
                ) {
                    steps.append(phoneContinuation.step)
                    continuation.yield(.step(phoneContinuation.step))
                    finalAnswer = phoneContinuation.text
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }

                if stepIndex == maxSteps - 1 {
                    finalAnswer = await synthesizeFallback(req: req, observations: observations, reason: .maxSteps)
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }
                continue
            }

            // Final path
            if let final = turn.final, !final.isEmpty {
                finalAnswer = final
                if streamedFinalLen == 0 {
                    continuation.yield(.finalDelta(final))
                } else if streamedFinalLen < final.count {
                    // Catch up any characters the streaming scanner missed (e.g. after escape).
                    let tail = String(final.suffix(final.count - streamedFinalLen))
                    if !tail.isEmpty { continuation.yield(.finalDelta(tail)) }
                }
                break stepsLoop
            }

            // Malformed / empty output - repair into a user-facing final and persist diagnostics.
            if let parseError = turn.parseError {
                recordParseFailure(
                    req: req,
                    parseError: parseError,
                    raw: raw,
                    scanner: scanner,
                    systemPrompt: sys,
                    userTurn: userTurn,
                    stepIndex: stepIndex
                )

                if parseError == .contextWindowExceeded {
                    let reflection = AgentStep(
                        kind: .reflection,
                        content: "Structured agent prompt exceeded the executor context window before generation."
                    )
                    steps.append(reflection)
                    continuation.yield(.step(reflection))
                    finalAnswer = "I couldn't run the structured agent turn because the prompt exceeded the local model context window."
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }

                if parseError == .empty, raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    let reason = generationDiagnostics.emptyOutputReason ?? "unknownEmptyStream"
                    let reflection = AgentStep(
                        kind: .reflection,
                        content: "Structured agent-json stream produced no usable text. Reason: \(reason)."
                    )
                    steps.append(reflection)
                    continuation.yield(.step(reflection))
                    finalAnswer = "I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: \(reason)."
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }

                if Self.hasUsableObservation(for: IntentRouter.classify(Self.sanitizedStructuredUserMessage(req.userMessage)).intent, observations: observations) {
                    let reflection = AgentStep(kind: .reflection, content: "Malformed structured turn repaired by synthesizing from existing tool observations.")
                    steps.append(reflection)
                    continuation.yield(.step(reflection))
                    finalAnswer = await synthesizeFallback(req: req, observations: observations, reason: .malformed)
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }

                if let recovery = await Self.structuredParseFailureRecovery(req: req, options: options) {
                    for step in recovery.steps {
                        steps.append(step)
                        continuation.yield(.step(step))
                    }
                    finalAnswer = recovery.text
                    continuation.yield(.finalDelta(finalAnswer))
                    break stepsLoop
                }

                let reflectionText = diagnosticReflection(for: parseError, raw: raw)
                let reflection = AgentStep(kind: .reflection, content: reflectionText)
                steps.append(reflection)
                continuation.yield(.step(reflection))

                let streamedFinal = scanner.final.trimmingCharacters(in: .whitespacesAndNewlines)
                if !streamedFinal.isEmpty {
                    finalAnswer = streamedFinal
                } else if !observations.isEmpty {
                    finalAnswer = await synthesizeFallback(req: req, observations: observations, reason: .malformed)
                } else {
                    finalAnswer = await synthesizeUnstructuredFallback(
                        req: req,
                        rawOutput: raw,
                        streamedThought: scanner.thought,
                        parseError: parseError
                    )
                }

                if streamedFinalLen == 0 {
                    continuation.yield(.finalDelta(finalAnswer))
                }
                break stepsLoop
            }

            // Nothing at all.
            finalAnswer = observations.isEmpty
                ? await synthesizeUnstructuredFallback(req: req, rawOutput: raw, streamedThought: scanner.thought, parseError: .empty)
                : await synthesizeFallback(req: req, observations: observations, reason: .empty)
            continuation.yield(.finalDelta(finalAnswer))
            break
        }

        if Task.isCancelled {
            continuation.finish()
            return
        }

        finalAnswer = Self.postprocessStructuredFinalAnswer(
            finalAnswer,
            req: req,
            observations: observations,
            steps: steps
        )
        continuation.yield(.done(finalText: finalAnswer, steps: steps))
        continuation.finish()
    }

    private nonisolated static func postprocessStructuredFinalAnswer(
        _ finalAnswer: String,
        req: AgentRequest,
        observations: [(tool: String, result: String)],
        steps: [AgentStep]
    ) -> String {
        let prompt = sanitizedStructuredUserMessage(req.userMessage)
        let routing = IntentRouter.classify(prompt)

        if routing.intent == .phoneCall,
           let contactObservation = observations.last(where: { ToolRouteGuard.canonicalToolID($0.tool) == "contacts.search" })?.result,
           let phoneContinuation = phoneCallContinuationAfterContactSearchIfNeeded(
                actionTool: "contacts.search",
                observation: contactObservation,
                prompt: prompt,
                availableToolIDs: Set(req.availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })
           ),
           structuredFinalContradictsContactPhoneContinuation(finalAnswer) {
            return phoneContinuation.text
        }

        if routing.intent == .weather,
           weatherFinalOverstatesPrecipitation(finalAnswer: finalAnswer, observations: observations) {
            let weatherObservation = observations
                .last(where: { ToolRouteGuard.canonicalToolID($0.tool) == "weather" })?
                .result
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let observationText = weatherObservation?.isEmpty == false ? weatherObservation! : "the weather observation"
            return "Weather update: \(observationText). No precipitation was reported in the weather observation."
        }

        if let memoryFinal = memorySaveRecallFinalIfApplicable(
            routing: routing,
            prompt: prompt,
            steps: steps
        ) {
            return memoryFinal
        }

        if routing.intent == .webSearch,
           hasUsableObservation(for: .webSearch, observations: observations),
           webFinalRequiresObservationFallback(finalAnswer) {
            return deterministicWebSummaryFallback(observations: observations)
                ?? deterministicWebResultFallback(observations: observations)
                ?? "I found web results, but could not synthesize a grounded final answer from them."
        }

        if routing.intent == .rag || routing.intent == .files {
            let outcome = retrievalOutcome(from: observations)
            if outcome.isEmptyRetrieval,
               let deterministic = deterministicObservationFallback(observations: observations, intent: routing.intent) {
                return deterministic
            }
            if structuredFinalIsGenericFallback(finalAnswer),
               hasUsableObservation(for: routing.intent, observations: observations),
               let deterministic = deterministicObservationFallback(observations: observations, intent: routing.intent) {
                return deterministic
            }
        }

        return finalAnswer
    }

    private nonisolated static func phoneCallContinuationAfterContactSearchIfNeeded(
        actionTool: String,
        observation: String,
        prompt: String,
        availableToolIDs: Set<String>
    ) -> AgentPhoneCallContinuation? {
        guard ToolRouteGuard.canonicalToolID(actionTool) == "contacts.search" else { return nil }
        let cleanPrompt = sanitizedStructuredUserMessage(prompt)
        let routing = IntentRouter.classify(cleanPrompt)
        guard routing.intent == .phoneCall else { return nil }
        return SlotAgentService.phoneCallContinuation(
            afterContactObservation: observation,
            availableToolIDs: availableToolIDs,
            routing: routing
        )
    }

    private nonisolated static func repairedMemoryActionIfNeeded(
        modelAction: AgentAction,
        memoryPlan: MemoryCommandPlan?,
        steps: [AgentStep]
    ) -> (action: AgentAction, reflection: AgentStep?) {
        guard let required = nextRequiredMemoryAction(memoryPlan: memoryPlan, steps: steps) else {
            return (modelAction, nil)
        }
        let modelTool = ToolRouteGuard.canonicalToolID(modelAction.tool)
        let requiredTool = ToolRouteGuard.canonicalToolID(required.tool)
        if modelTool == requiredTool, memoryActionArgumentsMatch(modelAction, required: required) {
            return (modelAction, nil)
        }
        let reflection = AgentStep(
            kind: .reflection,
            content: "Memory save-then-recall invariant repaired \(modelTool) into \(requiredTool) before tool execution."
        )
        return (required, reflection)
    }

    private nonisolated static func memoryActionArgumentsMatch(_ action: AgentAction, required: AgentAction) -> Bool {
        let requiredKeys = Set(required.args.keys)
        guard Set(action.args.keys).isSuperset(of: requiredKeys) else { return false }
        for key in requiredKeys {
            guard action.args[key]?.stringValue == required.args[key]?.stringValue else { return false }
        }
        return true
    }

    private nonisolated static func nextRequiredMemoryAction(
        memoryPlan: MemoryCommandPlan?,
        steps: [AgentStep]
    ) -> AgentAction? {
        guard let memoryPlan else { return nil }
        let actionToolIDs = steps
            .filter { $0.kind == .action }
            .compactMap(\.toolID)
            .map(ToolRouteGuard.canonicalToolID)
        if !actionToolIDs.contains("memory.save") {
            return AgentAction(tool: "memory.save", args: [
                "content": .string(memoryPlan.saveContent),
                "kind": .string("fact")
            ])
        }
        if !actionToolIDs.contains("memory.recall") {
            return AgentAction(tool: "memory.recall", args: ["query": .string(memoryPlan.recallQuery)])
        }
        return nil
    }

    private nonisolated static func approvalBoundaryFinal(for toolID: String, action: AgentAction) -> String {
        switch toolID {
        case "alarm.request_authorization":
            return "Approval required for alarm.request_authorization. I did not request alarm authorization yet."
        case "alarm.schedule", "alarm.countdown", "alarm.pause", "alarm.resume", "alarm.stop", "alarm.snooze", "alarm.cancel":
            return "Approval required for \(toolID). I did not change alarms yet."
        case "calendar.create":
            return "Approval required for calendar.create. I did not create an event yet."
        case "reminders.create":
            return "Approval required for reminders.create. I did not create a reminder yet."
        case "phone.call":
            return "Approval required for phone.call. I did not place the call yet."
        case "camera.capture":
            return "Approval required for camera.capture. I did not open the camera yet."
        default:
            return "Approval required for \(action.displayContent). I did not run it yet."
        }
    }

    private nonisolated static func structuredFinalContradictsContactPhoneContinuation(_ finalAnswer: String) -> Bool {
        let lower = finalAnswer.lowercased()
        return lower.contains("contact search is unavailable")
            || lower.contains("contacts are unavailable")
            || lower.contains("phone call tools unavailable")
            || lower.contains("phone call tools are unavailable")
            || lower.contains("limited local mode")
    }

    #if DEBUG
    nonisolated static func phoneCallContinuationAfterContactSearchForTests(
        actionTool: String,
        observation: String,
        prompt: String,
        availableToolIDs: Set<String>
    ) -> (text: String, step: AgentStep)? {
        guard let continuation = phoneCallContinuationAfterContactSearchIfNeeded(
            actionTool: actionTool,
            observation: observation,
            prompt: prompt,
            availableToolIDs: availableToolIDs
        ) else { return nil }
        return (continuation.text, continuation.step)
    }

    #endif

    private nonisolated static func weatherFinalOverstatesPrecipitation(
        finalAnswer: String,
        observations: [(tool: String, result: String)]
    ) -> Bool {
        let answer = finalAnswer.lowercased()
        let recommendsPrecipitationAction = answer.contains("umbrella")
            || answer.contains("likely raining")
            || answer.contains("it's raining")
            || answer.contains("it is raining")
        guard recommendsPrecipitationAction else { return false }
        let weatherObservation = observations
            .filter { ToolRouteGuard.canonicalToolID($0.tool) == "weather" }
            .map(\.result)
            .joined(separator: "\n")
            .lowercased()
        let precipitationSignals = [
            "rain", "raining", "drizzle", "precip", "precipitation", "shower",
            "forecasted rain", "chance of rain", "probability of precipitation",
            "freezing rain", "snow", "thunderstorm"
        ]
        return !precipitationSignals.contains(where: { weatherObservation.contains($0) })
    }

    private nonisolated static func memorySaveRecallFinalIfApplicable(
        routing: IntentRoutingDecision,
        prompt: String,
        steps: [AgentStep]
    ) -> String? {
        guard routing.intent == .memory else { return nil }
        let actionSteps = steps.filter { $0.kind == .action }
        let actionToolIDs = actionSteps.compactMap(\.toolID).map(ToolRouteGuard.canonicalToolID)
        guard actionToolIDs.contains("memory.save"), actionToolIDs.contains("memory.recall") else { return nil }
        let lowerPrompt = prompt.lowercased()
        guard lowerPrompt.contains("tell me what you remembered")
            || lowerPrompt.contains("what you remembered")
            || lowerPrompt.contains("what did you remember")
        else {
            return nil
        }
        guard let savedContent = actionSteps
            .first(where: { ToolRouteGuard.canonicalToolID($0.toolID ?? "") == "memory.save" })?
            .toolArgs?["content"]
        else {
            return nil
        }
        let remembered = rememberedPreference(from: savedContent)
        guard !remembered.isEmpty else { return nil }
        if remembered.lowercased().hasPrefix("you ") {
            return "I remember that \(remembered)."
        }
        return "I remember that \(remembered)."
    }

    private nonisolated static func rememberedPreference(from content: String) -> String {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "you prefer concise bullet points" }
        if let range = trimmed.range(of: "I prefer ", options: [.caseInsensitive]) {
            let preference = preferenceFragment(String(trimmed[range.upperBound...]))
            if !preference.isEmpty { return "you prefer \(preference)" }
        }
        if let range = trimmed.range(of: "prefer ", options: [.caseInsensitive]) {
            let preference = preferenceFragment(String(trimmed[range.upperBound...]))
            if !preference.isEmpty { return "you prefer \(preference)" }
        }
        if let range = trimmed.range(of: "Remember that ", options: [.caseInsensitive]) {
            let remembered = preferenceFragment(String(trimmed[range.upperBound...]))
            if !remembered.isEmpty { return remembered }
        }
        return preferenceFragment(trimmed)
    }

    private nonisolated static func preferenceFragment(_ text: String) -> String {
        var fragment = text
        if let range = fragment.range(of: ", then", options: [.caseInsensitive]) {
            fragment = String(fragment[..<range.lowerBound])
        }
        for separator in [".", "\n", ";", "?", "!"] {
            if let range = fragment.range(of: separator) {
                fragment = String(fragment[..<range.lowerBound])
            }
        }
        return fragment.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Constructs a system prompt that enforces strict JSON output and provides tool-routing guidance.
    ///
    /// The prompt specifies required JSON schemas, defines available tools, and includes routing heuristics for location-based queries, map searches, and web searches. It incorporates sanitized system context and attachment details from the request.
    ///
    /// - Parameter req: The agent request providing the base system prompt, available tools, and attachments.
    /// - Returns: The complete system prompt as a string.

    private func buildSystemPrompt(req: AgentRequest) -> String {
        var sys = """
        You are Lumen's structured routing executor. Emit one raw JSON object only.

        Response format contract: output exactly one valid JSON object matching this schema:
        \(Self.structuredAgentResponseSchema)

        Schemas:
        {"thought":"short","action":{"tool":"tool.id","args":{}}}
        {"thought":"short","final":"user-facing answer"}

        Rules:
        - Start with { and stop after the matching }.
        - No markdown, prose, code fences, XML, bullets, or hidden reasoning outside JSON.
        - Use double-quoted JSON. Use {} for empty args.
        - Choose exactly one of action or final.
        - action must be a JSON object, never a string. Invalid: {"action":"weather"}.
        - action.tool must be one available tool.
        - Use final when no tool is needed or observations already answer the user.
        - Keep thought under 12 words and final concise.
        """

        let appPrompt = Self.boundedStructuredContextNote(sanitizeSystemPromptForStructuredOutput(req.systemPrompt))
        if !appPrompt.isEmpty {
            sys += "\n\nContext note, lower priority than JSON/tool rules:\n"
            sys += appPrompt
            sys += "\n"
        }

        if !req.availableTools.isEmpty {
            sys += "\nAvailable tools:\n"
            sys += Self.compactStructuredToolList(req.availableTools, userMessage: req.userMessage)
            sys += "\n"
        } else {
            sys += "\nNo tools are available. Emit final JSON only.\n"
        }

        sys += "\nRouting hints: current web/research -> web.search; local files/notes -> rag.search; save user preference -> memory.save; recall stored memory -> memory.recall; weather -> weather; draft email -> mail.draft; scheduled agent run -> trigger.create. Do not include attachment bodies or local source snippets in this routing turn."
        return sys
    }

    private nonisolated static func compactStructuredToolList(_ tools: [ToolDefinition], userMessage: String) -> String {
        let routing = IntentRouter.classify(sanitizedStructuredUserMessage(userMessage))
        let preferredIDs = Set(routing.allowedToolIDs.map(ToolRouteGuard.canonicalToolID))
        let ordered = tools.enumerated().sorted { left, right in
            let leftPreferred = preferredIDs.contains(ToolRouteGuard.canonicalToolID(left.element.id))
            let rightPreferred = preferredIDs.contains(ToolRouteGuard.canonicalToolID(right.element.id))
            if leftPreferred != rightPreferred { return leftPreferred && !rightPreferred }
            return left.offset < right.offset
        }

        var seen: Set<String> = []
        var lines: [String] = []
        var used = 0
        for entry in ordered {
            let tool = entry.element
            let canonical = ToolRouteGuard.canonicalToolID(tool.id)
            guard !seen.contains(canonical) else { continue }
            seen.insert(canonical)
            guard lines.count < structuredToolCountCap else { break }

            let description = compactStructuredToolDescription(tool.description)
            let approval = tool.requiresApproval ? " approval" : ""
            let line = description.isEmpty
                ? "- \(canonical)\(approval)"
                : "- \(canonical)\(approval): \(description)"
            let cost = line.count + 1
            if !lines.isEmpty, used + cost > structuredToolListCharCap { break }
            lines.append(line)
            used += cost
        }

        if lines.isEmpty {
            return "- no usable tools after budget cap"
        }
        return lines.joined(separator: "\n")
    }

    private nonisolated static func compactStructuredToolDescription(_ description: String) -> String {
        var text = description
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        while text.contains("  ") {
            text = text.replacingOccurrences(of: "  ", with: " ")
        }
        guard !text.isEmpty else { return "" }

        let argsPart: String? = {
            guard let range = text.range(of: "Args:", options: .caseInsensitive) else { return nil }
            return String(text[range.lowerBound...])
                .split(separator: ".")
                .first
                .map(String.init)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }()
        let firstSentence = text
            .split(separator: ".")
            .first
            .map(String.init)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? text
        let combined = [firstSentence, argsPart]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
            .joined(separator: ". ")
        return String(combined.prefix(structuredToolDescriptionCharCap))
    }

    /// Constructs the user message for a structured agent turn.
    ///
    /// Incorporates conversation history, the current user request, and (for steps after the first) accumulated observations and reusable location context.
    /// - Returns: The complete user message prompting the agent to emit a JSON object.
    private func buildAgentUserTurn(req: AgentRequest, stepIndex: Int, scratchpad: String) -> String {
        var out = ""
        let context = sanitizedHistoryContext(req.history)
        let userMessage = Self.sanitizedStructuredUserMessage(req.userMessage)
        if !context.isEmpty {
            out += "Conversation context, for reference only. Do not imitate its formatting:\n"
            out += context
            out += "\n\n"
        }

        out += "User request:\n"
        out += userMessage

        if stepIndex > 0 {
            out += "\n\nPrior structured turns and observations:\n"
            out += Self.compactStructuredScratchpad(scratchpad)
            if let locationObservation = latestCurrentLocationObservation(in: scratchpad) {
                out += "\n\nReusable location context:\n"
                out += "Observation: \(locationObservation)"
            }
            out += "\n\nEmit the next JSON object now. If the observations already answer the user, choose final. If another tool is absolutely required, action must be an object like {\"tool\":\"tool.id\",\"args\":{}}; never emit action as a string."
        } else {
            out += "\n\nEmit the first JSON object now. Choose either action or final."
        }
        return out
    }

    private nonisolated static func compactStructuredScratchpad(_ scratchpad: String) -> String {
        let compact = scratchpad
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        guard compact.count > structuredScratchpadCharCap else { return compact }
        return String(compact.suffix(structuredScratchpadCharCap))
    }

    private nonisolated static func agentJSONEmptyOutputRetryRequest(
        from request: GenerateRequest,
        userTurn: String
    ) -> GenerateRequest {
        GenerateRequest(
            id: request.id,
            sessionID: request.sessionID,
            systemPrompt: request.systemPrompt,
            history: request.history,
            userMessage: agentJSONEmptyOutputRetryUserTurn(from: userTurn),
            temperature: min(request.temperature, 0.05),
            topP: min(request.topP, 0.6),
            repetitionPenalty: max(request.repetitionPenalty, 1.05),
            maxTokens: min(max(request.maxTokens, structuredTurnMinTokenCap), structuredTurnMaxTokenCap),
            modelName: request.modelName,
            relevantMemories: request.relevantMemories,
            attachments: request.attachments,
            responseFormat: request.responseFormat,
            seed: request.seed.map { $0 &+ 1 },
            developerTraceModeEnabled: request.developerTraceModeEnabled,
            reasoningCaptureEnabled: request.reasoningCaptureEnabled,
            reasoningTraceBudgetCharacters: request.reasoningTraceBudgetCharacters,
            allowsMemoryPressureContinuation: request.allowsMemoryPressureContinuation
        )
    }

    private nonisolated static func agentJSONEmptyOutputRetryUserTurn(from userTurn: String) -> String {
        """
        \(userTurn)

        Previous live agent-json attempt emitted no tokens. Do not stop silently.
        Emit exactly one non-empty JSON object now, with no prose, no markdown, and no code fence.
        The object must contain either:
        {"action":{"tool":"<allowed tool id>","args":{...}}}
        or:
        {"final":"<concise user-facing answer>"}
        Start the response with { and finish after the matching }.
        """
    }

    private func preflightAgentJSONPrompt(_ request: GenerateRequest) async -> StructuredPromptPreflight {
        let contextSize = await AppLlamaService.shared.contextSizeForDiagnostics(slot: Self.structuredAgentModelSlot)
        let promptBuild = await AppLlamaService.shared.buildMessagesForDiagnostics(
            req: request,
            contextSize: contextSize,
            slot: Self.structuredAgentModelSlot
        )
        let tokenLimit = max(
            128,
            contextSize - request.maxTokens - Self.structuredPromptPreflightSafetyTokens
        )
        let fits = promptBuild.estimatedPromptTokens <= tokenLimit
            && promptBuild.finalPromptChars <= PromptBudget.agentJSON(
                contextSize: contextSize,
                maxTokens: request.maxTokens
            ).totalChars + 256
        return StructuredPromptPreflight(
            request: request,
            contextSize: contextSize,
            finalPromptChars: promptBuild.finalPromptChars,
            estimatedPromptTokens: promptBuild.estimatedPromptTokens,
            fits: fits
        )
    }

    private nonisolated static func agentJSONContextCompactionRequest(from request: GenerateRequest) -> GenerateRequest {
        GenerateRequest(
            id: request.id,
            sessionID: request.sessionID,
            systemPrompt: truncateSystemPromptForAgentJSONPreflight(request.systemPrompt),
            history: [],
            userMessage: compactAgentJSONUserTurnForPreflight(request.userMessage),
            temperature: min(request.temperature, 0.05),
            topP: min(request.topP, 0.6),
            repetitionPenalty: max(request.repetitionPenalty, 1.05),
            maxTokens: min(max(request.maxTokens / 2, structuredTurnMinTokenCap), 224),
            modelName: request.modelName,
            relevantMemories: [],
            attachments: [],
            responseFormat: request.responseFormat,
            seed: request.seed,
            developerTraceModeEnabled: false,
            reasoningCaptureEnabled: false,
            reasoningTraceBudgetCharacters: request.reasoningTraceBudgetCharacters,
            allowsMemoryPressureContinuation: request.allowsMemoryPressureContinuation
        )
    }

    private nonisolated static func truncateSystemPromptForAgentJSONPreflight(_ systemPrompt: String) -> String {
        String(systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).prefix(1_200))
    }

    private nonisolated static func compactAgentJSONUserTurnForPreflight(_ userTurn: String) -> String {
        var requestText = userTurn
        if let marker = requestText.range(of: "User request:") {
            requestText = String(requestText[marker.upperBound...])
        }
        if let emit = requestText.range(of: "\n\nEmit ") {
            requestText = String(requestText[..<emit.lowerBound])
        }
        if let prior = requestText.range(of: "\n\nPrior structured turns and observations:") {
            requestText = String(requestText[..<prior.lowerBound])
        }
        requestText = requestText.trimmingCharacters(in: .whitespacesAndNewlines)
        requestText = String(requestText.prefix(600))
        return """
        User request:
        \(requestText)

        Emit one JSON object now: action with an available tool, or final if no tool is needed.
        """
    }

    private nonisolated static func runtimeFailureParseError(from raw: String) -> AgentTurnParseError? {
        let lower = raw.lowercased()
        if lower.contains("prompt exceeded context window before generation")
            || lower.contains("prompt exceeds shared chat context window")
            || lower.contains("failed to initialize context: prompt exceeds") {
            return .contextWindowExceeded
        }
        return nil
    }

    private func recordAgentModelTurnTrace(
        req: AgentRequest,
        userTurn: String,
        raw: String,
        turn: AgentTurn,
        stepIndex: Int,
        diagnostics: StructuredTurnGenerationDiagnostics
    ) {
        let routing = IntentRouter.classify(Self.sanitizedStructuredUserMessage(req.userMessage))
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "agent",
                stage: "agent-json-step-\(stepIndex)",
                scenarioID: req.scenarioID,
                e2eRunID: req.e2eRunID,
                agentRunID: req.agentRunID,
                conversationID: req.conversationID,
                turnID: req.turnID,
                intent: routing.intent.rawValue,
                promptPrefix: ModelOutputSanitizer.boundedPrefix(req.userMessage, limit: 1200),
                rawOutputPrefix: ModelOutputSanitizer.boundedPrefix(AgentThinkBlockSanitizer.redactedForDiagnostics(raw), limit: 1600),
                selectedToolID: turn.action.map { ToolRouteGuard.canonicalToolID($0.tool) },
                toolArguments: turn.action?.args.stringCoerced ?? [:],
                allowedToolIDs: req.availableTools.map { ToolRouteGuard.canonicalToolID($0.id) }.sorted(),
                requiresApproval: turn.action.map { ToolRouteGuard.requiresUserApproval(ToolRouteGuard.canonicalToolID($0.tool)) },
                approvalMode: nil,
                parseError: turn.parseError?.rawValue,
                emittedFinalInActionTurn: turn.final?.isEmpty == false,
                modelFamily: LumenModelFamily.persistedSelected.rawValue,
                adapterSlot: Self.structuredAgentModelSlot.rawValue,
                generationElapsedMs: diagnostics.generationElapsedMs,
                firstTokenLatencyMs: diagnostics.firstTokenLatencyMs,
                outputTokenCount: diagnostics.outputTokenCount,
                estimatedPromptTokenCount: diagnostics.estimatedPromptTokenCount,
                runtimePath: "agent-model",
                activeAdapterSlot: Self.structuredAgentModelSlot.rawValue,
                maxTokensRequested: diagnostics.maxTokensRequested,
                maxTokensEffective: diagnostics.maxTokensEffective,
                promptCharCount: diagnostics.promptCharCount ?? userTurn.count,
                emptyOutputReason: diagnostics.emptyOutputReason,
                streamStarted: diagnostics.streamStarted,
                selectedRuntime: diagnostics.selectedRuntime,
                selectedAdapter: diagnostics.selectedAdapter,
                modelIdentifier: diagnostics.modelIdentifier,
                modelLoaded: diagnostics.modelLoaded,
                stopSequences: diagnostics.stopSequences,
                temperature: diagnostics.temperature,
                topP: diagnostics.topP,
                cancellationStateBeforeStream: diagnostics.cancellationStateBeforeStream,
                firstChunkReceived: diagnostics.firstChunkReceived,
                textChunkCount: diagnostics.textChunkCount,
                finalChunkReceived: diagnostics.finalChunkReceived,
                streamTerminationReason: diagnostics.streamTerminationReason,
                selfModel: AgentBehaviorTrace.SelfModelDecisionSummary.fromPrompt(
                    userTurn,
                    selectedToolID: turn.action.map { ToolRouteGuard.canonicalToolID($0.tool) },
                    requiresApproval: turn.action.map { ToolRouteGuard.requiresUserApproval(ToolRouteGuard.canonicalToolID($0.tool)) },
                    approvalMode: nil
                )
            )
        )
    }

    private func compactScratchpadObservation(_ text: String) -> String {
        var compact = text.replacingOccurrences(of: "\n", with: " ")
        while compact.contains("  ") {
            compact = compact.replacingOccurrences(of: "  ", with: " ")
        }
        return compact.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func currentLocationScratchpadContext(from text: String) -> String? {
        let compact = compactScratchpadObservation(text)
        guard let range = compact.range(of: "Current location:", options: .caseInsensitive) else { return nil }
        let suffix = compact[range.lowerBound...]
        let normalized = String(suffix).trimmingCharacters(in: .whitespacesAndNewlines)
        guard normalized.contains(",") else { return nil }
        return normalized
    }

    private func latestCurrentLocationObservation(in scratchpad: String) -> String? {
        var latest: String?
        for line in scratchpad.split(separator: "\n") {
            let raw = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
            guard raw.hasPrefix("Observation:") || raw.hasPrefix("Context:") else { continue }
            if let context = currentLocationScratchpadContext(from: raw) {
                latest = context
            }
        }
        return latest
    }

    /// Formats recent conversation history as role-labeled lines, sanitizing and skipping empty content.
    /// - Returns: A newline-separated string of "Role: content" lines.
    private func sanitizedHistoryContext(_ history: [(role: MessageRole, content: String)]) -> String {
        let recent = history.suffix(3)
        var lines: [String] = []
        var used = 0
        for item in recent {
            let role: String
            switch item.role {
            case .user: role = "User"
            case .assistant: role = "Assistant"
            case .system: role = "System"
            case .tool: role = "Tool"
            }
            let content = sanitizeHistoryContent(item.content)
            guard !content.isEmpty else { continue }
            let line = "\(role): \(content)"
            let cost = line.count + 1
            if used + cost > Self.structuredHistoryTotalCharCap { break }
            lines.append(line)
            used += cost
        }
        return lines.joined(separator: "\n")
    }

    /// Sanitizes conversation history content for use in structured prompts by removing code blocks, XML-like tags, internal grounding markers, and redundant punctuation, while normalizing whitespace and capping the result.
    ///
    /// - Parameter content: The raw history message content to sanitize.
    /// - Returns: The sanitized content, with consecutive spaces collapsed to single spaces and truncated to 480 characters.
    private func sanitizeHistoryContent(_ content: String) -> String {
        var text = Self.stripInternalGrounding(from: content)
        text = text.replacingOccurrences(
            of: #"```[\s\S]*?```"#,
            with: " ",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"</?[A-Za-z_][A-Za-z0-9_.:-]*(?:\s+[^<>]*?)?/?>"#,
            with: " ",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"([{}\[\]`|])\1+"#,
            with: "$1",
            options: .regularExpression
        )
        text = text
            .replacingOccurrences(of: "<json>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "</json>", with: "", options: .caseInsensitive)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        text = text.replacingOccurrences(of: "\n", with: " ")
        while text.contains("  ") { text = text.replacingOccurrences(of: "  ", with: " ") }
        return String(text.prefix(Self.structuredHistoryTurnCharCap))
    }

    func sanitizeHistoryContentForTests(_ content: String) -> String {
        sanitizeHistoryContent(content)
    }

    /// Prepares a system prompt for structured JSON output by removing internal grounding markers and lines containing blocked formatting directives.
    /// - Returns: The sanitized system prompt with internal grounding removed and lines mentioning markdown, code fences, headings, or step-by-step instructions filtered out.
    private func sanitizeSystemPromptForStructuredOutput(_ systemPrompt: String) -> String {
        let trimmed = Self.stripInternalGrounding(from: systemPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        let blockedPhrases = [
            "markdown",
            "code fence",
            "code fences",
            "fenced code block",
            "fenced code blocks",
            "headings",
            "step-by-step",
            "step by step"
        ]

        let parts = trimmed.components(separatedBy: .newlines)
        var kept: [String] = []
        for part in parts {
            let sentence = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !sentence.isEmpty else { continue }
            let lowered = sentence.lowercased()
            if blockedPhrases.contains(where: { lowered.contains($0) }) { continue }
            kept.append(sentence)
        }
        return kept.joined(separator: "\n")
    }

    /// Removes internal grounding markers from the user message, trims whitespace, and caps the result to `structuredUserMessageCharCap`.
    /// If stripping yields an empty string, returns the capped trimmed original message instead.
    private nonisolated static func sanitizedStructuredUserMessage(_ userMessage: String) -> String {
        let stripped = stripInternalGrounding(from: userMessage)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else {
            let trimmed = userMessage.trimmingCharacters(in: .whitespacesAndNewlines)
            return String(trimmed.prefix(structuredUserMessageCharCap))
        }
        return String(stripped.prefix(structuredUserMessageCharCap))
    }

    /// Caps the text to the maximum context note length.
    /// - Returns: The trimmed text, capped to the maximum context note length.
    private nonisolated static func boundedStructuredContextNote(_ text: String) -> String {
        String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(structuredContextNoteCharCap))
    }

    /// Removes all content starting from the first occurrence of any internal grounding marker.
    /// - Returns: The text before the first internal grounding marker, or the original text if none is found.
    private nonisolated static func stripInternalGrounding(from text: String) -> String {
        var stripped = text
        let markers = [
            "<!-- LUMEN_GROUNDING_V1 -->",
            "[AVAILABLE LOCAL TOOLS]",
            "[RUNTIME POLICY]",
            "[LOCAL MEMORY]",
            "[LOCAL SOURCES]"
        ]
        for marker in markers {
            if let range = stripped.range(of: marker, options: [.caseInsensitive]) {
                stripped = String(stripped[..<range.lowerBound])
            }
        }
        return stripped
    }

    func sanitizeSystemPromptForStructuredOutputForTests(_ systemPrompt: String) -> String {
        sanitizeSystemPromptForStructuredOutput(systemPrompt)
    }

    private func agentTemperature(from userTemperature: Double) -> Double {
        min(max(userTemperature, 0.0), 0.15)
    }

    private func agentTopP(from userTopP: Double) -> Double {
        min(max(userTopP, 0.1), 0.85)
    }

    private func structuredTurnMaxTokens(from requestedMaxTokens: Int, req _: AgentRequest, stepIndex _: Int) -> Int {
        return min(max(requestedMaxTokens, Self.structuredTurnMinTokenCap), Self.structuredTurnMaxTokenCap)
    }

    private nonisolated static func agentJSONEmptyStreamReason(
        streamStarted: Bool,
        textChunkCount: Int,
        finalChunkReceived: Bool,
        taskCancelled: Bool,
        maxTokensEffective: Int
    ) -> String {
        if maxTokensEffective <= 0 {
            return "decodeBudgetZero"
        }
        if taskCancelled, textChunkCount == 0 {
            return "cancelledBeforeFirstToken"
        }
        if !streamStarted {
            return "runtimeUnavailable"
        }
        if finalChunkReceived, textChunkCount == 0 {
            return "completedWithoutText"
        }
        if textChunkCount == 0 {
            return "stoppedBeforeFirstToken"
        }
        return "unknownEmptyStream"
    }

    func structuredTurnMaxTokensForTests(from requestedMaxTokens: Int) -> Int {
        structuredTurnMaxTokens(
            from: requestedMaxTokens,
            req: AgentRequest(
                systemPrompt: "",
                history: [],
                userMessage: "test",
                temperature: 0.1,
                topP: 0.9,
                repetitionPenalty: 1.1,
                maxTokens: requestedMaxTokens,
                maxSteps: 1,
                availableTools: [],
                relevantMemories: []
            ),
            stepIndex: 0
        )
    }

    private func diagnosticReflection(for _: AgentTurnParseError, raw: String) -> String {
        let noise = AgentNoiseInspector.inspect(raw)
        var parts = ["I hit an internal formatting issue and repaired it into a plain answer."]
        if let prefix = noise.prefixNoise, !prefix.isEmpty, Self.isMeaningfulDiagnosticNoise(prefix) {
            parts.append("Prefix noise: \(String(prefix.prefix(120)))")
        }
        if let suffix = noise.suffixNoise, !suffix.isEmpty, Self.isMeaningfulDiagnosticNoise(suffix) {
            parts.append("Suffix noise: \(String(suffix.prefix(120)))")
        }
        if noise.selectedJSON == nil && raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
            parts.append("No valid JSON object found in raw model output.")
        }
        return parts.joined(separator: " ")
    }

    private func recordParseFailure(
        req: AgentRequest,
        parseError: AgentTurnParseError,
        raw: String,
        scanner: StreamingJSONScanner,
        systemPrompt: String,
        userTurn: String,
        stepIndex: Int
    ) {
        let snapshot = AgentNoiseInspector.inspect(raw)
        let trace = AgentParseFailureTrace(
            id: UUID(),
            createdAt: Date(),
            parseError: parseError.rawValue,
            modelName: "agent-json",
            temperature: agentTemperature(from: req.temperature),
            topP: agentTopP(from: req.topP),
            maxTokens: req.maxTokens,
            stepIndex: stepIndex,
            systemPromptPrefix: String(systemPrompt.prefix(2_000)),
            userTurnPrefix: String(userTurn.prefix(2_000)),
            rawOutputPrefix: String(AgentThinkBlockSanitizer.redactedForDiagnostics(raw).prefix(4_000)),
            streamedThoughtPrefix: String(scanner.thought.prefix(1_000)),
            streamedFinalPrefix: String(scanner.final.prefix(1_000)),
            selectedJSONPrefix: snapshot.selectedJSON.map { String($0.prefix(2_000)) },
            prefixNoise: snapshot.prefixNoise.map { String($0.prefix(1_000)) },
            suffixNoise: snapshot.suffixNoise.map { String($0.prefix(1_000)) }
        )
        AgentParseFailureRecorder.record(trace)
    }

    private func recordRecoverableNoise(
        req: AgentRequest,
        raw: String,
        systemPrompt: String,
        userTurn: String,
        stepIndex: Int
    ) {
        let snapshot = AgentNoiseInspector.inspect(raw)
        let trace = AgentParseNoiseTrace(
            id: UUID(),
            createdAt: Date(),
            modelName: "agent-json",
            temperature: agentTemperature(from: req.temperature),
            topP: agentTopP(from: req.topP),
            maxTokens: req.maxTokens,
            stepIndex: stepIndex,
            systemPromptPrefix: String(systemPrompt.prefix(2_000)),
            userTurnPrefix: String(userTurn.prefix(2_000)),
            rawOutputPrefix: String(AgentThinkBlockSanitizer.redactedForDiagnostics(raw).prefix(4_000)),
            selectedJSONPrefix: snapshot.selectedJSON.map { String($0.prefix(2_000)) },
            prefixNoise: snapshot.prefixNoise.map { String($0.prefix(1_000)) },
            suffixNoise: snapshot.suffixNoise.map { String($0.prefix(1_000)) }
        )
        AgentParseNoiseRecorder.record(trace)
    }

    /// Attempts to recover from a structured parse failure.
    /// - Returns: A tuple of the recovered text and steps if recovery succeeds, `nil` otherwise.
    private nonisolated static func structuredParseFailureRecovery(
        req: AgentRequest,
        options: LegacyAgentRunOptions
    ) async -> (text: String, steps: [AgentStep])? {
        guard options.allowDeterministicCompatibility,
              options.allowParseFailureDeterministicRecovery else {
            return nil
        }
        let prompt = sanitizedStructuredUserMessage(req.userMessage)
        guard !prompt.isEmpty else { return nil }

        let routing = await IntentClassifierService.shared.route(prompt)
        guard routing.intent != .unknown else { return nil }

        let recoveryOptions = LegacyAgentRunOptions(
            modelContext: options.modelContext,
            conversationID: options.conversationID ?? req.conversationID,
            turnID: options.turnID ?? req.turnID,
            scenarioID: options.scenarioID ?? req.scenarioID,
            e2eRunID: options.e2eRunID ?? req.e2eRunID,
            agentRunID: options.agentRunID ?? req.agentRunID,
            groundingMode: options.groundingMode,
            allowDegradedGrounding: options.allowDegradedGrounding,
            preventDoubleGrounding: options.preventDoubleGrounding,
            diagnosticsEnabled: options.diagnosticsEnabled || options.groundingMode == .slotAgent,
            allowDeterministicCompatibility: options.allowDeterministicCompatibility,
            allowParseFailureDeterministicRecovery: options.allowParseFailureDeterministicRecovery,
            allowsMemoryPressureContinuation: options.allowsMemoryPressureContinuation
        )
        let recovery = await SlotAgentService.deterministicCompatibilityResponseForRecovery(
            original: req,
            effective: req,
            options: recoveryOptions
        )
        let text = recovery.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenericParseRecoveryText(text) else { return nil }

        if IntentRouter.intentRequiresTool(routing), !routing.requiresClarification {
            let actionToolIDs = Set(recovery.steps
                .filter { $0.kind == .action || $0.kind == .approvalBoundary }
                .compactMap(\.toolID)
                .map(ToolRouteGuard.canonicalToolID))
            guard !actionToolIDs.isEmpty,
                  actionToolIDs.contains(where: { routing.allowedToolIDs.contains($0) }) else {
                return nil
            }
        }

        RuntimeFallbackLogger.record(
            source: "agent-service-structured-turn",
            primaryBehavior: "parse model output as structured agent turn",
            fallbackBehavior: "recover with deterministic route-scoped action/final",
            reason: "parse-failure-deterministic-recovery",
            consequence: "model-backed turn was preserved but repaired with policy-scoped execution",
            values: [
                "turnID": req.turnID?.uuidString ?? "none",
                "conversationID": req.conversationID?.uuidString ?? "none",
                "promptSHA256": RuntimeFallbackLogger.promptHash(req.userMessage),
                "intent": routing.intent.rawValue,
                "stepCount": String(recovery.steps.count)
            ]
        )
        return (text, recovery.steps)
    }

    private nonisolated static func isGenericParseRecoveryText(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("please ask again or tell me what you'd like to do next")
            || lower == "i'm ready. tell me what you want to do next."
            || lower == FinalOutputSanitizer.fallback.lowercased()
            || lower.contains("i couldn't produce a confident answer")
    }

#if DEBUG
    nonisolated static var structuredAgentModelSlotForTests: LumenModelSlot {
        structuredAgentModelSlot
    }

    nonisolated static func sanitizedStructuredUserMessageForTests(_ userMessage: String) -> String {
        sanitizedStructuredUserMessage(userMessage)
    }

    /// Generates the structured system prompt used for agent routing.
    /// - Parameter req: The agent request containing system context and available tools.
    /// - Returns: The system prompt string specifying the JSON-only output contract and routing instructions.
    func structuredSystemPromptForTests(req: AgentRequest) -> String {
        buildSystemPrompt(req: req)
    }

    func structuredAgentUserTurnForTests(req: AgentRequest, stepIndex: Int = 0, scratchpad: String = "") -> String {
        buildAgentUserTurn(req: req, stepIndex: stepIndex, scratchpad: scratchpad)
    }

    nonisolated static func agentJSONEmptyOutputRetryUserTurnForTests(from userTurn: String) -> String {
        agentJSONEmptyOutputRetryUserTurn(from: userTurn)
    }

    nonisolated static func agentJSONEmptyOutputRetryRequestForTests(
        from request: GenerateRequest,
        userTurn: String
    ) -> GenerateRequest {
        agentJSONEmptyOutputRetryRequest(from: request, userTurn: userTurn)
    }

    nonisolated static func agentJSONContextCompactionRequestForTests(from request: GenerateRequest) -> GenerateRequest {
        agentJSONContextCompactionRequest(from: request)
    }

    func recordAgentModelTurnTraceForTests(
        req: AgentRequest,
        raw: String,
        stepIndex: Int = 0,
        outputTokenCount: Int? = nil
    ) {
        recordAgentModelTurnTrace(
            req: req,
            userTurn: buildAgentUserTurn(req: req, stepIndex: stepIndex, scratchpad: ""),
            raw: raw,
            turn: AgentTurnParser.parse(raw),
            stepIndex: stepIndex,
            diagnostics: StructuredTurnGenerationDiagnostics(
                generationElapsedMs: 1,
                firstTokenLatencyMs: outputTokenCount == 0 ? nil : 1,
                outputTokenCount: outputTokenCount,
                estimatedPromptTokenCount: nil,
                maxTokensRequested: req.maxTokens,
                maxTokensEffective: min(max(req.maxTokens, Self.structuredTurnMinTokenCap), Self.structuredTurnMaxTokenCap),
                promptCharCount: nil,
                emptyOutputReason: outputTokenCount == 0 ? "agent-json-stream-completed-without-text" : nil,
                streamStarted: nil,
                selectedRuntime: nil,
                selectedAdapter: nil,
                modelIdentifier: nil,
                modelLoaded: nil,
                stopSequences: [],
                temperature: nil,
                topP: nil,
                cancellationStateBeforeStream: nil,
                firstChunkReceived: nil,
                textChunkCount: nil,
                finalChunkReceived: nil,
                streamTerminationReason: nil
            )
        )
    }

    nonisolated static func runtimeFailureParseErrorForTests(from raw: String) -> AgentTurnParseError? {
        runtimeFailureParseError(from: raw)
    }

    nonisolated static func observationFallbackPlainTextForTests(from raw: String, intent: UserIntent) -> String? {
        observationFallbackPlainText(from: raw, intent: intent)
    }

    nonisolated static func deterministicWebSummaryFallbackForTests(observations: [(tool: String, result: String)]) -> String? {
        deterministicWebSummaryFallback(observations: observations)
    }

    nonisolated static func repairMissingToolActionForTests(
        raw: String,
        req: AgentRequest,
        observations: [(tool: String, result: String)] = []
    ) -> (action: AgentAction, diagnostic: String)? {
        repairMissingToolActionIfPossible(raw: raw, req: req, observations: observations)
    }

    nonisolated static func postprocessStructuredFinalAnswerForTests(
        _ finalAnswer: String,
        req: AgentRequest,
        observations: [(tool: String, result: String)],
        steps: [AgentStep]
    ) -> String {
        postprocessStructuredFinalAnswer(finalAnswer, req: req, observations: observations, steps: steps)
    }

    nonisolated static func repairedMemoryActionForTests(
        modelAction: AgentAction,
        prompt: String,
        steps: [AgentStep]
    ) -> (action: AgentAction, reflection: AgentStep?) {
        repairedMemoryActionIfNeeded(
            modelAction: modelAction,
            memoryPlan: MemoryCommandPlan.saveThenRecall(from: prompt),
            steps: steps
        )
    }

    nonisolated static func nextRequiredMemoryActionForTests(
        prompt: String,
        steps: [AgentStep]
    ) -> AgentAction? {
        nextRequiredMemoryAction(
            memoryPlan: MemoryCommandPlan.saveThenRecall(from: prompt),
            steps: steps
        )
    }

    /// Exposes the internal structured parse failure recovery function for testing.
    /// - Returns: A tuple of recovery text and steps if recovery succeeds, otherwise `nil`.
    nonisolated static func structuredParseFailureRecoveryForTests(
        req: AgentRequest,
        options: LegacyAgentRunOptions
    ) async -> (text: String, steps: [AgentStep])? {
        await structuredParseFailureRecovery(req: req, options: options)
    }
#endif

    // MARK: - Fallback synthesis

    private enum FallbackReason {
        case duplicate, maxSteps, malformed, empty

        var hint: String {
            switch self {
            case .duplicate: return "You already called that tool with these arguments — summarize the existing observations."
            case .maxSteps: return "You've reached the maximum number of reasoning steps — give the best answer now."
            case .malformed: return "Prior output was not valid structured JSON — summarize the observations cleanly."
            case .empty: return "Summarize the observations into a direct answer."
            }
        }

        var diagnosticReason: String {
            switch self {
            case .duplicate: return "duplicate-tool-call"
            case .maxSteps: return "max-agent-steps"
            case .malformed: return "malformed-agent-turn"
            case .empty: return "empty-agent-turn"
            }
        }
    }

    /// Synthesizes a final answer from gathered tool observations.
    ///
    /// When structured reasoning cannot produce a direct action or explicit final answer, this method composes a fallback response by summarizing tool results into a user-facing answer. If no observations are available, returns a generic retry message.
    ///
    /// - Returns: A synthesized final answer text, or a retry message if no observations exist.
    private func synthesizeFallback(req: AgentRequest, observations: [(tool: String, result: String)], reason: FallbackReason) async -> String {
        RuntimeFallbackLogger.record(
            source: "agent-service-structured-turn",
            primaryBehavior: "continue structured agent turn JSON",
            fallbackBehavior: "synthesize final answer from observations",
            reason: reason.diagnosticReason,
            consequence: "agent stopped using primary structured action loop",
            values: [
                "turnID": req.turnID?.uuidString ?? "none",
                "conversationID": req.conversationID?.uuidString ?? "none",
                "promptSHA256": RuntimeFallbackLogger.promptHash(req.userMessage),
                "observationCount": String(observations.count)
            ]
        )
        guard !observations.isEmpty else {
            return "I couldn't find a confident answer. Try rephrasing the question."
        }
        let userMessage = Self.sanitizedStructuredUserMessage(req.userMessage)
        let routing = IntentRouter.classify(userMessage)
        let prompt = Self.observationFallbackPrompt(
            userMessage: userMessage,
            observations: observations,
            intent: routing.intent,
            reason: reason
        )

        if let deterministic = Self.deterministicObservationFallback(observations: observations, intent: routing.intent) {
            if routing.intent == .webSearch || Self.retrievalOutcome(from: observations).isEmptyRetrieval {
                return deterministic
            }
        }

        let genReq = GenerateRequest(
            systemPrompt: Self.observationFallbackSystemPrompt(intent: routing.intent),
            history: [],
            userMessage: prompt,
            temperature: 0.2,
            topP: min(req.topP, 0.85),
            repetitionPenalty: req.repetitionPenalty,
            maxTokens: 256,
            modelName: "agent-summary",
            relevantMemories: []
        )
        var out = ""
        for await token in await AppLlamaService.shared.stream(genReq) {
            if Task.isCancelled { break }
            if case .text(let s) = token { out += s }
            if case .done = token { break }
        }
        let trimmed = out.trimmingCharacters(in: .whitespacesAndNewlines)
        if let cleaned = Self.observationFallbackPlainText(from: trimmed, intent: routing.intent) {
            return cleaned
        }
        if let deterministic = Self.deterministicObservationFallback(observations: observations, intent: routing.intent) {
            return deterministic
        }
        return "I couldn't produce a confident answer."
    }

    private nonisolated static func observationFallbackPrompt(
        userMessage: String,
        observations: [(tool: String, result: String)],
        intent: UserIntent,
        reason: FallbackReason
    ) -> String {
        var prompt = "The user asked: \"\(userMessage)\"\n\nYou gathered these tool observations:\n"
        for (i, obs) in observations.enumerated() {
            prompt += "\n[\(i + 1)] \(obs.tool):\n\(obs.result)\n"
        }
        prompt += "\n\(reason.hint)\n"
        if intent == .rag || intent == .files {
            prompt += "Write the final answer in plain text with exactly these sections: Summary, Key modules.\n"
            prompt += "In Key modules, name concrete modules/components/services/packages from the observations when available; if none are explicit, say that clearly.\n"
            prompt += "Every bullet/sentence must include at least one bracketed source marker like [1], [2] that maps to the observation numbers above.\n"
            prompt += "Ground every claim in the listed observations and avoid generic advice. No preamble, no JSON, no prefixes, no apology. If observations conflict, prefer the most recent."
        } else {
            prompt += "Write a concise direct answer in plain text. Do not use JSON, markdown tables, code fences, or a Key modules section. "
            prompt += "Do not add facts that are not present in the observations. If observations conflict, prefer the most recent."
        }
        return prompt
    }

    private nonisolated static func observationFallbackSystemPrompt(intent: UserIntent) -> String {
        if intent == .rag || intent == .files {
            return "You summarize local retrieval results into a concise user-facing answer. Output plain text only. Include Summary and Key modules sections grounded in provided snippets/sources."
        }
        return "You summarize tool results into a concise user-facing answer. Output plain text only. Do not output JSON."
    }

    private nonisolated static func observationFallbackPlainText(from raw: String, intent: UserIntent) -> String? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        if let direct = firstUsefulPlainTextFallback(from: text) {
            return direct
        }

        guard text.first == "{",
              let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        guard let summary = jsonStringValue(object, keys: ["summary", "final", "answer"]) else {
            return nil
        }
        let cleanSummary = sanitizeInternalErrorNoise(from: summary)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard cleanSummary.count >= 8 else { return nil }

        if intent == .rag || intent == .files {
            var parts = [String(cleanSummary.prefix(4_000))]
            if let modules = jsonStringListValue(object, keys: ["key modules", "Key modules", "modules"]), !modules.isEmpty {
                parts.append("Key modules: \(modules.joined(separator: ", "))")
            }
            return parts.joined(separator: "\n\n")
        }
        return String(cleanSummary.prefix(4_000))
    }

    private nonisolated static func deterministicObservationFallback(
        observations: [(tool: String, result: String)],
        intent: UserIntent
    ) -> String? {
        guard !observations.isEmpty else { return nil }
        if intent == .rag || intent == .files {
            let outcome = retrievalOutcome(from: observations)
            if outcome.isEmptyRetrieval, let message = outcome.emptyMessage {
                return message
            }
            let sourced = observations.prefix(3).enumerated().map { index, obs in
                "[\(index + 1)] \(compactObservationResult(obs.result, limit: 700))"
            }
            return "Summary\n\(sourced.joined(separator: "\n"))\n\nKey modules\nUse the cited observations above for concrete modules when available."
        }
        if intent == .webSearch {
            return deterministicWebSummaryFallback(observations: observations)
                ?? deterministicWebResultFallback(observations: observations)
        }
        guard let last = observations.last else { return nil }
        let compact = compactObservationResult(last.result, limit: 1_200)
        return compact.isEmpty ? nil : compact
    }

    private nonisolated static func deterministicWebSummaryFallback(observations: [(tool: String, result: String)]) -> String? {
        let joined = observations
            .filter {
                let tool = ToolRouteGuard.canonicalToolID($0.tool)
                return tool == "web.search" || tool == "web.fetch"
            }
            .map(\.result)
            .joined(separator: "\n")
        let candidates = webSummaryCandidates(from: joined)
        let useful: [String]
        if candidates.isEmpty {
            useful = joined
                .split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        } else {
            useful = candidates
        }
        let lines = useful
            .filter { line in
                let lower = line.lowercased()
                return line.count >= 24
                    && !lower.hasPrefix("search results for:")
                    && !lower.hasPrefix("web search results:")
                    && !lower.hasPrefix("http")
                    && !lower.contains("<lumen_web_payload")
                    && !lower.contains("\"mediakind\"")
                    && !webFinalRequiresObservationFallback(line)
            }
        let ordered = prioritizeWebCandidates(lines)
        let bullets = ordered.prefix(2).map { "- \(compactObservationResult($0, limit: 220))" }
        guard bullets.count >= 2 else { return nil }
        return "Summary:\n\(bullets.joined(separator: "\n"))"
    }

    private nonisolated static func deterministicWebResultFallback(observations: [(tool: String, result: String)]) -> String? {
        let joined = observations
            .filter {
                let tool = ToolRouteGuard.canonicalToolID($0.tool)
                return tool == "web.search" || tool == "web.fetch"
            }
            .map(\.result)
            .joined(separator: "\n")
        let candidates = prioritizeWebCandidates(webSummaryCandidates(from: joined))
            .filter { candidate in
                let lower = candidate.lowercased()
                return candidate.count >= 12
                    && !lower.hasPrefix("search results for:")
                    && !lower.hasPrefix("web search results:")
                    && !lower.contains("<lumen_web_payload")
                    && !webFinalRequiresObservationFallback(candidate)
            }
        let bullets = candidates.prefix(3).map { "- \(compactObservationResult($0, limit: 180))" }
        guard !bullets.isEmpty else {
            let compact = compactObservationResult(joined, limit: 500)
            return compact.isEmpty ? nil : "Web results found:\n- \(compact)"
        }
        if bullets.count == 1 {
            return """
            Web results found:
            \(bullets[0])
            - The remaining result payload did not include enough snippet detail for a stronger synthesis.
            """
        }
        return "Web results found:\n\(bullets.joined(separator: "\n"))"
    }

    private nonisolated static func webSummaryCandidates(from text: String) -> [String] {
        var candidates: [String] = []
        let patterns = [
            #"(?is)<lumen_web_payload[^>]*>(.*?)</lumen_web_payload>"#,
            #"(?is)\{[^{}]*"title"\s*:\s*"([^"]+)"[^{}]*"snippet"\s*:\s*"([^"]+)"[^{}]*\}"#,
            #"(?is)\{[^{}]*"snippet"\s*:\s*"([^"]+)"[^{}]*"title"\s*:\s*"([^"]+)"[^{}]*\}"#,
            #"(?is)"title"\s*:\s*"([^"]+)".{0,900}?"snippet"\s*:\s*"([^"]+)""#,
            #"(?is)"snippet"\s*:\s*"([^"]+)".{0,900}?"title"\s*:\s*"([^"]+)""#
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let ns = text as NSString
            for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
                if match.numberOfRanges >= 3 {
                    let first = ns.substring(with: match.range(at: 1))
                    let second = ns.substring(with: match.range(at: 2))
                    candidates.append("\(first): \(second)")
                } else if match.numberOfRanges >= 2 {
                    candidates.append(ns.substring(with: match.range(at: 1)))
                }
            }
        }
        if !candidates.isEmpty { return candidates.map(decodeJSONStringEscapes).filter { !$0.isEmpty } }
        return text
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    private nonisolated static func prioritizeWebCandidates(_ candidates: [String]) -> [String] {
        candidates.sorted { lhs, rhs in
            webCandidatePriority(lhs) > webCandidatePriority(rhs)
        }
    }

    private nonisolated static func webCandidatePriority(_ text: String) -> Int {
        let lower = text.lowercased()
        if lower.contains("developer.apple.com") || lower.contains("swift.org") || lower.contains("docs.swift.org") { return 3 }
        if lower.contains("swift") || lower.contains("concurrency") || lower.contains("actor") || lower.contains("task") { return 2 }
        return 1
    }

    private nonisolated static func decodeJSONStringEscapes(_ text: String) -> String {
        let quoted = "\"\(text.replacingOccurrences(of: "\"", with: "\\\""))\""
        guard let data = quoted.data(using: .utf8),
              let decoded = try? JSONSerialization.jsonObject(with: data) as? String else {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func webPromptRequiresSynthesis(_ prompt: String) -> Bool {
        let lower = prompt.lowercased()
        return lower.contains("summarize")
            || lower.contains("synthesize")
            || lower.contains("compare")
    }

    private nonisolated static func webFinalRequiresObservationFallback(_ finalAnswer: String) -> Bool {
        let text = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        if text.isEmpty { return true }
        if lower.contains("no direct answer from web search") { return true }
        if lower.hasPrefix("search results for:") || lower.hasPrefix("web search results:") { return true }
        if lower.contains("search results for:") && (lower.contains("\nhttp") || lower.contains("\n- http")) { return true }
        if lower.contains("\"intent\"")
            && lower.contains("\"nextmodel\"")
            && lower.contains("\"reasoningsummary\"")
            && lower.contains("\"requiresapproval\"")
            && lower.contains("\"sourcefile\"") {
            return true
        }
        if text.range(of: #"(?is)^\s*(https?://\S+)\s*$"#, options: .regularExpression) != nil {
            return true
        }
        if text.range(of: #"(?is)^\s*see\s+the\s+full\s+(tutorial|article|guide|post|result)\s+at\s+https?://\S+\s*\.?\s*$"#, options: .regularExpression) != nil {
            return true
        }
        if text.range(of: #"(?is)^\s*(?:check\s+out|see|read|visit|open|here(?:'s| is))\b[^\n]{0,180}https?://\S+\s*\.?\s*$"#, options: .regularExpression) != nil {
            return true
        }
        return false
    }

    private nonisolated static func structuredFinalIsGenericFallback(_ finalAnswer: String) -> Bool {
        let lower = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return lower.isEmpty
            || lower.contains("i'm ready. please ask again")
            || lower.contains("please ask again or tell me what you'd like to do next")
            || lower.contains("tool output could not be validated")
            || lower.contains("could not be validated")
            || lower.contains("i couldn't produce a confident answer")
            || lower.contains("i couldn't find a confident answer")
    }

    private enum RetrievalOutcome: Sendable, Equatable {
        case snippets
        case emptyIndex(String)
        case noMatches(String)
        case unavailable(String)

        var isEmptyRetrieval: Bool {
            switch self {
            case .emptyIndex, .noMatches: return true
            case .snippets, .unavailable: return false
            }
        }

        var emptyMessage: String? {
            switch self {
            case .emptyIndex:
                return "I searched your local files but found no matching architecture notes. The local index appears empty; import or create files and reindex."
            case .noMatches:
                return "I searched your local files but found no matching architecture notes."
            case .snippets, .unavailable:
                return nil
            }
        }
    }

    private nonisolated static func retrievalOutcome(from observations: [(tool: String, result: String)]) -> RetrievalOutcome {
        let ragText = observations
            .filter { ToolRouteGuard.canonicalToolID($0.tool) == "rag.search" || ToolRouteGuard.canonicalToolID($0.tool) == "files.read" }
            .map(\.result)
            .joined(separator: "\n")
        let lower = ragText.lowercased()
        if lower.contains("local index appears empty") || lower.contains("import or create local files") {
            return .emptyIndex(ragText)
        }
        if lower.contains("no matching files found") || lower.contains("no matching local snippets") {
            return .noMatches(ragText)
        }
        if lower.contains("unavailable") || lower.contains("disabled") || lower.contains("denied") {
            return .unavailable(ragText)
        }
        return .snippets
    }

    private nonisolated static func hasUsableObservation(for intent: UserIntent, observations: [(tool: String, result: String)]) -> Bool {
        switch intent {
        case .webSearch:
            return observations.contains {
                let tool = ToolRouteGuard.canonicalToolID($0.tool)
                let result = $0.result.trimmingCharacters(in: .whitespacesAndNewlines)
                return (tool == "web.search" || tool == "web.fetch") && !result.isEmpty
            }
        case .rag, .files:
            return observations.contains {
                let tool = ToolRouteGuard.canonicalToolID($0.tool)
                return tool == "rag.search" || tool == "files.read"
            }
        default:
            return !observations.isEmpty
        }
    }

    private nonisolated static func repairMissingToolActionIfPossible(
        raw: String,
        req: AgentRequest,
        observations: [(tool: String, result: String)]
    ) -> (action: AgentAction, diagnostic: String)? {
        guard observations.isEmpty else { return nil }
        switch AgentJSONCandidateSelector.select(from: raw) {
        case .failure:
            return nil
        case .success(let selection):
            let allowed = Array(Set(req.availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })).sorted()
            guard allowed.count == 1, let tool = allowed.first else { return nil }
            let actionObject = (selection.object["action"] as? [String: Any]) ?? selection.object
            let rawArgs = (actionObject["args"] ?? actionObject["arguments"] ?? actionObject["input"]) as? [String: Any] ?? [:]
            var args: AgentJSONArguments = [:]
            for (key, value) in rawArgs {
                guard let parsed = AgentJSONValue.parse(value) else { return nil }
                args[key] = parsed
            }
            return (
                AgentAction(tool: tool, args: args),
                "Structured action was missing action.tool; repaired to the only allowed tool \(tool)."
            )
        }
    }

    private nonisolated static func compactObservationResult(_ result: String, limit: Int) -> String {
        var text = sanitizeInternalErrorNoise(from: result)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        while text.contains("\n\n\n") {
            text = text.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        }
        return String(text.prefix(limit)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func jsonStringValue(_ object: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = object[key] as? String {
                return value
            }
            if let match = object.first(where: { $0.key.caseInsensitiveCompare(key) == .orderedSame })?.value as? String {
                return match
            }
        }
        return nil
    }

    private nonisolated static func jsonStringListValue(_ object: [String: Any], keys: [String]) -> [String]? {
        for key in keys {
            let value = object[key] ?? object.first(where: { $0.key.caseInsensitiveCompare(key) == .orderedSame })?.value
            if let list = value as? [String] {
                return list.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            }
            if let string = value as? String {
                let parts = string
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                return parts
            }
        }
        return nil
    }

    /// Recovers a plain-text final answer from failed structured-turn output.
    ///
    /// When the agent model output cannot be parsed as a structured turn, this function attempts to extract or synthesize a usable plain-text response that can be presented to the user.
    /// - Parameters:
    ///   - req: The original agent request.
    ///   - rawOutput: The model output that failed to parse.
    ///   - streamedThought: Any partial thought captured during streaming.
    ///   - parseError: The specific parsing error that triggered recovery.
    /// - Returns: A plain-text final answer.
    private func synthesizeUnstructuredFallback(
        req: AgentRequest,
        rawOutput: String,
        streamedThought: String,
        parseError: AgentTurnParseError
    ) async -> String {
        RuntimeFallbackLogger.record(
            source: "agent-service-unstructured-output",
            primaryBehavior: "parse model output as structured agent turn",
            fallbackBehavior: "repair or extract plain text final answer",
            reason: parseError.rawValue,
            consequence: "model output bypassed primary structured agent protocol",
            values: [
                "turnID": req.turnID?.uuidString ?? "none",
                "conversationID": req.conversationID?.uuidString ?? "none",
                "promptSHA256": RuntimeFallbackLogger.promptHash(req.userMessage),
                "rawOutputChars": String(rawOutput.count),
                "streamedThoughtChars": String(streamedThought.count)
            ]
        )
        if let direct = Self.firstUsefulPlainTextFallback(from: rawOutput) {
            return direct
        }

        let clippedRaw = String(rawOutput.trimmingCharacters(in: .whitespacesAndNewlines).prefix(4_000))
        let clippedThought = String(streamedThought.trimmingCharacters(in: .whitespacesAndNewlines).prefix(1_000))

        let userMessage = Self.sanitizedStructuredUserMessage(req.userMessage)
        var prompt = "The user asked:\n\(userMessage)\n\n"
        prompt += "The previous local model response could not be parsed as a structured agent turn (\(parseError.rawValue)).\n"
        if !clippedThought.isEmpty {
            prompt += "Partial thought captured from that response:\n\(clippedThought)\n\n"
        }
        if !clippedRaw.isEmpty {
            prompt += "Raw failed response:\n\(clippedRaw)\n\n"
        }
        prompt += "Write the final answer the user should see. Output plain text only. Do not mention JSON, parsing, schemas, tools, or internal errors. Do not include code fences."

        let genReq = GenerateRequest(
            systemPrompt: "You repair a failed agent turn into a concise user-facing final answer. Output plain text only.",
            history: [],
            userMessage: prompt,
            temperature: 0.2,
            topP: min(req.topP, 0.85),
            repetitionPenalty: req.repetitionPenalty,
            maxTokens: min(max(req.maxTokens, 128), 512),
            modelName: "agent-repair",
            relevantMemories: req.relevantMemories
        )

        var out = ""
        for await token in await AppLlamaService.shared.stream(genReq) {
            if Task.isCancelled { break }
            if case .text(let s) = token { out += s }
            if case .done = token { break }
        }

        if let repaired = Self.firstUsefulPlainTextFallback(from: out) {
            return repaired
        }
        if !clippedThought.isEmpty {
            return clippedThought
        }
        return "I couldn't produce a confident answer. Try rephrasing the question."
    }

    private nonisolated static func firstUsefulPlainTextFallback(from raw: String) -> String? {
        var text = sanitizeInternalErrorNoise(from: raw)
        guard !text.isEmpty else { return nil }

        text = text
            .replacingOccurrences(of: "```json", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let lower = text.lowercased()
        let looksLikeStructuredTurn =
            text.first == "{" ||
            lower.contains("\"thought\"") ||
            lower.contains("\"action\"") ||
            lower.contains("\"final\"") ||
            lower.contains("\"tool\"")

        guard !looksLikeStructuredTurn else { return nil }
        guard text.count >= 8 else { return nil }
        return String(text.prefix(4_000))
    }

    private nonisolated static func sanitizeInternalErrorNoise(from raw: String) -> String {
        var text = raw
        let knownNoisePatterns = [
            #"(?im)^\s*Generation error:.*(?:\R|$)"#,
            #"(?im)^\s*The operation couldn[’']t be completed\..*(?:\R|$)"#,
            #"(?im)^\s*\(SwiftLlama\.LlamaError error \d+\)\.?(?:\R|$)"#,
            #"(?im)^\s*I hit an internal formatting issue and repaired it into a plain answer\..*(?:\R|$)"#,
            #"(?im)^\s*Prefix noise:.*(?:\R|$)"#,
            #"(?im)^\s*Suffix noise:.*(?:\R|$)"#,
            #"(?im)^\s*No valid JSON object found in raw model output\..*(?:\R|$)"#
        ]

        for pattern in knownNoisePatterns {
            text = text.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func isMeaningfulDiagnosticNoise(_ text: String) -> Bool {
        let sanitized = sanitizeInternalErrorNoise(from: text)
        return !sanitized.isEmpty
    }

    func sanitizeInternalErrorNoiseForTests(_ raw: String) -> String {
        Self.sanitizeInternalErrorNoise(from: raw)
    }
}


private extension AgentService {
    static func applyLegacyGroundingAssembly(_ req: AgentRequest) -> AgentRequest {
        let memoryContent = req.relevantMemories.prefix(8).map { "- \($0.content)" }.joined(separator: "\n")
        let toolContent = req.availableTools.prefix(24).map { "- \($0.id): \($0.description)" }.joined(separator: "\n")
        let runtimeContent = "legacy-interactive"
        let sections: [PromptGroundingSection] = [
            .init(title: "Relevant memories", content: memoryContent, estimatedChars: memoryContent.count, sourceIDs: req.relevantMemories.prefix(8).map { $0.id.uuidString }, privacyLevel: .moderate),
            .init(title: "Available tools", content: toolContent, estimatedChars: toolContent.count, sourceIDs: req.availableTools.prefix(24).map { $0.id }, privacyLevel: .low),
            .init(title: "Runtime policy", content: runtimeContent, estimatedChars: runtimeContent.count, sourceIDs: [], privacyLevel: .low)
        ].filter { !$0.content.isEmpty }
        let budgetPlan = ContextBudgetAllocator.allocate(
            for: AssistantTurnContext(
                task: .agentPlan,
                input: req.userMessage,
                isForeground: true,
                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                thermalState: ProcessInfo.processInfo.thermalState
            ),
            maxInputTokens: 800
        )
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: req.systemPrompt, baseUserMessage: req.userMessage, sections: sections, policy: .rolePipeline, budgetPlan: budgetPlan)
        return AgentRequest(systemPrompt: assembled.systemPrompt, history: req.history, userMessage: assembled.userMessage, temperature: req.temperature, topP: req.topP, repetitionPenalty: req.repetitionPenalty, maxTokens: req.maxTokens, maxSteps: req.maxSteps, availableTools: req.availableTools, relevantMemories: req.relevantMemories, attachments: req.attachments, conversationID: req.conversationID, turnID: req.turnID, scenarioID: req.scenarioID, e2eRunID: req.e2eRunID, agentRunID: req.agentRunID)
    }
}
