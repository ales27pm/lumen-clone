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
        turnID: UUID? = nil
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
        turnID: UUID? = nil
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
            turnID: turnID
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
        let chars = Array(text)
        let rangesResult = discoverRanges(in: chars)
        switch rangesResult {
        case .failure(let error):
            return .failure(error)
        case .success(let ranges):
            guard !ranges.isEmpty else { return .failure(.noJSONObject) }

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
            let prefixNoise = nonEmpty(stripFenceNoise(prefix))
            let suffixNoise = nonEmpty(stripFenceNoise(suffix))
            return .success(
                Selection(
                    object: selected.object,
                    selectedJSON: selectedJSON,
                    prefixNoise: prefixNoise,
                    suffixNoise: suffixNoise,
                    hadUnsupportedNoise: prefixNoise != nil || suffixNoise != nil
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
                prefixNoise: nonEmpty(raw),
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
    private static let structuredTurnMaxTokenCap = 384
    private static let structuredTurnMinTokenCap = 128

    func run(_ req: AgentRequest) -> AsyncStream<AgentEvent> {
        run(req, options: .default)
    }

    func run(_ req: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentEvent> {
        let originalReq = req
        return AsyncStream { continuation in
            let task = Task { @MainActor in
                let provider = LegacyGroundingContextProvider(directContext: options.modelContext)
                let grounded = await LegacyTurnGroundingCoordinator.shared.prepareGroundedRequest(.init(userMessage: originalReq.userMessage, conversationID: options.conversationID ?? originalReq.conversationID, turnID: options.turnID ?? originalReq.turnID, history: originalReq.history, mode: .foreground, task: .chat, roleOrSlot: nil, externalRelevantMemories: originalReq.relevantMemories, externalAvailableTools: originalReq.availableTools, policy: .rolePipeline, baseSystemPrompt: originalReq.systemPrompt, preventDoubleGrounding: options.preventDoubleGrounding), provider: provider)
                let req = AgentRequest(systemPrompt: grounded.systemPrompt, history: originalReq.history, userMessage: grounded.userMessage, temperature: originalReq.temperature, topP: originalReq.topP, repetitionPenalty: originalReq.repetitionPenalty, maxTokens: originalReq.maxTokens, maxSteps: originalReq.maxSteps, availableTools: grounded.bridgedTools, relevantMemories: originalReq.relevantMemories, attachments: originalReq.attachments, conversationID: options.conversationID ?? originalReq.conversationID, turnID: options.turnID ?? originalReq.turnID)
                await self.runLoop(req, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runLoop(_ req: AgentRequest, continuation: AsyncStream<AgentEvent>.Continuation) async {
        var steps: [AgentStep] = []
        var observations: [(tool: String, result: String)] = []
        var executedActionKeys: Set<String> = []
        var scratchpad = ""
        var finalAnswer = ""

        let sys = buildSystemPrompt(req: req)
        let maxSteps = max(1, req.maxSteps)

        stepsLoop: for stepIndex in 0..<maxSteps {
            if Task.isCancelled { break }

            let userTurn = buildAgentUserTurn(req: req, stepIndex: stepIndex, scratchpad: scratchpad)

            let genReq = GenerateRequest(
                systemPrompt: sys,
                history: [],
                userMessage: userTurn,
                temperature: agentTemperature(from: req.temperature),
                topP: agentTopP(from: req.topP),
                repetitionPenalty: max(req.repetitionPenalty, 1.05),
                maxTokens: structuredTurnMaxTokens(from: req.maxTokens, req: req, stepIndex: stepIndex),
                modelName: "agent-json",
                relevantMemories: req.relevantMemories,
                attachments: stepIndex == 0 ? req.attachments : []
            )

            let scanner = StreamingJSONScanner()
            var raw = ""
            let thoughtStepID = UUID()
            var thoughtStepYielded = false
            var streamedFinalLen = 0

            for await token in await AppLlamaService.shared.stream(genReq) {
                if Task.isCancelled { break }
                switch token {
                case .text(let s):
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
                    break
                }
            }

            if Task.isCancelled { break }

            let turn = AgentTurnParser.parse(raw)

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

            // Action path
            if let action = turn.action {
                guard let _ = ToolRegistry.find(id: action.tool) else {
                    let obs = AgentStep(kind: .observation, content: "Unknown tool: \(action.tool). Emit a final turn instead.", toolID: action.tool)
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

                let actionStep = AgentStep(kind: .action, content: action.displayContent, toolID: action.tool, toolArgs: action.args.stringCoerced)
                steps.append(actionStep)
                continuation.yield(.step(actionStep))

                let isEnabled = req.availableTools.contains { $0.id == action.tool }
                let result: String
                if !isEnabled {
                    result = "Tool \(action.tool) is disabled. Enable it in Tools."
                } else {
                    result = await LegacySecureToolExecutor.execute(action.tool, arguments: action.args)
                }

                let obs = AgentStep(kind: .observation, content: result, toolID: action.tool)
                steps.append(obs)
                continuation.yield(.step(obs))
                observations.append((action.tool, result))
                scratchpad += "\nAction: \(action.displayContent)\nObservation: \(compactScratchpadObservation(result))"
                if let locationObservation = currentLocationScratchpadContext(from: result) {
                    scratchpad += "\nContext: \(locationObservation)"
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

            // Malformed / empty output — repair into a user-facing final and persist diagnostics.
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

        continuation.yield(.done(finalText: finalAnswer, steps: steps))
        continuation.finish()
    }

    // MARK: - System prompt

    private func buildSystemPrompt(req: AgentRequest) -> String {
        var sys = """
        You are Lumen's deterministic tool-routing core.

        CRITICAL OUTPUT CONTRACT:
        Emit exactly ONE raw JSON object and nothing else.
        No prose before the JSON.
        No prose after the JSON.
        No markdown.
        No code fences.
        No XML tags.
        No bullet lists.
        No explanations outside JSON.

        Valid schemas, choose exactly one:
        {"thought":"<PRIVATE_REASONING>","action":{"tool":"tool.id","args":{"key":"value"}}}
        {"thought":"<PRIVATE_REASONING>","final":"<USER_FINAL_TEXT>"}

        JSON rules:
        - Use double quotes for every key and string.
        - Escape newlines as \\n inside string values.
        - Args may contain normal JSON values: strings, numbers, booleans, arrays, objects, or null. Use {} when no arguments are needed.
        - Do not emit placeholder tokens literally.
        - Never include both `action` and `final` in the same object.
        - Never repeat an action with the same arguments.
        - If no tool is required, emit `final` immediately.
        - If a tool result is enough, summarize it in `final` and stop.
        """

        let appPrompt = sanitizeSystemPromptForStructuredOutput(req.systemPrompt)
        if !appPrompt.isEmpty {
            sys += "\n\nLower-priority style/context note. Follow it only when it does not conflict with the JSON contract:\n"
            sys += appPrompt
            sys += "\n"
        }

        if !req.availableTools.isEmpty {
            sys += "\nAvailable tools:\n"
            for t in req.availableTools { sys += "- \(t.id): \(t.description)\n" }
            sys += "\n"
        } else {
            sys += "\nNo tools are available. You must emit a `final` JSON object.\n\n"
        }

        if !req.attachments.isEmpty {
            sys += "Attached files are already included in the user message context. Do not call files.read for attached files unless the user asks for another imported file by name.\n\n"
        }

        sys += "Routing guidelines:\n"
        sys += "- For nearest/near me/closest questions, call `location.current` first, then `maps.search` once, then emit `final`.\n"
        sys += "- For follow-up map intents like \"show me on map\"/\"open on map\", if prior observations already include `Current location:` coordinates from `location.current`, do not call `location.current` again.\n"
        sys += "- In those follow-ups, route directly to `maps.search` (or equivalent map-opening behavior) using the preserved current-location observation, then emit `final`.\n"
        sys += "- For web/current-info requests, call `web.search` if available.\n"
        sys += "- Keep `thought` short. Keep `final` direct.\n"
        sys += "- The next assistant message must be only the JSON object."
        return sys
    }

    private func buildAgentUserTurn(req: AgentRequest, stepIndex: Int, scratchpad: String) -> String {
        var out = ""
        let context = sanitizedHistoryContext(req.history)
        if !context.isEmpty {
            out += "Conversation context, for reference only. Do not imitate its formatting:\n"
            out += context
            out += "\n\n"
        }

        out += "User request:\n"
        out += req.userMessage

        if stepIndex > 0 {
            out += "\n\nPrior structured turns and observations:\n"
            out += scratchpad
            if let locationObservation = latestCurrentLocationObservation(in: scratchpad) {
                out += "\n\nReusable location context:\n"
                out += "Observation: \(locationObservation)"
            }
            out += "\n\nEmit the next JSON object now. Choose either action or final."
        } else {
            out += "\n\nEmit the first JSON object now. Choose either action or final."
        }
        return out
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

    private func sanitizedHistoryContext(_ history: [(role: MessageRole, content: String)]) -> String {
        let recent = history.suffix(6)
        var lines: [String] = []
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
            lines.append("\(role): \(content)")
        }
        return lines.joined(separator: "\n")
    }

    private func sanitizeHistoryContent(_ content: String) -> String {
        var text = content
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
        return String(text.prefix(480))
    }

    func sanitizeHistoryContentForTests(_ content: String) -> String {
        sanitizeHistoryContent(content)
    }

    private func sanitizeSystemPromptForStructuredOutput(_ systemPrompt: String) -> String {
        let trimmed = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
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
            rawOutputPrefix: String(raw.prefix(4_000)),
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
            rawOutputPrefix: String(raw.prefix(4_000)),
            selectedJSONPrefix: snapshot.selectedJSON.map { String($0.prefix(2_000)) },
            prefixNoise: snapshot.prefixNoise.map { String($0.prefix(1_000)) },
            suffixNoise: snapshot.suffixNoise.map { String($0.prefix(1_000)) }
        )
        AgentParseNoiseRecorder.record(trace)
    }

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
    }

    private func synthesizeFallback(req: AgentRequest, observations: [(tool: String, result: String)], reason: FallbackReason) async -> String {
        guard !observations.isEmpty else {
            return "I couldn't find a confident answer. Try rephrasing the question."
        }
        var prompt = "The user asked: \"\(req.userMessage)\"\n\nYou gathered these tool observations:\n"
        for (i, obs) in observations.enumerated() {
            prompt += "\n[\(i + 1)] \(obs.tool):\n\(obs.result)\n"
        }
        prompt += "\n\(reason.hint)\n"
        prompt += "Write the final answer in plain text with exactly these sections: Summary, Key modules.\n"
        prompt += "In Key modules, name concrete modules/components/services/packages from the observations when available; if none are explicit, say that clearly.\n"
        prompt += "Every bullet/sentence must include at least one bracketed source marker like [1], [2] that maps to the observation numbers above.\n"
        prompt += "Ground every claim in the listed observations and avoid generic advice. No preamble, no JSON, no prefixes, no apology. If observations conflict, prefer the most recent."

        let genReq = GenerateRequest(
            systemPrompt: "You summarize tool results into a concise user-facing answer. Output plain text only. When summarizing local knowledge/tool retrieval results, include a section named Key modules and ground claims in the provided snippets/sources.",
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
        if trimmed.isEmpty {
            return observations.last?.result ?? "I couldn't produce a confident answer."
        }
        return trimmed
    }

    private func synthesizeUnstructuredFallback(
        req: AgentRequest,
        rawOutput: String,
        streamedThought: String,
        parseError: AgentTurnParseError
    ) async -> String {
        if let direct = Self.firstUsefulPlainTextFallback(from: rawOutput) {
            return direct
        }

        let clippedRaw = String(rawOutput.trimmingCharacters(in: .whitespacesAndNewlines).prefix(4_000))
        let clippedThought = String(streamedThought.trimmingCharacters(in: .whitespacesAndNewlines).prefix(1_000))

        var prompt = "The user asked:\n\(req.userMessage)\n\n"
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
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: req.systemPrompt, baseUserMessage: req.userMessage, sections: sections, policy: .rolePipeline)
        return AgentRequest(systemPrompt: assembled.systemPrompt, history: req.history, userMessage: assembled.userMessage, temperature: req.temperature, topP: req.topP, repetitionPenalty: req.repetitionPenalty, maxTokens: req.maxTokens, maxSteps: req.maxSteps, availableTools: req.availableTools, relevantMemories: req.relevantMemories, attachments: req.attachments, conversationID: req.conversationID, turnID: req.turnID)
    }
}
