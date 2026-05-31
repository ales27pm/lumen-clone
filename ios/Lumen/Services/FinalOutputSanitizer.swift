import Foundation

nonisolated struct SanitizedFinalOutput: Sendable, Equatable {
    let text: String
    let removedArtifacts: [FinalOutputArtifact]
    let hadUnsafeLeakage: Bool
    let artifactAudit: FinalOutputArtifactAudit
}

nonisolated struct FinalOutputArtifactAudit: Sendable, Equatable {
    let rawPrefix: String
    let sanitizedPrefix: String
    let hadUnsafeLeakage: Bool
    let removedArtifacts: Bool
    let removedArtifactTypes: [FinalOutputArtifact]
}

nonisolated enum FinalOutputArtifact: String, Codable, Sendable, Equatable {
    case thinkBlock
    case malformedThinkPrefix
    case lumenWebPayload
    case rawToolPayload
    case injectedFallbackPrefix
    case emptyAfterSanitization
}

private final class FinalOutputSanitizerRecoveryCache: @unchecked Sendable {
    private let lock = NSLock()
    private var recoveredBySanitizedText: [String: SanitizedFinalOutput] = [:]

    func remember(_ output: SanitizedFinalOutput, forSanitizedText sanitizedText: String) {
        guard output.hadUnsafeLeakage else { return }
        lock.lock()
        recoveredBySanitizedText[sanitizedText] = output
        if recoveredBySanitizedText.count > 64 {
            recoveredBySanitizedText.remove(at: recoveredBySanitizedText.startIndex)
        }
        lock.unlock()
    }

    func consumeRecovery(forSanitizedText sanitizedText: String) -> SanitizedFinalOutput? {
        lock.lock()
        defer { lock.unlock() }
        return recoveredBySanitizedText.removeValue(forKey: sanitizedText)
    }
}

nonisolated struct StreamingFinalOutputSanitizer: Sendable {
    nonisolated enum Finalization: Sendable {
        case append(final: SanitizedFinalOutput, remainingDelta: String)
        case replace(final: SanitizedFinalOutput)
    }

    private var rawBuffer = ""
    private var emittedSanitized = ""
    private var sawSanitizerGeneratedFallbackDuringStream = false
    private let holdbackCharacters = 192

    mutating func ingest(_ chunk: String) -> String {
        guard !chunk.isEmpty else { return "" }
        rawBuffer += chunk

        let cutoff = safeRawCutoffIndex(in: rawBuffer)
        guard cutoff > 0 else { return "" }

        let safeRawPrefix = String(rawBuffer.prefix(cutoff))
        guard !safeRawPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return ""
        }

        let sanitized = FinalOutputSanitizer.sanitizeUserVisibleText(safeRawPrefix)

        // A partial stream can have an empty/unsafe safe prefix while the holdback window
        // waits for split markers such as <think> or <lumen_web_payload>. The global
        // fallback is valid only after finalization proves the complete output is
        // unusable; emitting it here leaks a fake error prefix into otherwise valid
        // answers. Track this provenance so finish() can clean a later fallback prefix
        // only when the stream itself proved sanitizer-generated fallback ancestry.
        if sanitized.text == FinalOutputSanitizer.fallback {
            let rawPrefixWasFallback = safeRawPrefix
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
                .hasPrefix(FinalOutputSanitizer.fallback.lowercased())
            if sanitized.hadUnsafeLeakage || !rawPrefixWasFallback {
                sawSanitizerGeneratedFallbackDuringStream = true
            }
            return ""
        }

        let sanitizedPrefix = sanitized.text
        guard sanitizedPrefix.count > emittedSanitized.count else { return "" }

        let delta = String(sanitizedPrefix.dropFirst(emittedSanitized.count))
        emittedSanitized = sanitizedPrefix
        return delta
    }

    mutating func finish() -> Finalization {
        let final = FinalOutputSanitizer.sanitizeUserVisibleText(
            rawBuffer,
            isInjectedProvenance: sawSanitizerGeneratedFallbackDuringStream
        )
        if final.text.hasPrefix(emittedSanitized) {
            let remainingDelta = String(final.text.dropFirst(emittedSanitized.count))
            emittedSanitized = final.text
            return .append(final: final, remainingDelta: remainingDelta)
        }
        emittedSanitized = final.text
        return .replace(final: final)
    }

    private func safeRawCutoffIndex(in raw: String) -> Int {
        let lower = raw.lowercased()
        let holdbackStart = max(0, raw.count - holdbackCharacters)
        var unsafeStart = raw.count

        if let hiddenOpen = firstUnclosedHiddenMarkerStart(in: lower) {
            unsafeStart = min(unsafeStart, hiddenOpen)
        }

        if let payloadOpen = lower.range(of: "<lumen_web_payload")?.lowerBound,
           lower.range(of: "</lumen_web_payload>", range: payloadOpen..<lower.endIndex) == nil {
            unsafeStart = min(unsafeStart, lower.distance(from: lower.startIndex, to: payloadOpen))
        }

        if let rawJSONStart = rawToolPayloadUnclosedObjectStart(lower: lower) {
            unsafeStart = min(unsafeStart, rawJSONStart)
        }

        return max(0, min(unsafeStart, holdbackStart))
    }

    private func rawToolPayloadUnclosedObjectStart(lower: String) -> Int? {
        let markers = ["\"kind\":\"searchresults\"", "\"kind\" : \"searchresults\"", "\"mediakind\":\"page\"", "\"mediakind\" : \"page\"", "\"sourcepageurl\""]
        guard let markerRange = markers.compactMap({ lower.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) else { return nil }
        let markerIndex = lower.distance(from: lower.startIndex, to: markerRange.lowerBound)
        let prefix = lower.prefix(markerIndex)
        guard let bracePos = prefix.lastIndex(of: "{") else { return nil }
        let braceIndex = lower.distance(from: lower.startIndex, to: bracePos)
        let suffix = String(lower[lower.index(lower.startIndex, offsetBy: braceIndex)...])
        var depth = 0
        var inString = false
        var escaped = false
        for ch in suffix {
            if inString {
                if escaped { escaped = false; continue }
                if ch == "\\" { escaped = true; continue }
                if ch == "\"" { inString = false }
                continue
            }
            if ch == "\"" { inString = true; continue }
            if ch == "{" { depth += 1 }
            else if ch == "}" { depth -= 1; if depth == 0 { return nil } }
        }
        return braceIndex
    }

    private func firstUnclosedHiddenMarkerStart(in lower: String) -> Int? {
        let tagNames = ["think", "thinkresponse", "think_response", "thinking", "reasoning", "analysis", "chain_of_thought"]
        return tagNames.compactMap { tag -> Int? in
            guard let open = lower.range(of: "<\(tag)")?.lowerBound else { return nil }
            if lower.range(of: "</\(tag)>", range: open..<lower.endIndex) != nil { return nil }
            return lower.distance(from: lower.startIndex, to: open)
        }.min()
    }
}

nonisolated enum FinalOutputSanitizer {
    static let fallback = "I hit an internal response-format issue. Please try again."
    private static let recoveryCache = FinalOutputSanitizerRecoveryCache()
    private static let rawToolPayloadPattern = #"(?is)\{[^{}]{0,24000}(?:"kind"\s*:\s*"searchresults"|"mediakind"\s*:\s*"page"|"sourcepageurl"|"kind":"searchresults")[^{}]{0,24000}\}"#
    private static let cachedRawToolPayloadRegex: Result<NSRegularExpression, Error> = {
        do {
            return .success(try NSRegularExpression(pattern: rawToolPayloadPattern, options: []))
        } catch {
            return .failure(error)
        }
    }()

    static func sanitizeUserVisibleText(_ raw: String, isInjectedProvenance: Bool = false) -> SanitizedFinalOutput {
        var text = raw
        var removed: [FinalOutputArtifact] = []

        func mark(_ artifact: FinalOutputArtifact) {
            if !removed.contains(artifact) {
                removed.append(artifact)
            }
        }

        let originalLower = raw.lowercased()
        if containsHiddenReasoningMarker(originalLower) {
            mark(.thinkBlock)
        }
        if originalLower.contains("<lumen_web_payload") || originalLower.contains("</lumen_web_payload>") {
            mark(.lumenWebPayload)
        }
        if containsRawToolPayloadMarker(originalLower) {
            mark(.rawToolPayload)
        }

        let hiddenRemoval = removingHiddenReasoningBlocks(from: text)
        if hiddenRemoval.removedAny {
            text = hiddenRemoval.text
            mark(.thinkBlock)
        }
        let malformedHiddenRemoval = removingMalformedHiddenReasoningLines(from: text)
        if malformedHiddenRemoval.removedAny {
            text = malformedHiddenRemoval.text
            mark(.malformedThinkPrefix)
        }
        if beginsWithHiddenReasoningMarker(text) {
            text = ""
            mark(.malformedThinkPrefix)
        }
        text = text.replacingOccurrences(of: "(?is)<lumen_web_payload>.*?</lumen_web_payload>", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?is)<lumen_web_payload[^>]*>.*$", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?is)</lumen_web_payload>", with: " ", options: .regularExpression)

        let rawPayloadRemoval = removingRawToolPayloadObjects(from: text)
        if rawPayloadRemoval.removedAny {
            text = rawPayloadRemoval.text
            mark(.rawToolPayload)
        }

        let loosePayloadRemoval = removingRawToolPayloadFragments(from: text)
        if loosePayloadRemoval.removedAny {
            text = loosePayloadRemoval.text
            mark(.rawToolPayload)
        }

        let markerLineRemoval = removingRawToolPayloadMarkerLines(from: text)
        if markerLineRemoval.removedAny {
            text = markerLineRemoval.text
            mark(.rawToolPayload)
        }

        if containsRawToolPayloadMarker(text.lowercased()) {
            let trailingRemoval = removingTrailingRawToolPayload(from: text)
            if trailingRemoval.removedAny {
                text = trailingRemoval.text
                mark(.rawToolPayload)
            }
        }

        text = normalizeWhitespace(text)

        if isInjectedProvenance {
            let fallbackRemoval = removingInjectedFallbackPrefix(from: text)
            if fallbackRemoval.removedAny {
                text = fallbackRemoval.text
                mark(.injectedFallbackPrefix)
            }
        }

        if text.isEmpty {
            mark(.emptyAfterSanitization)
            text = fallback
        }

        let hadUnsafeLeakage = !removed.isEmpty
        let output = SanitizedFinalOutput(
            text: text,
            removedArtifacts: removed,
            hadUnsafeLeakage: hadUnsafeLeakage,
            artifactAudit: FinalOutputArtifactAudit(
                rawPrefix: String(raw.prefix(220)),
                sanitizedPrefix: String(text.prefix(220)),
                hadUnsafeLeakage: hadUnsafeLeakage,
                removedArtifacts: !removed.isEmpty,
                removedArtifactTypes: removed
            )
        )
        recoveryCache.remember(output, forSanitizedText: text)
        return output
    }

    static func consumeRecoveredUnsafeOutput(forSanitizedText text: String) -> SanitizedFinalOutput? {
        if let direct = recoveryCache.consumeRecovery(forSanitizedText: text) {
            return direct
        }

        let sanitizedKey = sanitizedRecoveryKey(for: text)
        guard sanitizedKey != text else { return nil }
        return recoveryCache.consumeRecovery(forSanitizedText: sanitizedKey)
    }

    private static func sanitizedRecoveryKey(for raw: String) -> String {
        var text = raw
        text = removingHiddenReasoningBlocks(from: text).text
        text = removingMalformedHiddenReasoningLines(from: text).text
        if beginsWithHiddenReasoningMarker(text) {
            text = ""
        }
        text = text.replacingOccurrences(of: "(?is)<lumen_web_payload>.*?</lumen_web_payload>", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?is)<lumen_web_payload[^>]*>.*$", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?is)</lumen_web_payload>", with: " ", options: .regularExpression)
        text = removingRawToolPayloadObjects(from: text).text
        text = normalizeWhitespace(text)
        return text.isEmpty ? fallback : text
    }

    private static let hiddenReasoningTagPattern = "(?:think[\\w_-]*|thinking|reasoning|analysis|chain_of_thought)"

    private static func containsHiddenReasoningMarker(_ lowercasedText: String) -> Bool {
        ["<think", "</think", "<thinking", "</thinking", "<reasoning", "</reasoning", "<analysis", "</analysis", "<chain_of_thought", "</chain_of_thought"].contains {
            lowercasedText.contains($0)
        }
    }

    private static func beginsWithHiddenReasoningMarker(_ text: String) -> Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).range(
            of: "^<\\s*\(hiddenReasoningTagPattern)\\b",
            options: [.regularExpression, .caseInsensitive]
        ) != nil
    }

    private static func removingHiddenReasoningBlocks(from source: String) -> (text: String, removedAny: Bool) {
        let pattern = "(?is)<\\s*(\(hiddenReasoningTagPattern))\\b[^>]*>.*?</\\s*\\1\\s*>"
        let redacted = source.replacingOccurrences(of: pattern, with: " ", options: .regularExpression)
        return (redacted, redacted != source)
    }

    private static func removingMalformedHiddenReasoningLines(from source: String) -> (text: String, removedAny: Bool) {
        let pattern = "(?im)^\\s*<\\s*\(hiddenReasoningTagPattern)\\b[^\\n]*(?:\\n|$)"
        let redacted = source.replacingOccurrences(of: pattern, with: " ", options: .regularExpression)
        return (redacted, redacted != source)
    }

    private static func normalizeWhitespace(_ text: String) -> String {
        text
            .replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
            .replacingOccurrences(of: " *\\n *", with: "\n", options: .regularExpression)
            .replacingOccurrences(of: "\\n{3,}", with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func removingInjectedFallbackPrefix(from source: String) -> (text: String, removedAny: Bool) {
        let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > fallback.count else { return (source, false) }
        guard trimmed.lowercased().hasPrefix(fallback.lowercased()) else { return (source, false) }
        let remainder = trimmed.dropFirst(fallback.count)
            .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ".:;-—–*")))
        guard !remainder.isEmpty else { return (source, false) }
        return (String(remainder), true)
    }

    private static func containsRawToolPayloadMarker(_ lowercasedText: String) -> Bool {
        lowercasedText.contains("{\"kind\":\"searchresults\"")
            || lowercasedText.contains("\"kind\" : \"searchresults\"")
            || lowercasedText.contains("\"mediakind\":\"page\"")
            || lowercasedText.contains("\"mediakind\" : \"page\"")
            || lowercasedText.contains("\"sourcepageurl\"")
    }

    private static func removingRawToolPayloadFragments(from source: String) -> (text: String, removedAny: Bool) {
        guard case let .success(regex) = cachedRawToolPayloadRegex else {
            return (source, false)
        }

        let range = NSRange(source.startIndex..<source.endIndex, in: source)
        if regex.firstMatch(in: source, options: [], range: range) == nil {
            return (source, false)
        }
        let redacted = regex.stringByReplacingMatches(in: source, options: [], range: range, withTemplate: " ")
        return (redacted, redacted != source)
    }

    private static func removingRawToolPayloadMarkerLines(from source: String) -> (text: String, removedAny: Bool) {
        let pattern = #"(?im)^.*("kind"\s*:\s*"searchresults"|"mediakind"\s*:\s*"page"|"sourcepageurl").*$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return (source, false)
        }
        let range = NSRange(source.startIndex..<source.endIndex, in: source)
        if regex.firstMatch(in: source, options: [], range: range) == nil {
            return (source, false)
        }
        let redacted = regex.stringByReplacingMatches(in: source, options: [], range: range, withTemplate: " ")
        return (redacted, redacted != source)
    }

    private static func removingTrailingRawToolPayload(from source: String) -> (text: String, removedAny: Bool) {
        let lower = source.lowercased()
        let markers = ["\"kind\":\"searchresults\"", "\"kind\" : \"searchresults\"", "\"mediakind\":\"page\"", "\"mediakind\" : \"page\"", "\"sourcepageurl\""]
        guard let markerIndex = markers.compactMap({ lower.range(of: $0)?.lowerBound }).min() else {
            return (source, false)
        }

        let lineStart = source[..<markerIndex].lastIndex(of: "\n").map { source.index(after: $0) } ?? source.startIndex
        let redacted = String(source[..<lineStart]) + " "
        return (redacted, true)
    }

    private static func removingRawToolPayloadObjects(from source: String) -> (text: String, removedAny: Bool) {
        var output = ""
        var index = source.startIndex
        var removedAny = false

        while index < source.endIndex {
            guard source[index] == "{" else {
                output.append(source[index])
                index = source.index(after: index)
                continue
            }

            guard let objectEnd = balancedJSONObjectEnd(in: source, from: index) else {
                output.append(source[index])
                index = source.index(after: index)
                continue
            }

            let nextIndex = source.index(after: objectEnd)
            let candidate = String(source[index..<nextIndex])
            if containsRawToolPayloadMarker(candidate.lowercased()) {
                output.append(" ")
                removedAny = true
            } else {
                output.append(candidate)
            }
            index = nextIndex
        }

        return (output, removedAny)
    }

    private static func balancedJSONObjectEnd(in source: String, from start: String.Index) -> String.Index? {
        var index = start
        var depth = 0
        var isInsideString = false
        var isEscaped = false

        while index < source.endIndex {
            let character = source[index]

            if isInsideString {
                if isEscaped {
                    isEscaped = false
                } else if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    isInsideString = false
                }
            } else {
                if character == "\"" {
                    isInsideString = true
                } else if character == "{" {
                    depth += 1
                } else if character == "}" {
                    depth -= 1
                    if depth == 0 {
                        return index
                    }
                }
            }

            index = source.index(after: index)
        }

        return nil
    }
}
