import Foundation

nonisolated struct AgentStep: Codable, Sendable, Identifiable, Hashable {
    var id: UUID = UUID()
    var kind: Kind
    var content: String
    var toolID: String?
    var toolArgs: [String: String]?

    nonisolated enum Kind: String, Codable, Sendable {
        case thought
        case action
        case approvalBoundary
        case observation
        case reflection
    }

    var icon: String {
        switch kind {
        case .thought: "brain"
        case .action: "wrench.and.screwdriver.fill"
        case .approvalBoundary: "checkmark.shield"
        case .observation: "eye.fill"
        case .reflection: "sparkle"
        }
    }

    var label: String {
        switch kind {
        case .thought: "Thought"
        case .action: "Action"
        case .approvalBoundary: "Approval"
        case .observation: "Observation"
        case .reflection: "Reflection"
        }
    }
}

nonisolated enum AgentStepContentBudget {
    static let maxPersistedSteps = 80
    static let maxStepCharacters = 2_000

    static func boundedSanitizedSteps(_ steps: [AgentStep]) -> [AgentStep] {
        let sanitized = AgentVisibleContentSanitizer.sanitizedSteps(steps)
        let bounded = sanitized.count > maxPersistedSteps ? Array(sanitized.suffix(maxPersistedSteps)) : sanitized
        return bounded.map { step in
            var copy = step
            copy.content = truncated(step.content)
            return copy
        }
    }

    static func truncated(_ text: String) -> String {
        guard text.count > maxStepCharacters else { return text }
        if let existingMarkerRange = existingTruncationMarkerRange(in: text) {
            let visibleSource = String(text[..<existingMarkerRange.lowerBound])
            return applyingTruncationMarker(String(text[existingMarkerRange]), to: visibleSource)
        }

        var marker = truncationMarker(hiddenCount: max(0, text.count - maxStepCharacters))
        var visible = boundedVisiblePrefix(from: text, marker: marker)
        for _ in 0..<3 {
            let refinedMarker = truncationMarker(hiddenCount: max(0, text.count - visible.count))
            let refinedVisible = boundedVisiblePrefix(from: text, marker: refinedMarker)
            if refinedMarker == marker, refinedVisible == visible { break }
            marker = refinedMarker
            visible = refinedVisible
        }
        return "\(visible)\(marker)"
    }

    private static func applyingTruncationMarker(_ marker: String, to visibleSource: String) -> String {
        let visible = boundedVisiblePrefix(from: visibleSource, marker: marker)
        return "\(visible)\(marker)"
    }

    private static func boundedVisiblePrefix(from text: String, marker: String) -> String {
        let visibleLimit = max(0, maxStepCharacters - marker.count)
        return String(text.prefix(visibleLimit))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func truncationMarker(hiddenCount: Int) -> String {
        "\n… \(hiddenCount.formatted()) more characters hidden."
    }

    private static func existingTruncationMarkerRange(in text: String) -> Range<String.Index>? {
        text.range(
            of: #"\n… [0-9,]+ more characters hidden\.$"#,
            options: .regularExpression
        )
    }
}

/// Final-answer placeholder filtering already exists in `ChatView`, but agent
/// steps are persisted and rendered through a separate path. Keep this sanitizer
/// close to the step model so every UI surface can reuse the same hard stop.
nonisolated enum AgentVisibleContentSanitizer {
    private static let literalSentinels: Set<String> = [
        "<private_reasoning>",
        "<user_final_text>",
        "private_reasoning",
        "user_final_text"
    ]

    private static let compactSentinels: Set<String> = [
        "privatereasoning",
        "userfinaltext",
        "answershowntotheuser",
        "youranswertotheuser",
        "shortprivateroutingnote",
        "shortreasoning"
    ]

    private static let compactPrefixes: [String] = [
        "privatereasoning",
        "userfinaltext",
        "answershowntotheuser",
        "youranswertotheuser",
        "shortprivateroutingnote",
        "shortreasoning"
    ]

    private static let internalNoiseMarkers: [String] = [
        "i hit an internal formatting issue",
        "internal formatting issue and repaired",
        "generation error:",
        "no valid json object found in raw model output",
        "swiftllama.llamaerror",
        "prefix noise:",
        "suffix noise:",
        "selected json:",
        "raw model output"
    ]

    static func sanitizedSteps(_ steps: [AgentStep]) -> [AgentStep] {
        steps.compactMap { step in
            guard let clean = sanitize(step.content, kind: step.kind) else {
                return nil
            }
            var copy = step
            copy.content = clean
            return copy
        }
    }

    static func sanitize(_ text: String, kind: AgentStep.Kind? = nil) -> String? {
        let withoutWebPayload = WebRichContentPayload.removingMarkers(from: text)
        let trimmed = withoutWebPayload.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard !isPrivateOrSchemaPlaceholder(trimmed) else { return nil }
        guard !isInternalRepairNoise(trimmed) else { return nil }

        let cleanedLines = trimmed
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .filter { !isPrivateOrSchemaPlaceholder($0) }
            .filter { !isInternalRepairNoise($0) }

        let cleaned = cleanedLines.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !cleaned.isEmpty else { return nil }
        guard !isPrivateOrSchemaPlaceholder(cleaned) else { return nil }
        guard !isInternalRepairNoise(cleaned) else { return nil }
        return cleaned
    }

    static func isPrivateOrSchemaPlaceholder(_ text: String) -> Bool {
        let literal = normalizedLiteral(text)
        if literalSentinels.contains(literal) { return true }
        if literal.count >= 6, literalSentinels.contains(where: { $0.hasPrefix(literal) }) { return true }

        let compact = compacted(text)
        if compactSentinels.contains(compact) { return true }
        if compact.count >= 6, compactPrefixes.contains(where: { $0.hasPrefix(compact) }) { return true }
        return false
    }

    static func isInternalRepairNoise(_ text: String) -> Bool {
        let lower = text.lowercased()
        return internalNoiseMarkers.contains { lower.contains($0) }
    }

    private static func normalizedLiteral(_ text: String) -> String {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: #"\s+"#, with: "", options: .regularExpression)
    }

    private static func compacted(_ text: String) -> String {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "", options: .regularExpression)
    }
}

nonisolated enum AgentStepCodec {
    static func encode(_ steps: [AgentStep]) -> String? {
        let sanitized = AgentVisibleContentSanitizer.sanitizedSteps(steps)
        guard !sanitized.isEmpty else { return nil }
        let enc = JSONEncoder()
        guard let data = try? enc.encode(sanitized) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func decode(_ string: String?) -> [AgentStep] {
        guard let string, let data = string.data(using: .utf8) else { return [] }
        let decoded = (try? JSONDecoder().decode([AgentStep].self, from: data)) ?? []
        return AgentVisibleContentSanitizer.sanitizedSteps(decoded)
    }
}
