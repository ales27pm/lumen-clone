import Foundation

public struct LiveRuntimeToolRegistryProvider: RuntimeToolRegistryProviding {
    public init() {}

    public func currentToolDefinitions() -> [RuntimeToolDefinition] {
        ToolRegistry.all.map { tool in
            RuntimeToolDefinition(
                id: tool.id,
                displayName: tool.name,
                description: tool.description,
                requiresApproval: tool.requiresApproval,
                permissionKey: tool.permissionKey,
                arguments: RuntimeToolArgumentInferencer.arguments(from: tool.description)
            )
        }
    }
}

private enum RuntimeToolArgumentInferencer {
    private static let numericMarkers = ["minutes", "seconds", "duration", "interval", "limit", "count", "months"]
    private static let typeHintWords: Set<String> = ["uuid", "fallback"]

    static func arguments(from description: String) -> [RuntimeToolArgument] {
        guard let argsBody = argsBody(from: description) else { return [] }
        let trimmed = argsBody.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let lowered = trimmed.lowercased()
        guard !["none", "no args", "n/a"].contains(lowered) else { return [] }

        let normalized = trimmed
            .replacingOccurrences(of: " or ", with: ", ")
            .replacingOccurrences(of: " plus ", with: ", ")
            .replacingOccurrences(of: " depending on ", with: ", ")

        var specs: [(name: String, required: Bool)] = []
        var optionalGroup = false

        for rawPart in normalized.split(whereSeparator: { character in
            character == "," || character == ";" || character == "/"
        }) {
            var token = String(rawPart).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !token.isEmpty else { continue }
            token = removeParentheticalText(from: token)

            let lowerToken = token.lowercased()
            let tokenOptional = optionalGroup || lowerToken.hasPrefix("optional ")
            if lowerToken.hasPrefix("optional ") {
                token = String(token.dropFirst("optional ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard !token.isEmpty else {
                optionalGroup = true
                continue
            }

            let pieces = token.split(whereSeparator: { $0.isWhitespace })
            guard let first = pieces.first else { continue }
            let name = String(first).trimmingCharacters(in: CharacterSet(charactersIn: "`'\".:"))
            let loweredName = name.lowercased()
            guard !["none", "args", "arg"].contains(loweredName) else { continue }
            if typeHintWords.contains(loweredName), pieces.count > 1 { continue }
            guard isValidArgumentName(name) else { continue }
            guard !specs.contains(where: { $0.name == name }) else {
                optionalGroup = tokenOptional
                continue
            }
            specs.append((name: name, required: !tokenOptional))
            optionalGroup = tokenOptional
        }

        return specs.map { spec in
            RuntimeToolArgument(
                name: spec.name,
                type: inferredType(for: spec.name),
                required: spec.required
            )
        }
    }

    private static func argsBody(from description: String) -> String? {
        guard let argsRange = description.range(of: "Args:", options: [.caseInsensitive]) else {
            return nil
        }
        let afterArgs = description[argsRange.upperBound...]
        if let sentenceEnd = afterArgs.firstIndex(of: ".") {
            return String(afterArgs[..<sentenceEnd])
        }
        return String(afterArgs)
    }

    private static func removeParentheticalText(from value: String) -> String {
        var result = ""
        var depth = 0
        for character in value {
            if character == "(" {
                depth += 1
                continue
            }
            if character == ")" {
                depth = max(0, depth - 1)
                continue
            }
            if depth == 0 {
                result.append(character)
            }
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func isValidArgumentName(_ value: String) -> Bool {
        guard let first = value.first, first == "_" || first.isLetter else { return false }
        return value.allSatisfy { character in
            character == "_" || character.isLetter || character.isNumber
        }
    }

    private static func inferredType(for name: String) -> String {
        let lowered = name.lowercased()
        if numericMarkers.contains(where: { lowered.contains($0) }) {
            return "number"
        }
        if lowered == "repeats" || lowered.hasPrefix("is") || lowered.hasPrefix("has") || lowered.hasPrefix("should") {
            return "bool"
        }
        return "string"
    }
}
