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

    /// Extracts argument definitions from a tool description, inferring their types and required status.
    /// - Parameter description: The tool description to parse for arguments.
    /// - Returns: An array of parsed arguments. Returns an empty array if no valid arguments are found.
    static func arguments(from description: String) -> [RuntimeToolArgument] {
        guard let argsBody = argsBody(from: description) else { return [] }
        let trimmed = argsBody.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let lowered = trimmed.lowercased()
        guard !["none", "no args", "n/a"].contains(lowered) else { return [] }

        var specs: [(name: String, required: Bool)] = []
        var optionalGroup = false

        for rawPart in trimmed.split(whereSeparator: { character in
            character == "," || character == ";"
        }) {
            var token = String(rawPart).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !token.isEmpty else { continue }
            token = removeParentheticalText(from: token)

            let lowerToken = token.lowercased()
            let declaredOptional = lowerToken.hasPrefix("optional ")
            let tokenOptional = optionalGroup || declaredOptional
            if declaredOptional {
                token = String(token.dropFirst("optional ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if token.lowercased().hasPrefix("plus ") {
                token = String(token.dropFirst("plus ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            token = removeDependencyPhrases(from: token)
            token = removeValueHintPhrases(from: token)
            guard !token.isEmpty else {
                optionalGroup = true
                continue
            }

            let aliases = argumentAliases(from: token)
            for alias in aliases {
                let pieces = alias.split(whereSeparator: { $0.isWhitespace })
                guard let first = pieces.first else { continue }
                let name = String(first).trimmingCharacters(in: CharacterSet(charactersIn: "`'\".:"))
                let loweredName = name.lowercased()
                guard !["none", "args", "arg"].contains(loweredName) else { continue }
                if typeHintWords.contains(loweredName), pieces.count > 1 { continue }
                guard isValidArgumentName(name) else { continue }
                guard !specs.contains(where: { $0.name == name }) else { continue }
                specs.append((name: name, required: !tokenOptional))
            }
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

    /// Removes trailing phrases indicating conditional dependency.
    /// - Returns: The trimmed string with dependency phrases removed.
    private static func removeDependencyPhrases(from value: String) -> String {
        var token = value.trimmingCharacters(in: .whitespacesAndNewlines)
        for marker in [" depending on schedule", " depending on the schedule"] {
            if let range = token.range(of: marker, options: [.caseInsensitive]) {
                token.removeSubrange(range.lowerBound..<token.endIndex)
            }
        }
        return token.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Removes boolean value-hint suffixes from a string.
    /// - Parameters:
    ///   - value: The string to process.
    /// - Returns: The string with boolean value-hint suffixes removed.
    private static func removeValueHintPhrases(from value: String) -> String {
        var token = value.trimmingCharacters(in: .whitespacesAndNewlines)
        for marker in [" true/false", " true or false"] {
            if let range = token.range(of: marker, options: [.caseInsensitive]) {
                token.removeSubrange(range.lowerBound..<token.endIndex)
            }
        }
        return token.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Produces alternate names for the same argument.
    /// - Parameter value: A string containing alternate argument names separated by " or " or "/".
    /// - Returns: An array of alternate names for the argument.
    private static func argumentAliases(from value: String) -> [String] {
        value
            .components(separatedBy: " or ")
            .flatMap { $0.components(separatedBy: "/") }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
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
