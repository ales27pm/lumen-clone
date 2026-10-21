import Foundation

/// Messages are rendered by the imported tokenizer's own Gemma chat template.
nonisolated struct CoreAIGemmaMessage: Sendable, Equatable {
    let role: String
    let content: String
}

nonisolated struct CoreAIGemmaPromptPreflight: Sendable, Equatable {
    let contextSize: Int
    let promptTokens: Int
    let promptChars: Int
}

/// Constant, content-free failures: underlying SDK errors can contain local paths.
nonisolated enum CoreAIGemmaError: Error, LocalizedError, Sendable, Equatable {
    case unsupported
    case busy
    case invalidModel
    case invalidMessages
    case invalidTemperature
    case invalidTokenBudget
    case contextFull
    case resourceDenied
    case modelNotConfigured
    case integrityFailure
    case loadFailed
    case inferenceFailed
    case emptyOutput

    var errorDescription: String? {
        switch self {
        case .unsupported:
            return "Core AI Gemma requires a supported physical device running iOS 27.2 or later."
        case .busy:
            return "A Core AI operation is already in progress. Wait for its cancellation to complete."
        case .invalidModel:
            return "The local Gemma bundle is incomplete or incompatible with the pinned Core AI runtime."
        case .invalidMessages:
            return "Gemma requires a bounded, alternating user and assistant conversation ending in a user message."
        case .invalidTemperature:
            return "The generation temperature must be finite and between 0 and 2."
        case .invalidTokenBudget:
            return "The output token budget must be positive."
        case .contextFull:
            return "The Gemma prompt leaves no space for a response in the imported model's context."
        case .resourceDenied:
            return "Core AI model work is unavailable under the current foreground and resource policy."
        case .modelNotConfigured:
            return "Import a local Gemma Core AI model in Settings before starting a conversation."
        case .integrityFailure:
            return "The imported Gemma model changed or is incomplete. Import the original package again."
        case .loadFailed:
            return "The local Core AI Gemma model could not be loaded."
        case .inferenceFailed:
            return "Core AI Gemma generation failed. The model must be loaded again."
        case .emptyOutput:
            return "Core AI Gemma returned no response text."
        }
    }
}

/// Adapted from ales27pm/assist at 6a26dcf27042d73833d8ce5b51178bdeb5f17562.
/// These are structural checks; the owned model store separately verifies byte identity.
nonisolated enum CoreAIGemmaPolicy {
    static let sourceModel = "google/gemma-4-E2B-it"
    static let supportedContexts: Set<Int> = [8_192, 32_768]
    static let maximumMessages = 256
    static let maximumPromptBytes = 2 * 1_024 * 1_024

    /// Verified against the pinned exported tokenizer's canonical chat_template.jinja
    /// (SHA-256 0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5).
    /// Its thought channel ends with <channel|>; final content follows without a
    /// separate final-channel marker. Native tool-call syntax is never executable JSON.
    private static let thoughtOpen = "<|channel>thought\n"
    private static let thoughtClose = "<channel|>"
    private static let modelTurnOpen = "<|turn>model\n"
    private static let nativeControls = [
        "<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
        "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
        "<|tool_response>", "<tool_response|>", "<|\"|>", "<|think|>",
        "<|channel>", "<channel|>", "<|turn>", "<turn|>",
        "<|image>", "<|audio>", "<|image|>", "<|audio|>",
        "<image|>", "<audio|>", "<|video|>"
    ]

    /// An incomplete thought/control prefix exposes no text. This must run before
    /// any streaming callback, rather than trying to erase already displayed reasoning.
    static func visibleSnapshot(_ raw: String) -> String? {
        try? normalizeOutput(raw, final: false)
    }

    static func visibleOutput(_ raw: String) throws -> String {
        guard let text = try normalizeOutput(raw, final: true),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw CoreAIGemmaError.emptyOutput
        }
        return text
    }

    private static func normalizeOutput(_ raw: String, final: Bool) throws -> String? {
        var body = raw[...]
        func withoutLeadingSpace(_ value: Substring) -> Substring { value.drop(while: \.isWhitespace) }
        var prefix = withoutLeadingSpace(body)
        if prefix.hasPrefix(modelTurnOpen) {
            body = prefix.dropFirst(modelTurnOpen.count)
            prefix = withoutLeadingSpace(body)
        } else if !prefix.isEmpty, modelTurnOpen.hasPrefix(prefix) {
            if final { throw CoreAIGemmaError.emptyOutput }
            return nil
        }
        if prefix.hasPrefix(thoughtOpen) {
            let reasoning = prefix.dropFirst(thoughtOpen.count)
            guard let close = reasoning.range(of: thoughtClose) else {
                if final { throw CoreAIGemmaError.emptyOutput }
                return nil
            }
            body = reasoning[close.upperBound...]
        } else if !prefix.isEmpty, thoughtOpen.hasPrefix(prefix) {
            if final { throw CoreAIGemmaError.emptyOutput }
            return nil
        }
        var text = String(body)

        // Preserve a complete JSON payload byte-for-byte, including control-token
        // spellings inside quoted data. Prefix control framing was already removed.
        if isCompleteJSON(text) { return text }

        // EOS tokens are normally consumed as token IDs before decoding. Handle a
        // decoded terminator too, so an echoed subsequent turn cannot become content.
        if let end = firstTurnTerminator(in: text) { text = String(text[..<end]) }
        if isCompleteJSON(text) { return text }
        guard !nativeControls.contains(where: { text.contains($0) }) else {
            throw CoreAIGemmaError.emptyOutput
        }
        if let opening = text.lastIndex(of: "<") {
            let tail = text[opening...]
            if nativeControls.contains(where: { $0.hasPrefix(tail) }) {
                if final { throw CoreAIGemmaError.emptyOutput }
                text = String(text[..<opening])
            }
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            if final { throw CoreAIGemmaError.emptyOutput }
            return nil
        }
        return text
    }

    private static func isCompleteJSON(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{") || trimmed.hasPrefix("["), let data = trimmed.data(using: .utf8) else { return false }
        return (try? JSONSerialization.jsonObject(with: data)) != nil
    }

    private static func firstTurnTerminator(in text: String) -> String.Index? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let json = trimmed.hasPrefix("{") || trimmed.hasPrefix("[")
        var inString = false
        var escaped = false
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if json, inString {
                if escaped { escaped = false }
                else if character == "\\" { escaped = true }
                else if character == "\"" { inString = false }
            } else if json, character == "\"" {
                inString = true
            } else if text[index...].hasPrefix("<turn|>") || text[index...].hasPrefix("<eos>") {
                return index
            }
            index = text.index(after: index)
        }
        return nil
    }

    static func validateMessages(_ messages: [CoreAIGemmaMessage]) throws {
        guard !messages.isEmpty, messages.count <= maximumMessages,
              messages.first?.role == "user", messages.last?.role == "user" else {
            throw CoreAIGemmaError.invalidMessages
        }
        var bytes = 0
        for (index, message) in messages.enumerated() {
            guard message.role == (index.isMultiple(of: 2) ? "user" : "assistant"),
                  !message.content.isEmpty else {
                throw CoreAIGemmaError.invalidMessages
            }
            let count = message.content.utf8.count
            guard count <= maximumPromptBytes - bytes else {
                throw CoreAIGemmaError.invalidMessages
            }
            bytes += count
        }
    }

    static func validateTemperature(_ temperature: Double) throws {
        guard temperature.isFinite, (0...2).contains(temperature) else {
            throw CoreAIGemmaError.invalidTemperature
        }
    }

    static func validateModel(tokenizer: String, context: Int, embeddedTokenizer: Bool) throws {
        guard tokenizer == sourceModel, supportedContexts.contains(context), embeddedTokenizer else {
            throw CoreAIGemmaError.invalidModel
        }
    }

    static func validateGraphs(_ names: [String], context: Int) throws {
        let decodeDimensions: [(Int, Int)] = names.compactMap { name in
            let pieces = name.split(separator: "_")
            guard pieces.count == 3, pieces[0] == "extend",
                  let capacity = Int(pieces[1]), let query = Int(pieces[2]),
                  capacity > 0, query > 0 else { return nil }
            return (capacity, query)
        }
        guard names.contains("load_embeddings"),
              decodeDimensions.map({ $0.0 }).max() == context,
              decodeDimensions.contains(where: { $0.0 == context && $0.1 == 8 }),
              names.contains("prompt_opt_\(context)_64") else {
            throw CoreAIGemmaError.invalidModel
        }
    }

    static func outputBudget(promptTokens: Int, capacity: Int, requested: Int) throws -> Int {
        guard requested > 0 else { throw CoreAIGemmaError.invalidTokenBudget }
        guard promptTokens > 0, promptTokens < capacity else { throw CoreAIGemmaError.contextFull }
        return min(requested, capacity - promptTokens)
    }

    static func stopTokenIDs(metadata: Data, vocabSize: Int, tokenizerEOS: Int?) throws -> Set<Int32> {
        struct Envelope: Decodable {
            struct Language: Decodable { let eos_token_ids: [Int] }
            let language: Language
        }
        let declared: [Int]
        do { declared = try JSONDecoder().decode(Envelope.self, from: metadata).language.eos_token_ids }
        catch { throw CoreAIGemmaError.invalidModel }
        guard !declared.isEmpty else { throw CoreAIGemmaError.invalidModel }
        // Additional special tokens include nonterminal role/thinking markers.
        let ids = Set(declared + (tokenizerEOS.map { [$0] } ?? []))
        guard ids.allSatisfy({ $0 >= 0 && $0 < vocabSize && $0 <= Int(Int32.max) }) else {
            throw CoreAIGemmaError.invalidModel
        }
        return Set(ids.map(Int32.init))
    }

    /// Reject metadata traversal and symbolic links before SDK model/tokenizer IO.
    static func confinedURL(_ relativePath: String, root: URL) throws -> URL {
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard root.isFileURL, root.host == nil || root.host == "",
              !relativePath.isEmpty, !(relativePath as NSString).isAbsolutePath,
              !relativePath.contains("\\"), !relativePath.utf8.contains(0),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw CoreAIGemmaError.invalidModel
        }
        let normalizedRoot = root.resolvingSymlinksInPath().standardizedFileURL
        var candidate = normalizedRoot
        do {
            for component in components {
                candidate.appendPathComponent(String(component))
                let values = try candidate.resourceValues(forKeys: [.isSymbolicLinkKey])
                guard values.isSymbolicLink == false else { throw CoreAIGemmaError.invalidModel }
            }
        } catch { throw CoreAIGemmaError.invalidModel }
        let resolved = candidate.resolvingSymlinksInPath().standardizedFileURL
        guard resolved.path.hasPrefix(normalizedRoot.path + "/") else {
            throw CoreAIGemmaError.invalidModel
        }
        return resolved
    }

    static func requireFile(_ relativePath: String, root: URL) throws -> URL {
        let url = try confinedURL(relativePath, root: root)
        do {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey, .isUbiquitousItemKey])
            guard values.isRegularFile == true, (values.fileSize ?? 0) > 0,
                  values.isUbiquitousItem != true, FileManager.default.isReadableFile(atPath: url.path) else {
                throw CoreAIGemmaError.invalidModel
            }
            return url
        } catch { throw CoreAIGemmaError.invalidModel }
    }
}

/// Decode complete prefixes to preserve byte-fallback UTF-8 and tokenizer normalization.
/// Geometrically spaced snapshots bound cumulative decode work for long responses.
nonisolated struct CoreAIGemmaOutputBuffer {
    private var ids: [Int] = []
    private var text = ""
    private var nextSnapshot = 1
    var count: Int { ids.count }

    mutating func append(_ token: Int) { ids.append(token) }

    mutating func progressSnapshot(decode: ([Int]) -> String) -> String {
        guard count >= nextSnapshot else { return text }
        return snapshot(decode: decode)
    }

    mutating func snapshot(decode: ([Int]) -> String) -> String {
        text = decode(ids)
        nextSnapshot = count + max(1, (count + 63) / 64)
        return text
    }
}
