import Foundation

nonisolated enum SemanticEmbeddingText {
    static let formatVersion = 3
    private static let maxContentCharacters = 4096
    private static let maxMetadataCharacters = 160

    static func query(_ text: String) -> String {
        "search_query: \(boundedInlineContent(text))"
    }

    static func document(
        content: String,
        sourceName: String? = nil,
        sourceType: String? = nil,
        chunkIndex: Int? = nil
    ) -> String {
        var metadata: [String] = []
        if let sourceType = cleanMetadata(sourceType) {
            metadata.append("Source type: \(sourceType)")
        }
        if let sourceName = cleanMetadata(sourceName) {
            metadata.append("Source name: \(sourceName)")
        }
        if let chunkIndex {
            metadata.append("Chunk index: \(chunkIndex)")
        }
        return prefixed("search_document", metadata: metadata, content: content)
    }

    static func memoryQuery(_ text: String) -> String {
        query(text)
    }

    static func memoryDocument(content: String, kind: MemoryKind, source: String, topic: String?) -> String {
        var metadata = ["Memory kind: \(kind.rawValue)"]
        if let source = cleanMetadata(source) {
            metadata.append("Memory source: \(source)")
        }
        if let topic = cleanMetadata(topic) {
            metadata.append("Topic: \(topic)")
        }
        return prefixed("search_document", metadata: metadata, content: content)
    }

    private static func prefixed(_ prefix: String, metadata: [String] = [], content: String) -> String {
        let body = boundedDocumentContent(content)
        guard !metadata.isEmpty else {
            return "\(prefix): \(body)"
        }
        return """
        \(prefix):
        \(metadata.joined(separator: "\n"))
        Content:
        \(body)
        """
    }

    private static func boundedInlineContent(_ text: String) -> String {
        let collapsed = collapseWhitespace(text)
        guard collapsed.count > maxContentCharacters else { return collapsed }
        return String(collapsed.prefix(maxContentCharacters))
    }

    private static func boundedDocumentContent(_ text: String) -> String {
        let normalized = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard normalized.count > maxContentCharacters else { return normalized }
        return String(normalized.prefix(maxContentCharacters))
    }

    private static func cleanMetadata(_ text: String?) -> String? {
        guard let text else { return nil }
        let collapsed = collapseWhitespace(text)
        guard !collapsed.isEmpty else { return nil }
        guard collapsed.count > maxMetadataCharacters else { return collapsed }
        return String(collapsed.prefix(maxMetadataCharacters))
    }

    private static func collapseWhitespace(_ text: String) -> String {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}

enum RAGEmbeddingMetadata {
    static let unidentifiedModelIdentifier = "assistant-kernel-embedding:unidentified"

    nonisolated static func modelIdentifier(forFileURL fileURL: URL) throws -> String {
        "llama:sha256:\(try SHA256FileHasher.sha256Hex(for: fileURL))"
    }
}
