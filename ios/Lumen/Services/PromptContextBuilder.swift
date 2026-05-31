import Foundation

nonisolated enum PromptContextBuilder {
    static func renderMemoryBlock(_ memories: [MemoryContextItem], limit: Int = 10) -> String {
        guard !memories.isEmpty else { return "" }
        let lines = memories.prefix(limit).map { item in
            "• [\(item.scope.rawValue) | \(item.authority.rawValue)] \(item.content)"
        }
        guard !lines.isEmpty else { return "" }
        return "\n\nRelevant memory from previous conversations:\n" + lines.joined(separator: "\n")
    }
}
