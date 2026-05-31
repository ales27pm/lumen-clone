import Foundation

struct LLMContextItem: Sendable, Codable, Equatable, Identifiable {
    let id: UUID
    let title: String?
    let content: String
    let source: String?
    let score: Double?
    let tokenEstimate: Int?
    let metadata: [String: String]

    init(
        id: UUID = UUID(),
        title: String? = nil,
        content: String,
        source: String? = nil,
        score: Double? = nil,
        tokenEstimate: Int? = nil,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.title = title
        self.content = content
        self.source = source
        self.score = score
        self.tokenEstimate = tokenEstimate.map { max(0, $0) }
        self.metadata = metadata
    }
}
