import Foundation

struct RAGRetrievalResult: Codable, Sendable, Hashable {
    let chunkID: UUID
    let source: RAGSource
    let excerpt: String
    let score: Double
    let retrievalMode: String
    let offsetStart: Int?
    let offsetEnd: Int?
}
