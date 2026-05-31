import Foundation

struct RAGDocument: Codable, Sendable { let source: RAGSource; let text: String; let metadata: [String:String] }
