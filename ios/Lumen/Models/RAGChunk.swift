import Foundation
import SwiftData

@Model
final class RAGChunk {
    var id: UUID = UUID()
    var content: String = ""
    var sourceType: String = "file"
    var sourceName: String = ""
    var sourceRef: String?
    var chunkIndex: Int = 0
    var createdAt: Date = Date()
    var embedding: [Double] = []

    init(content: String, sourceType: RAGSourceType, sourceName: String, sourceRef: String? = nil, chunkIndex: Int = 0, embedding: [Double] = []) {
        self.content = content
        self.sourceType = sourceType.rawValue
        self.sourceName = sourceName
        self.sourceRef = sourceRef
        self.chunkIndex = chunkIndex
        self.embedding = embedding
    }

    var kind: RAGSourceType { RAGSourceType(rawValue: sourceType) ?? .file }
}

enum RAGSourceType: String, Codable, CaseIterable, Sendable {
    case file, pdf, photo, note

    var label: String {
        switch self {
        case .file: "Files"
        case .pdf: "PDFs"
        case .photo: "Photos"
        case .note: "Notes"
        }
    }

    var icon: String {
        switch self {
        case .file: "doc.text.fill"
        case .pdf: "doc.richtext.fill"
        case .photo: "photo.stack.fill"
        case .note: "note.text"
        }
    }
}
