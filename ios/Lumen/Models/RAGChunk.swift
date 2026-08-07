import Foundation
import SwiftData

@Model
final class RAGChunk {
    static let replacementStagingSourceTypePrefix = "__lumen_rag_replacement_staging__:"

    var id: UUID = UUID()
    var content: String = ""
    var sourceType: String = "file"
    var sourceName: String = ""
    var sourceRef: String?
    var chunkIndex: Int = 0
    var createdAt: Date = Date()
    var embedding: [Double] = []
    var embeddingFormatVersion: Int = 0
    var embeddingModelIdentifier: String = ""
    var embeddingDimension: Int = 0

    init(
        content: String,
        sourceType: RAGSourceType,
        sourceName: String,
        sourceRef: String? = nil,
        chunkIndex: Int = 0,
        embedding: [Double] = [],
        embeddingFormatVersion: Int = SemanticEmbeddingText.formatVersion,
        embeddingModelIdentifier: String = RAGEmbeddingMetadata.unidentifiedModelIdentifier,
        embeddingDimension: Int? = nil
    ) {
        self.content = content
        self.sourceType = sourceType.rawValue
        self.sourceName = sourceName
        self.sourceRef = sourceRef
        self.chunkIndex = chunkIndex
        self.embedding = embedding
        self.embeddingFormatVersion = embeddingFormatVersion
        self.embeddingModelIdentifier = embeddingModelIdentifier
        self.embeddingDimension = embeddingDimension ?? embedding.count
    }

    var isReplacementStaging: Bool {
        sourceType.hasPrefix(Self.replacementStagingSourceTypePrefix)
    }

    var kind: RAGSourceType {
        if isReplacementStaging,
           let rawValue = sourceType
               .dropFirst(Self.replacementStagingSourceTypePrefix.count)
               .split(separator: ":", omittingEmptySubsequences: false)
               .first,
           let stagedKind = RAGSourceType(rawValue: String(rawValue)) {
            return stagedKind
        }
        return RAGSourceType(rawValue: sourceType) ?? .file
    }

    static func replacementStagingSourceType(id: UUID, kind: RAGSourceType) -> String {
        "\(replacementStagingSourceTypePrefix)\(kind.rawValue):\(id.uuidString)"
    }
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
