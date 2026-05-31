import Foundation

struct LocalLLMModel: Sendable, Codable, Equatable, Identifiable {
    let id: String
    let displayName: String
    let backend: LLMBackendKind
    let localURL: URL?
    let expectedSHA256: String?
    let parameterCountBillion: Double?
    let quantization: String?
    let contextLength: Int
    let fileSizeBytes: Int64?
    let createdAt: Date?
    let lastUsedAt: Date?

    init(
        id: String,
        displayName: String,
        backend: LLMBackendKind,
        localURL: URL? = nil,
        expectedSHA256: String? = nil,
        parameterCountBillion: Double? = nil,
        quantization: String? = nil,
        contextLength: Int,
        fileSizeBytes: Int64? = nil,
        createdAt: Date? = nil,
        lastUsedAt: Date? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.backend = backend
        self.localURL = localURL
        self.expectedSHA256 = expectedSHA256
        self.parameterCountBillion = parameterCountBillion
        self.quantization = quantization
        self.contextLength = max(1, contextLength)
        self.fileSizeBytes = fileSizeBytes
        self.createdAt = createdAt
        self.lastUsedAt = lastUsedAt
    }
}
