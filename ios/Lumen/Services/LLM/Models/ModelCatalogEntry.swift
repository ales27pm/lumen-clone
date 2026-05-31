import Foundation

struct ModelCatalogEntry: Sendable, Codable, Equatable, Identifiable {
    let id: String
    let displayName: String
    let backend: LLMBackendKind
    let recommendedUse: ModelRecommendedUse
    let source: ModelCatalogSource
    let parameterCountBillion: Double?
    let quantization: String?
    let contextLength: Int
    let expectedFileName: String?
    let expectedSHA256: String?
    let expectedSizeBytes: Int64?
    let minimumRecommendedTier: DevicePerformanceTier
    let tags: [String]
    let notes: String?

    init(
        id: String,
        displayName: String,
        backend: LLMBackendKind,
        recommendedUse: ModelRecommendedUse,
        source: ModelCatalogSource,
        parameterCountBillion: Double? = nil,
        quantization: String? = nil,
        contextLength: Int,
        expectedFileName: String? = nil,
        expectedSHA256: String? = nil,
        expectedSizeBytes: Int64? = nil,
        minimumRecommendedTier: DevicePerformanceTier,
        tags: [String] = [],
        notes: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.backend = backend
        self.recommendedUse = recommendedUse
        self.source = source
        self.parameterCountBillion = parameterCountBillion
        self.quantization = quantization
        self.contextLength = max(1, contextLength)
        self.expectedFileName = expectedFileName
        self.expectedSHA256 = expectedSHA256
        self.expectedSizeBytes = expectedSizeBytes
        self.minimumRecommendedTier = minimumRecommendedTier
        self.tags = tags
        self.notes = notes
    }

    func asLocalModel(localURL: URL?) -> LocalLLMModel {
        LocalLLMModel(
            id: id,
            displayName: displayName,
            backend: backend,
            localURL: localURL,
            expectedSHA256: expectedSHA256,
            parameterCountBillion: parameterCountBillion,
            quantization: quantization,
            contextLength: contextLength,
            fileSizeBytes: expectedSizeBytes
        )
    }
}

enum ModelRecommendedUse: String, Sendable, Codable, Equatable, CaseIterable {
    case tinyIntent
    case fastChat
    case standardChat
    case deepThink
    case embedding
    case reranking
    case vision
    case testing
}

enum ModelCatalogSource: Sendable, Codable, Equatable {
    case bundled
    case localImport
    case remote(url: URL)
    case huggingFace(repoID: String, fileName: String)
    case unknown

    private enum SourceType: String, Codable {
        case bundled
        case localImport
        case remote
        case huggingFace
        case unknown
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case url
        case repoID
        case fileName
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawType = try container.decode(String.self, forKey: .type)
        let type = SourceType(rawValue: rawType) ?? .unknown

        switch type {
        case .bundled:
            self = .bundled
        case .localImport:
            self = .localImport
        case .remote:
            self = .remote(url: try container.decode(URL.self, forKey: .url))
        case .huggingFace:
            self = .huggingFace(
                repoID: try container.decode(String.self, forKey: .repoID),
                fileName: try container.decode(String.self, forKey: .fileName)
            )
        case .unknown:
            self = .unknown
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .bundled:
            try container.encode(SourceType.bundled, forKey: .type)
        case .localImport:
            try container.encode(SourceType.localImport, forKey: .type)
        case .remote(let url):
            try container.encode(SourceType.remote, forKey: .type)
            try container.encode(url, forKey: .url)
        case .huggingFace(let repoID, let fileName):
            try container.encode(SourceType.huggingFace, forKey: .type)
            try container.encode(repoID, forKey: .repoID)
            try container.encode(fileName, forKey: .fileName)
        case .unknown:
            try container.encode(SourceType.unknown, forKey: .type)
        }
    }
}
