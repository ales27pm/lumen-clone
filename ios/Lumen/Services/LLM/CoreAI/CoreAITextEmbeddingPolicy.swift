import Foundation

nonisolated enum CoreAITextEmbeddingError: Error, LocalizedError, Equatable, Sendable {
    case unsupported, busy, notPrepared, notSelected
    case invalidRequest, inputTooLong, invalidDimensions, resourceDenied
    case modelNotConfigured, integrityFailure, invalidModel, loadFailed, inferenceFailed, invalidOutput

    var errorDescription: String? {
        switch self {
        case .unsupported: return "CoreAI text embeddings require a supported physical device running iOS 27.2 or later."
        case .busy: return "The text embedding model is busy. Retry after the current operation finishes."
        case .notPrepared: return "Prepare the selected text embedding model before using it."
        case .notSelected: return "CoreAI text embeddings are not selected."
        case .invalidRequest: return "The embedding request is invalid or lacks an explicit supported purpose."
        case .inputTooLong: return "The text exceeds the embedding model's 2048-token input limit. Split it into smaller passages."
        case .invalidDimensions: return "The embedding dimension must be 128, 256, 512, or 768."
        case .resourceDenied: return "Device resource policy currently prevents text embedding work."
        case .modelNotConfigured: return "Import the pinned CoreAI text embedding package first."
        case .integrityFailure: return "The imported text embedding package failed its integrity check."
        case .invalidModel: return "The imported package does not match the supported text embedding model."
        case .loadFailed: return "The local text embedding model could not be prepared."
        case .inferenceFailed: return "The local text embedding operation failed."
        case .invalidOutput: return "The text embedding model returned an invalid vector."
        }
    }
}

nonisolated struct CoreAITextEmbeddingConfiguration: Decodable, Equatable, Sendable {
    let base_model: String
    let format: String
    let asset: String
    let functions: [String]
    let buckets: [Int]
    let max_tokens: Int
    let precision_mode: String
    let inputs: [String: String]
    let output: [String: String]
    let hidden_size: Int
    let embedding_dim: Int
    let matryoshka_dims: [Int]
    let bos_token_id: Int32
    let eos_token_id: Int32
    let pad_token_id: Int32
    let prompts: [String: String]
}

/// Contract of the immutable EverPaper export, rather than a generic model loader.
nonisolated enum CoreAITextEmbeddingPolicy {
    static let revision = "305fc2a96790d9343f38038aeca211dfca56d9a5"
    static let graphDigest = "0b51638f2d17794825de21def6ab485d968fdf412853ea209617623a51f1bbde"
    static let tokenizerDigest = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"
    static let configurationDigest = "0204c8d4532f6930709c3c8eb7f7b25b54e4be42aa7b8c6c397fc17e872fb7bf"
    // The export revision binds graph/config/tokenizer hashes; format4 binds the
    // explicit task prefix, BOS/EOS, rejection policy, MRL slicing and L2 recipe.
    static let baseIdentifier = "coreai:EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5:text-mm:format4"
    static let formatVersion = 4
    static let defaultDimensions = 256
    static let dimensions = [128, 256, 512, 768]
    static let buckets = [32, 64, 128, 288, 512, 1024, 2048]
    static let functions = buckets.map { "mm_\($0)" }
    static let maximumTokens = 2048
    static let maximumInputBytes = 131_072
    static let maximumTitleBytes = 2_048
    static let queryPrefix = "task: search result | query: "
    static let documentPrefix = "title: none | text: "

    static func modelIdentifier(dimensions: Int = defaultDimensions) -> String {
        "\(baseIdentifier):mrl-\(dimensions)"
    }

    static func requestedDimensions(_ request: EmbeddingRequest) throws -> Int {
        let value = request.dimensions ?? defaultDimensions
        guard dimensions.contains(value) else { throw CoreAITextEmbeddingError.invalidDimensions }
        return value
    }

    /// The caller supplies raw text, never the legacy model's already-prefixed text.
    static func formattedText(_ request: EmbeddingRequest) throws -> String {
        guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              request.text.utf8.count <= maximumInputBytes else {
            throw CoreAITextEmbeddingError.invalidRequest
        }
        _ = try requestedDimensions(request)
        let taskPrefix: String
        switch request.purpose {
        case .legacyFormatted: throw CoreAITextEmbeddingError.invalidRequest
        case .retrievalQuery: taskPrefix = queryPrefix
        case .retrievalDocument(let title):
            if let title, title.utf8.count > maximumTitleBytes { throw CoreAITextEmbeddingError.invalidRequest }
            let label = title?.trimmingCharacters(in: .whitespacesAndNewlines)
            taskPrefix = "title: \(label.flatMap { $0.isEmpty ? nil : $0 } ?? "none") | text: "
        case .semanticSimilarity: taskPrefix = "task: sentence similarity | query: "
        case .codeQuery: taskPrefix = "task: code retrieval | query: "
        }
        return taskPrefix + request.text
    }

    static func tokenIDs(_ body: [Int32]) throws -> [Int32] {
        guard body.count <= maximumTokens - 2 else { throw CoreAITextEmbeddingError.inputTooLong }
        guard !body.isEmpty, body.allSatisfy({ $0 >= 0 && $0 < 262_144 }) else {
            throw CoreAITextEmbeddingError.invalidRequest
        }
        return [2] + body + [1]
    }

    static func bucket(forTokenCount count: Int) throws -> Int {
        guard count >= 2 else { throw CoreAITextEmbeddingError.invalidRequest }
        guard let bucket = buckets.first(where: { $0 >= count }) else { throw CoreAITextEmbeddingError.inputTooLong }
        return bucket
    }

    /// FP16 graph outputs are widened before MRL slicing and accumulation. Reject
    /// malformed/zero/nonfinite outputs instead of manufacturing retrieval success.
    static func normalizedVector(_ values: [Double], dimensions dimension: Int) throws -> [Double] {
        guard dimensions.contains(dimension) else { throw CoreAITextEmbeddingError.invalidDimensions }
        guard values.count == 768, values.allSatisfy(\.isFinite) else { throw CoreAITextEmbeddingError.invalidOutput }
        let selected = Array(values.prefix(dimension))
        let squaredNorm = selected.reduce(0.0) { $0 + $1 * $1 }
        guard squaredNorm.isFinite, squaredNorm > 1e-24 else { throw CoreAITextEmbeddingError.invalidOutput }
        let norm = sqrt(squaredNorm)
        let result = selected.map { $0 / norm }
        guard result.allSatisfy(\.isFinite) else { throw CoreAITextEmbeddingError.invalidOutput }
        return result
    }

    static func validateConfiguration(_ data: Data) throws -> CoreAITextEmbeddingConfiguration {
        let config: CoreAITextEmbeddingConfiguration
        do { config = try JSONDecoder().decode(CoreAITextEmbeddingConfiguration.self, from: data) }
        catch { throw CoreAITextEmbeddingError.invalidModel }
        guard config.base_model == "google/embeddinggemma-2", config.format == "coreai",
              config.asset == "embeddinggemma-2-text-mm.aimodel", config.precision_mode == "ane",
              config.functions == functions, config.buckets == buckets, config.max_tokens == maximumTokens,
              config.hidden_size == 512, config.embedding_dim == 768,
              config.matryoshka_dims == [768, 512, 256, 128],
              config.bos_token_id == 2, config.eos_token_id == 1, config.pad_token_id == 0,
              Set(config.inputs.keys) == ["input_ids", "attention_mask", "mm_embeds", "mm_mask"],
              Set(config.output.keys) == ["embedding"],
              config.prompts["document"] == documentPrefix,
              config.prompts["search_query"] == queryPrefix,
              config.prompts["sentence_similarity"] == "task: sentence similarity | query: ",
              config.prompts["code_retrieval"] == "task: code retrieval | query: " else {
            throw CoreAITextEmbeddingError.invalidModel
        }
        return config
    }
}
