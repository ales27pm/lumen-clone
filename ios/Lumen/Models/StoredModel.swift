import Foundation
import SwiftData
import OSLog

@Model
final class StoredModel {
    var id: UUID = UUID()
    var name: String = ""
    var repoId: String = ""
    var fileName: String = ""
    var sizeBytes: Int64 = 0
    var quantization: String = "Q4_K_M"
    var parameters: String = ""
    var role: String = "chat"
    var downloadedAt: Date = Date()
    var localPath: String = ""

    init(name: String, repoId: String, fileName: String, sizeBytes: Int64, quantization: String, parameters: String, role: ModelRole, localPath: String) {
        self.name = name
        self.repoId = repoId
        self.fileName = fileName
        self.sizeBytes = sizeBytes
        self.quantization = quantization
        self.parameters = parameters
        self.role = role.rawValue
        self.localPath = localPath
    }

    var modelRole: ModelRole { ModelRole(rawValue: role) ?? .chat }
}

enum ModelRole: String, Codable, CaseIterable, Sendable {
    case chat, embedding, roleAdapter
}

nonisolated struct CatalogModel: Identifiable, Hashable, Sendable {
    nonisolated enum DownloadURLError: LocalizedError, Equatable {
        case missingRepoPath
        case missingFileName
        case invalidRepoPathCharacters
        case invalidFileNameCharacters
        case invalidURLComponents

        var errorDescription: String? {
            switch self {
            case .missingRepoPath: return "Repository path is missing."
            case .missingFileName: return "File name is missing."
            case .invalidRepoPathCharacters: return "Repository path contains invalid characters."
            case .invalidFileNameCharacters: return "File name contains invalid characters."
            case .invalidURLComponents: return "Could not build a valid download URL."
            }
        }
    }

    private static let logger = Logger(subsystem: "ai.lumen.app", category: "model-catalog")

    let id: String
    let name: String
    let repoId: String
    let fileName: String
    let parameters: String
    let quantization: String
    let sizeBytes: Int64
    let role: ModelRole
    let description: String
    let tags: [String]

    var downloadURLResult: Result<URL, DownloadURLError> {
        let sanitizedRepoPath: String
        do {
            sanitizedRepoPath = try Self.sanitizeRepoPath(repoId)
        } catch let error as DownloadURLError {
            Self.logInvalidMetadata(modelID: id, repoId: repoId, fileName: fileName, reason: error)
            return .failure(error)
        } catch {
            Self.logInvalidMetadata(modelID: id, repoId: repoId, fileName: fileName, reason: .invalidRepoPathCharacters)
            return .failure(.invalidRepoPathCharacters)
        }

        let sanitizedFileName: String
        do {
            sanitizedFileName = try Self.sanitizeFileName(fileName)
        } catch let error as DownloadURLError {
            Self.logInvalidMetadata(modelID: id, repoId: repoId, fileName: fileName, reason: error)
            return .failure(error)
        } catch {
            Self.logInvalidMetadata(modelID: id, repoId: repoId, fileName: fileName, reason: .invalidFileNameCharacters)
            return .failure(.invalidFileNameCharacters)
        }

        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.percentEncodedPath = "/\(sanitizedRepoPath)/resolve/main/\(sanitizedFileName)"
        components.queryItems = [URLQueryItem(name: "download", value: "true")]

        guard let url = components.url else {
            Self.logInvalidMetadata(modelID: id, repoId: repoId, fileName: fileName, reason: .invalidURLComponents)
            return .failure(.invalidURLComponents)
        }
        return .success(url)
    }

    var downloadURL: URL? {
        if case .success(let url) = downloadURLResult {
            return url
        }
        return nil
    }

    private static func sanitizeRepoPath(_ value: String) throws -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw DownloadURLError.missingRepoPath }
        guard !trimmed.contains("//"), !trimmed.hasPrefix("/"), !trimmed.hasSuffix("/") else {
            throw DownloadURLError.invalidRepoPathCharacters
        }
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.~/")
        guard trimmed.unicodeScalars.allSatisfy(allowed.contains) else { throw DownloadURLError.invalidRepoPathCharacters }
        guard let encoded = trimmed.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)?
            .replacingOccurrences(of: "%2F", with: "/")
        else { throw DownloadURLError.invalidRepoPathCharacters }
        return encoded
    }

    private static func sanitizeFileName(_ value: String) throws -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw DownloadURLError.missingFileName }
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.~/")
        guard trimmed.unicodeScalars.allSatisfy(allowed.contains) else { throw DownloadURLError.invalidFileNameCharacters }
        guard let encoded = trimmed.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)?
            .replacingOccurrences(of: "%2F", with: "/")
        else { throw DownloadURLError.invalidFileNameCharacters }
        return encoded
    }

    private static func logInvalidMetadata(modelID: String, repoId: String, fileName: String, reason: DownloadURLError) {
        logger.error("invalid_catalog_metadata model_id=\(modelID, privacy: .public) repo_id=\(repoId, privacy: .public) file_name=\(fileName, privacy: .public) reason=\(String(describing: reason), privacy: .public)")
    }
}

nonisolated enum ModelCatalog {
    static let defaultOnboardingModelID = "fleet-bootstrap-qwen3-fast-shared-q4"
    static let featured: [CatalogModel] = uniqueByArtifact(LumenModelFleetCatalog.selectableBootstrapModels)

    // Keeps compatibility if a build expects the previous default identifiers.
    static let legacyDefaultOnboardingModelID = "fleet-v1-ft-cortex-qwen1.5b-q4"
    static let legacyGeneralDefaultOnboardingModelID = "qwen2.5-1.5b-q4"

    static var defaultOnboardingModel: CatalogModel {
        featured.first { $0.id == defaultOnboardingModelID }
        ?? featured.first { $0.id == legacyDefaultOnboardingModelID }
        ?? featured.first { $0.id == legacyGeneralDefaultOnboardingModelID }
        ?? legacyFeatured.first { $0.id == defaultOnboardingModelID }
        ?? legacyFeatured.first { $0.id == legacyDefaultOnboardingModelID }
        ?? legacyFeatured.first { $0.id == legacyGeneralDefaultOnboardingModelID }
        ?? legacyFeatured.first
        ?? featured[0]
    }

    static let legacyFeatured: [CatalogModel] = [
        CatalogModel(
            id: "qwen2.5-1.5b-q4",
            name: "Qwen 2.5 Instruct",
            repoId: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            fileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_117_000_000,
            role: .chat,
            description: "Fast, capable default. Great on any iPhone.",
            tags: ["recommended", "fast", "tools"]
        ),
        CatalogModel(
            id: "llama-3.2-3b-q4",
            name: "Llama 3.2 Instruct",
            repoId: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            fileName: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            parameters: "3B",
            quantization: "Q4_K_M",
            sizeBytes: 2_020_000_000,
            role: .chat,
            description: "Meta's smart compact model. Balanced quality.",
            tags: ["balanced", "tools"]
        ),
        CatalogModel(
            id: "dolphin-3.0-llama-3.2-3b-q4",
            name: "Dolphin 3.0 Llama 3.2",
            repoId: "itlwas/Dolphin3.0-Llama3.2-3B-Q4_K_M-GGUF",
            fileName: "dolphin3.0-llama3.2-3b-q4_k_m.gguf",
            parameters: "3B",
            quantization: "Q4_K_M",
            sizeBytes: 2_019_382_112,
            role: .chat,
            description: "Dolphin-tuned Llama 3.2 for assistant chat.",
            tags: ["chat", "tools"]
        ),
        CatalogModel(
            id: "gemma-2-2b-q4",
            name: "Gemma 2 Instruct",
            repoId: "bartowski/gemma-2-2b-it-GGUF",
            fileName: "gemma-2-2b-it-Q4_K_M.gguf",
            parameters: "2B",
            quantization: "Q4_K_M",
            sizeBytes: 1_630_000_000,
            role: .chat,
            description: "Google's polite and creative assistant.",
            tags: ["creative"]
        ),
        CatalogModel(
            id: "phi-3.5-mini-q4",
            name: "Phi 3.5 Mini",
            repoId: "bartowski/Phi-3.5-mini-instruct-GGUF",
            fileName: "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            parameters: "3.8B",
            quantization: "Q4_K_M",
            sizeBytes: 2_390_000_000,
            role: .chat,
            description: "Microsoft's reasoning-focused model.",
            tags: ["reasoning"]
        ),
        CatalogModel(
            id: "mistral-7b-q4",
            name: "Mistral 7B Instruct",
            repoId: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            fileName: "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            parameters: "7B",
            quantization: "Q4_K_M",
            sizeBytes: 4_370_000_000,
            role: .chat,
            description: "Large, heavy. Needs 8GB+ device RAM.",
            tags: ["large", "quality"]
        ),
        CatalogModel(
            id: "nomic-embed-q4",
            name: "Nomic Embed v1.5",
            repoId: "nomic-ai/nomic-embed-text-v1.5-GGUF",
            fileName: "nomic-embed-text-v1.5.Q4_K_M.gguf",
            parameters: "137M",
            quantization: "Q4_K_M",
            sizeBytes: 85_000_000,
            role: .embedding,
            description: "Tiny, fast vector embeddings for memory.",
            tags: ["embedding", "tiny"]
        ),
    ]

    private static func uniqueByArtifact(_ models: [CatalogModel]) -> [CatalogModel] {
        var seen: Set<String> = []
        var unique: [CatalogModel] = []
        unique.reserveCapacity(models.count)

        for model in models {
            let artifactKey = "\(model.repoId.lowercased())/\(model.fileName.lowercased())"
            guard seen.insert(artifactKey).inserted else { continue }
            unique.append(model)
        }

        return unique
    }
}
