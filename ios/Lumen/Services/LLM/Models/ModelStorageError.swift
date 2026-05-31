import Foundation

enum ModelStorageError: LocalizedError, Sendable {
    case applicationSupportUnavailable
    case failedToCreateDirectory(URL, String)
    case failedToSetResourceValues(URL, String)
    case fileNotFound(URL)
    case unreadableFile(URL)
    case invalidModelFileExtension(String)
    case hashMismatch(expected: String, actual: String)
    case metadataEncodingFailed(String)
    case metadataDecodingFailed(String)
    case metadataWriteFailed(String)
    case metadataReadFailed(String)
    case deleteFailed(URL, String)
    case importFailed(String)

    var errorDescription: String? {
        switch self {
        case .applicationSupportUnavailable:
            return "Application Support is unavailable for model storage."
        case .failedToCreateDirectory(let url, let reason):
            return "Failed to create model storage directory at \(url.path): \(reason)"
        case .failedToSetResourceValues(let url, let reason):
            return "Failed to set model storage resource values at \(url.path): \(reason)"
        case .fileNotFound(let url):
            return "Model file was not found at \(url.path)."
        case .unreadableFile(let url):
            return "Model file is not readable at \(url.path)."
        case .invalidModelFileExtension(let fileName):
            return "The model file extension is not supported: \(fileName)"
        case .hashMismatch(let expected, let actual):
            return "Model file hash mismatch. Expected \(expected), got \(actual)."
        case .metadataEncodingFailed(let reason):
            return "Failed to encode model metadata: \(reason)"
        case .metadataDecodingFailed(let reason):
            return "Failed to decode model metadata: \(reason)"
        case .metadataWriteFailed(let reason):
            return "Failed to write model metadata: \(reason)"
        case .metadataReadFailed(let reason):
            return "Failed to read model metadata: \(reason)"
        case .deleteFailed(let url, let reason):
            return "Failed to delete \(url.path): \(reason)"
        case .importFailed(let reason):
            return "Failed to import model file: \(reason)"
        }
    }
}
