import Foundation

enum LLMEngineError: LocalizedError, Sendable, Equatable {
    case modelNotFound
    case modelNotLoaded
    case wrongModelLoaded(expected: String, actual: String?)
    case backendUnavailable(String)
    case generationAlreadyRunning
    case generationCancelled
    case invalidRequest(String)
    case contextTooLarge(max: Int, actual: Int)
    case unsupportedFeature(String)
    case memoryBudgetExceeded
    case backgroundExecutionNotAllowed
    case internalFailure(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "The requested local model could not be found."
        case .modelNotLoaded:
            return "No local language model is currently loaded."
        case .wrongModelLoaded(let expected, let actual):
            let actualDescription = actual ?? "none"
            return "The loaded model does not match the request. Expected \(expected), got \(actualDescription)."
        case .backendUnavailable(let backend):
            return "The \(backend) language model backend is not available."
        case .generationAlreadyRunning:
            return "A generation is already running for this engine."
        case .generationCancelled:
            return "The language model generation was cancelled."
        case .invalidRequest(let reason):
            return "The language model request is invalid: \(reason)"
        case .contextTooLarge(let max, let actual):
            return "The request context is too large. Maximum \(max) tokens, got \(actual)."
        case .unsupportedFeature(let feature):
            return "This language model backend does not support \(feature)."
        case .memoryBudgetExceeded:
            return "The request exceeds the configured memory budget."
        case .backgroundExecutionNotAllowed:
            return "This language model backend is not allowed to run in the background."
        case .internalFailure(let reason):
            return "The language model engine failed internally: \(reason)"
        }
    }
}
