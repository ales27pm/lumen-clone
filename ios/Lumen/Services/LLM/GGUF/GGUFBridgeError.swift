import Foundation

nonisolated enum GGUFBridgeError: LocalizedError, Sendable, Equatable {
    case backendNotCompiled
    case invalidModelPath(String)
    case modelFileMissing(String)
    case modelNotLoaded
    case generationAlreadyRunning
    case generationCancelled
    case invalidPrompt
    case nativeFailure(String)

    var errorDescription: String? {
        switch self {
        case .backendNotCompiled:
            return "The GGUF native backend is not compiled into this build."
        case .invalidModelPath(let path):
            return "The GGUF model path is invalid: \(path)"
        case .modelFileMissing(let path):
            return "The GGUF model file is missing at \(path)."
        case .modelNotLoaded:
            return "No GGUF model is currently loaded."
        case .generationAlreadyRunning:
            return "A GGUF generation is already running."
        case .generationCancelled:
            return "The GGUF generation was cancelled."
        case .invalidPrompt:
            return "The GGUF prompt is empty or invalid."
        case .nativeFailure(let message):
            return "The GGUF native backend failed: \(message)"
        }
    }
}
