import Foundation

nonisolated enum GGUFEngineErrorMapper {
    static func map(_ error: Error) -> Error {
        if let engineError = error as? LLMEngineError {
            return engineError
        }

        if let bridgeError = error as? GGUFBridgeError {
            switch bridgeError {
            case .backendNotCompiled:
                return LLMEngineError.backendUnavailable("GGUF native backend is not compiled.")
            case .modelFileMissing:
                return LLMEngineError.modelNotFound
            case .modelNotLoaded:
                return LLMEngineError.modelNotLoaded
            case .generationAlreadyRunning:
                return LLMEngineError.generationAlreadyRunning
            case .generationCancelled:
                return LLMEngineError.generationCancelled
            case .invalidModelPath, .invalidPrompt, .nativeFailure:
                return LLMEngineError.internalFailure(bridgeError.localizedDescription)
            }
        }

        if let storageError = error as? ModelStorageError {
            switch storageError {
            case .fileNotFound, .unreadableFile:
                return LLMEngineError.modelNotFound
            case .invalidModelFileExtension:
                return LLMEngineError.invalidRequest(storageError.localizedDescription)
            default:
                return LLMEngineError.internalFailure(storageError.localizedDescription)
            }
        }

        return LLMEngineError.internalFailure(error.localizedDescription)
    }
}
