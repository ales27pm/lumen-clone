import Foundation

actor LLMEngineRouter {
    private var registry: [LLMBackendKind: any LLMEngine] = [:]

    func register(_ engine: any LLMEngine, for backend: LLMBackendKind) {
        registry[backend] = engine
    }

    func engine(for backend: LLMBackendKind) throws -> any LLMEngine {
        guard let engine = registry[backend] else {
            throw LLMEngineError.backendUnavailable(backend.rawValue)
        }
        return engine
    }

    func engine(for model: LocalLLMModel) throws -> any LLMEngine {
        try engine(for: model.backend)
    }

    func availableBackends() -> [LLMBackendKind] {
        LLMBackendKind.allCases.filter { registry[$0] != nil }
    }

    func hasBackend(_ backend: LLMBackendKind) -> Bool {
        registry[backend] != nil
    }
}
