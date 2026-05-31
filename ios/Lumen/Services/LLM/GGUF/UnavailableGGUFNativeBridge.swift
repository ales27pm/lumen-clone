import Foundation

actor UnavailableGGUFNativeBridge: GGUFNativeBridge {
    private var cancellationRequested = false

    func status() async -> GGUFBridgeStatus {
        .unavailable
    }

    func load(config: GGUFBridgeLoadConfig) async throws -> GGUFBridgeModelInfo {
        cancellationRequested = false
        throw GGUFBridgeError.backendNotCompiled
    }

    func unload() async {
    }

    nonisolated func generate(config: GGUFBridgeGenerateConfig) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: GGUFBridgeError.backendNotCompiled)
        }
    }

    func cancel() async {
        cancellationRequested = true
    }
}
