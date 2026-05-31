import Foundation

nonisolated protocol GGUFNativeBridge: Sendable {
    func status() async -> GGUFBridgeStatus
    func load(config: GGUFBridgeLoadConfig) async throws -> GGUFBridgeModelInfo
    func unload() async
    func generate(config: GGUFBridgeGenerateConfig) -> AsyncThrowingStream<String, Error>
    func cancel() async
}
