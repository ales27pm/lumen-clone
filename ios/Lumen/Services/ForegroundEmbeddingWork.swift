import Foundation
import SwiftData

/// Explicit user actions prepare the selected encoder before asking stores for
/// vectors. Stores and passive/background discovery keep their no-load contract.
@MainActor
enum ForegroundEmbeddingWork {
    enum Failure: LocalizedError {
        case notForeground
        case embeddingUnavailable

        var errorDescription: String? {
            switch self {
            case .notForeground:
                return "This operation requires the app to be active."
            case .embeddingUnavailable:
                return "The selected local embedding model could not be prepared. Check Models and available device resources."
            }
        }
    }

    static func run<Value>(
        isForeground: Bool,
        appState: AppState,
        context: ModelContext,
        operation: () async throws -> Value
    ) async throws -> Value {
        try await run(isForeground: isForeground, prepare: {
            let stored = try context.fetch(FetchDescriptor<StoredModel>())
            return await ModelLoader.ensureEmbedLoaded(appState: appState, stored: stored, intent: .userChat)
        }, operation: operation)
    }

    static func run<Value>(
        isForeground: Bool,
        prepare: () async throws -> Bool,
        operation: () async throws -> Value
    ) async throws -> Value {
        try Task.checkCancellation()
        guard isForeground else { throw Failure.notForeground }
        guard try await prepare() else { throw Failure.embeddingUnavailable }
        try Task.checkCancellation()
        let value = try await operation()
        try Task.checkCancellation()
        return value
    }
}
