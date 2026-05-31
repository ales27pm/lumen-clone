import Foundation
import SwiftData

enum EmbeddingBatcherError: Error { case unavailable(String) }

enum EmbeddingBatcher {
    static func embed(texts: [String], turnContext: AssistantTurnContext) async throws -> [[Double]] {
        let batch = (!turnContext.isForeground || turnContext.lowPowerMode) ? min(texts.count, 4) : min(texts.count, 12)
        var out: [[Double]] = []
        for t in texts.prefix(batch) {
            try Task.checkCancellation()
            do { out.append(try await AppLlamaService.shared.embed(t)) }
            catch is CancellationError { throw CancellationError() }
            catch { throw EmbeddingBatcherError.unavailable(String(describing: type(of: error))) }
        }
        return out
    }
}
