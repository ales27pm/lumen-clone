import Foundation
import SwiftData

@MainActor
struct LegacyGroundingContextProvider {
    var directContext: ModelContext?
    var allowSharedFallback: Bool = true

    func resolveContext() -> ModelContext? {
        if let directContext { return directContext }
        if allowSharedFallback, let shared = SharedContainer.shared { return ModelContext(shared) }
        return nil
    }

    var degradedReason: String? {
        resolveContext() == nil ? "model_context_unavailable" : nil
    }
}
