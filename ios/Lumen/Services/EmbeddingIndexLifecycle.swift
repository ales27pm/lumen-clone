import Foundation

/// Coordinates derived vector data only. A reset invalidates in-flight writers
/// even when the selected model's identity has not changed. The durable flag
/// keeps a partially cleared store unavailable for semantic search after relaunch.
@MainActor
enum EmbeddingIndexLifecycle {
    private static let pendingKey = "embeddingIndex.sourceRebuild.resetPending"
    private static var generation = UUID()

    static var resetPending: Bool { UserDefaults.standard.bool(forKey: pendingKey) }
    static var semanticReadsAllowed: Bool { !resetPending }

    static func captureWrite() -> UUID? { resetPending ? nil : generation }

    static func permitsWrite(_ captured: UUID?) -> Bool {
        guard let captured else { return false }
        return !resetPending && captured == generation
    }

    @discardableResult
    static func beginReset() -> UUID {
        generation = UUID()
        UserDefaults.standard.set(true, forKey: pendingKey)
        invalidateCaches()
        return generation
    }

    static func ownsReset(_ captured: UUID) -> Bool {
        resetPending && captured == generation
    }

    @discardableResult
    static func finishReset(_ captured: UUID) -> Bool {
        guard ownsReset(captured) else { return false }
        invalidateCaches()
        UserDefaults.standard.removeObject(forKey: pendingKey)
        return true
    }

    private static func invalidateCaches() {
        MemoryVectorIndex.shared.invalidate()
        RAGVectorIndex.shared.invalidate()
    }
}
