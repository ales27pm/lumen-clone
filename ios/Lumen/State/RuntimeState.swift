import Foundation
import Observation

nonisolated enum BootStepState: String, Codable, Sendable {
    case pending
    case running
    case complete
    case warning
    case failed
}

nonisolated struct BootStep: Identifiable, Hashable, Sendable {
    let id: String
    var title: String
    var detail: String
    var state: BootStepState
}

nonisolated enum ModelAutoloadState: Equatable, Sendable {
    case idle
    case loading
    case finished(chatLoaded: Bool, embeddingLoaded: Bool)

    var shouldRetryAfterLeavingActiveScene: Bool {
        switch self {
        case .idle:
            return false
        case .loading:
            return true
        case let .finished(chatLoaded, embeddingLoaded):
            return !chatLoaded || !embeddingLoaded
        }
    }
}

nonisolated struct ModelAutoloadRequestKey: Hashable, Sendable {
    let bootstrapReady: Bool
    let sceneIsActive: Bool
    let activeChatModelID: String?
    let activeEmbeddingModelID: String?
    let selectedModelFamily: LumenModelFamily
    let requestGeneration: UInt64

    var canStartAutoload: Bool {
        bootstrapReady && sceneIsActive
    }
}

nonisolated enum ModelAutoloadRetryPolicy {
    static let maximumRetryCount = 2
    static let minimumDelaySeconds: TimeInterval = 1
    static let maximumDelaySeconds: TimeInterval = 120

    static func boundedDelaySeconds(_ suggestedDelaySeconds: TimeInterval?) -> TimeInterval? {
        guard let suggestedDelaySeconds,
              suggestedDelaySeconds.isFinite,
              suggestedDelaySeconds > 0 else {
            return nil
        }
        return min(maximumDelaySeconds, max(minimumDelaySeconds, suggestedDelaySeconds))
    }

    static func shouldRetry(
        completedRetryCount: Int,
        suggestedDelaySeconds: TimeInterval?,
        sceneIsActive: Bool
    ) -> Bool {
        sceneIsActive
            && completedRetryCount < maximumRetryCount
            && boundedDelaySeconds(suggestedDelaySeconds) != nil
    }
}

/// Ephemeral, non-persisted UI state. Reset every launch. Do not persist any of
/// these to disk.
@Observable
final class RuntimeState {
    /// Whether a chat / agent generation is currently streaming.
    var isGenerating: Bool = false

    /// Whether the app has verified user notification permission for triggers.
    /// `nil` means not yet asked.
    var notificationPermissionGranted: Bool?

    /// Whether the boot overlay is visible.
    var bootSplashVisible: Bool = true

    /// Whether core launch work has finished and the user can continue into the app
    /// even if large model downloads are still running.
    var bootCoreComplete: Bool = false

    /// Human-readable boot status shown on the launch overlay.
    var bootHeadline: String = "Starting Lumen"

    /// Foreground launch restoration state for the persisted chat and embedding
    /// selections. This is intentionally ephemeral and contains no model paths.
    var modelAutoloadState: ModelAutoloadState = .idle

    /// Identifies the current foreground restoration attempt so a cancelled scene
    /// task cannot publish stale completion over a newer attempt.
    var modelAutoloadAttemptID: UUID?

    /// Model restoration must wait for detached grounding and trigger bootstrap to
    /// dismiss the splash. `bootCoreComplete` becomes true earlier, when the root UI
    /// first renders, so it is not sufficient on its own.
    var modelAutoloadBootstrapReady: Bool {
        bootCoreComplete && !bootSplashVisible
    }

    /// Invalidates the root autoload task when a model-family selection changes.
    /// Model IDs are already part of the task key, so they do not need to mutate
    /// this counter.
    var modelAutoloadRequestGeneration: UInt64 = 0

    /// Ordered boot steps. Kept ephemeral so a fresh launch always reflects the
    /// real current boot sequence.
    /// The blocking boot path stays lightweight. Selected chat and embedding
    /// runtimes are restored afterward from RootView while the scene is active.
    var bootSteps: [BootStep] = [
        BootStep(id: "container", title: "Storage", detail: "Preparing SwiftData container", state: .pending),
        BootStep(id: "grounding", title: "Knowledge", detail: "Loading agent manifests", state: .pending),
        BootStep(id: "triggers", title: "Triggers", detail: "Registering background tasks", state: .pending)
    ]

    func startBoot(headline: String = "Starting Lumen") {
        bootSplashVisible = true
        bootCoreComplete = false
        bootHeadline = headline
        for index in bootSteps.indices {
            bootSteps[index].state = .pending
        }
    }

    func updateBootStep(id: String, detail: String? = nil, state: BootStepState) {
        guard let index = bootSteps.firstIndex(where: { $0.id == id }) else { return }
        if let detail {
            bootSteps[index].detail = detail
        }
        bootSteps[index].state = state
    }

    func completeBootCore(headline: String = "Lumen is ready") {
        bootCoreComplete = true
        bootHeadline = headline
    }

    func dismissBootSplash() {
        bootSplashVisible = false
    }

    func requestModelAutoload() {
        modelAutoloadRequestGeneration &+= 1
    }
}
