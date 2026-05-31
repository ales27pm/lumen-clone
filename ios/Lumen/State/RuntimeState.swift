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

    /// Ordered boot steps. Kept ephemeral so a fresh launch always reflects the
    /// real current boot sequence.
    var bootSteps: [BootStep] = [
        BootStep(id: "container", title: "Storage", detail: "Preparing SwiftData container", state: .pending),
        BootStep(id: "models", title: "Models", detail: "Checking local GGUF files", state: .pending),
        BootStep(id: "loader", title: "Runtime", detail: "Loading active models", state: .pending),
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
}
