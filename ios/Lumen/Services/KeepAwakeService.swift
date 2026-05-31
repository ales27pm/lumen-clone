import SwiftUI
import UIKit

/// Prevents the device from auto-locking while one or more "reasons" are active.
///
/// Uses a reason-counted approach so multiple subsystems (e.g. voice mode + a long
/// model download) can independently request the screen to stay awake without
/// stomping on each other. The idle timer is only re-enabled when *all* reasons
/// are released.
@MainActor
@Observable
final class KeepAwakeService {
    static let shared = KeepAwakeService()

    private(set) var activeReasons: Set<String> = []

    var isActive: Bool { !activeReasons.isEmpty }

    private init() {}

    /// Activate keep-awake for a given reason. Idempotent per reason.
    func activate(reason: String) {
        let inserted = activeReasons.insert(reason).inserted
        guard inserted else { return }
        applyIdleTimerState()
    }

    /// Deactivate keep-awake for a given reason. No-op if not active.
    func deactivate(reason: String) {
        guard activeReasons.remove(reason) != nil else { return }
        applyIdleTimerState()
    }

    /// Release every reason and allow the device to auto-lock again.
    func deactivateAll() {
        guard !activeReasons.isEmpty else { return }
        activeReasons.removeAll()
        applyIdleTimerState()
    }

    /// Run an async block while the screen is kept awake.
    func withKeepAwake<T>(reason: String, _ operation: () async throws -> T) async rethrows -> T {
        activate(reason: reason)
        defer { deactivate(reason: reason) }
        return try await operation()
    }

    private func applyIdleTimerState() {
        UIApplication.shared.isIdleTimerDisabled = isActive
    }
}

// MARK: - SwiftUI helper

extension View {
    /// Keeps the device awake while this view is on screen and `isActive` is true.
    func keepScreenAwake(_ isActive: Bool, reason: String) -> some View {
        modifier(KeepAwakeModifier(isActive: isActive, reason: reason))
    }
}

private struct KeepAwakeModifier: ViewModifier {
    let isActive: Bool
    let reason: String

    func body(content: Content) -> some View {
        content
            .onAppear { sync() }
            .onDisappear { KeepAwakeService.shared.deactivate(reason: reason) }
            .onChange(of: isActive) { _, _ in sync() }
    }

    private func sync() {
        if isActive {
            KeepAwakeService.shared.activate(reason: reason)
        } else {
            KeepAwakeService.shared.deactivate(reason: reason)
        }
    }
}
