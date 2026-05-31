import SwiftUI
import UIKit
import Combine

/// Observes the device battery level, charging state, and Low Power Mode.
///
/// Usage:
/// ```swift
/// @State private var battery = BatteryService.shared
/// // or in a View:
/// Text("\(Int(battery.level * 100))%")
/// ```
@MainActor
@Observable
final class BatteryService {
    static let shared = BatteryService()

    enum ChargingState: String, Sendable {
        case unknown
        case unplugged
        case charging
        case full
    }

    /// Battery level in 0...1. `-1` means the value is unavailable (e.g. simulator).
    private(set) var level: Float = -1
    private(set) var state: ChargingState = .unknown
    private(set) var isLowPowerModeEnabled: Bool = false

    private var observers: [NSObjectProtocol] = []
    private var isStarted = false

    private init() {}

    func start() {
        guard !isStarted else { return }
        isStarted = true

        let device = UIDevice.current
        device.isBatteryMonitoringEnabled = true
        refresh()

        let center = NotificationCenter.default
        observers.append(center.addObserver(
            forName: UIDevice.batteryLevelDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        })
        observers.append(center.addObserver(
            forName: UIDevice.batteryStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        })
        observers.append(center.addObserver(
            forName: Notification.Name.NSProcessInfoPowerStateDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        })
    }

    func stop() {
        for token in observers {
            NotificationCenter.default.removeObserver(token)
        }
        observers.removeAll()
        UIDevice.current.isBatteryMonitoringEnabled = false
        isStarted = false
    }

    func refresh() {
        let device = UIDevice.current
        level = device.batteryLevel
        state = Self.mapState(device.batteryState)
        isLowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled
    }

    /// Convenience: percentage 0...100, or `nil` if unavailable.
    var percentage: Int? {
        guard level >= 0 else { return nil }
        return Int((level * 100).rounded())
    }

    var isCharging: Bool {
        state == .charging || state == .full
    }

    private static func mapState(_ state: UIDevice.BatteryState) -> ChargingState {
        switch state {
        case .unknown: return .unknown
        case .unplugged: return .unplugged
        case .charging: return .charging
        case .full: return .full
        @unknown default: return .unknown
        }
    }

}
