import Foundation

struct RuntimePowerState: Sendable, Codable, Equatable {
    let isLowPowerModeEnabled: Bool
    let thermalPressure: ThermalPressureLevel
    let isExternalPowerConnected: Bool?
    let appIsForeground: Bool

    init(
        isLowPowerModeEnabled: Bool,
        thermalPressure: ThermalPressureLevel,
        isExternalPowerConnected: Bool? = nil,
        appIsForeground: Bool
    ) {
        self.isLowPowerModeEnabled = isLowPowerModeEnabled
        self.thermalPressure = thermalPressure
        self.isExternalPowerConnected = isExternalPowerConnected
        self.appIsForeground = appIsForeground
    }

    var allowsHeavyForegroundInference: Bool {
        appIsForeground && thermalPressure != .critical
    }

    var allowsBackgroundInference: Bool {
        false
    }
}
