import Foundation
@testable import Lumen

struct TestDeviceCapabilityProvider: DeviceCapabilityProviding {
    let formFactor: DeviceFormFactor
    let performanceTier: DevicePerformanceTier
    let physicalMemoryBytes: UInt64
    let processorCount: Int
    let activeProcessorCount: Int
    let hasMetalSupport: Bool
    let isSimulator: Bool
    let osVersion: String
    let powerState: RuntimePowerState
    let capturedAt: Date

    init(
        formFactor: DeviceFormFactor = .iPhone,
        performanceTier: DevicePerformanceTier = .balanced,
        physicalMemoryBytes: UInt64 = 6 * 1_073_741_824,
        processorCount: Int = 6,
        activeProcessorCount: Int = 6,
        hasMetalSupport: Bool = true,
        isSimulator: Bool = false,
        osVersion: String = "Test OS",
        powerState: RuntimePowerState = RuntimePowerState(
            isLowPowerModeEnabled: false,
            thermalPressure: .nominal,
            isExternalPowerConnected: nil,
            appIsForeground: true
        ),
        capturedAt: Date = Date(timeIntervalSince1970: 1_700_000_000)
    ) {
        self.formFactor = formFactor
        self.performanceTier = performanceTier
        self.physicalMemoryBytes = physicalMemoryBytes
        self.processorCount = processorCount
        self.activeProcessorCount = activeProcessorCount
        self.hasMetalSupport = hasMetalSupport
        self.isSimulator = isSimulator
        self.osVersion = osVersion
        self.powerState = powerState
        self.capturedAt = capturedAt
    }

    func currentSnapshot(appIsForeground: Bool) async -> DeviceCapabilitySnapshot {
        let resolvedPowerState = RuntimePowerState(
            isLowPowerModeEnabled: powerState.isLowPowerModeEnabled,
            thermalPressure: powerState.thermalPressure,
            isExternalPowerConnected: powerState.isExternalPowerConnected,
            appIsForeground: appIsForeground
        )

        return DeviceCapabilitySnapshot(
            formFactor: formFactor,
            performanceTier: performanceTier,
            physicalMemoryBytes: physicalMemoryBytes,
            processorCount: processorCount,
            activeProcessorCount: activeProcessorCount,
            hasMetalSupport: hasMetalSupport,
            isSimulator: isSimulator,
            osVersion: osVersion,
            powerState: resolvedPowerState,
            capturedAt: capturedAt
        )
    }
}
