import Foundation
import Metal
import UIKit

protocol DeviceCapabilityProviding: Sendable {
    func currentSnapshot(appIsForeground: Bool) async -> DeviceCapabilitySnapshot
}

final class SystemDeviceCapabilityProvider: DeviceCapabilityProviding {
    init() {}

    func currentSnapshot(appIsForeground: Bool) async -> DeviceCapabilitySnapshot {
        let processInfo = ProcessInfo.processInfo
        let isSimulator = Self.isRunningInSimulator
        let physicalMemory = processInfo.physicalMemory
        let processorCount = processInfo.processorCount
        let activeProcessorCount = processInfo.activeProcessorCount
        let formFactor = Self.detectFormFactor(isSimulator: isSimulator)
        let tier = Self.performanceTier(
            isSimulator: isSimulator,
            physicalMemoryBytes: physicalMemory,
            processorCount: processorCount
        )
        let powerState = RuntimePowerState(
            isLowPowerModeEnabled: processInfo.isLowPowerModeEnabled,
            thermalPressure: Self.thermalPressure(from: processInfo.thermalState),
            isExternalPowerConnected: nil,
            appIsForeground: appIsForeground
        )

        return DeviceCapabilitySnapshot(
            formFactor: formFactor,
            performanceTier: tier,
            physicalMemoryBytes: physicalMemory,
            processorCount: processorCount,
            activeProcessorCount: activeProcessorCount,
            hasMetalSupport: MTLCreateSystemDefaultDevice() != nil,
            isSimulator: isSimulator,
            osVersion: processInfo.operatingSystemVersionString,
            powerState: powerState
        )
    }

    private static var isRunningInSimulator: Bool {
        #if targetEnvironment(simulator)
        true
        #else
        false
        #endif
    }

    private static func detectFormFactor(isSimulator: Bool) -> DeviceFormFactor {
        if isSimulator {
            return .simulator
        }

        #if targetEnvironment(macCatalyst)
        return .mac
        #else
        switch UIDevice.current.userInterfaceIdiom {
        case .phone:
            return .iPhone
        case .pad:
            return .iPad
        case .mac:
            return .mac
        default:
            return .unknown
        }
        #endif
    }

    private static func performanceTier(
        isSimulator: Bool,
        physicalMemoryBytes: UInt64,
        processorCount: Int
    ) -> DevicePerformanceTier {
        if isSimulator {
            return .simulator
        }

        let memoryGB = Double(physicalMemoryBytes) / 1_073_741_824.0
        if memoryGB >= 12 {
            return .extreme
        }
        if memoryGB >= 8 && processorCount >= 6 {
            return .high
        }
        if memoryGB >= 6 {
            return .balanced
        }
        if memoryGB > 0 && memoryGB < 6 {
            return .constrained
        }
        return .unknown
    }

    private static func thermalPressure(from state: ProcessInfo.ThermalState) -> ThermalPressureLevel {
        switch state {
        case .nominal:
            return .nominal
        case .fair:
            return .fair
        case .serious:
            return .serious
        case .critical:
            return .critical
        @unknown default:
            return .unknown
        }
    }
}
