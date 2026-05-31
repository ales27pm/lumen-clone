import Foundation

struct DeviceCapabilitySnapshot: Sendable, Codable, Equatable {
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
        formFactor: DeviceFormFactor,
        performanceTier: DevicePerformanceTier,
        physicalMemoryBytes: UInt64,
        processorCount: Int,
        activeProcessorCount: Int,
        hasMetalSupport: Bool,
        isSimulator: Bool,
        osVersion: String,
        powerState: RuntimePowerState,
        capturedAt: Date = Date()
    ) {
        self.formFactor = formFactor
        self.performanceTier = performanceTier
        self.physicalMemoryBytes = physicalMemoryBytes
        self.processorCount = max(1, processorCount)
        self.activeProcessorCount = max(1, activeProcessorCount)
        self.hasMetalSupport = hasMetalSupport
        self.isSimulator = isSimulator
        self.osVersion = osVersion
        self.powerState = powerState
        self.capturedAt = capturedAt
    }

    var physicalMemoryMB: Int {
        Int(physicalMemoryBytes / 1_048_576)
    }

    var recommendedLLMMemoryCeilingMB: Int {
        let physicalCeiling = Int(Double(physicalMemoryMB) * 0.65)
        let tierCap: Int
        switch performanceTier {
        case .constrained:
            tierCap = 1_536
        case .balanced:
            tierCap = 3_072
        case .high:
            tierCap = 4_096
        case .extreme:
            tierCap = 8_192
        case .simulator:
            tierCap = 1_024
        case .unknown:
            tierCap = 1_536
        }
        return max(512, min(physicalCeiling, tierCap))
    }
}
