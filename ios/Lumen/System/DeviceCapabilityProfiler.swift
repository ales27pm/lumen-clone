import Foundation
import UIKit
import Metal

struct DeviceCapabilityProfiler {
    @MainActor
    func captureSnapshot(processInfo: ProcessInfo = .processInfo) -> AssistantDeviceCapabilitySnapshot {
        let idiom: String
        switch UIDevice.current.userInterfaceIdiom {
        case .phone: idiom = "phone"
        case .pad: idiom = "pad"
        case .mac: idiom = "mac"
        case .tv: idiom = "tv"
        case .vision: idiom = "vision"
        default: idiom = "unspecified"
        }

        let bgStatus: String
        switch UIApplication.shared.backgroundRefreshStatus {
        case .available: bgStatus = "available"
        case .denied: bgStatus = "denied"
        case .restricted: bgStatus = "restricted"
        @unknown default: bgStatus = "unknown"
        }

        let foundationRuntime = FoundationModelsRuntimeAdapter()
        let coreMLRuntime = CoreMLRuntimeAdapter(modelURL: nil)

        return AssistantDeviceCapabilitySnapshot(
            osVersion: processInfo.operatingSystemVersionString,
            deviceIdiom: idiom,
            processorCount: processInfo.processorCount,
            physicalMemoryBytes: processInfo.physicalMemory,
            lowPowerModeEnabled: processInfo.isLowPowerModeEnabled,
            thermalState: .from(processThermalState: processInfo.thermalState),
            metalAvailable: MTLCreateSystemDefaultDevice() != nil,
            coreMLAvailable: coreMLRuntime.isAvailable,
            coreMLStatus: coreMLRuntime.availabilityStatus,
            foundationModelsAvailable: foundationRuntime.isAvailable,
            foundationModelsStatus: foundationRuntime.availabilityStatus,
            backgroundRefreshStatus: bgStatus
        )
    }
}
