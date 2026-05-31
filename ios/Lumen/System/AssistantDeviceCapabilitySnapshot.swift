import Foundation
import UIKit

enum DeviceThermalState: String, Codable, Sendable {
    case nominal
    case fair
    case serious
    case critical
    case unknown

    static func from(processThermalState: ProcessInfo.ThermalState) -> DeviceThermalState {
        switch processThermalState {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .unknown
        }
    }
}

struct AssistantDeviceCapabilitySnapshot: Codable, Sendable, Equatable {
    let osVersion: String
    let deviceIdiom: String
    let processorCount: Int
    let physicalMemoryBytes: UInt64
    let lowPowerModeEnabled: Bool
    let thermalState: DeviceThermalState
    let metalAvailable: Bool
    let coreMLAvailable: Bool
    let foundationModelsAvailable: Bool
    let backgroundRefreshStatus: String
}
