import Foundation

struct BackgroundDiagnosticsSnapshot: Sendable {
    let permittedIdentifiers: [String]
    let entitlementWarnings: [String]
    let entitlementStates: [RuntimeEntitlementState]
    let backgroundGPUSupported: Bool
    let continuedProcessingStatus: String
    let availableMemoryBytes: UInt64
    let energyKit: EnergyKitCapabilitySnapshot
    let storeKit: StoreKitCapabilitySnapshot
}
