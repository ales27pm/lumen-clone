import Foundation

#if canImport(EnergyKit)
import EnergyKit
#endif

struct EnergyKitCapabilitySnapshot: Sendable, Equatable {
    let frameworkAvailable: Bool
    let expectedEntitlementConfigured: Bool
    let status: String
    let venueCount: Int?
}

enum EnergyKitCapabilityService {
    static func snapshot() async -> EnergyKitCapabilitySnapshot {
        let expectedEntitlementConfigured = ExpectedEntitlementManifest.expectedBoolValue(for: ExpectedEntitlementKey.energyKit)
        #if canImport(EnergyKit)
        guard #available(iOS 26.1, *) else {
            return EnergyKitCapabilitySnapshot(
                frameworkAvailable: true,
                expectedEntitlementConfigured: expectedEntitlementConfigured,
                status: "requires_ios_26_1_for_venue_discovery",
                venueCount: nil
            )
        }
        do {
            let venues = try await EnergyVenue.venues()
            return EnergyKitCapabilitySnapshot(
                frameworkAvailable: true,
                expectedEntitlementConfigured: expectedEntitlementConfigured,
                status: "venues_available",
                venueCount: venues.count
            )
        } catch {
            return EnergyKitCapabilitySnapshot(
                frameworkAvailable: true,
                expectedEntitlementConfigured: expectedEntitlementConfigured,
                status: "venue_probe_failed:\(RuntimeMetricErrorSanitizer.code(for: error))",
                venueCount: nil
            )
        }
        #else
        return EnergyKitCapabilitySnapshot(
            frameworkAvailable: false,
            expectedEntitlementConfigured: expectedEntitlementConfigured,
            status: "framework_unavailable",
            venueCount: nil
        )
        #endif
    }
}
