import Foundation

#if canImport(EnergyKit)
import EnergyKit
#endif

struct EnergyKitCapabilitySnapshot: Sendable, Equatable {
    let frameworkAvailable: Bool
    let entitled: Bool
    let status: String
    let venueCount: Int?
}

enum EnergyKitCapabilityService {
    static func snapshot() async -> EnergyKitCapabilitySnapshot {
        let entitled = RuntimeEntitlementReader.boolValue(for: RuntimeEntitlementKey.energyKit)
        #if canImport(EnergyKit)
        guard #available(iOS 26.1, *) else {
            return EnergyKitCapabilitySnapshot(
                frameworkAvailable: true,
                entitled: entitled,
                status: "requires_ios_26_1_for_venue_discovery",
                venueCount: nil
            )
        }
        do {
            let venues = try await EnergyVenue.venues()
            return EnergyKitCapabilitySnapshot(
                frameworkAvailable: true,
                entitled: entitled,
                status: "venues_available",
                venueCount: venues.count
            )
        } catch {
            return EnergyKitCapabilitySnapshot(
                frameworkAvailable: true,
                entitled: entitled,
                status: "venue_probe_failed:\(RuntimeMetricErrorSanitizer.code(for: error))",
                venueCount: nil
            )
        }
        #else
        return EnergyKitCapabilitySnapshot(
            frameworkAvailable: false,
            entitled: entitled,
            status: "framework_unavailable",
            venueCount: nil
        )
        #endif
    }
}
