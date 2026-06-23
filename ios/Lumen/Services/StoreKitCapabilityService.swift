import Foundation
import StoreKit

struct StoreKitCapabilitySnapshot: Sendable, Equatable {
    let frameworkAvailable: Bool
    let status: String
    let environment: String
}

enum StoreKitCapabilityService {
    static func snapshot() async -> StoreKitCapabilitySnapshot {
        guard #available(iOS 16.0, *) else {
            return StoreKitCapabilitySnapshot(frameworkAvailable: true, status: "requires_ios_16", environment: "unavailable")
        }
        do {
            let verification = try await AppTransaction.shared
            let appTransaction = try verification.payloadValue
            return StoreKitCapabilitySnapshot(
                frameworkAvailable: true,
                status: "app_transaction_available",
                environment: String(describing: appTransaction.environment)
            )
        } catch {
            return StoreKitCapabilitySnapshot(frameworkAvailable: true, status: "app_transaction_unavailable:\(RuntimeMetricErrorSanitizer.code(for: error))", environment: "unavailable")
        }
    }
}
