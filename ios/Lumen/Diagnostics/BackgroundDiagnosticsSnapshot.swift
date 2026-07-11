import Foundation

struct BackgroundDiagnosticsSnapshot: Sendable {
    let permittedIdentifiers: [String]
    let entitlementWarnings: [String]
    let entitlementStates: [ExpectedEntitlementState]
    let backgroundGPUSupported: Bool
    let continuedProcessingStatus: String
    let continuedProcessingRegistrationIdentifier: String
    let continuedProcessingLastSubmittedIdentifier: String?
    let continuedProcessingRegistrationErrorDomain: String?
    let continuedProcessingRegistrationErrorCode: Int?
    let continuedProcessingSubmitErrorDomain: String?
    let continuedProcessingSubmitErrorCode: Int?
    let continuedProcessingRegisteredBeforeLaunchCompletion: Bool?
    let continuedProcessingProvisioningEntitlementPresent: Bool?
    let continuedProcessingExpectedEntitlementValue: String?
    let availableMemoryBytes: UInt64
    let energyKit: EnergyKitCapabilitySnapshot
    let storeKit: StoreKitCapabilitySnapshot
}

enum BackgroundDiagnosticsEntitlements {
    static func provisioningProfileContainsContinuedProcessingEntitlement(bundle: Bundle = .main) -> Bool? {
        guard let entitlements = embeddedProvisioningEntitlements(bundle: bundle) else { return nil }
        return entitlementIsEnabled(entitlements[ExpectedEntitlementKey.backgroundGPU])
    }

    static func expectedContinuedProcessingEntitlementValue() -> String? {
        ExpectedEntitlementManifest.valueDescription(for: ExpectedEntitlementKey.backgroundGPU)
    }

    private static func embeddedProvisioningEntitlements(bundle: Bundle) -> [String: Any]? {
        guard let url = bundle.url(forResource: "embedded", withExtension: "mobileprovision"),
              let data = try? Data(contentsOf: url),
              let raw = String(data: data, encoding: .isoLatin1) ?? String(data: data, encoding: .utf8),
              let plistStart = raw.range(of: "<plist"),
              let plistEnd = raw.range(of: "</plist>") else {
            return nil
        }
        let plistText = String(raw[plistStart.lowerBound..<plistEnd.upperBound])
        guard let plistData = plistText.data(using: .utf8),
              let plist = try? PropertyListSerialization.propertyList(from: plistData, options: [], format: nil) as? [String: Any],
              let entitlements = plist["Entitlements"] as? [String: Any] else {
            return nil
        }
        return entitlements
    }

    private static func entitlementIsEnabled(_ value: Any?) -> Bool {
        switch value {
        case let bool as Bool:
            return bool
        case let number as NSNumber:
            return number.boolValue
        case let string as String:
            let lower = string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return lower == "true" || lower == "1" || lower == "yes"
        case let array as [Any]:
            return !array.isEmpty
        default:
            return false
        }
    }
}
