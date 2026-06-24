import Foundation

enum ExpectedEntitlementKey {
    static let backgroundGPU = "com.apple.developer.background-tasks.continued-processing.gpu"
    static let carPlayVoiceConversation = "com.apple.developer.carplay-voice-based-conversation"
    static let energyKit = "com.apple.developer.energykit"
    static let healthKit = "com.apple.developer.healthkit"
    static let increasedDebuggingMemory = "com.apple.developer.kernel.increased-debugging-memory-limit"
    static let hardenedProcess = "com.apple.security.hardened-process"
    static let hardenedProcessCheckedAllocations = "com.apple.security.hardened-process.checked-allocations"
    static let hardenedProcessCheckedAllocationsSoftMode = "com.apple.security.hardened-process.checked-allocations.soft-mode"
    static let hardenedProcessDyldRO = "com.apple.security.hardened-process.dyld-ro"
    static let hardenedProcessEnhancedSecurityVersion = "com.apple.security.hardened-process.enhanced-security-version"
    static let hardenedProcessHardenedHeap = "com.apple.security.hardened-process.hardened-heap"
    static let hardenedProcessPlatformRestrictions = "com.apple.security.hardened-process.platform-restrictions"
}

struct ExpectedEntitlementState: Sendable, Equatable, Identifiable {
    let key: String
    let displayName: String
    let valueDescription: String
    let enabled: Bool

    var id: String { key }
}

enum ExpectedEntitlementManifest {
    static let productionEntitlements: [(key: String, displayName: String)] = [
        (ExpectedEntitlementKey.backgroundGPU, "Background GPU Access"),
        (ExpectedEntitlementKey.carPlayVoiceConversation, "CarPlay Voice Based Conversation"),
        (ExpectedEntitlementKey.energyKit, "EnergyKit"),
        (ExpectedEntitlementKey.healthKit, "HealthKit"),
        (ExpectedEntitlementKey.hardenedProcess, "Enhanced Security: Hardened Process"),
        (ExpectedEntitlementKey.hardenedProcessCheckedAllocations, "Enhanced Security: Checked Allocations"),
        (ExpectedEntitlementKey.hardenedProcessDyldRO, "Enhanced Security: dyld-ro"),
        (ExpectedEntitlementKey.hardenedProcessEnhancedSecurityVersion, "Enhanced Security: Version"),
        (ExpectedEntitlementKey.hardenedProcessHardenedHeap, "Enhanced Security: Hardened Heap"),
        (ExpectedEntitlementKey.hardenedProcessPlatformRestrictions, "Enhanced Security: Platform Restrictions")
    ]

    static let developmentOnlyEntitlements: [(key: String, displayName: String)] = [
        (ExpectedEntitlementKey.increasedDebuggingMemory, "Increased Debugging Memory Limit"),
        (ExpectedEntitlementKey.hardenedProcessCheckedAllocationsSoftMode, "Enhanced Security: Checked Allocations Soft Mode")
    ]

    static func currentStates(includeDevelopmentOnly: Bool = includesDevelopmentOnlyByDefault) -> [ExpectedEntitlementState] {
        let keys = includeDevelopmentOnly ? productionEntitlements + developmentOnlyEntitlements : productionEntitlements
        return keys.map { key, name in
            let value = valueDescription(for: key)
            return ExpectedEntitlementState(
                key: key,
                displayName: name,
                valueDescription: value ?? "missing",
                enabled: value.map(isEnabled(valueDescription:)) ?? false
            )
        }
    }

    static func expectedBoolValue(for key: String) -> Bool {
        guard let value = expectedEntitlementValues[key] else { return false }
        return isEnabled(valueDescription: value)
    }

    static func valueDescription(for key: String) -> String? {
        expectedEntitlementValues[key]
    }

    private static func isEnabled(valueDescription: String) -> Bool {
        switch valueDescription.lowercased() {
        case "true", "1", "2":
            return true
        case "false", "0", "missing":
            return false
        default:
            return !valueDescription.isEmpty
        }
    }

    private static let expectedEntitlementValues: [String: String] = {
        var values = [
            ExpectedEntitlementKey.backgroundGPU: "true",
            ExpectedEntitlementKey.carPlayVoiceConversation: "true",
            ExpectedEntitlementKey.energyKit: "true",
            ExpectedEntitlementKey.healthKit: "true",
            ExpectedEntitlementKey.hardenedProcess: "true",
            ExpectedEntitlementKey.hardenedProcessCheckedAllocations: "true",
            ExpectedEntitlementKey.hardenedProcessDyldRO: "true",
            ExpectedEntitlementKey.hardenedProcessEnhancedSecurityVersion: "1",
            ExpectedEntitlementKey.hardenedProcessHardenedHeap: "true",
            ExpectedEntitlementKey.hardenedProcessPlatformRestrictions: "2"
        ]

        #if DEBUG
        values[ExpectedEntitlementKey.increasedDebuggingMemory] = "true"
        values[ExpectedEntitlementKey.hardenedProcessCheckedAllocationsSoftMode] = "true"
        #endif

        return values
    }()

    private static var includesDevelopmentOnlyByDefault: Bool {
        #if DEBUG
        true
        #else
        false
        #endif
    }
}
