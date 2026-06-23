import Foundation

enum RuntimeEntitlementKey {
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

struct RuntimeEntitlementState: Sendable, Equatable, Identifiable {
    let key: String
    let displayName: String
    let valueDescription: String
    let enabled: Bool

    var id: String { key }
}

enum RuntimeEntitlementReader {
    static let requiredProductionEntitlements: [(key: String, displayName: String)] = [
        (RuntimeEntitlementKey.backgroundGPU, "Background GPU Access"),
        (RuntimeEntitlementKey.carPlayVoiceConversation, "CarPlay Voice Based Conversation"),
        (RuntimeEntitlementKey.energyKit, "EnergyKit"),
        (RuntimeEntitlementKey.healthKit, "HealthKit"),
        (RuntimeEntitlementKey.hardenedProcess, "Enhanced Security: Hardened Process"),
        (RuntimeEntitlementKey.hardenedProcessCheckedAllocations, "Enhanced Security: Checked Allocations"),
        (RuntimeEntitlementKey.hardenedProcessCheckedAllocationsSoftMode, "Enhanced Security: Checked Allocations Soft Mode"),
        (RuntimeEntitlementKey.hardenedProcessDyldRO, "Enhanced Security: dyld-ro"),
        (RuntimeEntitlementKey.hardenedProcessEnhancedSecurityVersion, "Enhanced Security: Version"),
        (RuntimeEntitlementKey.hardenedProcessHardenedHeap, "Enhanced Security: Hardened Heap"),
        (RuntimeEntitlementKey.hardenedProcessPlatformRestrictions, "Enhanced Security: Platform Restrictions")
    ]

    static let developmentOnlyEntitlements: [(key: String, displayName: String)] = [
        (RuntimeEntitlementKey.increasedDebuggingMemory, "Increased Debugging Memory Limit")
    ]

    static func currentStates(includeDevelopmentOnly: Bool = true) -> [RuntimeEntitlementState] {
        let keys = includeDevelopmentOnly ? requiredProductionEntitlements + developmentOnlyEntitlements : requiredProductionEntitlements
        return keys.map { key, name in
            let value = valueDescription(for: key)
            return RuntimeEntitlementState(
                key: key,
                displayName: name,
                valueDescription: value ?? "missing",
                enabled: value.map(isEnabled(valueDescription:)) ?? false
            )
        }
    }

    static func boolValue(for key: String) -> Bool {
        guard let value = configuredEntitlementValues[key] else { return false }
        return isEnabled(valueDescription: value)
    }

    static func valueDescription(for key: String) -> String? {
        configuredEntitlementValues[key]
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

    private static let configuredEntitlementValues: [String: String] = {
        var values = [
            RuntimeEntitlementKey.backgroundGPU: "true",
            RuntimeEntitlementKey.carPlayVoiceConversation: "true",
            RuntimeEntitlementKey.energyKit: "true",
            RuntimeEntitlementKey.healthKit: "true",
            RuntimeEntitlementKey.hardenedProcess: "true",
            RuntimeEntitlementKey.hardenedProcessCheckedAllocations: "true",
            RuntimeEntitlementKey.hardenedProcessCheckedAllocationsSoftMode: "true",
            RuntimeEntitlementKey.hardenedProcessDyldRO: "true",
            RuntimeEntitlementKey.hardenedProcessEnhancedSecurityVersion: "1",
            RuntimeEntitlementKey.hardenedProcessHardenedHeap: "true",
            RuntimeEntitlementKey.hardenedProcessPlatformRestrictions: "2"
        ]

        #if DEBUG
        values[RuntimeEntitlementKey.increasedDebuggingMemory] = "true"
        #endif

        return values
    }()
}
