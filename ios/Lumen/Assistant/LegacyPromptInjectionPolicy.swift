import Foundation

struct LegacyPromptInjectionPolicy: Sendable {
    let memoryMax: Int
    let ragMax: Int
    let toolMax: Int
    let runtimeMax: Int
    let allowSensitiveSections: Bool
    let backgroundSafeToolsOnly: Bool

    static let foregroundChat = Self(memoryMax: 900, ragMax: 1200, toolMax: 900, runtimeMax: 180, allowSensitiveSections: false, backgroundSafeToolsOnly: false)
    static let headlessTrigger = Self(memoryMax: 500, ragMax: 600, toolMax: 500, runtimeMax: 120, allowSensitiveSections: false, backgroundSafeToolsOnly: true)
    static let rolePipeline = Self(memoryMax: 700, ragMax: 900, toolMax: 700, runtimeMax: 120, allowSensitiveSections: false, backgroundSafeToolsOnly: false)
    static let slotAgent = Self(memoryMax: 700, ragMax: 900, toolMax: 700, runtimeMax: 120, allowSensitiveSections: false, backgroundSafeToolsOnly: false)
    static let diagnostics = Self(memoryMax: 400, ragMax: 400, toolMax: 400, runtimeMax: 120, allowSensitiveSections: true, backgroundSafeToolsOnly: true)
}
