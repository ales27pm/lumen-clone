import Foundation
import Observation

fileprivate nonisolated enum UserSettingsKeys {
    static let activeChatModelID = "activeChatModelID"
    static let activeEmbeddingModelID = "activeEmbeddingModelID"
    static let enabledToolIDs = "enabledToolIDs"
    static let disabledToolIDs = "disabledToolIDs"
    static let systemPrompt = "systemPrompt"
    static let temperature = "temperature"
    static let topP = "topP"
    static let repetitionPenalty = "repetitionPenalty"
    static let contextSize = "contextSize"
    static let maxTokens = "maxTokens"
    static let autoMemory = "autoMemory"
    static let selectedPresetID = "selectedPresetID"
    static let voiceID = "voiceID"
    static let speakingRate = "speakingRate"
    static let handsFree = "handsFree"
    static let maxAgentSteps = "maxAgentSteps"
    static let showThinkingByDefault = "showThinkingByDefault"
    static let agentModeEnabled = "agentModeEnabled"
    static let developerTraceModeEnabled = "developerTraceModeEnabled"
    static let developerReasoningCaptureEnabled = "developerReasoningCaptureEnabled"
    static let autoDownloadFleetModels = "autoDownloadFleetModels"
    static let confirmFleetDownloads = "confirmFleetDownloads"
}

/// Persistent user settings. Values are auto-persisted to UserDefaults whenever
/// they change. Initialization reads from UserDefaults; no didSet runs during init.
@Observable
final class UserSettings {
    // Model selection
    var activeChatModelID: String? { didSet { save() } }
    var activeEmbeddingModelID: String? { didSet { save() } }

    // Tools. Persisted as opt-out disabled IDs; enabled IDs are derived for call-site compatibility.
    private var disabledToolIDs: Set<String> { didSet { save() } }
    var enabledToolIDs: Set<String> {
        get { Self.currentToolIDs.subtracting(disabledToolIDs) }
        set { disabledToolIDs = Self.currentToolIDs.subtracting(newValue) }
    }

    // Prompting / generation
    var systemPrompt: String { didSet { save() } }
    var temperature: Double { didSet { save() } }
    var topP: Double { didSet { save() } }
    var repetitionPenalty: Double { didSet { save() } }
    var contextSize: Int { didSet { save() } }
    var maxTokens: Int { didSet { save() } }
    var autoMemory: Bool { didSet { save() } }
    var selectedPresetID: String { didSet { save() } }

    // Voice
    var voiceID: String? { didSet { save() } }
    var speakingRate: Double { didSet { save() } }
    var handsFree: Bool { didSet { save() } }

    // Agent
    var maxAgentSteps: Int { didSet { save() } }
    var showThinkingByDefault: Bool { didSet { save() } }
    var agentModeEnabled: Bool { didSet { save() } }
    var developerTraceModeEnabled: Bool { didSet { save() } }
    var developerReasoningCaptureEnabled: Bool { didSet { save() } }

    // Fleet bootstrap
    var autoDownloadFleetModels: Bool { didSet { save() } }
    var confirmFleetDownloads: Bool { didSet { save() } }

    @ObservationIgnored
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults

        activeChatModelID = defaults.string(forKey: UserSettingsKeys.activeChatModelID)
        activeEmbeddingModelID = defaults.string(forKey: UserSettingsKeys.activeEmbeddingModelID)

        disabledToolIDs = Self.loadDisabledToolIDs(defaults: defaults)

        systemPrompt = defaults.string(forKey: UserSettingsKeys.systemPrompt) ?? Presets.general.prompt
        temperature = defaults.object(forKey: UserSettingsKeys.temperature) as? Double ?? 0.7
        topP = defaults.object(forKey: UserSettingsKeys.topP) as? Double ?? 0.95
        repetitionPenalty = defaults.object(forKey: UserSettingsKeys.repetitionPenalty) as? Double ?? 1.1
        contextSize = defaults.object(forKey: UserSettingsKeys.contextSize) as? Int ?? 4096
        maxTokens = defaults.object(forKey: UserSettingsKeys.maxTokens) as? Int ?? 512
        autoMemory = defaults.object(forKey: UserSettingsKeys.autoMemory) as? Bool ?? true
        selectedPresetID = defaults.string(forKey: UserSettingsKeys.selectedPresetID) ?? Presets.general.id

        voiceID = defaults.string(forKey: UserSettingsKeys.voiceID)
        speakingRate = defaults.object(forKey: UserSettingsKeys.speakingRate) as? Double ?? 0.5
        handsFree = defaults.object(forKey: UserSettingsKeys.handsFree) as? Bool ?? false

        maxAgentSteps = defaults.object(forKey: UserSettingsKeys.maxAgentSteps) as? Int ?? 6
        showThinkingByDefault = defaults.object(forKey: UserSettingsKeys.showThinkingByDefault) as? Bool ?? false
        agentModeEnabled = defaults.object(forKey: UserSettingsKeys.agentModeEnabled) as? Bool ?? true
        #if DEBUG
        developerTraceModeEnabled = defaults.object(forKey: UserSettingsKeys.developerTraceModeEnabled) as? Bool ?? false
        developerReasoningCaptureEnabled = defaults.object(forKey: UserSettingsKeys.developerReasoningCaptureEnabled) as? Bool ?? false
        #else
        developerTraceModeEnabled = false
        developerReasoningCaptureEnabled = false
        #endif
        autoDownloadFleetModels = defaults.object(forKey: UserSettingsKeys.autoDownloadFleetModels) as? Bool ?? true
        confirmFleetDownloads = defaults.object(forKey: UserSettingsKeys.confirmFleetDownloads) as? Bool ?? false
    }

    private func save() {
        defaults.set(activeChatModelID, forKey: UserSettingsKeys.activeChatModelID)
        defaults.set(activeEmbeddingModelID, forKey: UserSettingsKeys.activeEmbeddingModelID)
        defaults.set(Array(disabledToolIDs), forKey: UserSettingsKeys.disabledToolIDs)
        defaults.set(Array(enabledToolIDs), forKey: UserSettingsKeys.enabledToolIDs)
        defaults.set(systemPrompt, forKey: UserSettingsKeys.systemPrompt)
        defaults.set(temperature, forKey: UserSettingsKeys.temperature)
        defaults.set(topP, forKey: UserSettingsKeys.topP)
        defaults.set(repetitionPenalty, forKey: UserSettingsKeys.repetitionPenalty)
        defaults.set(contextSize, forKey: UserSettingsKeys.contextSize)
        defaults.set(maxTokens, forKey: UserSettingsKeys.maxTokens)
        defaults.set(autoMemory, forKey: UserSettingsKeys.autoMemory)
        defaults.set(selectedPresetID, forKey: UserSettingsKeys.selectedPresetID)
        defaults.set(voiceID, forKey: UserSettingsKeys.voiceID)
        defaults.set(speakingRate, forKey: UserSettingsKeys.speakingRate)
        defaults.set(handsFree, forKey: UserSettingsKeys.handsFree)
        defaults.set(maxAgentSteps, forKey: UserSettingsKeys.maxAgentSteps)
        defaults.set(showThinkingByDefault, forKey: UserSettingsKeys.showThinkingByDefault)
        defaults.set(agentModeEnabled, forKey: UserSettingsKeys.agentModeEnabled)
        #if DEBUG
        defaults.set(developerTraceModeEnabled, forKey: UserSettingsKeys.developerTraceModeEnabled)
        defaults.set(developerReasoningCaptureEnabled, forKey: UserSettingsKeys.developerReasoningCaptureEnabled)
        #else
        defaults.set(false, forKey: UserSettingsKeys.developerTraceModeEnabled)
        defaults.set(false, forKey: UserSettingsKeys.developerReasoningCaptureEnabled)
        #endif
        defaults.set(autoDownloadFleetModels, forKey: UserSettingsKeys.autoDownloadFleetModels)
        defaults.set(confirmFleetDownloads, forKey: UserSettingsKeys.confirmFleetDownloads)
    }

    func toggleTool(_ id: String) {
        let canonical = ToolRouteGuard.canonicalToolID(id)
        if disabledToolIDs.contains(canonical) {
            disabledToolIDs.remove(canonical)
        } else {
            disabledToolIDs.insert(canonical)
        }
    }


    private static var currentToolIDs: Set<String> {
        Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
    }

    private static func loadDisabledToolIDs(defaults: UserDefaults) -> Set<String> {
        let current = currentToolIDs
        if let disabled = defaults.array(forKey: UserSettingsKeys.disabledToolIDs) as? [String] {
            return Set(disabled.map(ToolRouteGuard.canonicalToolID)).intersection(current)
        }
        if let legacyEnabled = defaults.array(forKey: UserSettingsKeys.enabledToolIDs) as? [String] {
            let enabled = Set(legacyEnabled.map(ToolRouteGuard.canonicalToolID))
            let disabled = current.subtracting(enabled)
            defaults.set(Array(disabled), forKey: UserSettingsKeys.disabledToolIDs)
            return disabled
        }
        return []
    }

    func applyPreset(_ preset: Preset) {        systemPrompt = preset.prompt
        selectedPresetID = preset.id
        temperature = preset.temperature
    }

    /// Snapshot for background / concurrency-safe consumers.
    var snapshot: SettingsSnapshot {
        SettingsSnapshot(
            activeChatModelID: activeChatModelID,
            activeEmbeddingModelID: activeEmbeddingModelID,
            enabledToolIDs: enabledToolIDs,
            systemPrompt: systemPrompt,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            contextSize: contextSize,
            maxTokens: maxTokens,
            autoMemory: autoMemory,
            voiceID: voiceID,
            speakingRate: speakingRate,
            handsFree: handsFree,
            maxAgentSteps: maxAgentSteps,
            agentModeEnabled: agentModeEnabled,
            developerTraceModeEnabled: developerTraceModeEnabled,
            developerReasoningCaptureEnabled: developerReasoningCaptureEnabled,
            autoDownloadFleetModels: autoDownloadFleetModels,
            confirmFleetDownloads: confirmFleetDownloads
        )
    }
}

/// Sendable, thread-safe snapshot of user settings. Safe to pass into background
/// tasks, detached actors, and BG task handlers.
nonisolated struct SettingsSnapshot: Sendable {
    let activeChatModelID: String?
    let activeEmbeddingModelID: String?
    let enabledToolIDs: Set<String>
    let systemPrompt: String
    let temperature: Double
    let topP: Double
    let repetitionPenalty: Double
    let contextSize: Int
    let maxTokens: Int
    let autoMemory: Bool
    let voiceID: String?
    let speakingRate: Double
    let handsFree: Bool
    let maxAgentSteps: Int
    let agentModeEnabled: Bool
    let developerTraceModeEnabled: Bool
    let developerReasoningCaptureEnabled: Bool
    let autoDownloadFleetModels: Bool
    let confirmFleetDownloads: Bool

    private static func debugDeveloperTraceEnabled(defaults: UserDefaults) -> Bool {
        #if DEBUG
        return defaults.object(forKey: UserSettingsKeys.developerTraceModeEnabled) as? Bool ?? false
        #else
        return false
        #endif
    }

    private static func debugDeveloperReasoningCaptureEnabled(defaults: UserDefaults) -> Bool {
        #if DEBUG
        return defaults.object(forKey: UserSettingsKeys.developerReasoningCaptureEnabled) as? Bool ?? false
        #else
        return false
        #endif
    }

    /// Loads a snapshot directly from UserDefaults without touching the
    /// in-memory `UserSettings` instance. Used by background tasks that may
    /// run before or without the main app scene.
    static func loadFromDisk(defaults: UserDefaults = .standard) -> SettingsSnapshot {
        let current = Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
        let disabled: Set<String>
        if let savedDisabled = defaults.array(forKey: UserSettingsKeys.disabledToolIDs) as? [String] {
            disabled = Set(savedDisabled.map(ToolRouteGuard.canonicalToolID)).intersection(current)
        } else if let legacyEnabled = defaults.array(forKey: UserSettingsKeys.enabledToolIDs) as? [String] {
            disabled = current.subtracting(Set(legacyEnabled.map(ToolRouteGuard.canonicalToolID)))
        } else {
            disabled = []
        }
        let enabled = current.subtracting(disabled)
        return SettingsSnapshot(
            activeChatModelID: defaults.string(forKey: UserSettingsKeys.activeChatModelID),
            activeEmbeddingModelID: defaults.string(forKey: UserSettingsKeys.activeEmbeddingModelID),
            enabledToolIDs: enabled,
            systemPrompt: defaults.string(forKey: UserSettingsKeys.systemPrompt) ?? Presets.general.prompt,
            temperature: defaults.object(forKey: UserSettingsKeys.temperature) as? Double ?? 0.7,
            topP: defaults.object(forKey: UserSettingsKeys.topP) as? Double ?? 0.95,
            repetitionPenalty: defaults.object(forKey: UserSettingsKeys.repetitionPenalty) as? Double ?? 1.1,
            contextSize: defaults.object(forKey: UserSettingsKeys.contextSize) as? Int ?? 4096,
            maxTokens: defaults.object(forKey: UserSettingsKeys.maxTokens) as? Int ?? 512,
            autoMemory: defaults.object(forKey: UserSettingsKeys.autoMemory) as? Bool ?? true,
            voiceID: defaults.string(forKey: UserSettingsKeys.voiceID),
            speakingRate: defaults.object(forKey: UserSettingsKeys.speakingRate) as? Double ?? 0.5,
            handsFree: defaults.object(forKey: UserSettingsKeys.handsFree) as? Bool ?? false,
            maxAgentSteps: defaults.object(forKey: UserSettingsKeys.maxAgentSteps) as? Int ?? 6,
            agentModeEnabled: defaults.object(forKey: UserSettingsKeys.agentModeEnabled) as? Bool ?? true,
            developerTraceModeEnabled: Self.debugDeveloperTraceEnabled(defaults: defaults),
            developerReasoningCaptureEnabled: Self.debugDeveloperReasoningCaptureEnabled(defaults: defaults),
            autoDownloadFleetModels: defaults.object(forKey: UserSettingsKeys.autoDownloadFleetModels) as? Bool ?? true,
            confirmFleetDownloads: defaults.object(forKey: UserSettingsKeys.confirmFleetDownloads) as? Bool ?? false
        )
    }
}
