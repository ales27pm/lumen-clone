import Foundation
import Observation

nonisolated enum UserSettingsStorageKeys {
    static let networkToolsEnabled = "networkToolsEnabled"
}

nonisolated struct PersistedModelSelectionV2: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let chatModelID: String?
    let embeddingModelID: String?
    let familyID: String
    let provisioningPlanID: String?
}

nonisolated enum PersistedModelSelectionStore {
    static let defaultsKey = "persistedModelSelectionV2"
    static let legacyChatKey = "activeChatModelID"
    static let legacyEmbeddingKey = "activeEmbeddingModelID"
    static let legacyFamilyKey = "selectedModelFamilyID"
    private static let schemaVersion = 2

    static func load(defaults: UserDefaults = .standard) -> PersistedModelSelectionV2? {
        guard let data = defaults.data(forKey: defaultsKey),
              let value = try? JSONDecoder().decode(PersistedModelSelectionV2.self, from: data),
              value.schemaVersion == schemaVersion,
              LumenModelFamily(rawValue: value.familyID) != nil
        else { return nil }
        return value
    }

    @discardableResult
    static func loadOrMigrate(defaults: UserDefaults = .standard) -> PersistedModelSelectionV2 {
        if let current = load(defaults: defaults) { return current }
        let migrated = PersistedModelSelectionV2(
            schemaVersion: schemaVersion,
            chatModelID: defaults.string(forKey: legacyChatKey),
            embeddingModelID: defaults.string(forKey: legacyEmbeddingKey),
            familyID: LumenModelFamily.fromStoredID(defaults.string(forKey: legacyFamilyKey)).rawValue,
            provisioningPlanID: nil
        )
        if let data = try? JSONEncoder().encode(migrated) {
            defaults.set(data, forKey: defaultsKey)
        }
        return migrated
    }

    static func selectedFamily(defaults: UserDefaults = .standard) -> LumenModelFamily {
        LumenModelFamily.fromStoredID(loadOrMigrate(defaults: defaults).familyID)
    }

    @discardableResult
    static func commit(
        chatModelID: String?,
        embeddingModelID: String?,
        family: LumenModelFamily,
        provisioningPlanID: String?,
        defaults: UserDefaults = .standard
    ) throws -> PersistedModelSelectionV2 {
        let value = PersistedModelSelectionV2(
            schemaVersion: schemaVersion,
            chatModelID: chatModelID,
            embeddingModelID: embeddingModelID,
            familyID: family.rawValue,
            provisioningPlanID: provisioningPlanID
        )
        let data = try JSONEncoder().encode(value)

        // The canonical record is written first. Readers prefer it over the legacy
        // mirrors, so an interruption cannot expose a half-updated chat/embed pair.
        defaults.set(data, forKey: defaultsKey)
        defaults.set(chatModelID, forKey: legacyChatKey)
        defaults.set(embeddingModelID, forKey: legacyEmbeddingKey)
        defaults.set(family.rawValue, forKey: legacyFamilyKey)
        return value
    }
}

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
    static let networkToolsEnabled = UserSettingsStorageKeys.networkToolsEnabled
}

fileprivate nonisolated enum ToolSettingsRegistrySnapshot {
    static var currentToolIDs: Set<String> {
        Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
    }

    static func loadDisabledToolIDs(defaults: UserDefaults, persistLegacyMigration: Bool) -> Set<String> {
        let current = currentToolIDs
        if let disabled = defaults.array(forKey: UserSettingsKeys.disabledToolIDs) as? [String] {
            return Set(disabled.map(ToolRouteGuard.canonicalToolID))
        }
        if let legacyEnabled = defaults.array(forKey: UserSettingsKeys.enabledToolIDs) as? [String] {
            let enabled = Set(legacyEnabled.map(ToolRouteGuard.canonicalToolID))
            let disabled = current.subtracting(enabled)
            if persistLegacyMigration {
                defaults.set(Array(disabled), forKey: UserSettingsKeys.disabledToolIDs)
            }
            return disabled
        }
        return []
    }
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
        get { ToolSettingsRegistrySnapshot.currentToolIDs.subtracting(disabledToolIDs) }
        set {
            let current = ToolSettingsRegistrySnapshot.currentToolIDs
            let canonicalizedEnabled = Set(newValue.map(ToolRouteGuard.canonicalToolID)).intersection(current)
            disabledToolIDs = current.subtracting(canonicalizedEnabled)
        }
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
    var networkToolsEnabled: Bool { didSet { save() } }

    // Fleet bootstrap
    var autoDownloadFleetModels: Bool { didSet { save() } }
    var confirmFleetDownloads: Bool { didSet { save() } }

    @ObservationIgnored
    private let defaults: UserDefaults

    @ObservationIgnored
    private var isApplyingAtomicModelSelection = false

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults

        // AppState owns one UserSettings instance created before provisioning can
        // start, making this the relaunch boundary for an interrupted final commit.
        _ = ModelProvisioningSwitchJournalStore.recoverIfNeeded(defaults: defaults)
        let persistedSelection = PersistedModelSelectionStore.loadOrMigrate(defaults: defaults)
        activeChatModelID = persistedSelection.chatModelID
        activeEmbeddingModelID = persistedSelection.embeddingModelID

        disabledToolIDs = ToolSettingsRegistrySnapshot.loadDisabledToolIDs(defaults: defaults, persistLegacyMigration: true)

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
        networkToolsEnabled = defaults.object(forKey: UserSettingsKeys.networkToolsEnabled) as? Bool ?? false
        autoDownloadFleetModels = defaults.object(forKey: UserSettingsKeys.autoDownloadFleetModels) as? Bool ?? true
        confirmFleetDownloads = defaults.object(forKey: UserSettingsKeys.confirmFleetDownloads) as? Bool ?? true
    }

    private func save() {
        if !isApplyingAtomicModelSelection {
            let currentSelection = PersistedModelSelectionStore.loadOrMigrate(defaults: defaults)
            let modelPairIsUnchanged = currentSelection.chatModelID == activeChatModelID
                && currentSelection.embeddingModelID == activeEmbeddingModelID
            _ = try? PersistedModelSelectionStore.commit(
                chatModelID: activeChatModelID,
                embeddingModelID: activeEmbeddingModelID,
                family: LumenModelFamily.fromStoredID(currentSelection.familyID),
                provisioningPlanID: modelPairIsUnchanged ? currentSelection.provisioningPlanID : nil,
                defaults: defaults
            )
        }
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
        defaults.set(networkToolsEnabled, forKey: UserSettingsKeys.networkToolsEnabled)
        defaults.set(autoDownloadFleetModels, forKey: UserSettingsKeys.autoDownloadFleetModels)
        defaults.set(confirmFleetDownloads, forKey: UserSettingsKeys.confirmFleetDownloads)
    }

    func commitActiveModelSelection(
        chatModelID: String?,
        embeddingModelID: String?,
        family: LumenModelFamily,
        provisioningPlanID: String?
    ) throws {
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: chatModelID,
            embeddingModelID: embeddingModelID,
            family: family,
            provisioningPlanID: provisioningPlanID,
            defaults: defaults
        )
        isApplyingAtomicModelSelection = true
        activeChatModelID = chatModelID
        activeEmbeddingModelID = embeddingModelID
        isApplyingAtomicModelSelection = false
    }

    func toggleTool(_ id: String) {
        let canonical = ToolRouteGuard.canonicalToolID(id)
        if disabledToolIDs.contains(canonical) {
            disabledToolIDs.remove(canonical)
        } else {
            disabledToolIDs.insert(canonical)
        }
    }

    func applyPreset(_ preset: Preset) {
        systemPrompt = preset.prompt
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
            networkToolsEnabled: networkToolsEnabled,
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
    let networkToolsEnabled: Bool
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
        let current = ToolSettingsRegistrySnapshot.currentToolIDs
        let disabled = ToolSettingsRegistrySnapshot.loadDisabledToolIDs(defaults: defaults, persistLegacyMigration: false)
        let enabled = current.subtracting(disabled)
        let persistedSelection = PersistedModelSelectionStore.loadOrMigrate(defaults: defaults)
        return SettingsSnapshot(
            activeChatModelID: persistedSelection.chatModelID,
            activeEmbeddingModelID: persistedSelection.embeddingModelID,
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
            networkToolsEnabled: defaults.object(forKey: UserSettingsKeys.networkToolsEnabled) as? Bool ?? false,
            autoDownloadFleetModels: defaults.object(forKey: UserSettingsKeys.autoDownloadFleetModels) as? Bool ?? true,
            confirmFleetDownloads: defaults.object(forKey: UserSettingsKeys.confirmFleetDownloads) as? Bool ?? true
        )
    }
}
