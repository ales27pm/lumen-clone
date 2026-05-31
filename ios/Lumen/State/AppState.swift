import SwiftUI
import Observation

/// Top-level observable app state. This is a thin facade that owns a
/// `UserSettings` (persistent) and a `RuntimeState` (ephemeral). Views can
/// observe either sub-object directly or go through `AppState` for convenience
/// during the transition from the old monolithic state.
@Observable
final class AppState {
    let settings: UserSettings
    let runtime: RuntimeState

    init(settings: UserSettings = UserSettings(), runtime: RuntimeState = RuntimeState()) {
        self.settings = settings
        self.runtime = runtime
    }

    // MARK: - Forwarded persistent settings (for existing call sites)

    var activeChatModelID: String? {
        get { settings.activeChatModelID }
        set { settings.activeChatModelID = newValue }
    }
    var activeEmbeddingModelID: String? {
        get { settings.activeEmbeddingModelID }
        set { settings.activeEmbeddingModelID = newValue }
    }
    var enabledToolIDs: Set<String> {
        get { settings.enabledToolIDs }
        set { settings.enabledToolIDs = newValue }
    }
    var systemPrompt: String {
        get { settings.systemPrompt }
        set { settings.systemPrompt = newValue }
    }
    var temperature: Double {
        get { settings.temperature }
        set { settings.temperature = newValue }
    }
    var topP: Double {
        get { settings.topP }
        set { settings.topP = newValue }
    }
    var repetitionPenalty: Double {
        get { settings.repetitionPenalty }
        set { settings.repetitionPenalty = newValue }
    }
    var contextSize: Int {
        get { settings.contextSize }
        set { settings.contextSize = newValue }
    }
    var maxTokens: Int {
        get { settings.maxTokens }
        set { settings.maxTokens = newValue }
    }
    var autoMemory: Bool {
        get { settings.autoMemory }
        set { settings.autoMemory = newValue }
    }
    var selectedPresetID: String {
        get { settings.selectedPresetID }
        set { settings.selectedPresetID = newValue }
    }
    var voiceID: String? {
        get { settings.voiceID }
        set { settings.voiceID = newValue }
    }
    var speakingRate: Double {
        get { settings.speakingRate }
        set { settings.speakingRate = newValue }
    }
    var handsFree: Bool {
        get { settings.handsFree }
        set { settings.handsFree = newValue }
    }
    var maxAgentSteps: Int {
        get { settings.maxAgentSteps }
        set { settings.maxAgentSteps = newValue }
    }
    var showThinkingByDefault: Bool {
        get { settings.showThinkingByDefault }
        set { settings.showThinkingByDefault = newValue }
    }
    var agentModeEnabled: Bool {
        get { settings.agentModeEnabled }
        set { settings.agentModeEnabled = newValue }
    }
    var developerTraceModeEnabled: Bool {
        get { settings.developerTraceModeEnabled }
        set { settings.developerTraceModeEnabled = newValue }
    }
    var developerReasoningCaptureEnabled: Bool {
        get { settings.developerReasoningCaptureEnabled }
        set { settings.developerReasoningCaptureEnabled = newValue }
    }
    var autoDownloadFleetModels: Bool {
        get { settings.autoDownloadFleetModels }
        set { settings.autoDownloadFleetModels = newValue }
    }
    var confirmFleetDownloads: Bool {
        get { settings.confirmFleetDownloads }
        set { settings.confirmFleetDownloads = newValue }
    }
    // MARK: - Forwarded runtime state

    var isGenerating: Bool {
        get { runtime.isGenerating }
        set { runtime.isGenerating = newValue }
    }

    // MARK: - Actions

    func toggleTool(_ id: String) {
        settings.toggleTool(id)
    }

    func applyPreset(_ preset: Preset) {
        settings.applyPreset(preset)
    }

    var snapshot: SettingsSnapshot { settings.snapshot }
}

nonisolated struct DownloadProgress: Sendable {
    var fractionCompleted: Double
    var bytesReceived: Int64
    var totalBytes: Int64
    var state: State

    enum State: Sendable { case queued, downloading, paused, completed, failed(String) }
}

nonisolated struct Preset: Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    let icon: String
    let prompt: String
    let temperature: Double
}

nonisolated enum Presets {
    static let general = Preset(
        id: "general",
        name: "General",
        icon: "sparkles",
        prompt: "You are Lumen, a helpful, concise on-device AI assistant. You have access to tools for calendar, reminders, contacts, location, messages, photos, camera, health, and motion. When a user request requires real-world actions or live data, call the appropriate tool using the tool-calling format. Respect user privacy — all data stays on this device.",
        temperature: 0.7
    )
    static let coder = Preset(
        id: "coder",
        name: "Coder",
        icon: "chevron.left.forwardslash.chevron.right",
        prompt: "You are Lumen in coder mode. Give precise, modern code answers. Prefer Swift/SwiftUI for iOS. Use fenced code blocks. Be terse.",
        temperature: 0.3
    )
    static let researcher = Preset(
        id: "researcher",
        name: "Researcher",
        icon: "books.vertical.fill",
        prompt: "You are Lumen in researcher mode. Be thorough, cite reasoning, and structure answers with headings. Think step-by-step.",
        temperature: 0.5
    )
    static let journal = Preset(
        id: "journal",
        name: "Journal",
        icon: "book.closed.fill",
        prompt: "You are Lumen in journal mode. Be warm, reflective, and ask open-ended questions. Help the user process thoughts privately.",
        temperature: 0.8
    )
    static let roleplay = Preset(
        id: "roleplay",
        name: "Roleplay",
        icon: "theatermasks.fill",
        prompt: "You are Lumen in creative mode. Be imaginative, vivid, and dramatic. Stay in character.",
        temperature: 1.0
    )

    static let all: [Preset] = [general, coder, researcher, journal, roleplay]
    static func find(id: String) -> Preset { all.first { $0.id == id } ?? general }
}
