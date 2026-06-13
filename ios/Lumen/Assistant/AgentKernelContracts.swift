import Foundation
import SwiftData

/// Canonical request envelope for the Agent Kernel migration.
///
/// All user-visible, voice, AppIntent, trigger, diagnostic, and benchmark turns
/// should eventually enter the assistant through `AssistantKernel.run(...)` using
/// this request. Runtime-specific and legacy service options should be adapted
/// before the kernel boundary, not threaded around it.
nonisolated struct AgentKernelRequest: Sendable, Equatable, Identifiable {
    let id: UUID
    let conversationID: UUID?
    let turnID: UUID?
    let userMessage: String
    let history: [AgentKernelMessage]
    let systemPrompt: String
    let task: AssistantTaskKind
    let source: AgentKernelSource
    let options: AgentKernelOptions

    init(
        id: UUID = UUID(),
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        userMessage: String,
        history: [AgentKernelMessage] = [],
        systemPrompt: String = "",
        task: AssistantTaskKind = .chat,
        source: AgentKernelSource = .chat,
        options: AgentKernelOptions = .chat
    ) {
        self.id = id
        self.conversationID = conversationID
        self.turnID = turnID
        self.userMessage = userMessage
        self.history = history
        self.systemPrompt = systemPrompt
        self.task = task
        self.source = source
        self.options = options
    }
}

nonisolated struct AgentKernelMessage: Codable, Sendable, Equatable, Hashable {
    enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    let role: Role
    let content: String
}

nonisolated enum AgentKernelSource: String, Codable, Sendable, Equatable {
    case chat
    case voice
    case appIntent
    case trigger
    case diagnostics
    case benchmark
}

nonisolated struct AgentKernelOptions: Codable, Sendable, Equatable {
    let allowHeavyRuntime: Bool
    let allowDegradedMode: Bool
    let requireUserVisibleFinal: Bool
    let diagnosticsEnabled: Bool
    let maxSteps: Int
    let prefersFoundationModels: Bool

    init(
        allowHeavyRuntime: Bool,
        allowDegradedMode: Bool,
        requireUserVisibleFinal: Bool,
        diagnosticsEnabled: Bool,
        maxSteps: Int,
        prefersFoundationModels: Bool
    ) {
        self.allowHeavyRuntime = allowHeavyRuntime
        self.allowDegradedMode = allowDegradedMode
        self.requireUserVisibleFinal = requireUserVisibleFinal
        self.diagnosticsEnabled = diagnosticsEnabled
        self.maxSteps = max(1, maxSteps)
        self.prefersFoundationModels = prefersFoundationModels
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            allowHeavyRuntime: try container.decode(Bool.self, forKey: .allowHeavyRuntime),
            allowDegradedMode: try container.decode(Bool.self, forKey: .allowDegradedMode),
            requireUserVisibleFinal: try container.decode(Bool.self, forKey: .requireUserVisibleFinal),
            diagnosticsEnabled: try container.decode(Bool.self, forKey: .diagnosticsEnabled),
            maxSteps: try container.decode(Int.self, forKey: .maxSteps),
            prefersFoundationModels: try container.decode(Bool.self, forKey: .prefersFoundationModels)
        )
    }

    private enum CodingKeys: String, CodingKey {
        case allowHeavyRuntime
        case allowDegradedMode
        case requireUserVisibleFinal
        case diagnosticsEnabled
        case maxSteps
        case prefersFoundationModels
    }

    static let chat = AgentKernelOptions(
        allowHeavyRuntime: true,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 8,
        prefersFoundationModels: true
    )

    static let voice = AgentKernelOptions(
        allowHeavyRuntime: true,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 6,
        prefersFoundationModels: true
    )

    static let headless = AgentKernelOptions(
        allowHeavyRuntime: false,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 4,
        prefersFoundationModels: false
    )

    static let diagnostics = AgentKernelOptions(
        allowHeavyRuntime: false,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 3,
        prefersFoundationModels: false
    )
}

nonisolated struct AgentKernelDiagnosticEvent: Codable, Sendable, Equatable, Identifiable {
    let id: UUID
    let createdAt: Date
    let stage: String
    let message: String
    let metadata: [String: String]

    init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        stage: String,
        message: String,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.createdAt = createdAt
        self.stage = stage
        self.message = message
        self.metadata = metadata
    }
}

nonisolated enum AgentKernelEvent: Sendable {
    case step(AgentStep)
    case stepDelta(id: UUID, text: String)
    case token(String)
    case finalDelta(String)
    case toolInvocation(ToolInvocation)
    case toolResult(ToolResult)
    case diagnostic(AgentKernelDiagnosticEvent)
    case final(String)
    case done(finalText: String, steps: [AgentStep])
    case error(String)
}

@MainActor
protocol AgentKernelRunning: AnyObject {
    func run(_ request: AgentKernelRequest, modelContext: ModelContext?) -> AsyncStream<AgentKernelEvent>
}

extension AgentKernelEvent {
    /// Temporary compatibility shim while ChatView/Voice/Headless entrypoints
    /// are migrated from `AgentEvent` to kernel-native events.
    var legacyAgentEvent: AgentEvent? {
        switch self {
        case .step(let step):
            return .step(step)
        case .stepDelta(let id, let text):
            return .stepDelta(id: id, text: text)
        case .token(let text), .finalDelta(let text):
            return .finalDelta(text)
        case .final:
            return nil
        case .done(let finalText, let steps):
            return .done(finalText: finalText, steps: steps)
        case .error(let message):
            return .error(message)
        case .toolInvocation, .toolResult, .diagnostic:
            return nil
        }
    }
}

extension AgentKernelSource {
    var toolInvocationSource: ToolInvocationSource {
        switch self {
        case .chat, .voice, .diagnostics, .benchmark:
            return .modelProposed
        case .appIntent:
            return .appIntent
        case .trigger:
            return .backgroundTrigger
        }
    }

    var isForeground: Bool {
        switch self {
        case .chat, .voice, .diagnostics, .benchmark:
            return true
        case .appIntent, .trigger:
            return false
        }
    }
}
