import Foundation
import SwiftData

nonisolated struct AgentTraceCorrelation: Codable, Sendable, Equatable, Hashable {
    let scenarioID: String?
    let e2eRunID: UUID?
    let agentRunID: UUID?
    let conversationID: UUID?
    let turnID: UUID?

    init(
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil
    ) {
        self.scenarioID = scenarioID
        self.e2eRunID = e2eRunID
        self.agentRunID = agentRunID
        self.conversationID = conversationID
        self.turnID = turnID
    }

    var diagnosticText: String {
        [
            "scenarioID=\(scenarioID ?? "nil")",
            "e2eRunID=\(e2eRunID?.uuidString ?? "nil")",
            "agentRunID=\(agentRunID?.uuidString ?? "nil")",
            "conversationID=\(conversationID?.uuidString ?? "nil")",
            "turnID=\(turnID?.uuidString ?? "nil")"
        ].joined(separator: ",")
    }

    var hasAnyIdentifier: Bool {
        scenarioID?.isEmpty == false || e2eRunID != nil || agentRunID != nil || conversationID != nil || turnID != nil
    }
}

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
    let relevantMemories: [MemoryContextItem]
    let attachments: [ChatAttachment]
    let task: AssistantTaskKind
    let source: AgentKernelSource
    let options: AgentKernelOptions
    let traceCorrelation: AgentTraceCorrelation?

    init(
        id: UUID = UUID(),
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        userMessage: String,
        history: [AgentKernelMessage] = [],
        systemPrompt: String = "",
        relevantMemories: [MemoryContextItem] = [],
        attachments: [ChatAttachment] = [],
        task: AssistantTaskKind = .chat,
        source: AgentKernelSource = .chat,
        options: AgentKernelOptions = .chat,
        traceCorrelation: AgentTraceCorrelation? = nil
    ) {
        self.id = id
        self.conversationID = conversationID
        self.turnID = turnID
        self.userMessage = userMessage
        self.history = history
        self.systemPrompt = systemPrompt
        self.relevantMemories = relevantMemories
        self.attachments = attachments
        self.task = task
        self.source = source
        self.options = options
        self.traceCorrelation = traceCorrelation
    }
}

nonisolated struct AgentKernelMessage: Codable, Sendable, Equatable, Hashable {
    enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool

        init(messageRole: MessageRole) {
            switch messageRole {
            case .system: self = .system
            case .user: self = .user
            case .assistant: self = .assistant
            case .tool: self = .tool
            }
        }

        var messageRole: MessageRole {
            switch self {
            case .system: return .system
            case .user: return .user
            case .assistant: return .assistant
            case .tool: return .tool
            }
        }
    }

    let role: Role
    let content: String

    init(role: Role, content: String) {
        self.role = role
        self.content = content
    }

    init(messageRole: MessageRole, content: String) {
        self.init(role: Role(messageRole: messageRole), content: content)
    }
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
    let temperature: Double
    let topP: Double
    let repetitionPenalty: Double
    let maxTokens: Int
    let forceModelBackedToolPlanning: Bool

    init(
        allowHeavyRuntime: Bool,
        allowDegradedMode: Bool,
        requireUserVisibleFinal: Bool,
        diagnosticsEnabled: Bool,
        maxSteps: Int,
        prefersFoundationModels: Bool,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        repetitionPenalty: Double = 1.1,
        maxTokens: Int = 1024,
        forceModelBackedToolPlanning: Bool = false
    ) {
        self.allowHeavyRuntime = allowHeavyRuntime
        self.allowDegradedMode = allowDegradedMode
        self.requireUserVisibleFinal = requireUserVisibleFinal
        self.diagnosticsEnabled = diagnosticsEnabled
        self.maxSteps = max(1, maxSteps)
        self.prefersFoundationModels = prefersFoundationModels
        self.temperature = min(max(temperature, 0.0), 2.0)
        self.topP = min(max(topP, 0.0), 1.0)
        self.repetitionPenalty = min(max(repetitionPenalty, 0.1), 3.0)
        self.maxTokens = max(1, maxTokens)
        self.forceModelBackedToolPlanning = forceModelBackedToolPlanning
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            allowHeavyRuntime: try container.decode(Bool.self, forKey: .allowHeavyRuntime),
            allowDegradedMode: try container.decode(Bool.self, forKey: .allowDegradedMode),
            requireUserVisibleFinal: try container.decode(Bool.self, forKey: .requireUserVisibleFinal),
            diagnosticsEnabled: try container.decode(Bool.self, forKey: .diagnosticsEnabled),
            maxSteps: try container.decode(Int.self, forKey: .maxSteps),
            prefersFoundationModels: try container.decode(Bool.self, forKey: .prefersFoundationModels),
            temperature: try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0.7,
            topP: try container.decodeIfPresent(Double.self, forKey: .topP) ?? 0.9,
            repetitionPenalty: try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? 1.1,
            maxTokens: try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 1024,
            forceModelBackedToolPlanning: try container.decodeIfPresent(Bool.self, forKey: .forceModelBackedToolPlanning) ?? false
        )
    }

    private enum CodingKeys: String, CodingKey {
        case allowHeavyRuntime
        case allowDegradedMode
        case requireUserVisibleFinal
        case diagnosticsEnabled
        case maxSteps
        case prefersFoundationModels
        case temperature
        case topP
        case repetitionPenalty
        case maxTokens
        case forceModelBackedToolPlanning
    }

    static let chat = AgentKernelOptions(
        allowHeavyRuntime: true,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 8,
        prefersFoundationModels: true,
        maxTokens: 1024
    )

    static let voice = AgentKernelOptions(
        allowHeavyRuntime: true,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 6,
        prefersFoundationModels: true,
        maxTokens: 1024
    )

    static let headless = AgentKernelOptions(
        allowHeavyRuntime: false,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 4,
        prefersFoundationModels: false,
        maxTokens: 512
    )

    static let diagnostics = AgentKernelOptions(
        allowHeavyRuntime: false,
        allowDegradedMode: true,
        requireUserVisibleFinal: true,
        diagnosticsEnabled: true,
        maxSteps: 3,
        prefersFoundationModels: false,
        maxTokens: 512
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
    /// Temporary event-shape adapter while UI entrypoints finish adopting
    /// kernel-native events directly.
    nonisolated var legacyAgentEvent: AgentEvent? {
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
        case .chat, .voice, .appIntent, .diagnostics, .benchmark:
            return true
        case .trigger:
            return false
        }
    }
}
