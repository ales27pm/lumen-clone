import Foundation
import SwiftData

@Model
final class ChatMessage {
    var id: UUID = UUID()
    var role: String = "user"
    var content: String = ""
    var createdAt: Date = Date()
    var toolName: String?
    var toolStatus: String?
    var toolResult: String?
    var agentStepsJSON: String?
    var wasStopped: Bool = false
    var visibleContent: String?
    var reasoningTrace: String?
    var rawModelOutput: String?
    var developerTraceID: UUID?
    var developerTraceJSON: String?
    var conversation: Conversation?

    init(
        role: MessageRole,
        content: String,
        toolName: String? = nil,
        toolStatus: ToolStatus? = nil,
        toolResult: String? = nil,
        agentSteps: [AgentStep] = [],
        wasStopped: Bool = false,
        visibleContent: String? = nil,
        reasoningTrace: String? = nil,
        rawModelOutput: String? = nil,
        developerTraceID: UUID? = nil,
        developerTrace: DeveloperTrace? = nil
    ) {
        self.role = role.rawValue
        self.content = content
        self.toolName = toolName
        self.toolStatus = toolStatus?.rawValue
        self.toolResult = toolResult
        self.agentStepsJSON = AgentStepCodec.encode(agentSteps)
        self.wasStopped = wasStopped
        self.visibleContent = visibleContent
        self.reasoningTrace = reasoningTrace
        self.rawModelOutput = rawModelOutput
        self.developerTraceID = developerTraceID ?? developerTrace?.id
        self.developerTraceJSON = developerTrace.flatMap(DeveloperTraceCodec.encode)
    }

    var agentSteps: [AgentStep] {
        get { AgentStepCodec.decode(agentStepsJSON) }
        set { agentStepsJSON = AgentStepCodec.encode(newValue) }
    }

    var messageRole: MessageRole { MessageRole(rawValue: role) ?? .user }
    var status: ToolStatus? { toolStatus.flatMap(ToolStatus.init(rawValue:)) }
    var assistantRenderContent: String { visibleContent ?? content }
    var developerTrace: DeveloperTrace? { DeveloperTraceCodec.decode(developerTraceJSON) }
}

enum MessageRole: String, Codable, CaseIterable, Sendable {
    case system, user, assistant, tool
}

enum ToolStatus: String, Codable, Sendable {
    case pendingApproval
    case running
    case completed
    case denied
    case failed
}
