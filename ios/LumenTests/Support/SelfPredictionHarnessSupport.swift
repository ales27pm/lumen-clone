#if SELF_PREDICTION_PORTABLE_TESTS
import Foundation
import SwiftUI

// Standalone Mac regression collaborators only. Inactive in app/Xcode targets.
// The runner uses the real ledger, parser, argument validator, tool registry,
// approval policy and result limiter. Permission/platform adapters, telemetry
// stores and the unused default native-tool catalogue below are doubles.
// These tests do not establish live permissions, model inference or device IO.
@MainActor final class AppState {}
enum Theme { static let textSecondary = Color.secondary }

@MainActor final class PermissionRegistry {
    static let shared = PermissionRegistry()
    func diagnostics() async -> [PermissionDomain: AssistantPermissionState] { [:] }
}

actor RuntimeMetricsStore { static let shared = RuntimeMetricsStore() }
actor ToolMetricsRecorder {
    init(store: RuntimeMetricsStore = .shared) {}
    @discardableResult
    func record(toolID: ToolID, status: ToolResultStatus, success: Bool,
                errorCode: String? = nil, memoryWarningCount: Int = 0) async -> Bool { true }
}

@MainActor final class MemoryPressureMonitor {
    static let shared = MemoryPressureMonitor()
    let warningCount = 0
}

// Integration tests always supply explicit tools. Do not turn absent native
// adapters into a pretend successful execution in this standalone target.
struct PortableUnusedNativeTool: LocalTool {
    var definition: SecureToolDefinition {
        .init(id: "portable.unavailable", displayName: "Unavailable", description: "Platform double",
              category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: false,
              requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .low,
              maxOutputCharacters: 100)
    }
    func validateArguments(_ arguments: [String: String]) throws {}
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        .init(invocationID: invocation.id, status: .unavailable, displayText: "Platform double",
              modelText: "Platform double", structuredPayload: nil, privacyLevel: .low,
              metricsSummary: "portable_unavailable", errorCode: "portable_unavailable")
    }
}
typealias DeviceStatusTool = PortableUnusedNativeTool
typealias MemorySearchTool = PortableUnusedNativeTool
typealias RAGSearchTool = PortableUnusedNativeTool
typealias CalendarReadTool = PortableUnusedNativeTool
typealias ContactsLookupTool = PortableUnusedNativeTool
typealias LocationSnapshotTool = PortableUnusedNativeTool
typealias OpenURLTool = PortableUnusedNativeTool
typealias NotificationTool = PortableUnusedNativeTool
enum ProductivityLocalTool { static let all: [any LocalTool] = [] }
enum CommunicationLocalTool { static let all: [any LocalTool] = [] }
enum LocationMediaHealthLocalTool { static let all: [any LocalTool] = [] }
enum KnowledgeLocalTool { static let all: [any LocalTool] = [] }

// The portable context tests use real HostSelfModelContext, snapshot shapes,
// context rendering and PromptAssembler, with empty attachments and memories.
// These unrelated adapters must never turn accidental IO/trace use into success.
nonisolated enum AttachmentResolver {
    struct AttachmentExtractionResult {
        let text: String?
        let mode: String
        let diagnostic: String?
    }
    static func extractTextWithDiagnostics(_ attachment: ChatAttachment) -> AttachmentExtractionResult {
        preconditionFailure("Attachment IO is outside the portable self-prediction suite")
    }
}

nonisolated enum AgentBehaviorTrace {
    struct SelfModelDecisionSummary {
        static func fromHost(snapshot: SelfModelSnapshot, included: Bool,
                             selectedToolID: String?, requiresApproval: Bool?, approvalMode: String?) -> SelfModelDecisionSummary {
            preconditionFailure("Trace production is outside the portable self-prediction suite")
        }
    }
}
#endif
