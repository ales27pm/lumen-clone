import Foundation
import UserNotifications

struct NotificationTool: LocalTool {
    let definition = SecureToolDefinition(id: "notify.local", displayName: "Create Notification", description: "Post local notification", category: .userVisibleAction, requiredPermissions: [.notifications], supportsBackgroundExecution: false, requiresUserApproval: true, argumentSchemaDescription: "{title:string,body:string}", resultPrivacyLevel: .moderate, maxOutputCharacters: 300)
    func validateArguments(_ arguments: [String : String]) throws {
        guard !(arguments["title"] ?? "").isEmpty, !(arguments["body"] ?? "").isEmpty else { throw ToolExecutionError.invalidArguments("title/body required") }
    }
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        do { try validateArguments(invocation.arguments) } catch { return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid notification arguments.", modelText: "Notification arguments rejected.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "invalid_args", errorCode: "invalid_notification") }
        let content = UNMutableNotificationContent(); content.title = invocation.arguments["title"] ?? "Lumen"; content.body = invocation.arguments["body"] ?? ""; content.sound = .default
        do { try await UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: invocation.id.uuidString, content: content, trigger: nil))
            return .init(invocationID: invocation.id, status: .success, displayText: "Notification scheduled.", modelText: "Notification scheduled.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "scheduled", errorCode: nil)
        } catch { return .init(invocationID: invocation.id, status: .failed, displayText: "Could not schedule notification.", modelText: "Notification failed.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "failed", errorCode: "notification_failed") }
    }
}
