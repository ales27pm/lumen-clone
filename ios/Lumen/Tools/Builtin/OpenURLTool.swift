import Foundation
import UIKit

struct OpenURLTool: LocalTool {
    let definition = SecureToolDefinition(id: "open.url", displayName: "Open URL", description: "Open a URL with approval", category: .sensitiveAction, requiredPermissions: [], supportsBackgroundExecution: false, requiresUserApproval: true, argumentSchemaDescription: "{url:string}", resultPrivacyLevel: .moderate, maxOutputCharacters: 300)
    func validateArguments(_ arguments: [String : String]) throws {
        guard let raw = arguments["url"], let url = URL(string: raw), let scheme = url.scheme else { throw ToolExecutionError.invalidArguments("Missing url") }
        if !["http","https"].contains(scheme.lowercased()) { throw ToolExecutionError.invalidArguments("Only http/https allowed") }
    }
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        do { try validateArguments(invocation.arguments) } catch {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid URL.", modelText: "URL rejected.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "invalid_args", errorCode: "invalid_url")
        }
        guard let raw = invocation.arguments["url"], let url = URL(string: raw) else { return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid URL.", modelText: "URL rejected.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "invalid_args", errorCode: "invalid_url") }
        let opened = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            Task { @MainActor in
                UIApplication.shared.open(url, options: [:]) { success in
                    continuation.resume(returning: success)
                }
            }
        }
        guard opened else {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Unable to open URL.", modelText: "URL open was rejected by the system.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "open_rejected", errorCode: "open_rejected")
        }
        return .init(invocationID: invocation.id, status: .success, displayText: "Opened URL.", modelText: "Opened URL successfully.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "opened", errorCode: nil)
    }
}
