import Foundation

protocol LocalTool {
    var definition: SecureToolDefinition { get }
    func validateArguments(_ arguments: [String: String]) throws
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult
}
