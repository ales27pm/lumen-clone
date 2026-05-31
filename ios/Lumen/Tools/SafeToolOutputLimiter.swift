import Foundation

enum SafeToolOutputLimiter {
    static func clamp(_ text: String, max: Int) -> String {
        guard max > 0 else { return "" }
        guard text.count > max else { return text }
        let suffix = "\n…(truncated)"
        guard max > suffix.count else { return String(text.prefix(max)) }
        return String(text.prefix(max - suffix.count)) + suffix
    }

    static func limit(result: ToolResult, maxOutput: Int) -> ToolResult {
        ToolResult(invocationID: result.invocationID, status: result.status, displayText: clamp(result.displayText, max: maxOutput), modelText: clamp(result.modelText, max: maxOutput), structuredPayload: result.structuredPayload, privacyLevel: result.privacyLevel, metricsSummary: result.metricsSummary, errorCode: result.errorCode)
    }
}
