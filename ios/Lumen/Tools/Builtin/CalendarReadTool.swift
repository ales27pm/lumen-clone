import Foundation

struct CalendarReadTool: LocalTool {
    let definition = SecureToolDefinition(id: "calendar.read", displayName: "Read Calendar", description: "List events in date range", category: .permissionRead, requiredPermissions: [], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "{startDate?:iso8601,endDate?:iso8601,limit?:1...20,calendarTitleFilter?:string}", resultPrivacyLevel: .moderate, maxOutputCharacters: 1800)
    let provider: CalendarTools.EventProvider
    init() { self.provider = CalendarTools.EventKitProvider() }
    init(provider: CalendarTools.EventProvider) { self.provider = provider }

    func validateArguments(_ arguments: [String : String]) throws { _ = try parse(arguments) }
    func parse(_ arguments: [String: String], now: Date = Date()) throws -> CalendarTools.CalendarListQuery {
        try CalendarTools.parseListArguments(arguments, now: now)
    }

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        if !context.isForeground { return .init(invocationID: invocation.id, status: .denied, displayText: "Calendar read is unavailable in background.", modelText: "Calendar read denied in background.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "bg_denied", errorCode: "bg_denied") }
        let response = await CalendarTools.listEventsResult(arguments: invocation.arguments, provider: provider)
        return SafeToolOutputLimiter.limit(
            result: .init(
                invocationID: invocation.id,
                status: response.status,
                displayText: response.displayText,
                modelText: response.modelText,
                structuredPayload: response.structuredPayload,
                privacyLevel: .moderate,
                metricsSummary: response.metricsSummary,
                errorCode: response.errorCode
            ),
            maxOutput: definition.maxOutputCharacters
        )
    }
}
