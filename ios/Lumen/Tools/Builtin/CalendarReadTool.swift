import Foundation
import EventKit

struct CalendarReadTool: LocalTool {
    protocol Provider { func events(start: Date, end: Date, limit: Int, titleFilter: String?) async throws -> [[String:String]] }
    struct EventKitProvider: Provider {
        func events(start: Date, end: Date, limit: Int, titleFilter: String?) async throws -> [[String : String]] {
            let store = EKEventStore()
            let pred = store.predicateForEvents(withStart: start, end: end, calendars: nil)
            return store.events(matching: pred)
                .filter { titleFilter == nil || $0.calendar.title.localizedCaseInsensitiveContains(titleFilter!) }
                .sorted { $0.startDate < $1.startDate }
                .prefix(limit)
                .map { ["title": $0.title, "start": ISO8601DateFormatter().string(from: $0.startDate), "end": ISO8601DateFormatter().string(from: $0.endDate), "calendar": $0.calendar.title, "location": String(($0.location ?? "").prefix(60))] }
        }
    }

    let definition = SecureToolDefinition(id: "calendar.read", displayName: "Read Calendar", description: "List events in date range", category: .permissionRead, requiredPermissions: [.calendars], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "{startDate?:iso8601,endDate?:iso8601,limit?:1...20,calendarTitleFilter?:string}", resultPrivacyLevel: .moderate, maxOutputCharacters: 1800)
    let provider: Provider
    init(provider: Provider = EventKitProvider()) { self.provider = provider }

    func validateArguments(_ arguments: [String : String]) throws { _ = try parse(arguments) }
    func parse(_ a:[String:String], now: Date = Date()) throws -> (Date,Date,Int,String?) {
        let limit = Int(a["limit"] ?? "10") ?? 10; guard (1...20).contains(limit) else { throw ToolExecutionError.invalidArguments("limit") }
        let f = ISO8601DateFormatter()
        let start: Date
        if let rawStart = a["startDate"] {
            guard let parsed = f.date(from: rawStart) else { throw ToolExecutionError.invalidArguments("startDate") }
            start = parsed
        } else { start = now }
        let end: Date
        if let rawEnd = a["endDate"] {
            guard let parsed = f.date(from: rawEnd) else { throw ToolExecutionError.invalidArguments("endDate") }
            end = parsed
        } else { end = now.addingTimeInterval(7*24*3600) }
        guard end > start else { throw ToolExecutionError.invalidArguments("endDate") }
        guard end.timeIntervalSince(start) <= 31*24*3600 else { throw ToolExecutionError.invalidArguments("date range max 31d") }
        let title = a["calendarTitleFilter"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let title, title.count > 120 { throw ToolExecutionError.invalidArguments("calendarTitleFilter") }
        return (start,end,limit,title)
    }

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        if !context.isForeground { return .init(invocationID: invocation.id, status: .denied, displayText: "Calendar read is unavailable in background.", modelText: "Calendar read denied in background.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "bg_denied", errorCode: "bg_denied") }
        let st = await context.permissionRegistry.currentStatus(for: .calendars)
        let gate = PermissionGate.evaluate(domain: .calendars, state: st, isForeground: context.isForeground)
        guard gate.allowed else { return .init(invocationID: invocation.id, status: .denied, displayText: gate.reason ?? "Calendar access denied.", modelText: "Calendar permission required.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "permission_denied", errorCode: "permission") }
        do {
            let (start,end,limit,title) = try parse(invocation.arguments)
            let events = try await provider.events(start: start, end: end, limit: limit, titleFilter: title)
            let lines = events.map { "- \($0["title"] ?? "(untitled)") | \($0["start"] ?? "") → \($0["end"] ?? "") | \($0["calendar"] ?? "")" }
            let text = lines.isEmpty ? "No events found in range." : lines.joined(separator: "\n")
            return SafeToolOutputLimiter.limit(result: .init(invocationID: invocation.id, status: .success, displayText: text, modelText: text, structuredPayload: ["count":"\(events.count)"], privacyLevel: .moderate, metricsSummary: "eventkit", errorCode: nil), maxOutput: definition.maxOutputCharacters)
        } catch {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid calendar query.", modelText: "Calendar input invalid.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "invalid_args", errorCode: "invalid")
        }
    }
}
