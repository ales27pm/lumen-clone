import Foundation

struct ProductivityLocalTool: LocalTool {
    static let nativeToolIDs: Set<String> = [
        "calendar.create",
        "calendar.list",
        "reminders.create",
        "reminders.list",
        "trigger.create",
        "trigger.list",
        "trigger.cancel",
        "alarm.authorization_status",
        "alarm.request_authorization",
        "alarm.schedule",
        "alarm.countdown",
        "alarm.list",
        "alarm.pause",
        "alarm.resume",
        "alarm.stop",
        "alarm.snooze",
        "alarm.cancel"
    ]

    @MainActor static var all: [ProductivityLocalTool] {
        ToolRegistry.all
            .filter { nativeToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            .map(ProductivityLocalTool.init)
    }

    let definition: SecureToolDefinition
    private let toolID: String

    init(_ catalogTool: ToolDefinition) {
        let canonical = ToolRouteGuard.canonicalToolID(catalogTool.id)
        self.toolID = canonical
        self.definition = SecureToolDefinition(
            id: canonical,
            displayName: catalogTool.name,
            description: catalogTool.description,
            category: Self.secureCategory(for: canonical, catalogTool: catalogTool),
            requiredPermissions: [],
            supportsBackgroundExecution: Self.supportsBackgroundExecution(canonical),
            requiresUserApproval: catalogTool.requiresApproval,
            argumentSchemaDescription: Self.argumentSchemaDescription(from: catalogTool.description),
            resultPrivacyLevel: .moderate,
            maxOutputCharacters: 2_400
        )
    }

    func validateArguments(_ arguments: [String: String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let approval: ToolExecutionApproval = invocation.source == .userApproved ? .userApproved : .autonomous
        let args = ToolRouteGuard.normalizedArguments(
            for: toolID,
            rawToolID: toolID,
            arguments: invocation.arguments
        )
        let calendarCreateArguments: CalendarCreateArguments?
        if toolID == "calendar.create" {
            guard let parsedArguments = Self.calendarCreateArguments(from: args) else {
                return result(
                    invocation: invocation,
                    response: CalendarTools.invalidArgumentsResponse(
                        displayText: "The calendar create request is missing a non-empty title or valid startsInMinutes argument."
                    )
                )
            }
            calendarCreateArguments = parsedArguments
        } else {
            calendarCreateArguments = nil
        }

        guard ToolRouteGuard.canExecuteTool(toolID, arguments: args, approval: approval) else {
            return result(
                invocation: invocation,
                text: ToolRouteGuard.approvalRequiredMessage(for: toolID),
                status: .requiresApproval,
                metricsSummary: "approval_required"
            )
        }

        if toolID.hasPrefix("calendar.") {
            if Self.shouldRequestCalendarPermission(toolID: toolID, isForeground: context.isForeground) {
                await requestCalendarPermissionIfNeeded()
            }
        } else {
            if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args, isForeground: context.isForeground) {
                return result(
                    invocation: invocation,
                    text: permissionFailure,
                    status: .denied,
                    metricsSummary: "permission_denied"
                )
            }
        }

        let text: String
        switch toolID {
        case "calendar.create":
            guard let calendarCreateArguments else {
                return result(
                    invocation: invocation,
                    response: CalendarTools.invalidArgumentsResponse(
                        displayText: "The calendar create request is missing a non-empty title or valid startsInMinutes argument."
                    )
                )
            }
            let response = await CalendarTools.createEventResult(
                title: calendarCreateArguments.title,
                startsInMinutes: calendarCreateArguments.startsInMinutes
            )
            return result(invocation: invocation, response: response)
        case "calendar.list":
            let response = await CalendarTools.listEventsResult(arguments: args)
            return result(invocation: invocation, response: response)
        case "reminders.create":
            text = await CalendarTools.createReminder(title: args["title"] ?? "Reminder")
        case "reminders.list":
            text = await CalendarTools.listReminders()
        case "trigger.create":
            text = await TriggerTools.create(args: args)
        case "trigger.list":
            text = await TriggerTools.list()
        case "trigger.cancel":
            text = await TriggerTools.cancel(title: args["title"] ?? args["id"] ?? "")
        case "alarm.authorization_status":
            text = await AlarmTools.authorizationStatus()
        case "alarm.request_authorization":
            text = await AlarmTools.requestAuthorization()
        case "alarm.schedule":
            text = await AlarmTools.schedule(args: args)
        case "alarm.countdown":
            text = await AlarmTools.countdown(args: args)
        case "alarm.list":
            text = await AlarmTools.list()
        case "alarm.pause":
            text = await AlarmTools.pause(id: args["id"] ?? "")
        case "alarm.resume":
            text = await AlarmTools.resume(id: args["id"] ?? "")
        case "alarm.stop":
            text = await AlarmTools.stop(id: args["id"] ?? "")
        case "alarm.snooze":
            text = await AlarmTools.snooze(id: args["id"] ?? "")
        case "alarm.cancel":
            text = await AlarmTools.cancel(id: args["id"] ?? args["title"] ?? "")
        default:
            text = "Unsupported native productivity tool: \(toolID)."
        }

        return result(
            invocation: invocation,
            text: text,
            status: ToolResultStatusClassifier.status(from: text),
            metricsSummary: "native_productivity_tool"
        )
    }

    private func result(
        invocation: ToolInvocation,
        text: String,
        status: ToolResultStatus,
        metricsSummary: String,
        structuredPayload: [String: String]? = nil,
        modelText: String? = nil,
        errorCode: String? = nil
    ) -> ToolResult {
        ToolResult(
            invocationID: invocation.id,
            status: status,
            displayText: text,
            modelText: modelText ?? text,
            structuredPayload: Self.resultPayload(toolID: toolID, structuredPayload: structuredPayload),
            privacyLevel: definition.resultPrivacyLevel,
            metricsSummary: status == .success ? metricsSummary : "\(metricsSummary)_\(status.rawValue)",
            errorCode: status == .success ? nil : (errorCode ?? status.rawValue)
        )
    }

    @MainActor
    private func requestCalendarPermissionIfNeeded() async {
        let permissions = PermissionsCenter.shared
        guard permissions.state(.calendar) == .notDetermined else { return }
        await permissions.request(.calendar)
    }

    static func resultPayload(toolID: String, structuredPayload: [String: String]?) -> [String: String] {
        [
            "toolID": toolID,
            "implementation": "ProductivityLocalTool"
        ].merging(structuredPayload ?? [:]) { canonicalValue, _ in canonicalValue }
    }

    static func shouldRequestCalendarPermission(toolID: String, isForeground: Bool) -> Bool {
        isForeground && toolID.hasPrefix("calendar.")
    }

    struct CalendarCreateArguments: Equatable {
        let title: String
        let startsInMinutes: Int
    }

    static func calendarCreateArguments(from args: [String: String]) -> CalendarCreateArguments? {
        guard let rawTitle = args["title"] else { return nil }
        let title = rawTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return nil }
        guard let rawStartsInMinutes = args["startsInMinutes"],
              let startsInMinutes = Int(rawStartsInMinutes),
              startsInMinutes >= 0,
              startsInMinutes <= CalendarTools.maximumSafeStartsInMinutes else {
            return nil
        }
        return CalendarCreateArguments(title: title, startsInMinutes: startsInMinutes)
    }

    private func result(invocation: ToolInvocation, response: CalendarTools.CalendarToolResponse) -> ToolResult {
        result(
            invocation: invocation,
            text: response.displayText,
            status: response.status,
            metricsSummary: response.metricsSummary,
            structuredPayload: response.structuredPayload,
            modelText: response.modelText,
            errorCode: response.errorCode
        )
    }

    private static func secureCategory(for canonical: String, catalogTool: ToolDefinition) -> SecureToolCategory {
        if canonical.contains("cancel") || canonical.contains("stop") {
            return .destructiveAction
        }
        if catalogTool.requiresApproval {
            return .sensitiveAction
        }
        switch canonical {
        case "calendar.list", "reminders.list", "trigger.list", "alarm.authorization_status", "alarm.list":
            return .readOnly
        default:
            return .userVisibleAction
        }
    }

    private static func supportsBackgroundExecution(_ canonical: String) -> Bool {
        switch canonical {
        case "calendar.list", "reminders.list", "trigger.list", "alarm.authorization_status", "alarm.list":
            return true
        default:
            return false
        }
    }

    private static func argumentSchemaDescription(from description: String) -> String {
        guard let range = description.range(of: "Args:") else { return "{}" }
        return String(description[range.lowerBound...])
    }
}
