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
        let approval: ToolExecutionApproval = invocation.source == .userInitiated ? .userApproved : .autonomous
        let args = ToolRouteGuard.normalizedArguments(
            for: toolID,
            rawToolID: toolID,
            arguments: invocation.arguments
        )

        guard ToolRouteGuard.canExecuteTool(toolID, arguments: args, approval: approval) else {
            return result(
                invocation: invocation,
                text: ToolRouteGuard.approvalRequiredMessage(for: toolID),
                status: .requiresApproval,
                metricsSummary: "approval_required"
            )
        }

        if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
            return result(
                invocation: invocation,
                text: permissionFailure,
                status: .denied,
                metricsSummary: "permission_denied"
            )
        }

        let text: String
        switch toolID {
        case "calendar.create":
            text = await CalendarTools.createEvent(
                title: args["title"] ?? "New Event",
                startsInMinutes: Int(args["startsInMinutes"] ?? "60") ?? 60
            )
        case "calendar.list":
            text = await CalendarTools.listEvents()
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
        metricsSummary: String
    ) -> ToolResult {
        ToolResult(
            invocationID: invocation.id,
            status: status,
            displayText: text,
            modelText: text,
            structuredPayload: ["toolID": toolID, "implementation": "ProductivityLocalTool"],
            privacyLevel: definition.resultPrivacyLevel,
            metricsSummary: status == .success ? metricsSummary : "\(metricsSummary)_\(status.rawValue)",
            errorCode: status == .success ? nil : status.rawValue
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
