import Foundation

struct CommunicationLocalTool: LocalTool {
    static let nativeToolIDs: Set<String> = [
        "contacts.search",
        "messages.draft",
        "mail.draft",
        "phone.call",
        "outlook.status",
        "outlook.folders.list",
        "outlook.messages.list",
        "outlook.messages.search",
        "outlook.message.read",
        "outlook.attachments.list",
        "outlook.draft.create",
        "outlook.mail.send",
        "outlook.message.mark_read",
        "outlook.message.mark_unread",
        "outlook.message.move",
        "outlook.message.archive",
        "outlook.message.delete",
        "outlook.message.reply",
        "outlook.message.reply_all",
        "outlook.message.forward"
    ]

    @MainActor static var all: [CommunicationLocalTool] {
        ToolRegistry.all
            .filter { nativeToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            .map(CommunicationLocalTool.init)
    }

    let definition: SecureToolDefinition
    private let legacyToolID: String

    init(_ legacy: ToolDefinition) {
        let canonical = ToolRouteGuard.canonicalToolID(legacy.id)
        self.legacyToolID = canonical
        self.definition = SecureToolDefinition(
            id: canonical,
            displayName: legacy.name,
            description: legacy.description,
            category: Self.secureCategory(for: canonical, legacy: legacy),
            requiredPermissions: [],
            supportsBackgroundExecution: !legacy.requiresApproval,
            requiresUserApproval: legacy.requiresApproval,
            argumentSchemaDescription: Self.argumentSchemaDescription(from: legacy.description),
            resultPrivacyLevel: .sensitive,
            maxOutputCharacters: Self.maxOutputCharacters(for: canonical)
        )
    }

    func validateArguments(_ arguments: [String: String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let approval: ToolExecutionApproval = invocation.source == .userInitiated ? .userApproved : .autonomous
        var args = ToolRouteGuard.normalizedArguments(
            for: legacyToolID,
            rawToolID: legacyToolID,
            arguments: invocation.arguments
        )

        guard ToolRouteGuard.canExecuteTool(legacyToolID, arguments: args, approval: approval) else {
            return result(
                invocation: invocation,
                text: ToolRouteGuard.approvalRequiredMessage(for: legacyToolID),
                status: .requiresApproval,
                metricsSummary: "approval_required"
            )
        }

        if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: legacyToolID, arguments: args) {
            return result(
                invocation: invocation,
                text: permissionFailure,
                status: .denied,
                metricsSummary: "permission_denied"
            )
        }

        let text: String
        switch legacyToolID {
        case "contacts.search":
            text = await ContactsTools.searchContacts(query: args["query"] ?? "")
        case "messages.draft":
            text = await ContactsTools.composeMessage(arguments: args)
        case "mail.draft":
            text = await ContactsTools.composeMail(arguments: args)
        case "phone.call":
            text = await ContactsTools.call(number: args["number"] ?? "")
        case "outlook.status":
            text = await OutlookTools.status()
        case "outlook.folders.list":
            text = await OutlookTools.listFolders(args: args)
        case "outlook.messages.list":
            text = await OutlookTools.listMessages(args: args)
        case "outlook.messages.search":
            text = await OutlookTools.searchMessages(args: args)
        case "outlook.message.read":
            text = await OutlookTools.readMessage(args: args)
        case "outlook.attachments.list":
            text = await OutlookTools.listAttachments(args: args)
        case "outlook.draft.create":
            text = await OutlookTools.createDraft(args: args)
        case "outlook.mail.send":
            text = await OutlookTools.sendMail(args: args)
        case "outlook.message.mark_read":
            text = await OutlookTools.markRead(args: args, isRead: true)
        case "outlook.message.mark_unread":
            text = await OutlookTools.markRead(args: args, isRead: false)
        case "outlook.message.move":
            text = await OutlookTools.moveMessage(args: args)
        case "outlook.message.archive":
            args["destination"] = "archive"
            text = await OutlookTools.moveMessage(args: args)
        case "outlook.message.delete":
            text = await OutlookTools.deleteMessage(args: args)
        case "outlook.message.reply":
            text = await OutlookTools.reply(args: args, replyAll: false)
        case "outlook.message.reply_all":
            text = await OutlookTools.reply(args: args, replyAll: true)
        case "outlook.message.forward":
            text = await OutlookTools.forward(args: args)
        default:
            text = "Tool unavailable pending native communication migration: \(legacyToolID)."
        }

        return result(
            invocation: invocation,
            text: text,
            status: LegacyToolExecutorLocalTool.status(from: text),
            metricsSummary: "native_communication_tool"
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
            structuredPayload: ["toolID": legacyToolID, "implementation": "CommunicationLocalTool"],
            privacyLevel: definition.resultPrivacyLevel,
            metricsSummary: status == .success ? metricsSummary : "\(metricsSummary)_\(status.rawValue)",
            errorCode: status == .success ? nil : status.rawValue
        )
    }

    private static func secureCategory(for canonical: String, legacy: ToolDefinition) -> SecureToolCategory {
        if canonical.contains("delete") || canonical.contains("archive") || canonical.contains("move") {
            return .destructiveAction
        }
        if legacy.requiresApproval {
            return .sensitiveAction
        }
        return .readOnly
    }

    private static func maxOutputCharacters(for canonical: String) -> Int {
        switch canonical {
        case "outlook.message.read":
            return 6_000
        case "outlook.messages.list", "outlook.messages.search":
            return 4_000
        default:
            return 2_400
        }
    }

    private static func argumentSchemaDescription(from description: String) -> String {
        guard let range = description.range(of: "Args:") else { return "{}" }
        return String(description[range.lowerBound...])
    }
}
