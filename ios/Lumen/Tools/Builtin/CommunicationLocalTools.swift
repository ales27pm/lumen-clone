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
            supportsBackgroundExecution: !catalogTool.requiresApproval,
            requiresUserApproval: catalogTool.requiresApproval,
            argumentSchemaDescription: Self.argumentSchemaDescription(from: catalogTool.description),
            resultPrivacyLevel: .sensitive,
            maxOutputCharacters: Self.maxOutputCharacters(for: canonical)
        )
    }

    func validateArguments(_ arguments: [String: String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let approval: ToolExecutionApproval = invocation.source == .userApproved ? .userApproved : .autonomous
        var args = ToolRouteGuard.normalizedArguments(
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

        if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args, isForeground: context.isForeground) {
            return result(
                invocation: invocation,
                text: permissionFailure,
                status: .denied,
                metricsSummary: "permission_denied"
            )
        }

        let text: String
        let outlookOutcome: OutlookToolOutcome?
        switch toolID {
        case "contacts.search":
            text = await ContactsTools.searchContacts(query: args["query"] ?? "")
            outlookOutcome = nil
        case "messages.draft":
            text = await ContactsTools.composeMessage(arguments: args)
            outlookOutcome = nil
        case "mail.draft":
            text = await ContactsTools.composeMail(arguments: args)
            outlookOutcome = nil
        case "phone.call":
            text = await ContactsTools.call(number: args["number"] ?? "")
            outlookOutcome = nil
        case "outlook.status":
            outlookOutcome = await OutlookTools.status()
            text = outlookOutcome?.text ?? ""
        case "outlook.folders.list":
            outlookOutcome = await OutlookTools.listFolders(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.messages.list":
            outlookOutcome = await OutlookTools.listMessages(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.messages.search":
            outlookOutcome = await OutlookTools.searchMessages(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.read":
            outlookOutcome = await OutlookTools.readMessage(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.attachments.list":
            outlookOutcome = await OutlookTools.listAttachments(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.draft.create":
            outlookOutcome = await OutlookTools.createDraft(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.mail.send":
            outlookOutcome = await OutlookTools.sendMail(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.mark_read":
            outlookOutcome = await OutlookTools.markRead(args: args, isRead: true)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.mark_unread":
            outlookOutcome = await OutlookTools.markRead(args: args, isRead: false)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.move":
            outlookOutcome = await OutlookTools.moveMessage(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.archive":
            args["destination"] = "archive"
            outlookOutcome = await OutlookTools.moveMessage(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.delete":
            outlookOutcome = await OutlookTools.deleteMessage(args: args)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.reply":
            outlookOutcome = await OutlookTools.reply(args: args, replyAll: false)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.reply_all":
            outlookOutcome = await OutlookTools.reply(args: args, replyAll: true)
            text = outlookOutcome?.text ?? ""
        case "outlook.message.forward":
            outlookOutcome = await OutlookTools.forward(args: args)
            text = outlookOutcome?.text ?? ""
        default:
            text = "Unsupported native communication tool: \(toolID)."
            outlookOutcome = nil
        }

        if let outlookOutcome {
            return result(
                invocation: invocation,
                outcome: outlookOutcome,
                metricsSummary: "native_communication_tool"
            )
        }

        return result(
            invocation: invocation,
            text: text,
            status: ToolResultStatusClassifier.status(from: text),
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
            structuredPayload: ["toolID": toolID, "implementation": "CommunicationLocalTool"],
            privacyLevel: definition.resultPrivacyLevel,
            metricsSummary: status == .success ? metricsSummary : "\(metricsSummary)_\(status.rawValue)",
            errorCode: status == .success ? nil : status.rawValue
        )
    }

    private func result(
        invocation: ToolInvocation,
        outcome: OutlookToolOutcome,
        metricsSummary: String
    ) -> ToolResult {
        var payload = outcome.structuredPayload
        payload["toolID"] = toolID
        payload["implementation"] = "CommunicationLocalTool"
        return ToolResult(
            invocationID: invocation.id,
            status: outcome.status,
            displayText: outcome.text,
            modelText: outcome.text,
            structuredPayload: payload,
            privacyLevel: definition.resultPrivacyLevel,
            metricsSummary: outcome.metricsSummary(base: metricsSummary),
            errorCode: outcome.errorCode
        )
    }

    private static func secureCategory(for canonical: String, catalogTool: ToolDefinition) -> SecureToolCategory {
        if canonical.contains("delete") || canonical.contains("archive") || canonical.contains("move") {
            return .destructiveAction
        }
        if catalogTool.requiresApproval {
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
