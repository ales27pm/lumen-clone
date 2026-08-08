import Testing
@testable import Lumen

struct ToolApprovalBoundaryTests {
    private let expectedApprovalTools: Set<String> = [
        "calendar.create","reminders.create","messages.draft","mail.draft","outlook.draft.create","outlook.mail.send",
        "outlook.message.mark_read","outlook.message.mark_unread","outlook.message.move","outlook.message.archive",
        "outlook.message.delete","outlook.message.reply","outlook.message.reply_all","outlook.message.forward","phone.call",
        "camera.capture","trigger.create","trigger.cancel","alarm.request_authorization","alarm.schedule","alarm.countdown",
        "alarm.pause","alarm.resume","alarm.stop","alarm.snooze","alarm.cancel","rag.index_files","rag.index_photos"
    ]

    @Test func requiresApprovalMatrixMatchesRegistry() {
        let actual = Set(ToolRegistry.all.filter(\.requiresApproval).map(\.id))
        #expect(actual == expectedApprovalTools)
    }

    @Test func capabilityContractsCoverRegistryArgumentsAndPolicy() {
        let expectedArgumentlessTools: Set<String> = [
            "calendar.list", "reminders.list", "outlook.status", "location.current", "camera.capture",
            "health.summary", "motion.activity", "rag.index_files", "trigger.list",
            "alarm.authorization_status", "alarm.request_authorization", "alarm.list"
        ]
        let runtimeTools = Dictionary(
            uniqueKeysWithValues: LiveRuntimeToolRegistryProvider().currentToolDefinitions().map { ($0.id, $0) }
        )

        var missingContracts: [String] = []
        for tool in ToolRegistry.all {
            let contract = tool.capabilityContract
            let runtimeTool = runtimeTools[tool.id]
            #expect(runtimeTool != nil)
            #expect(contract.toolID == tool.id)
            #expect(contract.requiresApproval == tool.requiresApproval)
            #expect(contract.confirmationMode == (tool.requiresApproval ? .userApproval : .none))
            #expect(contract.permissionKey == tool.permissionKey)
            #expect(contract.runtimeArguments == runtimeTool?.arguments)
            if !expectedArgumentlessTools.contains(tool.id), contract.arguments.isEmpty {
                missingContracts.append(tool.id)
            }
        }

        #expect(missingContracts.isEmpty)
    }

    @Test func capabilityContractsUseTypedArgumentsForAmbiguousValues() {
        let tools = Dictionary(
            uniqueKeysWithValues: LiveRuntimeToolRegistryProvider().currentToolDefinitions().map { ($0.id, $0) }
        )
        let folderArgs = Dictionary(uniqueKeysWithValues: (tools["outlook.folders.list"]?.arguments ?? []).map { ($0.name, $0) })
        let messageListArgs = Dictionary(uniqueKeysWithValues: (tools["outlook.messages.list"]?.arguments ?? []).map { ($0.name, $0) })
        let alarmArgs = Dictionary(uniqueKeysWithValues: (tools["alarm.schedule"]?.arguments ?? []).map { ($0.name, $0) })
        let triggerArgs = Dictionary(uniqueKeysWithValues: (tools["trigger.create"]?.arguments ?? []).map { ($0.name, $0) })

        #expect(folderArgs["includeHidden"]?.type == "bool")
        #expect(folderArgs["false"] == nil)
        #expect(messageListArgs["unreadOnly"]?.type == "bool")
        #expect(alarmArgs["repeats"]?.type == "bool")
        #expect(triggerArgs["inMinutes"]?.type == "number")
        #expect(triggerArgs["intervalSeconds"]?.type == "number")
        #expect(triggerArgs["schedule"]?.type == "enum")
        #expect(triggerArgs["schedule"]?.allowedValues == ["absolute", "interval", "relative"])
    }

    @MainActor @Test func ragReindexToolsAreDestructiveAndApprovalGated() throws {
        let secureTools = Dictionary(uniqueKeysWithValues: KnowledgeLocalTool.all.map { ($0.definition.id, $0.definition) })

        for id in ["rag.index_files", "rag.index_photos"] {
            let catalog = try #require(ToolRegistry.find(id: id))
            let secure = try #require(secureTools[id])
            #expect(catalog.requiresApproval)
            #expect(secure.requiresUserApproval)
            #expect(secure.category == .destructiveAction)
            #expect(!secure.supportsBackgroundExecution)
        }
    }

    @Test func finalizerRejectsApprovalToolObservationWithoutTrustedApproval() throws {
        let trigger = try #require(ToolRegistry.find(id: "trigger.create"))
        let blocked = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .trigger,
            tool: trigger,
            observation: "Runs tonight at 8 PM.",
            originalPrompt: "Schedule a trigger tonight.",
            trustedApprovalCaptured: false
        )

        #expect(blocked.accepted == false)
        #expect(blocked.text == nil)
        #expect(blocked.rejectionReason == "approval-required")
        #expect(ToolObservationFinalizer.finalizerCoverageKind(for: trigger) == "action-only-approval-boundary")

        let approved = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .trigger,
            tool: trigger,
            observation: "Runs tonight at 8 PM.",
            originalPrompt: "Schedule a trigger tonight.",
            trustedApprovalCaptured: true
        )
        #expect(approved.accepted == true)
        #expect(approved.text?.lowercased().contains("trigger scheduled") == true)
    }
}
