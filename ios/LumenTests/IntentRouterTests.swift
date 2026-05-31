import Testing
@testable import Lumen

struct IntentRouterTests {

    @Test func weatherHereRoutesToWeatherOnly() async throws {
        let decision = IntentRouter.classify("What is the weather here")
        #expect(decision.intent == .weather)
        #expect(IntentRouter.isToolAllowed("calendar.create", for: decision) == false)
        #expect(IntentRouter.isToolAllowed("web.search", for: decision) == false)
        #expect(IntentRouter.isToolAllowed("weather", for: decision))
    }

    @Test func draftEmailRequiresClarificationWhenUnderspecified() async throws {
        let decision = IntentRouter.classify("Draft a email")
        #expect(decision.intent == .emailDraft)
        #expect(decision.requiresClarification)
        #expect(decision.clarificationPrompt == "Who should I send it to, and what should it say?")
    }

    @Test func webSearchCannotUseCalendarOrReminderTools() async throws {
        let decision = IntentRouter.classify("Search web for diy underground shelter")
        #expect(decision.intent == .webSearch)
        #expect(IntentRouter.isToolAllowed("calendar.create", for: decision) == false)
        #expect(IntentRouter.isToolAllowed("reminders.create", for: decision) == false)
        #expect(IntentRouter.isToolAllowed("web.search", for: decision))
    }

    @Test func calendarPhraseRoutesToCalendarIntent() async throws {
        let decision = IntentRouter.classify("Create an event tomorrow at 5")
        #expect(decision.intent == .calendar)
    }

    @Test func unknownChatDoesNotForceTools() async throws {
        let decision = IntentRouter.classify("How are you today?")
        #expect(decision.intent == .chat || decision.intent == .unknown)
        #expect(decision.allowedToolIDs.isEmpty)
        #expect(!decision.requiresClarification)
    }

    @Test func slotAgentBlocksCalendarActionForWebSearchIntent() async throws {
        let routing = IntentRouter.classify("Search web for diy underground shelter")
        #expect(!SlotAgentService.isActionAllowed("calendar.create", routing: routing))
        #expect(SlotAgentService.isActionAllowed("web.search", routing: routing))
    }

    @Test func explicitChatGreetingRoutesToChatNoTools() async throws {
        let decision = IntentRouter.classify("Hi. How are you")
        #expect(decision.intent == .chat)
        #expect(decision.allowedToolIDs.isEmpty)
        #expect(!IntentRouter.intentRequiresTool(decision))
    }

    @Test func currentLocationPromptsRouteToMapsLocationOnly() async throws {
        let first = IntentRouter.classify("Where are we")
        #expect(first.intent == .maps)
        #expect(first.allowedToolIDs == ["location.current"])
        #expect(IntentRouter.intentRequiresTool(first))
        let second = IntentRouter.classify("Where am I")
        #expect(second.intent == .maps)
        #expect(second.allowedToolIDs == ["location.current"])
        #expect(IntentRouter.intentRequiresTool(second))
    }

    @Test func mailboxReadPromptsRouteToOutlook() async throws {
        for prompt in ["Read new emails", "Check unread emails", "Read the latest email", "Check my unread outlook emails", "Search Outlook for invoices"] {
            let decision = IntentRouter.classify(prompt)
            #expect(decision.intent == .outlook)
            #expect(IntentRouter.intentRequiresTool(decision))
        }
    }

    @Test func emailDraftAndOutlookSendDifferentiation() async throws {
        #expect(IntentRouter.classify("Draft an email to bob@example.com saying hello").intent == .emailDraft)
        #expect(IntentRouter.classify("Send an Outlook email to bob@example.com saying hello").intent == .outlook)
    }
}

extension IntentRouterTests {
    @Test func chatIntentCannotCallPhoneOrMailTools() async throws {
        let decision = IntentRouter.classify("Tell me a joke")
        #expect(decision.intent == .chat)
        #expect(!SlotAgentService.isActionAllowed("phone.call", routing: decision))
        #expect(!SlotAgentService.isActionAllowed("mail.draft", routing: decision))
    }

    @Test func memoryIntentRequiresSaveAndRecallTools() async throws {
        let decision = IntentRouter.classify("Remember that my favorite color is blue")
        #expect(decision.intent == .memory)
        let required = SlotAgentService.requiredTools(for: decision.intent)
        #expect(required == ["memory.save", "memory.recall"])
        #expect(required.isSubset(of: decision.allowedToolIDs))
    }

    @Test func concreteFileReadBeatsRAGArchitectureKeyword() async throws {
        let decision = IntentRouter.classify("Open and read architecture-notes.md.")
        #expect(decision.intent == .files)
        #expect(decision.allowedToolIDs == ["files.read"])

        let action = DeterministicToolPlanner.plan(
            routing: decision,
            prompt: "Open and read architecture-notes.md.",
            availableToolIDs: decision.allowedToolIDs
        )
        #expect(action?.tool == "files.read")
        #expect(action?.args["name"]?.stringValue == "architecture-notes.md")
    }

    @Test func ragIndexingPlansCorrectIndexTools() async throws {
        let fileDecision = IntentRouter.classify("Refresh the file retrieval index.")
        let fileAction = DeterministicToolPlanner.plan(
            routing: fileDecision,
            prompt: "Refresh the file retrieval index.",
            availableToolIDs: fileDecision.allowedToolIDs
        )
        #expect(fileDecision.intent == .rag)
        #expect(fileAction?.tool == "rag.index_files")

        let photoDecision = IntentRouter.classify("Refresh the photo retrieval index.")
        let photoAction = DeterministicToolPlanner.plan(
            routing: photoDecision,
            prompt: "Refresh the photo retrieval index.",
            availableToolIDs: photoDecision.allowedToolIDs
        )
        #expect(photoDecision.intent == .rag)
        #expect(photoAction?.tool == "rag.index_photos")
    }

    @Test func phoneCallFromContactsBeatsContactSearchAndStartsWithLookup() async throws {
        let decision = IntentRouter.classify("Place a call to Alex from contacts.")
        let action = DeterministicToolPlanner.plan(
            routing: decision,
            prompt: "Place a call to Alex from contacts.",
            availableToolIDs: decision.allowedToolIDs
        )
        #expect(decision.intent == .phoneCall)
        #expect(action?.tool == "contacts.search")
        #expect(action?.args["query"]?.stringValue == "Alex")
    }

    @Test func reminderPhrasesBeatEmbeddedCallTextMessageAndEmailVerbs() async throws {
        let prompts = [
            "Remind me to call Alex tomorrow",
            "Remind me to text Alex tomorrow",
            "Remind me to email Sarah next week",
            "Create a reminder to call the supplier",
            "Set a reminder for tomorrow to call Alex",
            "Can you set a reminder"
        ]

        for prompt in prompts {
            let decision = IntentRouter.classify(prompt)
            #expect(decision.intent == .reminder)
            #expect(decision.allowedToolIDs.contains("reminders.create"))
            #expect(decision.allowedToolIDs.contains("reminders.list"))
        }
    }
}
