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

    @Test func nearbySupportGroupMeetingRoutesToMapsNotCalendar() async throws {
        let prompts = [
            "Find the nearest Alcoholics Anonymous meeting tonight",
            "I have to go to an alcoholic anonymous meeting tonight. Can you help me find the closest one?"
        ]

        for prompt in prompts {
            let decision = IntentRouter.classify(prompt)
            #expect(decision.intent == .maps, "Prompt \(prompt) routed as \(decision.intent.rawValue)")
            #expect(IntentRouter.isToolAllowed("maps.search", for: decision))
            #expect(IntentRouter.isToolAllowed("location.current", for: decision))
            #expect(IntentRouter.isToolAllowed("calendar.list", for: decision) == false)
            #expect(IntentRouter.isToolAllowed("calendar.create", for: decision) == false)
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


extension IntentRouterTests {
    @Test func liveScenarioFailurePromptsUseDeterministicPriorityRoutes() async throws {
        let cases: [(String, UserIntent, String)] = [
            ("Keep in mind that I like short answers.", .memory, "memory.save"),
            ("Tell me what style I asked you to use.", .memory, "memory.recall"),
            ("Text 5551234567 that I am late.", .messageDraft, "messages.draft"),
            ("Am I signed in to Outlook?", .outlook, "outlook.status"),
            ("Search my photos for receipts.", .photos, "photos.search"),
            ("Find Lumen architecture notes in my local files.", .rag, "rag.search"),
            ("Search my local files for the latest Lumen diagnostics report.", .rag, "rag.search"),
            ("Run an agent reminder summary tonight in the background.", .trigger, "trigger.create"),
            ("Fetch the page at https://example.com.", .webSearch, "web.fetch")
        ]

        for item in cases {
            let decision = await IntentClassifierService.shared.route(item.0)
            #expect(decision.intent == item.1, "Prompt \(item.0) routed as \(decision.intent.rawValue)")
            #expect(IntentRouter.isToolAllowed(item.2, for: decision), "Prompt \(item.0) did not allow \(item.2)")
        }
    }

    @Test func liveScenarioGeneratedPromptsStayExecutableForSpecificTools() async throws {
        let entries = Dictionary(uniqueKeysWithValues: ToolScenarioBank.entries().map { ($0.expectedToolID + ":" + $0.kind.rawValue, $0) })
        let expected: [(String, String)] = [
            ("web.fetch:missingArgument", "web.fetch"),
            ("maps.directions:missingArgument", "maps.directions"),
            ("outlook.mail.send:approvalBoundary", "outlook.mail.send"),
            ("trigger.create:approvalBoundary", "trigger.create")
        ]

        for item in expected {
            let entry = try #require(entries[item.0])
            let decision = await IntentClassifierService.shared.route(entry.prompt)
            #expect(IntentRouter.isToolAllowed(item.1, for: decision), "Prompt \(entry.prompt) did not allow \(item.1)")
            let action = DeterministicToolPlanner.plan(
                routing: decision,
                prompt: entry.prompt,
                availableToolIDs: decision.allowedToolIDs
            )
            #expect(action?.tool == item.1, "Prompt \(entry.prompt) planned \(action?.tool ?? "nil")")
        }
    }

    @Test func identityRecallPromptsRouteToMemoryRecall() async throws {
        for prompt in ["What is my name?", "Who am I?", "Do you know my name?", "What's my name?", "What did I say my name was?"] {
            let decision = IntentRouter.classify(prompt)
            #expect(decision.intent == .memory)
            #expect(decision.allowedToolIDs.contains("memory.recall"))
        }
    }

    @Test func identitySavePromptRoutesToMemorySave() async throws {
        let decision = IntentRouter.classify("Remember my name is Alexis")
        #expect(decision.intent == .memory)
        #expect(decision.allowedToolIDs.contains("memory.save"))
    }
}


extension IntentRouterTests {
    @Test func profilePhrasesInsideExplicitEmailDraftDoNotPreemptDraftIntent() async throws {
        let decision = IntentRouter.classify("Draft an email to bob@example.com saying my name is Alexis")
        #expect(decision.intent == .emailDraft)
    }

    @Test func directCallMeProfilePromptStillRoutesToMemory() async throws {
        let decision = IntentRouter.classify("Call me Alexis")
        #expect(decision.intent == .memory)
        #expect(decision.allowedToolIDs.contains("memory.save"))
    }
}
