import Testing
@testable import Lumen

struct DeterministicToolPlannerTests {
    @Test func outlookReadNewEmailsPlansList() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.messages.list"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "Read new emails", availableToolIDs: ["outlook.messages.list"])
        #expect(action?.tool == "outlook.messages.list")
        #expect(action?.args["limit"]?.stringValue == "10")
    }

    @Test func outlookUnreadPlansListUnread() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.messages.list"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "Check unread emails", availableToolIDs: ["outlook.messages.list"])
        #expect(action?.args["unreadOnly"]?.stringValue == "true")
    }

    @Test func outlookLatestPlansListBeforeRead() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.messages.list", "outlook.message.read"], requiresClarification: false, clarificationPrompt: nil)
        let steps = DeterministicToolPlanner.planSteps(routing: routing, prompt: "Read the latest email", availableToolIDs: ["outlook.messages.list", "outlook.message.read"])
        #expect(steps.map(\.tool) == ["outlook.messages.list", "outlook.message.read"])
        #expect(steps.first?.args["limit"]?.stringValue == "1")
        #expect(steps.last?.args["messageId"]?.stringValue == "latest")
        #expect(steps.last?.args["id"]?.stringValue == "latest")
    }

    @Test func outlookLatestPlansReadOnlyWhenListUnavailable() async throws {
        let routing = IntentRoutingDecision(
            intent: .outlook,
            allowedToolIDs: ["outlook.messages.list", "outlook.message.read"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let steps = DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: "Read the latest email",
            availableToolIDs: ["outlook.message.read"]
        )
        #expect(steps.count == 1)
        #expect(steps.first?.tool == "outlook.message.read")
        #expect(steps.first?.args["messageId"]?.stringValue == "latest")
        #expect(steps.first?.args["id"]?.stringValue == "latest")
    }

    @Test func whereAreWePlansCurrentLocation() async throws {
        let routing = IntentRoutingDecision(intent: .maps, allowedToolIDs: ["location.current"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "Where are we", availableToolIDs: ["location.current"])
        #expect(action?.tool == "location.current")
    }

    @Test func unreadEmailsPreferMailboxListOverRead() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.message.read", "outlook.messages.list"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "Check unread emails", availableToolIDs: ["outlook.message.read", "outlook.messages.list"])
        #expect(action?.tool == "outlook.messages.list")
        #expect(action?.args["unreadOnly"]?.stringValue == "true")
    }

    @Test func nearbyQueryExtractionAvoidsNearMeTail() async throws {
        let routing = IntentRoutingDecision(intent: .maps, allowedToolIDs: ["maps.search"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "find restaurants near me", availableToolIDs: ["maps.search"])
        #expect(action?.tool == "maps.search")
        #expect(action?.args["query"]?.stringValue == "restaurants")
    }

    @Test func liveCoverageFailedPromptsPlanDeterministicActions() async throws {
        let cases: [(String, UserIntent, [String])] = [
            ("Give me directions to the nearest hardware store.", .maps, ["maps.directions"]),
            ("Find coffee near me.", .maps, ["location.current", "maps.search"]),
            ("Tell me what style I asked you to use.", .memory, ["memory.recall"]),
            ("Keep in mind that I like short answers.", .memory, ["memory.save"]),
            ("Am I walking or stationary right now?", .motion, ["motion.activity"]),
            ("Show attachments on the latest Outlook email.", .outlook, ["outlook.messages.list", "outlook.attachments.list"]),
            ("Show Outlook mail folders.", .outlook, ["outlook.folders.list"]),
            ("Compose an Outlook draft to alex@example.com subject Update body Done.", .outlook, ["outlook.draft.create"])
        ]

        for (prompt, expectedIntent, expectedTools) in cases {
            let routing = IntentRouter.classify(prompt)
            let steps = DeterministicToolPlanner.planSteps(
                routing: routing,
                prompt: prompt,
                availableToolIDs: routing.allowedToolIDs
            )
            #expect(routing.intent == expectedIntent, "Prompt \(prompt) routed as \(routing.intent.rawValue)")
            #expect(steps.map(\.tool) == expectedTools, "Prompt \(prompt) planned \(steps.map(\.tool))")
        }
    }

    @Test func messageDraftPhonePromptsExtractRecipientAndBody() async throws {
        let cases: [(String, String)] = [
            ("Text 5551234567 that I am late.", "I am late"),
            ("Text 5551234567 that approval boundary works.", "approval boundary works"),
            ("Draft a message to 5551234567 saying I am running late.", "I am running late")
        ]

        for (prompt, expectedBody) in cases {
            let routing = IntentRouter.classify(prompt)
            let action = DeterministicToolPlanner.plan(
                routing: routing,
                prompt: prompt,
                availableToolIDs: routing.allowedToolIDs
            )
            #expect(routing.intent == .messageDraft)
            #expect(!routing.requiresClarification)
            #expect(action?.tool == "messages.draft")
            #expect(action?.args["to"]?.stringValue == "5551234567")
            #expect(action?.args["body"]?.stringValue == expectedBody)
        }
    }

    @Test func scheduledSupportGroupMeetingSearchPlansLocationThenWebSearch() async throws {
        let routing = IntentRouter.classify("Find the nearest Alcoholics Anonymous meeting tonight")
        let steps = DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: "Find the nearest Alcoholics Anonymous meeting tonight",
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .webSearch)
        #expect(steps.map(\.tool) == ["location.current", "web.search"])
        #expect(steps.last?.args["query"]?.stringValue == "the nearest Alcoholics Anonymous meeting tonight near me")
    }

    @Test func dynamicLocalPublicLookupPlansLocationThenWebSearch() async throws {
        let prompt = "Where is the nearest free tax clinic tomorrow?"
        let routing = IntentRouter.classify(prompt)
        let steps = DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: prompt,
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .webSearch)
        #expect(steps.map(\.tool) == ["location.current", "web.search"])
        #expect(steps.last?.args["query"]?.stringValue == "the nearest free tax clinic tomorrow near me")
    }

    @Test func moveIntentIncludesDestination() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.message.move"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "move latest email to inbox", availableToolIDs: ["outlook.message.move"])
        #expect(action?.tool == "outlook.message.move")
        #expect(action?.args["destination"]?.stringValue == "inbox")
    }
    @Test func weatherInCityKeepsExplicitLocation() async throws {
        let routing = IntentRoutingDecision(intent: .weather, allowedToolIDs: ["weather"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "weather in Montreal", availableToolIDs: ["weather"])
        #expect(action?.tool == "weather")
        #expect(action?.args["location"]?.stringValue == "Montreal")
    }

    @Test func triggerSchedulePlansCreateAction() async throws {
        let routing = IntentRoutingDecision(intent: .trigger, allowedToolIDs: ["trigger.create", "trigger.list"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "Schedule a trigger to summarize reminders tonight and confirm what will run.",
            availableToolIDs: ["trigger.create", "trigger.list"]
        )
        #expect(action?.tool == "trigger.create")
        #expect(action?.args["title"]?.stringValue == "Reminder summary")
        #expect(action?.args["prompt"]?.stringValue.contains("summarize reminders") == true)
        #expect(action?.args["schedule"]?.stringValue == "once")
    }

    @Test func calendarAppointmentTomorrowMorningPlansCreateNotList() async throws {
        let routing = IntentRouter.classify("Set an appointment for tomorrow morning at nine in my calendar")
        let action = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "Set an appointment for tomorrow morning at nine in my calendar",
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .calendar)
        #expect(action?.tool == "calendar.create")
        #expect(action?.args["title"]?.stringValue == "Appointment")
        #expect(Int(action?.args["startsInMinutes"]?.stringValue ?? "0") ?? 0 > 0)
    }

    @Test func calendarUpcomingPromptStillPlansList() async throws {
        let routing = IntentRouter.classify("List my upcoming calendar events")
        let action = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "List my upcoming calendar events",
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .calendar)
        #expect(action?.tool == "calendar.list")
    }

    @Test func calendarShowAppointmentsDoesNotCreateEvent() async throws {
        let routing = IntentRouter.classify("Show my appointments tomorrow")
        let action = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "Show my appointments tomorrow",
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .calendar)
        #expect(action?.tool == "calendar.list")
    }

    @Test func calendarSearchNextEventPlansList() async throws {
        let routing = IntentRouter.classify("Search my calendar for next event")
        let action = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "Search my calendar for next event",
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .calendar)
        #expect(action?.tool == "calendar.list")
    }

    @Test func calendarReadOnlyPromptsDefaultToList() async throws {
        let prompts = [
            "When is my next meeting?",
            "What's on my schedule today?",
            "Do I have any appointments tomorrow?"
        ]

        for prompt in prompts {
            let routing = IntentRouter.classify(prompt)
            let action = DeterministicToolPlanner.plan(
                routing: routing,
                prompt: prompt,
                availableToolIDs: routing.allowedToolIDs
            )
            #expect(routing.intent == .calendar)
            #expect(action?.tool == "calendar.list")
        }
    }

    @Test func calendarStandaloneNumberWordDoesNotBecomeHour() async throws {
        let routing = IntentRouter.classify("Schedule three appointments tomorrow")
        let action = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "Schedule three appointments tomorrow",
            availableToolIDs: routing.allowedToolIDs
        )
        let baseline = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "Schedule appointments tomorrow",
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .calendar)
        #expect(action?.tool == "calendar.create")
        #expect(action?.args["startsInMinutes"]?.stringValue == baseline?.args["startsInMinutes"]?.stringValue)
    }


    @Test func memoryNameSaveNormalizesFact() async throws {
        let routing = IntentRouter.classify("Can you remember that my name is Alexis?")
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "Can you remember that my name is Alexis?", availableToolIDs: routing.allowedToolIDs)
        #expect(action?.tool == "memory.save")
        #expect(action?.args["content"]?.stringValue == "User's name is Alexis")
        #expect(action?.args["kind"]?.stringValue == "fact")
    }

    @Test func memoryNameRecallUsesProfileQuery() async throws {
        let routing = IntentRouter.classify("Who am I?")
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "Who am I?", availableToolIDs: routing.allowedToolIDs)
        #expect(action?.tool == "memory.recall")
        #expect(action?.args["query"]?.stringValue == "user name")
    }

    @Test func memorySaveThenRecallPlansBothActions() async throws {
        let prompt = "Remember that I prefer concise bullet points, then tell me what you remembered."
        let routing = IntentRouter.classify(prompt)
        let steps = DeterministicToolPlanner.planSteps(routing: routing, prompt: prompt, availableToolIDs: routing.allowedToolIDs)
        #expect(steps.map(\.tool) == ["memory.save", "memory.recall"])
        #expect(steps.first?.args["content"]?.stringValue.contains("prefer concise bullet points") == true)
        #expect(steps.last?.args["query"]?.stringValue == "prefer concise bullet points")
    }

    @Test func alarmCommandFamilyPlansExpectedTools() async throws {
        let cases: [(String, String)] = [
            ("Check alarm authorization status", "alarm.authorization_status"),
            ("Request alarm authorization", "alarm.request_authorization"),
            ("Set an alarm for tomorrow at 7", "alarm.schedule"),
            ("Start a countdown timer for 10 minutes", "alarm.countdown"),
            ("List alarms", "alarm.list"),
            ("Pause alarm 00000000-0000-0000-0000-000000000000", "alarm.pause"),
            ("Resume alarm 00000000-0000-0000-0000-000000000000", "alarm.resume"),
            ("Stop alarm 00000000-0000-0000-0000-000000000000", "alarm.stop"),
            ("Snooze alarm 00000000-0000-0000-0000-000000000000", "alarm.snooze"),
            ("Cancel alarm 00000000-0000-0000-0000-000000000000", "alarm.cancel")
        ]

        for (prompt, expectedTool) in cases {
            let routing = IntentRouter.classify(prompt)
            let action = DeterministicToolPlanner.plan(routing: routing, prompt: prompt, availableToolIDs: routing.allowedToolIDs)
            #expect(routing.intent == .alarm)
            #expect(action?.tool == expectedTool)
        }
    }

    @Test func alarmPlannerSuppliesRequiredArguments() async throws {
        let countdownRouting = IntentRouter.classify("Start a countdown timer for 10 minutes")
        let countdown = DeterministicToolPlanner.plan(routing: countdownRouting, prompt: "Start a countdown timer for 10 minutes", availableToolIDs: countdownRouting.allowedToolIDs)
        #expect(countdown?.args["durationSeconds"]?.stringValue == "600")

        let pauseRouting = IntentRouter.classify("Pause alarm 00000000-0000-0000-0000-000000000000")
        let pause = DeterministicToolPlanner.plan(routing: pauseRouting, prompt: "Pause alarm 00000000-0000-0000-0000-000000000000", availableToolIDs: pauseRouting.allowedToolIDs)
        #expect(pause?.args["id"]?.stringValue == "00000000-0000-0000-0000-000000000000")

        let cancelRouting = IntentRouter.classify("Cancel alarm 00000000-0000-0000-0000-000000000000")
        let cancel = DeterministicToolPlanner.plan(routing: cancelRouting, prompt: "Cancel alarm 00000000-0000-0000-0000-000000000000", availableToolIDs: cancelRouting.allowedToolIDs)
        #expect(cancel?.args["id"]?.stringValue == "00000000-0000-0000-0000-000000000000")
    }

    @Test func alarmCancelRequiresUUID() async throws {
        let routing = IntentRouter.classify("Cancel alarm named morning wakeup")
        let action = DeterministicToolPlanner.plan(
            routing: routing,
            prompt: "Cancel alarm named morning wakeup",
            availableToolIDs: routing.allowedToolIDs
        )
        #expect(routing.intent == .alarm)
        #expect(action == nil)
    }

    @Test func alarmPlannerDoesNotScheduleFromBareAlarmPrompt() async throws {
        let routing = IntentRouter.classify("alarm")
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "alarm", availableToolIDs: routing.allowedToolIDs)
        #expect(routing.intent == .alarm)
        #expect(action == nil)
    }

    @Test func triggerCancelRequiresIdentifier() async throws {
        let ambiguousPrompt = "Cancel that scheduled run"
        let ambiguousRouting = IntentRouter.classify(ambiguousPrompt)
        let ambiguous = DeterministicToolPlanner.plan(
            routing: ambiguousRouting,
            prompt: ambiguousPrompt,
            availableToolIDs: ambiguousRouting.allowedToolIDs
        )
        #expect(ambiguousRouting.intent == .trigger)
        #expect(ambiguous == nil)

        let namedPrompt = "Cancel trigger named morning summary"
        let namedRouting = IntentRouter.classify(namedPrompt)
        let named = DeterministicToolPlanner.plan(
            routing: namedRouting,
            prompt: namedPrompt,
            availableToolIDs: namedRouting.allowedToolIDs
        )
        #expect(named?.tool == "trigger.cancel")
        #expect(named?.args["id"]?.stringValue == "morning summary")
    }

}
