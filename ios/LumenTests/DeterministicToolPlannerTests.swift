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

}
