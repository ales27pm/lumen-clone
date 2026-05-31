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

    @Test func outlookLatestPlansRead() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.message.read"], requiresClarification: false, clarificationPrompt: nil)
        let action = DeterministicToolPlanner.plan(routing: routing, prompt: "Read the latest email", availableToolIDs: ["outlook.message.read"])
        #expect(action?.tool == "outlook.message.read")
        #expect(action?.args["message"]?.stringValue == "latest")
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

}
