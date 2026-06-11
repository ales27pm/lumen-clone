import Testing
@testable import Lumen

struct AlarmMutationUUIDContractTests {
    @Test func alarmMutationPlannerAcceptsOnlyUUIDArguments() async throws {
        let uuid = "00000000-0000-0000-0000-000000000000"
        let cases: [(String, String)] = [
            ("Pause alarm \(uuid)", "alarm.pause"),
            ("Resume alarm \(uuid)", "alarm.resume"),
            ("Stop alarm \(uuid)", "alarm.stop"),
            ("Snooze alarm \(uuid)", "alarm.snooze"),
            ("Cancel alarm \(uuid)", "alarm.cancel")
        ]

        for (prompt, expectedTool) in cases {
            let routing = IntentRouter.classify(prompt)
            let action = DeterministicToolPlanner.plan(
                routing: routing,
                prompt: prompt,
                availableToolIDs: routing.allowedToolIDs
            )
            #expect(routing.intent == .alarm)
            #expect(action?.tool == expectedTool)
            #expect(action?.args["id"]?.stringValue == uuid)
            #expect(action?.args["title"] == nil)
        }
    }

    @Test func alarmMutationPlannerRejectsTitleOnlyMutations() async throws {
        let prompts = [
            "Pause alarm named test",
            "Resume alarm named test",
            "Stop alarm named test",
            "Snooze alarm named test",
            "Cancel alarm named test"
        ]

        for prompt in prompts {
            let routing = IntentRouter.classify(prompt)
            let action = DeterministicToolPlanner.plan(
                routing: routing,
                prompt: prompt,
                availableToolIDs: routing.allowedToolIDs
            )
            #expect(routing.intent == .alarm)
            #expect(action == nil)
        }
    }
}
