import Testing
@testable import Lumen

struct AlarmMutationUUIDContractTests {
    @Test func alarmMutationPlannerAcceptsOnlyUUIDArguments() {
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

    @Test func alarmMutationPlannerRejectsTitleOnlyMutations() {
        let missingTitle = "alarm-title-that-definitely-does-not-exist-7B5899F1-DFEE-4C39-9F56-A25AF4D70C9D"
        let prompts = [
            "Pause alarm named \(missingTitle)",
            "Resume alarm named \(missingTitle)",
            "Stop alarm named \(missingTitle)",
            "Snooze alarm named \(missingTitle)",
            "Cancel alarm named \(missingTitle)"
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
