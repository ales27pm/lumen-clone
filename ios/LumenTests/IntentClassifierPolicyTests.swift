import Testing
@testable import Lumen

struct IntentClassifierPolicyTests {
    @Test func bundledUnavailableFallsBackToDeterministic() {
        let fallback = DeterministicIntentFallback.classify("explain this Swift code")
        let result = IntentClassifierPolicy.resolve(modelResult: nil, deterministic: fallback)
        #expect(result.intent == fallback.intent)
        #expect(result.source == .deterministicFallback)
    }

    @Test func highConfidenceWeatherBeatsWeakDeterministicChat() {
        let fallback = IntentClassificationResult(intent: .chat, confidence: 0.75, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .deterministicFallback, diagnostics: nil)
        let model = IntentClassificationResult(intent: .weather, confidence: 0.9, alternatives: [IntentAlternative(intent: .weather, confidence: 0.9)], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == .weather)
    }

    @Test func lowConfidenceFallsBackToDeterministic() {
        let fallback = DeterministicIntentFallback.classify("set alarm for 6")
        let model = IntentClassificationResult(intent: .weather, confidence: 0.2, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == fallback.intent)
    }

    @Test func mediumConfidenceSameIntentMerges() {
        let fallback = DeterministicIntentFallback.classify("weather here")
        let model = IntentClassificationResult(intent: .weather, confidence: 0.55, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.source == .policyMerged)
        #expect(result.confidence > model.confidence)
    }

    @Test func writeIntentDisagreementPrefersDeterministic() {
        let fallback = DeterministicIntentFallback.classify("create an event tomorrow")
        let model = IntentClassificationResult(intent: .maps, confidence: 0.68, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == .calendar)
    }

    @Test func highConfidenceModelDoesNotOverrideApprovalSensitiveAlarmFallback() {
        let fallback = DeterministicIntentFallback.classify("Set an alarm for tomorrow at 7.")
        let model = IntentClassificationResult(intent: .calendar, confidence: 0.92, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == .alarm)
        #expect(result.source == .policyMerged)
    }

    @Test func closeModelAlternativesAskClarification() {
        let fallback = IntentClassificationResult(intent: .maps, confidence: 0.60, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .deterministicFallback, diagnostics: nil)
        let model = IntentClassificationResult(
            intent: .calendar,
            confidence: 0.64,
            alternatives: [
                IntentAlternative(intent: .calendar, confidence: 0.64),
                IntentAlternative(intent: .maps, confidence: 0.59)
            ],
            requiresClarification: false,
            clarificationPrompt: nil,
            source: .bundledModel,
            diagnostics: nil
        )
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.requiresClarification)
        #expect(result.clarificationPrompt?.contains("calendar event") == true)
        #expect(result.clarificationPrompt?.contains("nearby place") == true)
    }

    @Test func chatIntentHasNoAllowedTools() {
        let chat = IntentClassificationResult(intent: .chat, confidence: 0.9, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        #expect(chat.asRoutingDecision().allowedToolIDs.isEmpty)
    }

    @Test func selectedIntentNeverExposesToolsOutsideMapping() {
        let model = IntentClassificationResult(intent: .weather, confidence: 0.9, alternatives: [IntentAlternative(intent: .calendar, confidence: 0.2)], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let resolved = IntentClassifierPolicy.resolve(modelResult: model, deterministic: DeterministicIntentFallback.classify("hello"))
        let routing = resolved.asRoutingDecision()
        #expect(routing.allowedToolIDs == IntentToolMapping.allowedToolIDs(for: .weather))
        #expect(!routing.allowedToolIDs.contains("calendar.create"))
    }

    @Test func policySanitizesConfidenceAndDeduplicatesAlternatives() {
        let model = IntentClassificationResult(
            intent: .weather,
            confidence: 1.7,
            alternatives: [
                IntentAlternative(intent: .weather, confidence: 0.8),
                IntentAlternative(intent: .weather, confidence: 0.6),
                IntentAlternative(intent: .maps, confidence: 0.2)
            ],
            requiresClarification: false,
            clarificationPrompt: nil,
            source: .bundledModel,
            diagnostics: nil
        )
        let resolved = IntentClassifierPolicy.resolve(modelResult: model, deterministic: DeterministicIntentFallback.classify("hi"))
        #expect(resolved.confidence == 1.0)
        let weatherCount = resolved.alternatives.filter { $0.intent == .weather }.count
        #expect(weatherCount == 1)
    }

    @Test func umbrellaPromptCanBeWeatherFromModel() {
        let model = IntentClassificationResult(intent: .weather, confidence: 0.82, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let fallback = DeterministicIntentFallback.classify("should I bring an umbrella")
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == .weather)
    }

    @Test func explainSwiftCodeStaysChatWhenModelUnavailable() {
        let fallback = DeterministicIntentFallback.classify("explain this Swift code")
        let result = IntentClassifierPolicy.resolve(modelResult: nil, deterministic: fallback)
        #expect(result.intent == .chat)
    }

    @Test func weatherRouteRunsBundledPredictionOffMainActor() async {
        await BundledIntentClassifier.shared.resetPredictionMainThreadViolationCountForTesting()

        let routing = await Task.detached {
            await IntentClassifierService.shared.route("What is the weather here?")
        }.value

        #expect(routing.intent == .weather)
        let violationCount = await BundledIntentClassifier.shared.predictionMainThreadViolationCountForTesting()
        #expect(violationCount == 0)
    }

    @Test func priorityOverridesRunBeforeBundledModel() async {
        let result = await IntentClassifierService.shared.classify("Help me message Jordan with a complete ETA and apology.")
        #expect(result.intent == .messageDraft)
        #expect(result.diagnostics == "deterministic_priority_override")
    }

    @Test func semanticRouteUsesWebForScheduledSupportMeeting() async {
        let routing = await IntentClassifierService.shared.route("Find the nearest Alcoholics Anonymous meeting tonight")
        #expect(routing.intent == .webSearch)
        #expect(!routing.requiresClarification)
        #expect(routing.allowedToolIDs.contains("web.search"))
        #expect(routing.allowedToolIDs.contains("location.current"))
        #expect(routing.allowedToolIDs.contains("maps.search") == false)
        #expect(routing.allowedToolIDs.contains("calendar.list") == false)
    }

    @Test func semanticRouteUsesWebForDynamicLocalPublicLookup() async {
        let routing = await IntentClassifierService.shared.route("Where is the nearest free tax clinic tomorrow?")
        #expect(routing.intent == .webSearch)
        #expect(!routing.requiresClarification)
        #expect(routing.allowedToolIDs.contains("web.search"))
        #expect(routing.allowedToolIDs.contains("location.current"))
        #expect(routing.allowedToolIDs.contains("maps.search") == false)
    }

    @Test func wakeMeUpRoutesToAlarmClarification() async {
        let routing = await IntentClassifierService.shared.route("Wake me up")
        #expect(routing.intent == .alarm)
        #expect(routing.requiresClarification)
        #expect(routing.clarificationPrompt == "What time should I use for the alarm?")
    }

    @Test func ambiguousPromptAsksClarificationBeforeTools() async {
        let meeting = await IntentClassifierService.shared.route("Find my meeting tonight")
        #expect(meeting.requiresClarification)
        #expect(meeting.clarificationPrompt == "Do you mean a calendar event or a nearby meeting location?")

        let reference = await IntentClassifierService.shared.route("Book that")
        #expect(reference.requiresClarification)
        #expect(reference.clarificationPrompt == "What would you like me to act on?")
        #expect(reference.allowedToolIDs.isEmpty)
    }

    @Test func selectedIntentWithMissingRequiredSlotAsksClarification() async {
        let web = await IntentClassifierService.shared.classify("Search the web")
        #expect(web.intent == .webSearch)
        #expect(web.requiresClarification)
        #expect(web.clarificationPrompt == "What should I search for?")
        #expect(web.diagnostics?.contains("slot_clarification") == true)

        let contact = await IntentClassifierService.shared.classify("Find contact")
        #expect(contact.intent == .contactSearch)
        #expect(contact.requiresClarification)
        #expect(contact.clarificationPrompt == "Which contact should I look up?")

        let alarm = await IntentClassifierService.shared.classify("Set an alarm")
        #expect(alarm.intent == .alarm)
        #expect(alarm.requiresClarification)
        #expect(alarm.clarificationPrompt == "What time should I use for the alarm?")

        let email = await IntentClassifierService.shared.classify("Email Sarah")
        #expect(email.intent == .emailDraft)
        #expect(email.requiresClarification)
        #expect(email.clarificationPrompt == "What should the email say?")

        let message = await IntentClassifierService.shared.classify("Message Jordan")
        #expect(message.intent == .messageDraft)
        #expect(message.requiresClarification)
        #expect(message.clarificationPrompt == "What should the message say?")

        let file = IntentRouter.classify("Use Read File, but ask for clarification if required details are missing.")
        #expect(file.intent == .files)
        #expect(file.requiresClarification)
        #expect(file.clarificationPrompt == "Which file should I read?")

        let rag = await IntentClassifierService.shared.classify("Use Search Personal Data, but ask for clarification if required details are missing.")
        #expect(rag.intent == .rag)
        #expect(rag.requiresClarification)
        #expect(rag.clarificationPrompt == "What should I search for?")
    }

    @Test func clearDirectCommandsDoNotOverAskClarification() async {
        let message = await IntentClassifierService.shared.classify("Help me message Jordan with a complete ETA and apology.")
        #expect(message.intent == .messageDraft)
        #expect(!message.requiresClarification)

        let web = await IntentClassifierService.shared.classify("Search the web for Qwen3 local inference benchmarks")
        #expect(web.intent == .webSearch)
        #expect(!web.requiresClarification)

        let alarm = await IntentClassifierService.shared.classify("Set an alarm for 6")
        #expect(alarm.intent == .alarm)
        #expect(!alarm.requiresClarification)
    }

    @Test func reminderContentMatchingDiagnosticClarificationTextIsPreserved() async {
        let prompts = [
            "Remind me to ask for clarification if required details are missing",
            "Remind me to call Alex and ask for clarification if required details are missing."
        ]

        for prompt in prompts {
            let reminder = await IntentClassifierService.shared.classify(prompt)
            #expect(reminder.intent == .reminder)
            #expect(!reminder.requiresClarification)
        }
    }

    @Test func alarmReadAndPermissionCommandsDoNotAskForTimeClarification() async {
        let prompts = [
            "Show alarm permission status.",
            "Check alarm authorization status.",
            "Use Alarm Auth Status, but ask for clarification if required details are missing.",
            "List active alarms.",
            "Show active alarms.",
            "Use List Alarms, but ask for clarification if required details are missing.",
            "Request permission to use alarms.",
            "Ask for alarm authorization.",
            "Cancel alarm 00000000-0000-0000-0000-000000000000.",
            "Pause alarm 00000000-0000-0000-0000-000000000000.",
            "Resume alarm 00000000-0000-0000-0000-000000000000.",
            "Stop alarm 00000000-0000-0000-0000-000000000000.",
            "Snooze alarm 00000000-0000-0000-0000-000000000000."
        ]

        for prompt in prompts {
            let routing = await IntentClassifierService.shared.route(prompt)
            #expect(routing.intent == .alarm, "Prompt \(prompt) routed as \(routing.intent.rawValue)")
            #expect(!routing.requiresClarification, "Prompt \(prompt) incorrectly asked \(routing.clarificationPrompt ?? "nil")")
        }
    }

    @Test func alarmBareCommandsAskForOperationSpecificMissingDetails() async {
        let alarm = await IntentClassifierService.shared.classify("Set alarm.")
        #expect(alarm.intent == .alarm)
        #expect(alarm.requiresClarification)
        #expect(alarm.clarificationPrompt?.lowercased().contains("time") == true)

        let timer = await IntentClassifierService.shared.classify("Start a timer.")
        #expect(timer.intent == .alarm)
        #expect(timer.requiresClarification)
        #expect(timer.clarificationPrompt?.lowercased().contains("duration") == true)

        let cancel = await IntentClassifierService.shared.classify("Cancel my alarm.")
        #expect(cancel.intent == .alarm)
        #expect(cancel.requiresClarification)
        #expect(cancel.clarificationPrompt == "Which alarm should I cancel?")
    }

    @Test func liveE2ERoutingRegressionsUsePriorityOverrides() async {
        let cases: [(String, UserIntent)] = [
            ("Draft a quick email update to Taylor about the delay and ask one question.", .emailDraft),
            ("Place a call to Alex from contacts.", .phoneCall),
            ("Remind me to call Alex tomorrow.", .reminder),
            ("Remind me to text Alex tomorrow.", .reminder),
            ("Remind me to email Sarah next week.", .reminder),
            ("Create a reminder to call the supplier.", .reminder),
            ("Open the camera and prepare to take a photo.", .camera),
            ("Show whether I was walking or driving recently.", .motion),
            ("Read this web URL: https://example.com.", .webSearch),
            ("Open and read architecture-notes.md.", .files),
            ("Save this note: prioritize bullet points.", .memory),
            ("Reindex local files for retrieval.", .rag),
            ("Refresh the file retrieval index.", .rag),
            ("Refresh the photo retrieval index.", .rag)
        ]

        for (prompt, expectedIntent) in cases {
            let result = await IntentClassifierService.shared.classify(prompt)
            #expect(result.intent == expectedIntent)
            #expect(result.diagnostics == "deterministic_priority_override")
        }
    }
}
