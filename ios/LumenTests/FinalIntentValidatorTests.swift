import Testing
@testable import Lumen

struct FinalIntentValidatorTests {
    @Test func preservesWeatherGpsTimeoutObservation() async throws {
        let routing = IntentRoutingDecision(intent: .weather, allowedToolIDs: ["weather", "location.current"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("GPS signal timeout while getting current location.", routing: routing, fallback: nil)
        #expect(text.contains("GPS signal timeout"))
        #expect(!text.contains("Please enable location"))
    }

    @Test func preservesOutlookMissingContextObservation() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.messages.list", "outlook.message.read"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("Missing Outlook message context. Ask me to list or search Outlook messages first.", routing: routing, fallback: nil)
        #expect(text.contains("Missing Outlook message context"))
    }

    @Test func preservesMemoryNoMatchObservation() async throws {
        let routing = IntentRoutingDecision(intent: .memory, allowedToolIDs: ["memory.recall"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("No matching memories.", routing: routing, fallback: nil)
        #expect(text == "No matching memories.")
    }

    @Test func preservesCleanMemoryRecallObservation() async throws {
        let routing = IntentRoutingDecision(intent: .memory, allowedToolIDs: ["memory.recall"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("I remember that you prefer concise bullet points.", routing: routing, fallback: nil)
        #expect(text == "I remember that you prefer concise bullet points.")
    }

    @Test func preservesWrappedMemoryRecallObservation() async throws {
        let routing = IntentRoutingDecision(intent: .memory, allowedToolIDs: ["memory.recall"], requiresClarification: false, clarificationPrompt: nil)
        let candidate = "Memory recall:\nI prefer concise bullet points"
        let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: nil)
        #expect(text == candidate)
    }

    @Test func preservesMapsSearchResultsObservation() async throws {
        let routing = IntentRoutingDecision(intent: .maps, allowedToolIDs: ["location.current", "maps.search"], requiresClarification: false, clarificationPrompt: nil)
        let candidate = "Maps search results:\n• Tim Hortons — Avenue de la Plaza, Sorel-Tracy"
        let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: "Maps/location tools are unavailable in this build right now.")
        #expect(text == candidate)
    }

    @Test func preservesRAGSearchResultsObservation() async throws {
        let routing = IntentRoutingDecision(intent: .rag, allowedToolIDs: ["rag.search"], requiresClarification: false, clarificationPrompt: nil)
        let candidate = "RAG search results:\nNo matching files found. Source: local RAG index; no matching module snippets were retrieved."
        let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: "Local search/indexing tools are unavailable in this build right now.")
        #expect(text == candidate)
    }

    @Test func preservesTrustedRAGClarificationPrompt() async throws {
        let clarification = "What should I search for?"
        let routing = IntentRoutingDecision(
            intent: .rag,
            allowedToolIDs: ["rag.search"],
            requiresClarification: true,
            clarificationPrompt: clarification
        )

        let accepted = FinalIntentValidator.validateWithOutcome(
            clarification,
            routing: routing,
            fallback: nil
        )
        #expect(accepted.text == clarification)
        #expect(accepted.acceptedCandidate)

        let recovered = FinalIntentValidator.validate("", routing: routing, fallback: nil)
        #expect(recovered == clarification)
    }

    @Test func alarmSafeFallbackRemainsUserSafeButNotExecutionEvidence() async throws {
        let routing = IntentRoutingDecision(intent: .alarm, allowedToolIDs: ["alarm.authorization_status", "alarm.list"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("Created a calendar event instead.", routing: routing, fallback: nil)
        #expect(text == "I couldn’t safely complete the alarm/timer request.")
        #expect(!E2ETestRunner.isSafeToolObservationFinalForTests(text, expectedToolID: "alarm.authorization_status"))
    }
}


extension FinalIntentValidatorTests {
    @Test func safeObservationStillRejectsCrossIntentLeaks() async throws {
        let routing = IntentRoutingDecision(intent: .weather, allowedToolIDs: ["weather"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("Created a new event. GPS signal timeout.", routing: routing, fallback: nil)
        #expect(!text.contains("Created a new event"))
    }

    @Test func outlookObservationRejectsTokenBearingOutput() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.messages.list"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate(
            "Outlook token: eyJhbGciOiJIUzI1NiJ9.eyJvdXQiOiJsdW1lbiJ9.signature123456",
            routing: routing,
            fallback: nil
        )
        #expect(!text.contains("eyJhbGciOi"))
        #expect(text == "Outlook tool output could not be validated.")
    }

    @Test func outlookObservationAcceptsExpiredAuthWithoutRawToken() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.messages.list"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("Outlook tool failed: authentication expired. Sign in again.", routing: routing, fallback: nil)
        #expect(text.contains("authentication expired"))
    }
}


extension FinalIntentValidatorTests {
    @Test func semanticVersionIsNotCredentialLeak() async throws {
        let routing = IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("The installed version is 1.2.3.", routing: routing, fallback: nil)
        #expect(text == "The installed version is 1.2.3.")
    }

    @Test func jwtShapedCredentialIsRejected() async throws {
        let routing = IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate(
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456ghi789",
            routing: routing,
            fallback: nil
        )
        #expect(!text.contains("eyJhbGciOi"))
    }
}


extension FinalIntentValidatorTests {
    @Test func quotedJsonAccessTokenIsRejected() async throws {
        let routing = IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate(
            #"{"access_token":"abcd1234efgh5678"}"#,
            routing: routing,
            fallback: nil
        )
        #expect(!text.contains("abcd1234efgh5678"))
    }

    @Test func bearerTokenEndingWithPaddingIsRejected() async throws {
        let routing = IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate(
            "Authorization: Bearer abcdefghijklmnop=",
            routing: routing,
            fallback: nil
        )
        #expect(!text.contains("abcdefghijklmnop="))
    }
}


extension FinalIntentValidatorTests {
    @Test func bearerTokenFollowedByColonIsRejected() async throws {
        let routing = IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate(
            "Authorization: Bearer abcdefghijklmnop=: rotate it now",
            routing: routing,
            fallback: nil
        )
        #expect(!text.contains("abcdefghijklmnop="))
    }

    @Test func jsonAccessTokenFollowedByColonDelimiterIsRejected() async throws {
        let routing = IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate(
            #"{"access_token":"abcd1234efgh5678": "unexpected delimiter"}"#,
            routing: routing,
            fallback: nil
        )
        #expect(!text.contains("abcd1234efgh5678"))
    }
}


extension FinalIntentValidatorTests {
    @Test func reportsToolJSONLeakReplacementReason() async throws {
        let routing = IntentRoutingDecision(intent: .weather, allowedToolIDs: ["weather"], requiresClarification: false, clarificationPrompt: nil)
        let outcome = FinalIntentValidator.validateWithOutcome(
            #"{"thought":"check weather","action":{"tool":"weather","args":{"city":"Montreal"}}}"#,
            routing: routing,
            fallback: nil
        )

        #expect(!outcome.acceptedCandidate)
        #expect(outcome.replacementSource == "safeMessage")
        #expect(outcome.rejectionReason == "tool-json-leak")
        #expect(!outcome.text.contains(#""action""#))
    }

    @Test func reportsCrossIntentLeakWhenFallbackIsUsed() async throws {
        let routing = IntentRoutingDecision(intent: .calendar, allowedToolIDs: ["calendar.list"], requiresClarification: false, clarificationPrompt: nil)
        let outcome = FinalIntentValidator.validateWithOutcome(
            "Weather for Montreal: clear sky, temperature 22 C.",
            routing: routing,
            fallback: "Calendar events:\nNo upcoming events"
        )

        #expect(!outcome.acceptedCandidate)
        #expect(outcome.replacementSource == "fallback")
        #expect(outcome.rejectionReason == "weather-leak")
        #expect(outcome.text == "Calendar events:\nNo upcoming events")
    }
}
