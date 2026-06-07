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
}


extension FinalIntentValidatorTests {
    @Test func safeObservationStillRejectsCrossIntentLeaks() async throws {
        let routing = IntentRoutingDecision(intent: .weather, allowedToolIDs: ["weather"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("Created a new event. GPS signal timeout.", routing: routing, fallback: nil)
        #expect(!text.contains("Created a new event"))
    }

    @Test func outlookObservationRejectsTokenBearingOutput() async throws {
        let routing = IntentRoutingDecision(intent: .outlook, allowedToolIDs: ["outlook.messages.list"], requiresClarification: false, clarificationPrompt: nil)
        let text = FinalIntentValidator.validate("Outlook token: eyJhbGciOi.fake.token", routing: routing, fallback: nil)
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
