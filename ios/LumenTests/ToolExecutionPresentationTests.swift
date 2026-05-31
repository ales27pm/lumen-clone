import Testing
@testable import Lumen

struct ToolExecutionPresentationTests {
    @Test func outlookAADSTS70000FailureMapsToFailedReconnectMessage() async throws {
        let raw = "Outlook tool failed: AADSTS70000: The request was denied because one or more scopes requested are unauthorized or expired. The user must first sign in and grant the client application access to the requested scope."
        let presentation = ToolExecutionPresentation.presentation(for: "outlook.mail.send", rawResult: raw)

        #expect(presentation.status == .failed)
        #expect(presentation.message.contains("Outlook needs to be reconnected"))
        #expect(presentation.message.contains("No Outlook email was sent"))
    }

    @Test func successfulOutlookSendMapsToCompleted() async throws {
        let presentation = ToolExecutionPresentation.presentation(
            for: "outlook.mail.send",
            rawResult: "Sent Outlook email to alexis.boulet@example.com."
        )

        #expect(presentation.status == .completed)
        #expect(presentation.message == "Sent Outlook email to alexis.boulet@example.com.")
    }

    @Test func successfulOutlookContentWithExpiredAndPermissionWordsStaysCompleted() async throws {
        let raw = "Subject: Contract permission review\nPreview: The coupon expired yesterday, but this is normal email content."
        let presentation = ToolExecutionPresentation.presentation(for: "outlook.message.read", rawResult: raw)

        #expect(presentation.status == .completed)
        #expect(presentation.message == raw)
    }

    @Test func pendingActionIDIsRemovedFromDisplayedToolArguments() async throws {
        let raw = "body: Hello, Alexis. This is a test email., pendingActionID: 4EDE847E-F619-49D6-9E8D-7889B13A36B5, subject: Test Email, to: alexis.boulet@example.com"
        let redacted = ToolArgumentRedactor.redactDisplayContent(raw)

        #expect(!redacted.contains("pendingActionID"))
        #expect(!redacted.contains("4EDE847E-F619-49D6-9E8D-7889B13A36B5"))
        #expect(redacted.contains("body: Hello, Alexis. This is a test email."))
        #expect(redacted.contains("subject: Test Email"))
        #expect(redacted.contains("to: alexis.boulet@example.com"))
    }

    @Test func jsonPendingActionIDIsRemovedFromDisplayedToolArguments() async throws {
        let raw = #"{"body":"Hello, Alexis.","pendingActionID":"4EDE847E-F619-49D6-9E8D-7889B13A36B5","subject":"Test Email"}"#
        let redacted = ToolArgumentRedactor.redactDisplayContent(raw)

        #expect(!redacted.contains("pendingActionID"))
        #expect(!redacted.contains("4EDE847E-F619-49D6-9E8D-7889B13A36B5"))
        #expect(redacted.contains(#""body":"Hello, Alexis.""#))
        #expect(redacted.contains(#""subject":"Test Email""#))
    }

    @Test func fullyRedactedSensitiveOnlyResultDoesNotFallBackToRawValue() async throws {
        let raw = "access_token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        let presentation = ToolExecutionPresentation.presentation(for: "outlook.mail.send", rawResult: raw)

        #expect(presentation.status == .completed)
        #expect(presentation.message == "(no displayable result)")
        #expect(!presentation.message.contains("access_token"))
        #expect(!presentation.message.contains("eyJhbGci"))
    }
}
