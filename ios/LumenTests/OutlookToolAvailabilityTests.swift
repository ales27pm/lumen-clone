import Foundation
import Testing
@testable import Lumen

struct OutlookToolAvailabilityTests {
    @Test func missingMSALConfigIsUnavailableAndNonSuccessful() {
        let outcome = OutlookToolOutcome.failure(from: MicrosoftGraphAuthError.missingClientID)

        #expect(outcome.status == .unavailable)
        #expect(outcome.availability == .notConfigured)
        #expect(outcome.errorCode == "outlook_not_configured")
        #expect(!outcome.text.localizedCaseInsensitiveContains("success"))
    }

    @Test func missingAuthIsUnavailableAndNonSuccessful() {
        let outcome = OutlookToolOutcome.failure(from: MicrosoftGraphAuthError.noAccount)

        #expect(outcome.status == .unavailable)
        #expect(outcome.availability == .authUnavailable)
        #expect(outcome.errorCode == "outlook_auth_unavailable")
        #expect(outcome.text.localizedCaseInsensitiveContains("not signed in"))
    }

    @Test func networkFailureIsUnavailableAndNonSuccessful() {
        let outcome = OutlookToolOutcome.failure(from: URLError(.notConnectedToInternet))

        #expect(outcome.status == .unavailable)
        #expect(outcome.availability == .networkUnavailable)
        #expect(outcome.errorCode == "outlook_network_unavailable")
        #expect(outcome.diagnostics["urlErrorCode"] == String(URLError.Code.notConnectedToInternet.rawValue))
    }

    @Test func validEmptyMailboxAndFolderResultsRemainSuccessful() {
        let mailbox = OutlookToolOutcome.validEmpty("No Outlook messages found.")
        let folders = OutlookToolOutcome.validEmpty("No Outlook mail folders found.")

        #expect(mailbox.status == .success)
        #expect(mailbox.availability == .validEmptyResult)
        #expect(mailbox.errorCode == nil)
        #expect(folders.status == .success)
        #expect(folders.availability == .validEmptyResult)
        #expect(folders.errorCode == nil)
    }

    @Test func permissionDeniedGraphErrorIsDeniedWithoutProviderMessage() {
        let rawProviderMessage = "AADSTS65001: body contains private calendar and mailbox content"
        let error = GraphAPIErrorEnvelope(
            error: .init(
                code: "Authorization_RequestDenied",
                message: rawProviderMessage,
                innerError: .init(requestId: "request-123", date: "2026-06-22")
            )
        )
        let outcome = OutlookToolOutcome.failure(from: error)

        #expect(outcome.status == .denied)
        #expect(outcome.availability == .permissionDenied)
        #expect(outcome.errorCode == "outlook_permission_denied")
        #expect(!outcome.text.contains(rawProviderMessage))
        #expect(!outcome.diagnostics.values.joined(separator: " ").contains(rawProviderMessage))
    }

    @Test func providerErrorRedactsRawMessageAndMessageBodies() {
        let rawProviderMessage = "Graph failed with access_token=secret-token and body=private message body"
        let error = GraphAPIErrorEnvelope(
            error: .init(
                code: "MailboxNotEnabledForRESTAPI",
                message: rawProviderMessage,
                innerError: .init(requestId: "request-456", date: "2026-06-22")
            )
        )
        let outcome = OutlookToolOutcome.failure(from: error, diagnostics: ["bundleID": "com.27pm.lumenclone"])
        let diagnosticText = outcome.diagnostics.values.joined(separator: " ")

        #expect(outcome.status == .failed)
        #expect(outcome.availability == .providerError)
        #expect(outcome.errorCode == "outlook_provider_error")
        #expect(outcome.diagnostics["providerCode"] == "MailboxNotEnabledForRESTAPI")
        #expect(!outcome.text.contains(rawProviderMessage))
        #expect(!diagnosticText.contains("secret-token"))
        #expect(!diagnosticText.contains("private message body"))
    }
}
