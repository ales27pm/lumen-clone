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

    @Test func expiredGrantRequiresReauthenticationInsteadOfBuildReconfiguration() {
        let outcome = OutlookToolOutcome.failure(from: MicrosoftGraphAuthError.invalidGrant)
        let interaction = OutlookToolOutcome.failure(from: MicrosoftGraphAuthError.interactionRequired)

        #expect(outcome.status == .unavailable)
        #expect(outcome.availability == .authUnavailable)
        #expect(outcome.errorCode == "outlook_reauthentication_required")
        #expect(outcome.text.localizedCaseInsensitiveContains("reconnect"))
        #expect(!outcome.text.localizedCaseInsensitiveContains("not configured"))
        #expect(interaction.errorCode == "outlook_interaction_required")
        #expect(!interaction.text.localizedCaseInsensitiveContains("not configured"))
    }

    @Test func consentAndScopeFailuresRemainDistinctPermissionFailures() {
        let consent = OutlookToolOutcome.failure(from: MicrosoftGraphAuthError.consentRequired)
        let scope = OutlookToolOutcome.failure(from: MicrosoftGraphAuthError.invalidScope)

        #expect(consent.status == .denied)
        #expect(consent.availability == .permissionDenied)
        #expect(consent.errorCode == "outlook_consent_required")
        #expect(scope.status == .denied)
        #expect(scope.availability == .permissionDenied)
        #expect(scope.errorCode == "outlook_scope_not_granted")
    }

    @Test func tokenEndpointFailuresHaveSafeSpecificTaxonomy() {
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(errorCode: "invalid_grant", httpStatus: 400) == .invalidGrant)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(errorCode: "interaction_required", httpStatus: 400) == .interactionRequired)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(errorCode: "consent_required", httpStatus: 400) == .consentRequired)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(errorCode: "invalid_scope", httpStatus: 400) == .invalidScope)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(errorCode: "server_error", httpStatus: 400) == .tokenEndpointUnavailable)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(errorCode: nil, httpStatus: 429) == .tokenEndpointThrottled)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(errorCode: nil, httpStatus: 503) == .tokenEndpointUnavailable)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(
            errorCode: "invalid_grant",
            errorDescription: "AADSTS65001: consent_required for the requested resource",
            httpStatus: 400
        ) == .consentRequired)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(
            errorCode: "invalid_grant",
            errorDescription: "AADSTS50076: interaction_required",
            httpStatus: 400
        ) == .interactionRequired)
        #expect(NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(
            errorCode: "invalid_request",
            errorDescription: "AADSTS70011: invalid_scope",
            httpStatus: 400
        ) == .invalidScope)

        let staticConfiguration = NativeMicrosoftOAuthClient.authErrorForTokenEndpointFailure(
            errorCode: "invalid_client",
            httpStatus: 401
        )
        guard case .some(.invalidConfiguration) = staticConfiguration else {
            Issue.record("Expected invalid_client to remain a static configuration failure")
            return
        }
    }

    @Test func authorizationAccessDeniedRequiresConsentWhileSheetCancellationRemainsSeparate() {
        #expect(NativeMicrosoftOAuthClient.authErrorForAuthorizationFailure(errorCode: "access_denied") == .consentRequired)
        #expect(NativeMicrosoftOAuthClient.authErrorForAuthorizationFailure(errorCode: "consent_required") == .consentRequired)
    }

    @Test func cachedAccessTokenDoesNotRequireGrantOnlyOfflineAccessScope() {
        #expect(NativeMicrosoftOAuthClient.accessTokenScopesSatisfy(
            grantedScopes: "User.Read Mail.Read",
            requestedScopes: MicrosoftGraphScope.inboxRead
        ))
        #expect(!NativeMicrosoftOAuthClient.accessTokenScopesSatisfy(
            grantedScopes: "User.Read",
            requestedScopes: MicrosoftGraphScope.inboxRead
        ))
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

    @Test func outlookHTMLReadFinalizationProducesBoundedPlainTextWithoutProviderIdentifiersOrTrackers() throws {
        let graphMessageID = "AQMkSYNTHETIC-MESSAGE-ID-0001=="
        let observation = """
        Subject: Synthetic device launch bulletin
        ID: \(graphMessageID)
        From: Example Devices
        Received: 2030-01-02T03:04:05Z
        Unread: true
        Has attachments: false
        Preview: This is synthetic test content.
        Body:
        <html><head>
        <meta name="unsubscribe_url" content="https://tracking.example.invalid/optout?message=fixture-0001&amp;token=synthetic-tracker-token">
        <style>body { margin: 0 } .preheader { display: none }</style>
        </head><body>
        <img src="https://tracking.example.invalid/pixel/synthetic-tracker-token" width="1" height="1">
        <h1>Synthetic device launch bulletin</h1>
        <p>The synthetic demo starts on January 15, 2030.</p>
        <p>ID: CASE-42</p>
        <p>Join at https://meet.example.invalid/synthetic-demo</p>
        <a href="https://tracking.example.invalid/click/synthetic-tracker-token">Read the synthetic announcement</a>
        </body></html>
        """

        let final = try #require(ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.message.read",
            observation: observation,
            originalPrompt: "Read my latest Outlook email."
        ))

        #expect(final.contains("Subject: Synthetic device launch bulletin"))
        #expect(final.contains("From: Example Devices"))
        #expect(final.contains("Received: 2030-01-02T03:04:05Z"))
        #expect(final.contains("Unread: true"))
        #expect(final.contains("Has attachments: false"))
        #expect(final.contains("Preview: This is synthetic test content."))
        #expect(final.contains("The synthetic demo starts on January 15, 2030."))
        #expect(final.contains("ID: CASE-42"))
        #expect(final.contains("https://meet.example.invalid/synthetic-demo"))
        #expect(final.contains("Read the synthetic announcement"))
        #expect(!final.contains(graphMessageID))
        #expect(!final.localizedCaseInsensitiveContains("<html"))
        #expect(!final.localizedCaseInsensitiveContains("<meta"))
        #expect(!final.contains("synthetic-tracker-token"))
        #expect(final.count <= OutlookToolUserVisibleOutput.maxFinalCharacters + 32)
    }

    @Test func outlookListFolderAndAttachmentFinalsHideRawGraphIdentifiers() throws {
        let graphID = "AQMkSYNTHETIC-MESSAGE-ID-0002=="
        let listFinal = try #require(ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.messages.list",
            observation: """
            Cached references: use ordinal args like {"message":"first"}, or the raw Message ID for follow-up tools.

            ---

            Index: 1
            Subject: Synthetic status
            ID: \(graphID)
            From: Example Sender
            Received: 2030-01-02T03:04:05Z
            Unread: false
            Has attachments: false
            Preview: All good.
            """,
            originalPrompt: "List my latest Outlook emails."
        ))
        let folderFinal = try #require(ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.folders.list",
            observation: "- Inbox — id: \(graphID), unread: 2, total: 8",
            originalPrompt: "List my Outlook folders."
        ))
        let attachmentFinal = try #require(ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.attachments.list",
            observation: "No attachments found for message \(graphID).",
            originalPrompt: "Show attachments on the latest Outlook email."
        ))

        #expect(listFinal.contains("Subject: Synthetic status"))
        #expect(!listFinal.contains(graphID))
        #expect(!listFinal.localizedCaseInsensitiveContains("raw Message ID"))
        #expect(folderFinal.contains("Inbox"))
        #expect(folderFinal.contains("unread: 2"))
        #expect(!folderFinal.contains(graphID))
        #expect(attachmentFinal.contains("selected Outlook message"))
        #expect(!attachmentFinal.contains(graphID))
    }
}
