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
}
