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

    @Test func nativeOAuthRefreshPreservesRotatableStateWhenMicrosoftOmitsIt() {
        let previous = NativeMicrosoftOAuthTokenSet(
            accessToken: "old-access",
            refreshToken: "durable-refresh",
            idToken: "durable-id",
            expiresOn: Date(timeIntervalSince1970: 100),
            scope: "User.Read Mail.Read",
            tokenType: "Bearer"
        )
        let response = NativeMicrosoftOAuthTokenSet(
            accessToken: "new-access",
            refreshToken: nil,
            idToken: nil,
            expiresOn: Date(timeIntervalSince1970: 200),
            scope: nil,
            tokenType: "Bearer"
        )

        let merged = NativeMicrosoftOAuthClient.preservingRefreshState(response, from: previous)

        #expect(merged.accessToken == "new-access")
        #expect(merged.refreshToken == "durable-refresh")
        #expect(merged.idToken == "durable-id")
        #expect(merged.scope == "User.Read Mail.Read")
        #expect(merged.expiresOn == Date(timeIntervalSince1970: 200))

        let rotatedResponse = NativeMicrosoftOAuthTokenSet(
            accessToken: "rotated-access",
            refreshToken: "rotated-refresh",
            idToken: "rotated-id",
            expiresOn: Date(timeIntervalSince1970: 300),
            scope: "User.Read Mail.Read Mail.Send",
            tokenType: "Bearer"
        )
        let rotated = NativeMicrosoftOAuthClient.preservingRefreshState(rotatedResponse, from: previous)
        #expect(rotated.refreshToken == "rotated-refresh")
        #expect(rotated.idToken == "rotated-id")
        #expect(rotated.scope == "User.Read Mail.Read Mail.Send")
    }

    @Test func nativeOAuthKeychainReplacementUpdatesWithoutDeleteFirst() throws {
        let store = NativeMicrosoftOAuthKeychainStore(
            service: "ai.lumen.tests.microsoftgraph.\(UUID().uuidString)",
            account: "replacement"
        )
        defer { try? store.clear() }
        let account = MicrosoftGraphAccountSnapshot(
            id: "synthetic-account",
            username: "person@example.invalid",
            name: "Synthetic Person",
            environment: "test",
            tenantID: nil
        )
        let first = NativeMicrosoftOAuthSession(
            account: account,
            token: NativeMicrosoftOAuthTokenSet(
                accessToken: "first-access",
                refreshToken: "first-refresh",
                idToken: nil,
                expiresOn: Date(timeIntervalSince1970: 100),
                scope: "User.Read",
                tokenType: "Bearer"
            )
        )
        let replacement = NativeMicrosoftOAuthSession(
            account: account,
            token: NativeMicrosoftOAuthTokenSet(
                accessToken: "replacement-access",
                refreshToken: "replacement-refresh",
                idToken: nil,
                expiresOn: Date(timeIntervalSince1970: 200),
                scope: "User.Read Mail.Read",
                tokenType: "Bearer"
            )
        )

        try store.save(first)
        try store.save(replacement)

        let loaded = try #require(store.load())
        #expect(loaded.account.id == "synthetic-account")
        #expect(loaded.token.accessToken == "replacement-access")
        #expect(loaded.token.refreshToken == "replacement-refresh")
        try store.clear()
        #expect(store.load() == nil)
    }

    @Test @MainActor func staleNativeRefreshCompletionCannotRestoreSignedOutSession() async throws {
        let store = NativeMicrosoftOAuthKeychainStore(
            service: "ai.lumen.tests.microsoftgraph.\(UUID().uuidString)",
            account: "stale-refresh"
        )
        defer { try? store.clear() }
        let account = MicrosoftGraphAccountSnapshot(
            id: "synthetic-account-a",
            username: "person-a@example.invalid",
            name: "Synthetic Person A",
            environment: "test",
            tenantID: nil
        )
        let initial = NativeMicrosoftOAuthSession(
            account: account,
            token: NativeMicrosoftOAuthTokenSet(
                accessToken: "initial-access",
                refreshToken: "initial-refresh",
                idToken: nil,
                expiresOn: Date(timeIntervalSince1970: 100),
                scope: "User.Read Mail.Read",
                tokenType: "Bearer"
            )
        )
        let staleRefresh = NativeMicrosoftOAuthSession(
            account: account,
            token: NativeMicrosoftOAuthTokenSet(
                accessToken: "stale-access",
                refreshToken: "stale-rotated-refresh",
                idToken: nil,
                expiresOn: Date(timeIntervalSince1970: 200),
                scope: "User.Read Mail.Read",
                tokenType: "Bearer"
            )
        )
        try store.save(initial)

        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: account.id)
        let requestEpoch = epoch.capture()
        let (completionStream, releaseCompletion) = AsyncStream<Void>.makeStream()
        var persistenceRan = false
        let completion = Task { @MainActor () -> MicrosoftGraphAuthEpochError? in
            do {
                _ = try await epoch.commitCurrentCompletion(
                    expected: requestEpoch,
                    accountID: account.id,
                    operation: {
                        for await _ in completionStream { break }
                        return staleRefresh
                    },
                    commit: { candidate in
                        persistenceRan = true
                        try store.save(candidate)
                    }
                )
                return nil
            } catch let error as MicrosoftGraphAuthEpochError {
                return error
            } catch {
                Issue.record("Unexpected stale refresh error: \(error)")
                return nil
            }
        }

        epoch.invalidateForSignOut()
        try store.clear()
        releaseCompletion.yield(())
        releaseCompletion.finish()

        #expect(await completion.value == .staleCompletion)
        #expect(!persistenceRan)
        #expect(store.load() == nil)

        // Legitimate same-account completion still commits when its epoch remains current.
        let currentEpoch = MicrosoftGraphAuthEpoch(initialAccountID: account.id)
        let currentRequest = currentEpoch.capture()
        let committed = try await currentEpoch.commitCurrentCompletion(
            expected: currentRequest,
            accountID: account.id,
            operation: { staleRefresh },
            commit: { candidate in try store.save(candidate) }
        )
        #expect(committed.token.refreshToken == "stale-rotated-refresh")
        #expect(store.load()?.token.refreshToken == "stale-rotated-refresh")
    }

    @Test @MainActor func lateRotatedRefreshResponseCannotOverwriteNewerProtectedSession() throws {
        let store = NativeMicrosoftOAuthKeychainStore(
            service: "ai.lumen.tests.microsoftgraph.\(UUID().uuidString)",
            account: "refresh-linearization"
        )
        defer { try? store.clear() }
        let account = MicrosoftGraphAccountSnapshot(
            id: "account-a",
            username: "person-a@example.invalid",
            name: "Synthetic Person A",
            environment: "test",
            tenantID: nil
        )
        let dispatchedSession = NativeMicrosoftOAuthSession(
            account: account,
            token: NativeMicrosoftOAuthTokenSet(
                accessToken: "access-r0",
                refreshToken: "refresh-r0",
                idToken: "id-r0",
                expiresOn: Date(timeIntervalSince1970: 100),
                scope: "User.Read Mail.Read",
                tokenType: "Bearer"
            )
        )
        try store.save(dispatchedSession)
        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: account.id)
        let authorization = epoch.capture()
        let client = NativeMicrosoftOAuthClient(keychainStore: store, authEpoch: epoch)

        let rotated = NativeMicrosoftOAuthTokenSet(
            accessToken: "access-r1",
            refreshToken: "refresh-r1",
            idToken: "id-r1",
            expiresOn: Date(timeIntervalSince1970: 200),
            scope: "User.Read Mail.Read Mail.Send",
            tokenType: "Bearer"
        )
        _ = try client.commitRefreshedTokenIfCurrent(
            rotated,
            expectedEpoch: authorization,
            expectedSession: dispatchedSession
        )

        // A second request was also dispatched from R0. Microsoft processed it
        // earlier and returned R0.5, but the response arrived after R1 committed.
        // Its explicit rotated token must not replace the newer protected token.
        let lateOlderRotation = NativeMicrosoftOAuthTokenSet(
            accessToken: "late-access",
            refreshToken: "refresh-r0.5",
            idToken: "id-r0.5",
            expiresOn: Date(timeIntervalSince1970: 150),
            scope: "User.Read Mail.Read",
            tokenType: "Bearer"
        )
        do {
            _ = try client.commitRefreshedTokenIfCurrent(
                lateOlderRotation,
                expectedEpoch: authorization,
                expectedSession: dispatchedSession
            )
            Issue.record("Expected the late refresh completion to lose the session CAS")
        } catch let error as MicrosoftGraphAuthEpochError {
            #expect(error == .staleCompletion)
        } catch {
            Issue.record("Unexpected refresh CAS error: \(error)")
        }

        #expect(store.load()?.token.accessToken == "access-r1")
        #expect(store.load()?.token.refreshToken == "refresh-r1")
        #expect(store.load()?.token.idToken == "id-r1")
        #expect(store.load()?.token.scope == "User.Read Mail.Read Mail.Send")
    }

    @Test @MainActor func concurrentNativeRefreshCallersSingleflightOneProviderRequest() async throws {
        let store = NativeMicrosoftOAuthKeychainStore(
            service: "ai.lumen.tests.microsoftgraph.\(UUID().uuidString)",
            account: "refresh-singleflight"
        )
        defer { try? store.clear() }
        let account = MicrosoftGraphAccountSnapshot(
            id: "account-a",
            username: "person-a@example.invalid",
            name: "Synthetic Person A",
            environment: "test",
            tenantID: nil
        )
        try store.save(
            NativeMicrosoftOAuthSession(
                account: account,
                token: NativeMicrosoftOAuthTokenSet(
                    accessToken: "expired-access",
                    refreshToken: "refresh-r0",
                    idToken: "id-r0",
                    expiresOn: Date.distantPast,
                    scope: "User.Read Mail.Read offline_access",
                    tokenType: "Bearer"
                )
            )
        )

        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: account.id)
        let authorization = epoch.capture()
        let coordinator = NativeMicrosoftOAuthRefreshCoordinator()
        let probe = SuspendedNativeMicrosoftRefreshProbe()
        let client = NativeMicrosoftOAuthClient(
            keychainStore: store,
            authEpoch: epoch,
            refreshCoordinator: coordinator,
            refreshOperation: { refreshToken, scopes in
                await probe.refresh(refreshToken: refreshToken, scopes: scopes)
            }
        )

        let first = Task { @MainActor in
            try await client.acquireToken(
                scopes: MicrosoftGraphScope.inboxRead,
                forceRefresh: true,
                expectedEpoch: authorization
            )
        }
        await probe.waitUntilFirstRequestStarted()
        let second = Task { @MainActor in
            try await client.acquireToken(
                scopes: MicrosoftGraphScope.inboxRead,
                forceRefresh: true,
                expectedEpoch: authorization
            )
        }

        for _ in 0..<100 where coordinator.joinedRequestCount == 0 {
            await Task.yield()
        }
        #expect(coordinator.joinedRequestCount == 1)
        #expect(probe.requestCount == 1)

        probe.releaseFirstRequest()
        let firstToken = try await first.value
        let secondToken = try await second.value

        #expect(probe.requestCount == 1)
        #expect(probe.observedRefreshTokens == ["refresh-r0"])
        #expect(firstToken == secondToken)
        #expect(firstToken.refreshToken == "refresh-r1")
        #expect(store.load()?.token.refreshToken == "refresh-r1")
    }

    @Test @MainActor func interactiveReplacementAdvancesEpochForTheSameAccount() throws {
        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let refreshEpoch = epoch.capture()
        let signInEpoch = epoch.capture()
        var persistenceRan = false

        let committed = try epoch.replaceAccountIfCurrent(signInEpoch, with: "account-a") {
            persistenceRan = true
        }

        #expect(persistenceRan)
        #expect(!epoch.isCurrent(refreshEpoch, accountID: "account-a"))
        #expect(epoch.isCurrent(committed, accountID: "account-a"))
    }

    @Test @MainActor func staleInteractiveAccountCachedBeforeReturnCannotReloadAfterSignOut() throws {
        let epoch = MicrosoftGraphAuthEpoch()
        let requestEpoch = epoch.capture()
        let providerCache = FakeMicrosoftAccountCache(removalFails: true)

        // MSAL stores the identity before returning the interactive result.
        providerCache.cacheBeforeReturning(accountID: "account-a")
        epoch.invalidateForSignOut()

        do {
            _ = try epoch.replaceAccountIfCurrent(
                requestEpoch,
                with: "account-a",
                discardStale: {
                    // Simulate provider-cache cleanup failing. The process-local
                    // quarantine must still keep reload fail-closed.
                    try? providerCache.remove(accountID: "account-a")
                }
            ) {}
            Issue.record("Expected the completion from before sign-out to be rejected")
        } catch let error as MicrosoftGraphAuthEpochError {
            #expect(error == .staleCompletion)
        }

        #expect(providerCache.cachedAccountIDs == ["account-a"])
        let reloadable = providerCache.cachedAccountIDs.filter(epoch.isAccountReloadable)
        let reloaded = epoch.adoptCachedAccount(reloadable.first)
        #expect(reloadable.isEmpty)
        #expect(reloaded.accountID == nil)

        // A new, current interactive sign-in can intentionally admit A again.
        let explicitSignIn = epoch.capture()
        let admitted = try epoch.replaceAccountIfCurrent(explicitSignIn, with: "account-a") {}
        #expect(epoch.isCurrent(admitted, accountID: "account-a"))
        #expect(epoch.isAccountReloadable("account-a"))
    }

    @Test @MainActor func staleInteractiveQuarantineSurvivesRelaunchAndBlocksOtherCachedAccounts() throws {
        let suiteName = "OutlookAuthEpochPersistenceTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let firstProcess = MicrosoftGraphAuthEpoch(persistence: defaults)
        let pendingSignIn = firstProcess.capture()
        firstProcess.invalidateForSignOut()
        do {
            _ = try firstProcess.replaceAccountIfCurrent(
                pendingSignIn,
                with: "account-a"
            ) {}
            Issue.record("Expected stale interactive completion to be rejected")
        } catch let error as MicrosoftGraphAuthEpochError {
            #expect(error == .staleCompletion)
        }

        let relaunched = MicrosoftGraphAuthEpoch(persistence: defaults)
        #expect(!relaunched.isAccountReloadable("account-a"))
        #expect(relaunched.preferredCachedAccountID(
            from: ["account-a", "account-b"],
            previousAccountID: nil
        ) == nil)

        // Only a new explicit interactive replacement re-enables cached restore
        // and admits the chosen account across a subsequent process launch.
        _ = try relaunched.replaceAccountIfCurrent(
            relaunched.capture(),
            with: "account-a"
        ) {}
        let afterExplicitSignIn = MicrosoftGraphAuthEpoch(persistence: defaults)
        #expect(afterExplicitSignIn.isAccountReloadable("account-a"))
        #expect(afterExplicitSignIn.preferredCachedAccountID(
            from: ["account-a", "account-b"],
            previousAccountID: "account-a"
        ) == "account-a")
    }

    @Test @MainActor func cachedAccountAdoptionUsesDurableHashAndFailsClosedOnAmbiguity() throws {
        let suiteName = "OutlookPreferredAccountPersistenceTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let firstProcess = MicrosoftGraphAuthEpoch(persistence: defaults)
        _ = try firstProcess.replaceAccountIfCurrent(
            firstProcess.capture(),
            with: "account-a"
        ) {}
        let persistedValues = defaults.dictionaryRepresentation().values
        #expect(!persistedValues.contains { String(describing: $0).contains("account-a") })

        let relaunched = MicrosoftGraphAuthEpoch(persistence: defaults)
        let selected = relaunched.preferredCachedAccountID(
            from: ["account-b", "account-a"],
            previousAccountID: nil
        )
        #expect(selected == "account-a")
        #expect(relaunched.adoptCachedAccount(selected).accountID == "account-a")

        defaults.removePersistentDomain(forName: suiteName)
        let ambiguous = MicrosoftGraphAuthEpoch(persistence: defaults)
        #expect(ambiguous.preferredCachedAccountID(
            from: ["account-b", "account-a"],
            previousAccountID: nil
        ) == nil)
        #expect(ambiguous.preferredCachedAccountID(
            from: ["account-b"],
            previousAccountID: nil
        ) == nil)
        _ = try ambiguous.replaceAccountIfCurrent(
            ambiguous.capture(),
            with: "account-b"
        ) {}
        let admittedRelaunch = MicrosoftGraphAuthEpoch(persistence: defaults)
        #expect(admittedRelaunch.preferredCachedAccountID(
            from: ["account-a", "account-b"],
            previousAccountID: nil
        ) == "account-b")
    }

    @Test @MainActor func missingActiveCachedAccountCannotFallThroughToAnotherIdentity() throws {
        let suiteName = "OutlookMissingPreferredAccountTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let active = MicrosoftGraphAuthEpoch(initialAccountID: "account-a", persistence: defaults)

        #expect(active.preferredCachedAccountID(
            from: ["account-b"],
            previousAccountID: "account-a"
        ) == nil)
        #expect(active.adoptCachedAccount(nil).accountID == nil)

        let relaunched = MicrosoftGraphAuthEpoch(persistence: defaults)
        #expect(relaunched.preferredCachedAccountID(
            from: ["account-b"],
            previousAccountID: nil
        ) == nil)
    }

    @Test @MainActor func missingOrDuplicateStableMSALIdentifiersFailClosed() throws {
        #expect(MicrosoftGraphAuthManager.normalizedStableAccountID(nil) == nil)
        #expect(MicrosoftGraphAuthManager.normalizedStableAccountID("") == nil)
        #expect(MicrosoftGraphAuthManager.normalizedStableAccountID("   ") == nil)
        #expect(MicrosoftGraphAuthManager.normalizedStableAccountID(" stable-id ") == "stable-id")
        #expect(MicrosoftGraphAuthManager.validatedStableAccountIDs(["account-a", nil]) == nil)
        #expect(MicrosoftGraphAuthManager.validatedStableAccountIDs(["account-a", " "]) == nil)
        #expect(MicrosoftGraphAuthManager.validatedStableAccountIDs([" account-a "]) == ["account-a"])

        let poisonedDiscovery = MicrosoftGraphAuthEpoch()
        poisonedDiscovery.rejectCachedAccountDiscovery()
        #expect(poisonedDiscovery.preferredCachedAccountID(
            from: ["account-a"],
            previousAccountID: nil
        ) == nil)

        let suiteName = "OutlookDuplicateStableAccountTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let firstUse = MicrosoftGraphAuthEpoch(persistence: defaults)
        #expect(firstUse.preferredCachedAccountID(
            from: ["duplicate-id", "duplicate-id"],
            previousAccountID: nil
        ) == nil)

        _ = try firstUse.replaceAccountIfCurrent(
            firstUse.capture(),
            with: "duplicate-id"
        ) {}
        let relaunched = MicrosoftGraphAuthEpoch(persistence: defaults)
        #expect(relaunched.preferredCachedAccountID(
            from: ["duplicate-id", "duplicate-id"],
            previousAccountID: nil
        ) == nil)
    }

    @Test @MainActor func mailCacheSaveAndPurgeFailuresAreTypedAndSurfaced() {
        let saveState = FakeMicrosoftGraphMailCacheIOState(writeFails: true)
        let saveStore = MicrosoftGraphMailCacheStore(fileIO: saveState.fileIO(fileExists: false))
        let snapshot = MicrosoftGraphMailCacheStore.Snapshot(
            messages: [],
            deltaLink: nil,
            updatedAt: Date(timeIntervalSince1970: 100)
        )

        do {
            try saveStore.save(snapshot, accountID: "account-a")
            Issue.record("Expected protected cache save failure")
        } catch let error as MicrosoftGraphMailCacheError {
            #expect(error == .saveFailed)
        } catch {
            Issue.record("Unexpected protected cache save error: \(error)")
        }
        #expect(saveState.writeCount == 1)

        let purgeState = FakeMicrosoftGraphMailCacheIOState(removalFails: true)
        let purgeStore = MicrosoftGraphMailCacheStore(fileIO: purgeState.fileIO(fileExists: true))
        do {
            try purgeStore.clearAll()
            Issue.record("Expected protected cache purge failure")
        } catch let error as MicrosoftGraphMailCacheError {
            #expect(error == .purgeFailed)
        } catch {
            Issue.record("Unexpected protected cache purge error: \(error)")
        }
        #expect(purgeState.removeCount == 1)
    }

    @Test @MainActor func suspendedInboxRefreshCannotWriteOrPublishAfterSignOut() async {
        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let auth = FakeMicrosoftGraphInboxAuth(accountID: "account-a", epoch: epoch)
        let client = SuspendedMicrosoftGraphInboxClient()
        let cacheState = FakeMicrosoftGraphMailCacheIOState()
        let cache = MicrosoftGraphMailCacheStore(fileIO: cacheState.fileIO(fileExists: false))
        let viewModel = MicrosoftGraphInboxViewModel(auth: auth, client: client, cache: cache)
        let privateMessage = GraphMailMessage(
            id: "message-a",
            subject: "Synthetic private mailbox data from account A",
            bodyPreview: "Synthetic private preview",
            receivedDateTime: "2030-01-02T03:04:05Z",
            sentDateTime: nil,
            isRead: false,
            hasAttachments: false,
            from: nil,
            toRecipients: nil,
            ccRecipients: nil,
            body: nil,
            removed: nil
        )

        let refresh = Task { @MainActor in
            await viewModel.refresh()
        }
        await client.waitUntilFetchStarted()
        auth.signOut()
        await client.release(
            GraphMailPage(value: [privateMessage], odataNextLink: nil, odataDeltaLink: "https://graph.microsoft.com/v1.0/delta")
        )
        await refresh.value

        #expect(cacheState.writeCount == 0)
        #expect(viewModel.messages.isEmpty)
        #expect(viewModel.lastSyncDate == nil)
        #expect(viewModel.error as? MicrosoftGraphAuthEpochError == .staleCompletion)
    }

    @Test @MainActor func staleStatusCannotExposeOrActivatePreviousAccount() throws {
        let suiteName = "OutlookStatusEpochTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let referenceStore = OutlookRecentMessageReferenceStore(legacyDefaults: defaults)
        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let accountAAuthorization = epoch.capture()
        let staleAccount = MicrosoftGraphAccountSnapshot(
            id: "account-a",
            username: "private-a@example.invalid",
            name: "Private Account A",
            environment: "test",
            tenantID: nil
        )
        let accountBAuthorization = try epoch.replaceAccountIfCurrent(
            epoch.capture(),
            with: "account-b"
        ) {}
        referenceStore.activate(
            accountID: "account-b",
            authorization: accountBAuthorization
        )

        let outcome = OutlookTools.completedStatusOutcome(
            account: staleAccount,
            authorizationEpoch: accountAAuthorization,
            authEpoch: epoch,
            authProviderDescription: "Synthetic",
            diagnostics: ["authProvider": "synthetic"],
            referenceStore: referenceStore
        )

        #expect(outcome.status == .unavailable)
        #expect(outcome.errorCode == "outlook_account_changed")
        #expect(!outcome.text.contains("private-a@example.invalid"))
        #expect(!outcome.text.contains("Private Account A"))
        #expect(referenceStore.isActive(accountID: "account-b", authorization: accountBAuthorization))
        #expect(!referenceStore.isActive(accountID: "account-a", authorization: accountAAuthorization))
    }

    @Test @MainActor func sharedEpochInvalidatesTwoManagersAndTheirPresentationGate() {
        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let managerA = MicrosoftGraphAuthManager(authEpoch: epoch)
        let managerB = MicrosoftGraphAuthManager(authEpoch: epoch)
        let accountAAuthorization = epoch.capture()
        let presentation = OutlookPresentationAccountGate(
            accountID: "account-a",
            authorization: accountAAuthorization
        )

        #expect(managerA.authorizationState == accountAAuthorization)
        #expect(managerB.authorizationState == accountAAuthorization)
        #expect(presentation.isCurrent(managerA.authorizationState))
        #expect(presentation.isCurrent(managerB.authorizationState))

        epoch.invalidateForSignOut()

        #expect(managerA.authorizationState == managerB.authorizationState)
        #expect(managerA.authorizationState.accountID == nil)
        #expect(!presentation.isCurrent(managerA.authorizationState))
        #expect(!presentation.isCurrent(managerB.authorizationState))
    }

    @Test @MainActor func suspendedMessageBodyFetchCannotPublishAfterAccountInvalidation() async {
        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let auth = FakeMicrosoftGraphInboxAuth(accountID: "account-a", epoch: epoch)
        let client = SuspendedMicrosoftGraphMessageBodyClient()
        var publishedMessages: [GraphMailMessage] = []
        let privateBody = GraphMailMessage(
            id: "message-a",
            subject: "Private subject from account A",
            bodyPreview: "Private preview from account A",
            receivedDateTime: "2030-01-02T03:04:05Z",
            sentDateTime: nil,
            isRead: false,
            hasAttachments: false,
            from: nil,
            toRecipients: nil,
            ccRecipients: nil,
            body: .init(contentType: "Text", content: "Private body from account A"),
            removed: nil
        )

        let load = Task { @MainActor () -> Error? in
            do {
                try await MicrosoftGraphMessageBodyLoader.loadAndPublish(
                    messageID: privateBody.id,
                    accountID: "account-a",
                    auth: auth,
                    client: client,
                    publish: { publishedMessages.append($0) }
                )
                return nil
            } catch {
                return error
            }
        }
        await client.waitUntilFetchStarted()
        auth.signOut()
        await client.release(privateBody)

        #expect(await load.value as? MicrosoftGraphAuthEpochError == .staleCompletion)
        #expect(publishedMessages.isEmpty)
    }

    @Test @MainActor func completedMutationAfterAccountSwitchIsNonRetryableAndIndeterminate() async throws {
        let suiteName = "OutlookCompletedMutationAccountTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let referenceStore = OutlookRecentMessageReferenceStore(legacyDefaults: defaults)
        let authEpoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let authorization = authEpoch.capture()
        let fakeMutation = SuspendedOutlookMutation()

        referenceStore.activate(accountID: "account-a", authorization: authorization)
        let dispatch = Task { await fakeMutation.dispatch() }
        await fakeMutation.waitUntilDispatched()
        let accountBAuthorization = try authEpoch.replaceAccountIfCurrent(
            authEpoch.capture(),
            with: "account-b"
        ) {}
        referenceStore.activate(
            accountID: "account-b",
            authorization: accountBAuthorization
        )
        await fakeMutation.release()

        let rawOutcome = await dispatch.value
        let guarded = OutlookTools.completedOperationOutcome(
            rawOutcome,
            authorizedAccountID: "account-a",
            authorizationEpoch: authorization,
            authEpoch: authEpoch,
            diagnostics: ["authProvider": "synthetic"],
            referenceStore: referenceStore,
            effect: .mutation
        )

        #expect(await fakeMutation.dispatchCount == 1)
        #expect(guarded.status == .failed)
        #expect(guarded.errorCode == "outlook_mutation_indeterminate")
        #expect(guarded.structuredPayload["retryable"] == "false")
        #expect(guarded.text.localizedCaseInsensitiveContains("may have completed"))
        #expect(guarded.text.localizedCaseInsensitiveContains("verify in Outlook"))
        #expect(!guarded.text.localizedCaseInsensitiveContains("try again"))
        #expect(!guarded.text.contains("Synthetic mutation response from account A"))

        let providerFailureAfterDispatch = OutlookTools.completedOperationOutcome(
            OutlookToolOutcome(
                text: "Retry the synthetic mutation",
                status: .failed,
                availability: .providerError,
                errorCode: "synthetic_decode_failure",
                diagnostics: ["privateResult": "Synthetic private mutation result"]
            ),
            authorizedAccountID: "account-a",
            authorizationEpoch: authorization,
            authEpoch: authEpoch,
            diagnostics: ["authProvider": "synthetic"],
            referenceStore: referenceStore,
            effect: .mutation
        )
        #expect(providerFailureAfterDispatch.errorCode == "outlook_mutation_indeterminate")
        #expect(providerFailureAfterDispatch.structuredPayload["retryable"] == "false")
        #expect(!providerFailureAfterDispatch.text.localizedCaseInsensitiveContains("retry"))
        #expect(!providerFailureAfterDispatch.diagnostics.values.contains { $0.contains("private mutation result") })
    }

    @Test @MainActor func recentMessageReferencesAreTransientAndBoundToAuthorizationEpoch() throws {
        let suiteName = "OutlookRecentMessageReferenceStoreTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set(Data("legacy mailbox metadata".utf8), forKey: "OutlookTools.recentMessageReferences.v1")

        let store = OutlookRecentMessageReferenceStore(ttl: 30 * 60, legacyDefaults: defaults)
        #expect(defaults.object(forKey: "OutlookTools.recentMessageReferences.v1") == nil)

        let now = Date(timeIntervalSince1970: 1_000)
        let accountAReference = OutlookMessageReference(
            ordinal: 1,
            id: "message-a",
            subject: "Synthetic A",
            sender: "Sender A",
            receivedDateTime: nil,
            source: "test",
            cachedAt: now
        )
        let epoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let accountAFirstAuthorization = epoch.capture()

        store.activate(accountID: "account-a", authorization: accountAFirstAuthorization)
        store.save(
            [accountAReference],
            accountID: "account-a",
            authorization: accountAFirstAuthorization
        )
        #expect(store.load(
            accountID: "account-a",
            authorization: accountAFirstAuthorization,
            now: now
        ).map(\.id) == ["message-a"])

        let accountBAuthorization = try epoch.replaceAccountIfCurrent(
            epoch.capture(),
            with: "account-b"
        ) {}
        store.activate(accountID: "account-b", authorization: accountBAuthorization)
        let accountASecondAuthorization = try epoch.replaceAccountIfCurrent(
            epoch.capture(),
            with: "account-a"
        ) {}
        store.activate(accountID: "account-a", authorization: accountASecondAuthorization)

        // An A/g0 request completes after B/g1 and A/g2. Account-only binding
        // would accept this write; exact authorization binding must discard it.
        store.save(
            [accountAReference],
            accountID: "account-a",
            authorization: accountAFirstAuthorization
        )
        #expect(store.load(
            accountID: "account-a",
            authorization: accountASecondAuthorization,
            now: now
        ).isEmpty)

        store.save(
            [accountAReference],
            accountID: "account-a",
            authorization: accountASecondAuthorization
        )
        #expect(store.load(
            accountID: "account-a",
            authorization: accountASecondAuthorization,
            now: now
        ).map(\.id) == ["message-a"])
        store.clearAll()
        #expect(store.load(
            accountID: "account-a",
            authorization: accountASecondAuthorization,
            now: now
        ).isEmpty)
    }

    @Test @MainActor func completedOldAccountOutcomeIsSuppressedAfterAccountSwitch() throws {
        let suiteName = "OutlookCompletedOperationAccountTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = OutlookRecentMessageReferenceStore(legacyDefaults: defaults)
        let oldAccountOutcome = OutlookToolOutcome.success(
            "Synthetic private mailbox data from account A",
            diagnostics: ["providerPayload": "Synthetic private mailbox diagnostic from account A"]
        )
        let diagnostics = ["authProvider": "synthetic"]
        let authEpoch = MicrosoftGraphAuthEpoch(initialAccountID: "account-a")
        let accountAAuthorization = authEpoch.capture()

        store.activate(accountID: "account-a", authorization: accountAAuthorization)
        let accountBAuthorization = try authEpoch.replaceAccountIfCurrent(
            authEpoch.capture(),
            with: "account-b"
        ) {}
        store.activate(accountID: "account-b", authorization: accountBAuthorization)
        let suppressed = OutlookTools.completedOperationOutcome(
            oldAccountOutcome,
            authorizedAccountID: "account-a",
            authorizationEpoch: accountAAuthorization,
            authEpoch: authEpoch,
            diagnostics: diagnostics,
            referenceStore: store
        )

        #expect(suppressed.status == .unavailable)
        #expect(suppressed.availability == .authUnavailable)
        #expect(suppressed.errorCode == "outlook_account_changed")
        #expect(suppressed.diagnostics == diagnostics)
        #expect(!suppressed.text.contains("private mailbox data"))
        #expect(!suppressed.diagnostics.values.contains { $0.contains("private mailbox diagnostic") })

        let currentAccountOutcome = OutlookToolOutcome.success("Current account result")
        let allowed = OutlookTools.completedOperationOutcome(
            currentAccountOutcome,
            authorizedAccountID: "account-b",
            authorizationEpoch: accountBAuthorization,
            authEpoch: authEpoch,
            diagnostics: diagnostics,
            referenceStore: store
        )
        #expect(allowed.status == .success)
        #expect(allowed.text == "Current account result")
        #expect(allowed.diagnostics == diagnostics)
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

@MainActor
private final class SuspendedNativeMicrosoftRefreshProbe {
    private(set) var requestCount = 0
    private(set) var observedRefreshTokens: [String] = []
    private var firstRequestStarted = false
    private var firstRequestStartedContinuation: CheckedContinuation<Void, Never>?
    private var firstResponseContinuation: CheckedContinuation<NativeMicrosoftOAuthTokenSet, Never>?

    func refresh(
        refreshToken: String,
        scopes: [String]
    ) async -> NativeMicrosoftOAuthTokenSet {
        requestCount += 1
        observedRefreshTokens.append(refreshToken)
        if requestCount > 1 {
            return NativeMicrosoftOAuthTokenSet(
                accessToken: "unexpected-second-access",
                refreshToken: "unexpected-second-refresh",
                idToken: "unexpected-second-id",
                expiresOn: Date.distantFuture,
                scope: scopes.joined(separator: " "),
                tokenType: "Bearer"
            )
        }

        firstRequestStarted = true
        firstRequestStartedContinuation?.resume()
        firstRequestStartedContinuation = nil
        return await withCheckedContinuation { continuation in
            firstResponseContinuation = continuation
        }
    }

    func waitUntilFirstRequestStarted() async {
        guard !firstRequestStarted else { return }
        await withCheckedContinuation { continuation in
            firstRequestStartedContinuation = continuation
        }
    }

    func releaseFirstRequest() {
        firstResponseContinuation?.resume(
            returning: NativeMicrosoftOAuthTokenSet(
                accessToken: "access-r1",
                refreshToken: "refresh-r1",
                idToken: "id-r1",
                expiresOn: Date.distantFuture,
                scope: "User.Read Mail.Read offline_access",
                tokenType: "Bearer"
            )
        )
        firstResponseContinuation = nil
    }
}

private enum SyntheticMicrosoftGraphCacheFailure: Error {
    case write
    case remove
}

@MainActor
private final class FakeMicrosoftAccountCache {
    private(set) var cachedAccountIDs: [String] = []
    private let removalFails: Bool

    init(removalFails: Bool) {
        self.removalFails = removalFails
    }

    func cacheBeforeReturning(accountID: String) {
        cachedAccountIDs.append(accountID)
    }

    func remove(accountID: String) throws {
        if removalFails {
            throw SyntheticMicrosoftGraphCacheFailure.remove
        }
        cachedAccountIDs.removeAll { $0 == accountID }
    }
}

private final class FakeMicrosoftGraphMailCacheIOState {
    private(set) var writeCount = 0
    private(set) var removeCount = 0
    private let writeFails: Bool
    private let removalFails: Bool

    init(writeFails: Bool = false, removalFails: Bool = false) {
        self.writeFails = writeFails
        self.removalFails = removalFails
    }

    func fileIO(fileExists: Bool) -> MicrosoftGraphMailCacheFileIO {
        MicrosoftGraphMailCacheFileIO(
            cacheDirectory: {
                URL(fileURLWithPath: "/synthetic-tests/MicrosoftGraphMail", isDirectory: true)
            },
            fileExists: { _ in fileExists },
            readData: { _ in Data() },
            createDirectory: { _ in },
            writeData: { [self] _, _ in
                writeCount += 1
                if writeFails {
                    throw SyntheticMicrosoftGraphCacheFailure.write
                }
            },
            removeItem: { [self] _ in
                removeCount += 1
                if removalFails {
                    throw SyntheticMicrosoftGraphCacheFailure.remove
                }
            }
        )
    }
}

@MainActor
private final class FakeMicrosoftGraphInboxAuth: MicrosoftGraphInboxAuthorizing {
    private(set) var account: MicrosoftGraphAccountSnapshot?
    private(set) var token: MicrosoftGraphTokenSnapshot?
    private let epoch: MicrosoftGraphAuthEpoch

    init(accountID: String, epoch: MicrosoftGraphAuthEpoch) {
        self.epoch = epoch
        self.account = MicrosoftGraphAccountSnapshot(
            id: accountID,
            username: "synthetic@example.invalid",
            name: "Synthetic Account",
            environment: "test",
            tenantID: nil
        )
        self.token = MicrosoftGraphTokenSnapshot(
            accessToken: "synthetic-access-token",
            expiresOn: Date.distantFuture,
            scopes: MicrosoftGraphScope.inboxRead
        )
    }

    func acquireToken(
        scopes: [String],
        preferredAccountID: String?,
        forceRefresh: Bool
    ) async throws -> String {
        guard let account else { throw MicrosoftGraphAuthError.noAccount }
        let snapshot = epoch.capture()
        try epoch.requireCurrent(snapshot, accountID: account.id)
        if let preferredAccountID, preferredAccountID != account.id {
            throw MicrosoftGraphAuthEpochError.staleCompletion
        }
        return token?.accessToken ?? "synthetic-access-token"
    }

    func captureAuthorization(for accountID: String) throws -> MicrosoftGraphAuthEpoch.Snapshot {
        let snapshot = epoch.capture()
        try epoch.requireCurrent(snapshot, accountID: accountID)
        return snapshot
    }

    func requireCurrentAuthorization(
        _ snapshot: MicrosoftGraphAuthEpoch.Snapshot,
        for accountID: String
    ) throws {
        try epoch.requireCurrent(snapshot, accountID: accountID)
    }

    func signOut() {
        epoch.invalidateForSignOut()
        account = nil
        token = nil
    }
}

private actor SuspendedMicrosoftGraphInboxClient: MicrosoftGraphInboxClientProtocol {
    private var fetchStarted = false
    private var fetchStartedContinuation: CheckedContinuation<Void, Never>?
    private var fetchContinuation: CheckedContinuation<GraphMailPage, Never>?

    func fetchInboxPage(
        accessToken: String,
        pageSize: Int,
        nextOrDeltaLink: String?
    ) async throws -> GraphMailPage {
        await withCheckedContinuation { continuation in
            fetchContinuation = continuation
            fetchStarted = true
            fetchStartedContinuation?.resume()
            fetchStartedContinuation = nil
        }
    }

    func sendMail(_ mail: GraphSendMailRequest, accessToken: String) async throws {}

    func waitUntilFetchStarted() async {
        guard !fetchStarted else { return }
        await withCheckedContinuation { continuation in
            fetchStartedContinuation = continuation
        }
    }

    func release(_ page: GraphMailPage) {
        fetchContinuation?.resume(returning: page)
        fetchContinuation = nil
    }
}

private actor SuspendedMicrosoftGraphMessageBodyClient: MicrosoftGraphMessageBodyClientProtocol {
    private var fetchStarted = false
    private var fetchStartedContinuation: CheckedContinuation<Void, Never>?
    private var fetchContinuation: CheckedContinuation<GraphMailMessage, Never>?

    func fetchMessageBody(
        messageID: String,
        accessToken: String
    ) async throws -> GraphMailMessage {
        await withCheckedContinuation { continuation in
            fetchContinuation = continuation
            fetchStarted = true
            fetchStartedContinuation?.resume()
            fetchStartedContinuation = nil
        }
    }

    func waitUntilFetchStarted() async {
        guard !fetchStarted else { return }
        await withCheckedContinuation { continuation in
            fetchStartedContinuation = continuation
        }
    }

    func release(_ message: GraphMailMessage) {
        fetchContinuation?.resume(returning: message)
        fetchContinuation = nil
    }
}

private actor SuspendedOutlookMutation {
    private(set) var dispatchCount = 0
    private var dispatched = false
    private var dispatchedContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func dispatch() async -> OutlookToolOutcome {
        dispatchCount += 1
        dispatched = true
        dispatchedContinuation?.resume()
        dispatchedContinuation = nil
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
        return .success("Synthetic mutation response from account A")
    }

    func waitUntilDispatched() async {
        guard !dispatched else { return }
        await withCheckedContinuation { continuation in
            dispatchedContinuation = continuation
        }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}
