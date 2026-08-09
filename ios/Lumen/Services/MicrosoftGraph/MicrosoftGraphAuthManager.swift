import Foundation
import Observation
import OSLog
import UIKit
#if canImport(MSAL)
import MSAL
#endif

nonisolated enum MicrosoftGraphAuthEpochError: LocalizedError, Equatable, Sendable {
    case staleCompletion

    var errorDescription: String? {
        "The Microsoft account changed while authentication was in progress. Try again."
    }
}

/// Process-wide compare-and-swap guard for Microsoft authentication work.
///
/// Every async authentication operation captures a snapshot before it starts. A
/// sign-out or interactive sign-in advances the epoch, so an older completion can
/// no longer persist credentials, mutate manager state, or reactivate mail context.
@MainActor
@Observable
final class MicrosoftGraphAuthEpoch {
    struct Snapshot: Equatable, Sendable {
        fileprivate let generation: UInt64
        let accountID: String?
    }

    static let shared = MicrosoftGraphAuthEpoch(persistence: .standard)

    private static let quarantinedAccountHashesKey = "lumen.microsoftGraph.authEpoch.quarantinedAccountHashes.v1"
    private static let cachedAdoptionDisabledKey = "lumen.microsoftGraph.authEpoch.cachedAdoptionDisabled.v1"
    private static let preferredAccountHashKey = "lumen.microsoftGraph.authEpoch.preferredAccountHash.v1"

    private var generation: UInt64 = 0
    private var activeAccountID: String?
    private var quarantinedAccountIDs: Set<String> = []
    private var quarantinedAccountHashes: Set<String>
    private var cachedAccountAdoptionAllowed: Bool
    private var preferredAccountHash: String?
    private let persistence: UserDefaults?

    init(initialAccountID: String? = nil, persistence: UserDefaults? = nil) {
        activeAccountID = initialAccountID
        self.persistence = persistence
        quarantinedAccountHashes = Set(
            persistence?.stringArray(forKey: Self.quarantinedAccountHashesKey) ?? []
        )
        cachedAccountAdoptionAllowed = !(persistence?.bool(forKey: Self.cachedAdoptionDisabledKey) ?? false)
        preferredAccountHash = persistence?.string(forKey: Self.preferredAccountHashKey)
        if let initialAccountID {
            preferredAccountHash = Self.accountHash(initialAccountID)
            cachedAccountAdoptionAllowed = true
            persistQuarantineState()
        }
    }

    func capture() -> Snapshot {
        Snapshot(generation: generation, accountID: activeAccountID)
    }

    func isCurrent(_ snapshot: Snapshot, accountID: String? = nil) -> Bool {
        guard snapshot.generation == generation,
              snapshot.accountID == activeAccountID else { return false }
        return accountID.map { $0 == activeAccountID } ?? true
    }

    func requireCurrent(_ snapshot: Snapshot, accountID: String? = nil) throws {
        guard isCurrent(snapshot, accountID: accountID) else {
            throw MicrosoftGraphAuthEpochError.staleCompletion
        }
    }

    /// A stale interactive MSAL completion can enter the provider cache before
    /// Lumen gets a chance to compare its auth epoch. Quarantined identities are
    /// never eligible for a later cached-account reload; an explicit current
    /// interactive sign-in is required to admit the identity again.
    func isAccountReloadable(_ accountID: String) -> Bool {
        !quarantinedAccountIDs.contains(accountID)
            && !quarantinedAccountHashes.contains(Self.accountHash(accountID))
    }

    /// Selects a cached identity without allowing sign-out to silently switch to
    /// another provider-cached account. A fresh process may restore its last cache;
    /// explicit sign-out durably disables all cached adoption until a new
    /// interactive sign-in succeeds.
    func preferredCachedAccountID(
        from candidateAccountIDs: [String],
        previousAccountID: String?
    ) -> String? {
        // Preserve multiplicity: two provider objects claiming the same stable ID
        // are ambiguous and must never collapse into a provider-order choice.
        let reloadable = candidateAccountIDs.filter(isAccountReloadable)
        if let activeAccountID {
            let activeMatches = reloadable.filter { $0 == activeAccountID }
            guard activeMatches.count == 1 else {
                disableCachedAdoption()
                return nil
            }
            rememberPreferredAccount(activeAccountID)
            return activeAccountID
        }
        guard cachedAccountAdoptionAllowed else { return nil }

        if let preferredAccountHash {
            let exactMatches = reloadable.filter {
                Self.accountHash($0) == preferredAccountHash
            }
            guard exactMatches.count == 1 else {
                // A missing preferred identity or an ambiguous hash match must not
                // silently fall back to a provider-defined account order.
                disableCachedAdoption()
                return nil
            }
            return exactMatches[0]
        }

        // Migration/first use admits only one unambiguous provider identity. The
        // raw process-local previous ID is intentionally not an identity selector.
        _ = previousAccountID
        guard reloadable.count <= 1 else {
            disableCachedAdoption()
            return nil
        }
        guard let onlyAccountID = reloadable.first else { return nil }
        rememberPreferredAccount(onlyAccountID)
        return onlyAccountID
    }

    /// Reconciles synchronous cached-account discovery with the process-wide
    /// account. A real identity change advances the epoch; repeated discovery of
    /// the same account does not invalidate legitimate work.
    @discardableResult
    func adoptCachedAccount(_ accountID: String?) -> Snapshot {
        let reloadableAccountID = accountID.flatMap { candidate in
            isAccountReloadable(candidate) ? candidate : nil
        }
        if let reloadableAccountID {
            guard cachedAccountAdoptionAllowed else { return capture() }
            let candidateHash = Self.accountHash(reloadableAccountID)
            if let preferredAccountHash, preferredAccountHash != candidateHash {
                disableCachedAdoption()
                return capture()
            }
            rememberPreferredAccount(reloadableAccountID)
        }
        if activeAccountID != reloadableAccountID {
            generation &+= 1
            activeAccountID = reloadableAccountID
        }
        return capture()
    }

    /// An unaddressable or otherwise ambiguous provider-cache entry poisons the
    /// whole discovery result. Dropping it could make a mixed cache appear unique.
    @discardableResult
    func rejectCachedAccountDiscovery() -> Snapshot {
        disableCachedAdoption()
        if activeAccountID != nil {
            generation &+= 1
            activeAccountID = nil
        }
        return capture()
    }

    /// Invalidates every in-flight completion, even when no account is currently
    /// active. This also blocks a sign-in that began before a sign-out request.
    @discardableResult
    func invalidateForSignOut() -> String? {
        let previousAccountID = activeAccountID
        if let previousAccountID {
            quarantine(previousAccountID)
        }
        cachedAccountAdoptionAllowed = false
        preferredAccountHash = nil
        persistQuarantineState()
        generation &+= 1
        activeAccountID = nil
        return previousAccountID
    }

    /// Commits an interactive authentication replacement. The persistence closure
    /// runs only after the epoch comparison and without an actor suspension; the
    /// epoch then advances even when the selected account ID is unchanged.
    func replaceAccountIfCurrent(
        _ expected: Snapshot,
        with accountID: String,
        discardStale: () -> Void = {},
        persist: () throws -> Void
    ) throws -> Snapshot {
        guard isCurrent(expected) else {
            // Do not evict an identity already admitted by a newer successful
            // sign-in. Otherwise a late duplicate completion could remove the
            // newer session for the same account.
            if activeAccountID != accountID {
                quarantine(accountID)
                discardStale()
            }
            throw MicrosoftGraphAuthEpochError.staleCompletion
        }
        try persist()
        if let previousAccountID = activeAccountID, previousAccountID != accountID {
            quarantine(previousAccountID)
        }
        admit(accountID)
        generation &+= 1
        activeAccountID = accountID
        return capture()
    }

    /// Awaits untrusted/external work, then commits its value only if the captured
    /// account generation is still current. The commit cannot interleave with a
    /// sign-out or account switch because it is synchronous on the main actor.
    func commitCurrentCompletion<Value>(
        expected: Snapshot,
        accountID: String,
        operation: () async throws -> Value,
        commit: (Value) throws -> Void
    ) async throws -> Value {
        let value = try await operation()
        try requireCurrent(expected, accountID: accountID)
        try commit(value)
        return value
    }

    private func quarantine(_ accountID: String) {
        quarantinedAccountIDs.insert(accountID)
        quarantinedAccountHashes.insert(Self.accountHash(accountID))
        persistQuarantineState()
    }

    private func admit(_ accountID: String) {
        quarantinedAccountIDs.remove(accountID)
        quarantinedAccountHashes.remove(Self.accountHash(accountID))
        cachedAccountAdoptionAllowed = true
        preferredAccountHash = Self.accountHash(accountID)
        persistQuarantineState()
    }

    private func rememberPreferredAccount(_ accountID: String) {
        let accountHash = Self.accountHash(accountID)
        guard preferredAccountHash != accountHash else { return }
        preferredAccountHash = accountHash
        persistQuarantineState()
    }

    private func disableCachedAdoption() {
        guard cachedAccountAdoptionAllowed else { return }
        cachedAccountAdoptionAllowed = false
        persistQuarantineState()
    }

    private func persistQuarantineState() {
        guard let persistence else { return }
        persistence.set(
            quarantinedAccountHashes.sorted(),
            forKey: Self.quarantinedAccountHashesKey
        )
        persistence.set(
            !cachedAccountAdoptionAllowed,
            forKey: Self.cachedAdoptionDisabledKey
        )
        if let preferredAccountHash {
            persistence.set(preferredAccountHash, forKey: Self.preferredAccountHashKey)
        } else {
            persistence.removeObject(forKey: Self.preferredAccountHashKey)
        }
    }

    private nonisolated static func accountHash(_ accountID: String) -> String {
        RuntimeFallbackLogger.promptHash(accountID)
    }
}

@MainActor
@Observable
final class MicrosoftGraphAuthManager {
    private let logger = Logger(subsystem: "ai.lumen.microsoftgraph", category: "auth")
    private let nativeOAuth: NativeMicrosoftOAuthClient
    private let authEpoch: MicrosoftGraphAuthEpoch
    private(set) var account: MicrosoftGraphAccountSnapshot?
    private(set) var token: MicrosoftGraphTokenSnapshot?
    private(set) var tokensByScopeSet: [String: MicrosoftGraphTokenSnapshot] = [:]
    private(set) var accounts: [MicrosoftGraphAccountSnapshot] = []
    private(set) var isAuthenticating = false
    private(set) var lastError: Error?
    private let cachedForceNativeOAuth: Bool

    var isSignedIn: Bool {
        guard let accountID = account?.id else { return false }
        return authEpoch.isCurrent(authEpoch.capture(), accountID: accountID)
    }
    var authorizationState: MicrosoftGraphAuthEpoch.Snapshot {
        authEpoch.capture()
    }
    private var shouldUseNativeOAuth: Bool {
        cachedForceNativeOAuth
    }
    var canUseMSAL: Bool {
        guard !shouldUseNativeOAuth else { return false }
        #if canImport(MSAL)
        return true
        #else
        return false
        #endif
    }
    var authProviderDescription: String { canUseMSAL ? "MSAL" : "Native OAuth PKCE" }
    var authProviderPath: String { canUseMSAL ? "MSAL" : "Native fallback (ASWebAuthenticationSession + PKCE)" }
    var activeClientID: String {
        (try? MicrosoftGraphConfiguration.load().clientID) ?? "Unavailable"
    }
    var activeRedirectURI: String {
        if let configured = try? MicrosoftGraphConfiguration.load().redirectURI {
            return configured
        }
        return "msauth.\(Bundle.main.bundleIdentifier ?? "com.27pm.lumenclone")://auth"
    }
    var bundleIdentifier: String {
        Bundle.main.bundleIdentifier ?? "Unavailable"
    }

    convenience init() {
        self.init(authEpoch: .shared)
    }

    init(authEpoch: MicrosoftGraphAuthEpoch) {
        self.authEpoch = authEpoch
        self.nativeOAuth = NativeMicrosoftOAuthClient(
            keychainStore: NativeMicrosoftOAuthKeychainStore(),
            authEpoch: authEpoch
        )
        cachedForceNativeOAuth = (try? MicrosoftGraphConfiguration.load().forceNativeOAuth) ?? false
    }

    func bootstrap() async {
        lastError = nil
        await reloadCachedAccounts()
        guard lastError == nil else { return }
        guard let selectedAccountID = account?.id else { return }
        do {
            _ = try await acquireToken(
                scopes: MicrosoftGraphScope.inboxRead,
                preferredAccountID: selectedAccountID,
                forceRefresh: false
            )
        } catch {
            logger.info("Silent Microsoft token bootstrap failed: \(String(describing: error), privacy: .private)")
        }
    }

    func reloadCachedAccounts() async {
        let previousAccountID = account?.id
        let globallyActiveAccountID = authEpoch.capture().accountID
        if canUseMSAL {
            #if canImport(MSAL)
            do {
                let application = try makeMSALApplication()
                let cached = try application.allAccounts()
                guard Self.validatedStableAccountIDs(cached.map(\.identifier)) != nil else {
                    authEpoch.rejectCachedAccountDiscovery()
                    accounts = []
                    account = nil
                    token = nil
                    tokensByScopeSet = [:]
                    lastError = MicrosoftGraphAuthError.noAccount
                    synchronizeRecentReferenceAccount()
                    return
                }
                let reloadedAccounts = cached
                    .compactMap(Self.snapshot(from:))
                    .filter { authEpoch.isAccountReloadable($0.id) }
                let preferredID = authEpoch.preferredCachedAccountID(
                    from: reloadedAccounts.map(\.id),
                    previousAccountID: globallyActiveAccountID ?? previousAccountID
                )
                let selected = preferredID.flatMap { preferredID in
                    reloadedAccounts.first(where: { $0.id == preferredID })
                }
                authEpoch.adoptCachedAccount(selected?.id)
                accounts = reloadedAccounts
                account = selected
                if selected == nil {
                    token = nil
                    tokensByScopeSet = [:]
                }
                lastError = nil
            } catch {
                lastError = error
            }
            #endif
        } else {
            let session = nativeOAuth.loadCachedSession()
            let reloadedAccounts = session
                .map { [$0.account] }
                .map { $0.filter { authEpoch.isAccountReloadable($0.id) } } ?? []
            let preferredID = authEpoch.preferredCachedAccountID(
                from: reloadedAccounts.map(\.id),
                previousAccountID: globallyActiveAccountID ?? previousAccountID
            )
            let selected = preferredID.flatMap { preferredID in
                reloadedAccounts.first(where: { $0.id == preferredID })
            }
            authEpoch.adoptCachedAccount(selected?.id)
            accounts = reloadedAccounts
            account = selected
            if let session, selected?.id == session.account.id {
                let restored = Self.tokenSnapshot(from: session.token, scopes: MicrosoftGraphScope.inboxRead)
                token = restored
                tokensByScopeSet[Self.scopeCacheKey(for: MicrosoftGraphScope.inboxRead)] = restored
                lastError = nil
            } else {
                token = nil
                tokensByScopeSet = [:]
            }
        }
        synchronizeRecentReferenceAccount()
    }

    func signIn(scopes: [String] = MicrosoftGraphScope.inboxRead, presentationViewController: UIViewController) async throws {
        let requestEpoch = authEpoch.capture()
        isAuthenticating = true
        defer { isAuthenticating = false }
        do {
            let result = try await interactiveToken(
                scopes: scopes,
                presentationViewController: presentationViewController,
                expectedEpoch: requestEpoch
            )
            try authEpoch.requireCurrent(result.authorizationEpoch, accountID: result.account.id)
            token = result.token
            tokensByScopeSet[Self.scopeCacheKey(for: scopes)] = result.token
            account = result.account
            accounts = [result.account]
            lastError = nil
            synchronizeRecentReferenceAccount()
            await reloadCachedAccounts()
        } catch {
            let normalizedError = normalize(error)
            if authEpoch.isCurrent(requestEpoch) {
                lastError = normalizedError
            }
            throw normalizedError
        }
    }

    func acquireToken(scopes: [String], preferredAccountID: String? = nil, forceRefresh: Bool = false) async throws -> String {
        let requestEpoch = authEpoch.capture()
        guard let activeAccountID = requestEpoch.accountID else {
            throw MicrosoftGraphAuthError.noAccount
        }
        if let preferredAccountID, preferredAccountID != activeAccountID {
            throw MicrosoftGraphAuthEpochError.staleCompletion
        }

        #if canImport(MSAL)
        if canUseMSAL {
            let application = try makeMSALApplication()
            let cachedAccounts = try application.allAccounts()
            let selectedMatches = cachedAccounts.filter {
                Self.snapshot(from: $0)?.id == activeAccountID
            }
            guard selectedMatches.count == 1,
                  let selected = selectedMatches.first else {
                throw MicrosoftGraphAuthError.noAccount
            }

            do {
                let result = try await application.acquireTokenSilentAsync(scopes: scopes, account: selected, forceRefresh: forceRefresh)
                guard let resultAccount = Self.snapshot(from: result.account) else {
                    throw MicrosoftGraphAuthError.noAccount
                }
                try authEpoch.requireCurrent(requestEpoch, accountID: resultAccount.id)
                let mapped = Self.tokenSnapshot(from: result, scopes: scopes)
                token = mapped
                tokensByScopeSet[Self.scopeCacheKey(for: scopes)] = mapped
                account = resultAccount
                accounts = cachedAccounts.compactMap(Self.snapshot(from:))
                synchronizeRecentReferenceAccount()
                lastError = nil
                return mapped.accessToken
            } catch let error as NSError where Self.isInteractionRequired(error) {
                try authEpoch.requireCurrent(requestEpoch, accountID: activeAccountID)
                token = nil
                tokensByScopeSet = [:]
                lastError = MicrosoftGraphAuthError.interactionRequired
                throw MicrosoftGraphAuthError.interactionRequired
            } catch {
                try authEpoch.requireCurrent(requestEpoch, accountID: activeAccountID)
                lastError = error
                throw error
            }
        }
        #endif

        do {
            let nativeToken = try await nativeOAuth.acquireToken(
                scopes: scopes,
                forceRefresh: forceRefresh,
                expectedEpoch: requestEpoch
            )
            try authEpoch.requireCurrent(requestEpoch, accountID: activeAccountID)
            guard let session = nativeOAuth.loadCachedSession(), session.account.id == activeAccountID else {
                throw MicrosoftGraphAuthEpochError.staleCompletion
            }
            let mapped = Self.tokenSnapshot(from: nativeToken, scopes: scopes)
            token = mapped
            tokensByScopeSet[Self.scopeCacheKey(for: scopes)] = mapped
            account = session.account
            accounts = [session.account]
            synchronizeRecentReferenceAccount()
            lastError = nil
            return mapped.accessToken
        } catch {
            try authEpoch.requireCurrent(requestEpoch, accountID: activeAccountID)
            lastError = error
            throw error
        }
    }
    func cachedToken(for scopes: [String]) -> MicrosoftGraphTokenSnapshot? {
        tokensByScopeSet[Self.scopeCacheKey(for: scopes)]
    }

    func captureAuthorization(for accountID: String) throws -> MicrosoftGraphAuthEpoch.Snapshot {
        let snapshot = authEpoch.capture()
        try authEpoch.requireCurrent(snapshot, accountID: accountID)
        return snapshot
    }

    func requireCurrentAuthorization(
        _ snapshot: MicrosoftGraphAuthEpoch.Snapshot,
        for accountID: String
    ) throws {
        try authEpoch.requireCurrent(snapshot, accountID: accountID)
    }

    func registerExternalError(_ error: Error) {
        lastError = normalize(error)
    }

    func signOutCurrentAccount() async throws {
        let accountIDToRemove = authEpoch.invalidateForSignOut() ?? account?.id
        // Purge mail-derived context before any account transition. These caches
        // must never survive a sign-out attempt or be reused by another account.
        OutlookRecentMessageReferenceStore.shared.clearAll()
        var signOutError: Error?
        do {
            try MicrosoftGraphMailCacheStore.shared.clearAll()
        } catch {
            // Continue removing credentials and local account state, but retain
            // the typed purge failure for the UI instead of reporting success.
            signOutError = error
        }
        token = nil
        tokensByScopeSet = [:]
        account = nil
        accounts = []
        if canUseMSAL {
            #if canImport(MSAL)
            do {
                let application = try makeMSALApplication()
                let cached = try application.allAccounts()
                if let accountIDToRemove {
                    let matches = cached.filter {
                        Self.snapshot(from: $0)?.id == accountIDToRemove
                    }
                    guard matches.count == 1, let match = matches.first else {
                        throw MicrosoftGraphAuthError.noAccount
                    }
                    try application.remove(match)
                }
            } catch {
                signOutError = signOutError ?? error
            }
            #endif
        } else {
            do {
                try nativeOAuth.signOut()
            } catch {
                signOutError = signOutError ?? error
            }
        }
        lastError = signOutError
        if let signOutError {
            throw signOutError
        }
    }

    func handleOpenURL(_ url: URL) -> Bool {
        #if canImport(MSAL)
        return MSALPublicClientApplication.handleMSALResponse(url, sourceApplication: nil)
        #else
        return false
        #endif
    }

    private func interactiveToken(
        scopes: [String],
        presentationViewController: UIViewController,
        expectedEpoch: MicrosoftGraphAuthEpoch.Snapshot
    ) async throws -> (
        token: MicrosoftGraphTokenSnapshot,
        account: MicrosoftGraphAccountSnapshot,
        authorizationEpoch: MicrosoftGraphAuthEpoch.Snapshot
    ) {
        #if canImport(MSAL)
        if canUseMSAL {
            let application = try makeMSALApplication()
            let webParams = MSALWebviewParameters(authPresentationViewController: presentationViewController)
            let params = MSALInteractiveTokenParameters(scopes: scopes, webviewParameters: webParams)
            params.promptType = .selectAccount
            let result = try await application.acquireTokenAsync(with: params)
            guard let account = Self.snapshot(from: result.account) else {
                do {
                    try application.remove(result.account)
                } catch {
                    logger.error("Failed to remove an MSAL account without a stable identifier.")
                }
                throw MicrosoftGraphAuthError.noAccount
            }
            let committedEpoch = try authEpoch.replaceAccountIfCurrent(
                expectedEpoch,
                with: account.id,
                discardStale: {
                    do {
                        try application.remove(result.account)
                    } catch {
                        // The process-local quarantine remains authoritative even
                        // if the provider cache cannot be updated immediately.
                        logger.error("Failed to remove a stale MSAL account from the provider cache.")
                    }
                }
            ) {}
            return (Self.tokenSnapshot(from: result, scopes: scopes), account, committedEpoch)
        }
        #endif

        let result = try await nativeOAuth.signIn(
            scopes: scopes,
            presentationViewController: presentationViewController,
            expectedEpoch: expectedEpoch
        )
        return (
            Self.tokenSnapshot(from: result.session.token, scopes: scopes),
            result.session.account,
            result.authorizationEpoch
        )
    }

    private nonisolated static func tokenSnapshot(from token: NativeMicrosoftOAuthTokenSet, scopes: [String]) -> MicrosoftGraphTokenSnapshot {
        MicrosoftGraphTokenSnapshot(accessToken: token.accessToken, expiresOn: token.expiresOn, scopes: scopes)
    }

    #if canImport(MSAL)
    private nonisolated static func isInteractionRequired(_ error: NSError) -> Bool {
        error.domain == MSALErrorDomain && error.code == MSALError.interactionRequired.rawValue
    }

    private func makeMSALApplication() throws -> MSALPublicClientApplication {
        let config = try MicrosoftGraphConfiguration.load()
        let authority = try MSALAuthority(url: config.authorityURL)
        let appConfig = MSALPublicClientApplicationConfig(clientId: config.clientID, redirectUri: config.redirectURI, authority: authority)
        if !config.keychainSharingGroup.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           config.keychainSharingGroup != "com.microsoft.adalcache" {
            appConfig.cacheConfig.keychainSharingGroup = config.keychainSharingGroup
        }

        configureMSALLoggingIfNeeded()
        return try MSALPublicClientApplication(configuration: appConfig)
    }

    private func configureMSALLoggingIfNeeded() {
        // MSAL exposes global logger configuration per process. We intentionally configure
        // it once and share it across all MicrosoftGraphAuthManager instances.
        _ = Self.configureMSALLoggingOnce
    }

    private nonisolated static let configureMSALLoggingOnce: Void = {
        MSALGlobalConfig.loggerConfig.logLevel = .warning
        MSALGlobalConfig.loggerConfig.setLogCallback { level, message, containsPII in
            guard !containsPII else { return }
            Logger(subsystem: "ai.lumen.microsoftgraph", category: "msal").debug("[MSAL \(level.rawValue, privacy: .public)] \(message ?? "", privacy: .private)")
        }
    }()

    private nonisolated static func snapshot(from account: MSALAccount) -> MicrosoftGraphAccountSnapshot? {
        guard let identifier = normalizedStableAccountID(account.identifier) else {
            return nil
        }
        return MicrosoftGraphAccountSnapshot(
            id: identifier,
            username: account.username,
            name: account.accountClaims?["name"] as? String,
            environment: account.environment,
            tenantID: nil
        )
    }

    private nonisolated static func tokenSnapshot(from result: MSALResult, scopes: [String]) -> MicrosoftGraphTokenSnapshot {
        MicrosoftGraphTokenSnapshot(accessToken: result.accessToken, expiresOn: result.expiresOn, scopes: scopes)
    }
    #endif

    nonisolated static func normalizedStableAccountID(_ identifier: String?) -> String? {
        guard let identifier = identifier?.trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty else { return nil }
        return identifier
    }

    nonisolated static func validatedStableAccountIDs(
        _ identifiers: [String?]
    ) -> [String]? {
        let normalized = identifiers.compactMap(normalizedStableAccountID)
        guard normalized.count == identifiers.count else { return nil }
        return normalized
    }

    private nonisolated static func scopeCacheKey(for scopes: [String]) -> String {
        scopes.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.sorted().joined(separator: " ")
    }

    private func synchronizeRecentReferenceAccount() {
        let currentEpoch = authEpoch.capture()
        if let accountID = account?.id,
           authEpoch.isCurrent(currentEpoch, accountID: accountID) {
            OutlookRecentMessageReferenceStore.shared.activate(
                accountID: accountID,
                authorization: currentEpoch
            )
        } else if currentEpoch.accountID == nil {
            OutlookRecentMessageReferenceStore.shared.clearAll()
        }
    }

    private func normalize(_ error: Error) -> Error {
        #if canImport(MSAL)
        guard let nsError = error as NSError?, nsError.domain == MSALErrorDomain else { return error }

        if nsError.code == MSALError.internal.rawValue {
            let message = """
            Microsoft sign-in failed with an internal MSAL error (\(nsError.code)). This usually means the Entra app configuration does not match this iOS build. Verify the Client ID, Redirect URI, and iOS bundle ID in Azure App Registration.
            Diagnostics:
            - Active client ID: \(activeClientID)
            - Redirect URI: \(activeRedirectURI)
            - Bundle ID: \(bundleIdentifier)
            - Auth provider path: \(authProviderPath)
            """
            return MicrosoftGraphAuthError.invalidConfiguration(message)
        }
        #endif
        return error
    }
}

#if canImport(MSAL)
nonisolated extension MSALPublicClientApplication {
    func acquireTokenSilentAsync(scopes: [String], account: MSALAccount, forceRefresh: Bool = false) async throws -> MSALResult {
        let params = MSALSilentTokenParameters(scopes: scopes, account: account)
        params.forceRefresh = forceRefresh
        return try await withCheckedThrowingContinuation { continuation in
            acquireTokenSilent(with: params) { result, error in
                if let result {
                    continuation.resume(returning: result)
                } else {
                    continuation.resume(throwing: error ?? MicrosoftGraphAuthError.interactionRequired)
                }
            }
        }
    }

    func acquireTokenAsync(with parameters: MSALInteractiveTokenParameters) async throws -> MSALResult {
        try await withCheckedThrowingContinuation { continuation in
            acquireToken(with: parameters) { result, error in
                if let result {
                    continuation.resume(returning: result)
                } else {
                    continuation.resume(throwing: error ?? MicrosoftGraphAuthError.signInCancelled)
                }
            }
        }
    }
}
#endif

nonisolated enum MicrosoftGraphPresenter {
    @MainActor
    static func topViewController() -> UIViewController? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let activeScene = scenes.first { $0.activationState == .foregroundActive } ?? scenes.first
        let root = activeScene?.windows.first { $0.isKeyWindow }?.rootViewController
        return top(from: root)
    }

    @MainActor
    private static func top(from controller: UIViewController?) -> UIViewController? {
        if let nav = controller as? UINavigationController { return top(from: nav.visibleViewController) }
        if let tab = controller as? UITabBarController { return top(from: tab.selectedViewController) }
        if let presented = controller?.presentedViewController { return top(from: presented) }
        return controller
    }
}

nonisolated enum MicrosoftGraphURLHandler {
    @MainActor
    @discardableResult
    static func handle(_ url: URL) -> Bool {
        #if canImport(MSAL)
        return MSALPublicClientApplication.handleMSALResponse(url, sourceApplication: nil)
        #else
        return false
        #endif
    }
}
