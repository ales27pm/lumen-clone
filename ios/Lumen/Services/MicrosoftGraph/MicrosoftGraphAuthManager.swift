import Foundation
import Observation
import OSLog
import UIKit
#if canImport(MSAL)
import MSAL
#endif

@MainActor
@Observable
final class MicrosoftGraphAuthManager {
    private let logger = Logger(subsystem: "ai.lumen.microsoftgraph", category: "auth")
    private let nativeOAuth = NativeMicrosoftOAuthClient()
    private(set) var account: MicrosoftGraphAccountSnapshot?
    private(set) var token: MicrosoftGraphTokenSnapshot?
    private(set) var tokensByScopeSet: [String: MicrosoftGraphTokenSnapshot] = [:]
    private(set) var accounts: [MicrosoftGraphAccountSnapshot] = []
    private(set) var isAuthenticating = false
    private(set) var lastError: Error?
    private let cachedForceNativeOAuth: Bool

    var isSignedIn: Bool { account != nil }
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
        return "msauth.\(Bundle.main.bundleIdentifier ?? "com.27pm.lumen")://auth"
    }
    var bundleIdentifier: String {
        Bundle.main.bundleIdentifier ?? "Unavailable"
    }

    init() {
        cachedForceNativeOAuth = (try? MicrosoftGraphConfiguration.load().forceNativeOAuth) ?? false
    }

    func bootstrap() async {
        await reloadCachedAccounts()
        guard let first = accounts.first else { return }
        do {
            _ = try await acquireToken(scopes: MicrosoftGraphScope.inboxRead, preferredAccountID: first.id, forceRefresh: false)
        } catch {
            lastError = error
            logger.info("Silent Microsoft token bootstrap failed: \(String(describing: error), privacy: .private)")
        }
    }

    func reloadCachedAccounts() async {
        if canUseMSAL {
            #if canImport(MSAL)
            do {
                let application = try makeMSALApplication()
                let cached = try application.allAccounts()
                let previousAccountID = account?.id
                accounts = cached.map(Self.snapshot(from:))
                if let previousAccountID {
                    account = accounts.first(where: { $0.id == previousAccountID }) ?? accounts.first
                } else {
                    account = accounts.first
                }
            } catch {
                lastError = error
            }
            #endif
        } else {
            let previousAccountID = account?.id
            accounts = nativeOAuth.cachedAccounts()
            if let previousAccountID {
                account = accounts.first(where: { $0.id == previousAccountID }) ?? accounts.first
            } else {
                account = accounts.first
            }
            if let session = nativeOAuth.loadCachedSession() {
                let restored = Self.tokenSnapshot(from: session.token, scopes: MicrosoftGraphScope.inboxRead)
                token = restored
                tokensByScopeSet[Self.scopeCacheKey(for: MicrosoftGraphScope.inboxRead)] = restored
                lastError = nil
            }
        }
    }

    func signIn(scopes: [String] = MicrosoftGraphScope.inboxRead, presentationViewController: UIViewController) async throws {
        isAuthenticating = true
        defer { isAuthenticating = false }
        do {
            let result = try await interactiveToken(scopes: scopes, presentationViewController: presentationViewController)
            token = result.token
            tokensByScopeSet[Self.scopeCacheKey(for: scopes)] = result.token
            account = result.account
            await reloadCachedAccounts()
        } catch {
            let normalizedError = normalize(error)
            lastError = normalizedError
            throw normalizedError
        }
    }

    func acquireToken(scopes: [String], preferredAccountID: String? = nil, forceRefresh: Bool = false) async throws -> String {
        #if canImport(MSAL)
        if canUseMSAL {
            let application = try makeMSALApplication()
            let cachedAccounts = try application.allAccounts()
            let selected = cachedAccounts.first { Self.snapshot(from: $0).id == preferredAccountID } ?? cachedAccounts.first
            guard let selected else { throw MicrosoftGraphAuthError.noAccount }

            do {
                let result = try await application.acquireTokenSilentAsync(scopes: scopes, account: selected, forceRefresh: forceRefresh)
                let mapped = Self.tokenSnapshot(from: result, scopes: scopes)
                token = mapped
                tokensByScopeSet[Self.scopeCacheKey(for: scopes)] = mapped
                account = Self.snapshot(from: result.account)
                return mapped.accessToken
            } catch let error as NSError where Self.isInteractionRequired(error) {
                token = nil
                tokensByScopeSet = [:]
                lastError = MicrosoftGraphAuthError.interactionRequired
                throw MicrosoftGraphAuthError.interactionRequired
            } catch {
                lastError = error
                throw error
            }
        }
        #endif

        do {
            let nativeToken = try await nativeOAuth.acquireToken(scopes: scopes, forceRefresh: forceRefresh)
            let mapped = Self.tokenSnapshot(from: nativeToken, scopes: scopes)
            token = mapped
            tokensByScopeSet[Self.scopeCacheKey(for: scopes)] = mapped
            if let session = nativeOAuth.loadCachedSession() {
                account = session.account
                accounts = [session.account]
            }
            return mapped.accessToken
        } catch {
            lastError = error
            throw error
        }
    }
    func cachedToken(for scopes: [String]) -> MicrosoftGraphTokenSnapshot? {
        tokensByScopeSet[Self.scopeCacheKey(for: scopes)]
    }

    func registerExternalError(_ error: Error) {
        lastError = normalize(error)
    }

    func signOutCurrentAccount() async {
        if canUseMSAL {
            #if canImport(MSAL)
            do {
                let application = try makeMSALApplication()
                let cached = try application.allAccounts()
                if let currentID = account?.id, let match = cached.first(where: { Self.snapshot(from: $0).id == currentID }) {
                    try application.remove(match)
                }
            } catch {
                lastError = error
            }
            token = nil
            tokensByScopeSet = [:]
            account = nil
            await reloadCachedAccounts()
            MicrosoftGraphMailCacheStore.shared.clearAll()
            #endif
        } else {
            nativeOAuth.signOut()
            token = nil
            tokensByScopeSet = [:]
            account = nil
            accounts = []
            MicrosoftGraphMailCacheStore.shared.clearAll()
        }
    }

    func handleOpenURL(_ url: URL) -> Bool {
        #if canImport(MSAL)
        return MSALPublicClientApplication.handleMSALResponse(url, sourceApplication: nil)
        #else
        return false
        #endif
    }

    private func interactiveToken(scopes: [String], presentationViewController: UIViewController) async throws -> (token: MicrosoftGraphTokenSnapshot, account: MicrosoftGraphAccountSnapshot) {
        #if canImport(MSAL)
        if canUseMSAL {
            let application = try makeMSALApplication()
            let webParams = MSALWebviewParameters(authPresentationViewController: presentationViewController)
            let params = MSALInteractiveTokenParameters(scopes: scopes, webviewParameters: webParams)
            params.promptType = .selectAccount
            let result = try await application.acquireTokenAsync(with: params)
            return (Self.tokenSnapshot(from: result, scopes: scopes), Self.snapshot(from: result.account))
        }
        #endif

        let session = try await nativeOAuth.signIn(scopes: scopes, presentationViewController: presentationViewController)
        return (Self.tokenSnapshot(from: session.token, scopes: scopes), session.account)
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
            Logger(subsystem: "ai.lumen.microsoftgraph", category: "msal").debug("[MSAL \(level.rawValue, privacy: .public)] \(message ?? "", privacy: .public)")
        }
    }()

    private nonisolated static func snapshot(from account: MSALAccount) -> MicrosoftGraphAccountSnapshot {
        let identifier = account.identifier ?? account.username ?? "unknown-msal-account"
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

    private nonisolated static func scopeCacheKey(for scopes: [String]) -> String {
        scopes.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.sorted().joined(separator: " ")
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
