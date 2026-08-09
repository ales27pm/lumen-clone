import AuthenticationServices
import CryptoKit
import Foundation
import OSLog
import Security

nonisolated struct NativeMicrosoftOAuthTokenSet: Codable, Hashable, Sendable {
    let accessToken: String
    let refreshToken: String?
    let idToken: String?
    let expiresOn: Date
    let scope: String?
    let tokenType: String

    var shouldRefreshProactively: Bool {
        expiresOn.timeIntervalSinceNow < 300
    }
}

nonisolated struct NativeMicrosoftOAuthProfile: Codable, Sendable {
    let id: String
    let displayName: String?
    let userPrincipalName: String?
    let mail: String?
}

nonisolated struct NativeMicrosoftOAuthSession: Codable, Hashable, Sendable {
    let account: MicrosoftGraphAccountSnapshot
    let token: NativeMicrosoftOAuthTokenSet
}

typealias NativeMicrosoftOAuthRefreshOperation = @MainActor (
    _ refreshToken: String,
    _ scopes: [String]
) async throws -> NativeMicrosoftOAuthTokenSet

/// Process-wide singleflight and serialization for native refresh-token use.
///
/// Refresh-token rotation is ordered by the provider, not by response arrival.
/// Allowing two requests to leave with the same token can therefore leave the app
/// holding a server-obsolete token even if local persistence uses compare-and-swap.
@MainActor
final class NativeMicrosoftOAuthRefreshCoordinator {
    static let shared = NativeMicrosoftOAuthRefreshCoordinator()

    private struct InFlightRefresh {
        let id: UUID
        let scopes: Set<String>
        let task: Task<NativeMicrosoftOAuthTokenSet, Error>
    }

    private var inFlightByAccountID: [String: InFlightRefresh] = [:]
    private(set) var joinedRequestCount = 0

    func refreshToken(
        for accountID: String,
        scopes: [String],
        operation: @escaping @MainActor () async throws -> NativeMicrosoftOAuthTokenSet
    ) async throws -> NativeMicrosoftOAuthTokenSet {
        let requestedScopes = Set(
            scopes.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        )
        if let inFlight = inFlightByAccountID[accountID] {
            if inFlight.scopes.isSuperset(of: requestedScopes) {
                joinedRequestCount += 1
                return try await inFlight.task.value
            }

            // A narrower request is already using this account's refresh token.
            // Wait for it to commit (or fail), then re-read the protected session
            // in a new serialized operation for the broader scope set.
            _ = try? await inFlight.task.value
            return try await refreshToken(
                for: accountID,
                scopes: scopes,
                operation: operation
            )
        }

        let id = UUID()
        let task = Task { @MainActor [self] in
            defer { finishRefresh(id: id, accountID: accountID) }
            return try await operation()
        }
        inFlightByAccountID[accountID] = InFlightRefresh(
            id: id,
            scopes: requestedScopes,
            task: task
        )
        return try await task.value
    }

    private func finishRefresh(id: UUID, accountID: String) {
        guard inFlightByAccountID[accountID]?.id == id else { return }
        inFlightByAccountID[accountID] = nil
    }
}

@MainActor
final class NativeMicrosoftOAuthClient: NSObject, ASWebAuthenticationPresentationContextProviding {
    private let logger = Logger(subsystem: "ai.lumen.microsoftgraph", category: "native-oauth")
    private let callbackScheme: String
    private var activeSession: ASWebAuthenticationSession?
    private weak var presentationAnchor: UIWindow?
    private let keychainStore: NativeMicrosoftOAuthKeychainStore
    private let authEpoch: MicrosoftGraphAuthEpoch
    private let refreshCoordinator: NativeMicrosoftOAuthRefreshCoordinator
    private let refreshOperation: NativeMicrosoftOAuthRefreshOperation?

    override convenience init() {
        self.init(
            keychainStore: NativeMicrosoftOAuthKeychainStore(),
            authEpoch: .shared
        )
    }

    convenience init(keychainStore: NativeMicrosoftOAuthKeychainStore) {
        self.init(keychainStore: keychainStore, authEpoch: .shared)
    }

    convenience init(
        keychainStore: NativeMicrosoftOAuthKeychainStore,
        authEpoch: MicrosoftGraphAuthEpoch
    ) {
        self.init(
            keychainStore: keychainStore,
            authEpoch: authEpoch,
            refreshCoordinator: .shared,
            refreshOperation: nil
        )
    }

    init(
        keychainStore: NativeMicrosoftOAuthKeychainStore,
        authEpoch: MicrosoftGraphAuthEpoch,
        refreshCoordinator: NativeMicrosoftOAuthRefreshCoordinator,
        refreshOperation: NativeMicrosoftOAuthRefreshOperation?
    ) {
        let bundleID = Bundle.main.bundleIdentifier ?? "com.27pm.lumenclone"
        self.callbackScheme = "msauth.\(bundleID)"
        self.keychainStore = keychainStore
        self.authEpoch = authEpoch
        self.refreshCoordinator = refreshCoordinator
        self.refreshOperation = refreshOperation
        super.init()
    }

    func loadCachedSession() -> NativeMicrosoftOAuthSession? {
        keychainStore.load()
    }

    func cachedAccounts() -> [MicrosoftGraphAccountSnapshot] {
        guard let session = loadCachedSession() else { return [] }
        return [session.account]
    }

    func signIn(
        scopes: [String],
        presentationViewController: UIViewController,
        expectedEpoch: MicrosoftGraphAuthEpoch.Snapshot
    ) async throws -> (
        session: NativeMicrosoftOAuthSession,
        authorizationEpoch: MicrosoftGraphAuthEpoch.Snapshot
    ) {
        let config = try MicrosoftGraphConfiguration.load()
        let verifier = try Self.makeCodeVerifier()
        let challenge = Self.makeCodeChallenge(verifier: verifier)
        let state = UUID().uuidString
        let redirectURI = config.redirectURI ?? "msauth.\(Bundle.main.bundleIdentifier ?? "com.27pm.lumenclone")://auth"
        let authURL = try authorizationURL(config: config, scopes: scopes, redirectURI: redirectURI, state: state, codeChallenge: challenge)
        let callbackURL = try await authenticate(
            url: authURL,
            callbackScheme: URL(string: redirectURI)?.scheme ?? callbackScheme,
            presentationViewController: presentationViewController
        )
        let code = try Self.authorizationCode(from: callbackURL, expectedState: state)
        let token = try await exchangeCode(config: config, code: code, redirectURI: redirectURI, verifier: verifier, scopes: scopes)
        let account = try await fetchAccount(accessToken: token.accessToken)
        let session = NativeMicrosoftOAuthSession(account: account, token: token)
        let committedEpoch = try authEpoch.replaceAccountIfCurrent(expectedEpoch, with: account.id) {
            try keychainStore.save(session)
        }
        return (session, committedEpoch)
    }

    func acquireToken(
        scopes: [String],
        forceRefresh: Bool,
        expectedEpoch: MicrosoftGraphAuthEpoch.Snapshot
    ) async throws -> NativeMicrosoftOAuthTokenSet {
        guard let observedSession = loadCachedSession() else {
            throw MicrosoftGraphAuthError.noAccount
        }
        let accountID = observedSession.account.id
        try authEpoch.requireCurrent(expectedEpoch, accountID: accountID)
        if !forceRefresh,
           !observedSession.token.shouldRefreshProactively,
           token(observedSession.token, satisfies: scopes) {
            return observedSession.token
        }

        let refreshed = try await refreshCoordinator.refreshToken(
            for: accountID,
            scopes: scopes
        ) { [self] in
            try authEpoch.requireCurrent(expectedEpoch, accountID: accountID)
            guard let currentSession = loadCachedSession(),
                  currentSession.account.id == accountID else {
                throw MicrosoftGraphAuthEpochError.staleCompletion
            }
            if !forceRefresh,
               !currentSession.token.shouldRefreshProactively,
               token(currentSession.token, satisfies: scopes) {
                return currentSession.token
            }
            guard let currentRefreshToken = currentSession.token.refreshToken else {
                throw MicrosoftGraphAuthError.interactionRequired
            }
            let responseToken: NativeMicrosoftOAuthTokenSet
            if let refreshOperation {
                responseToken = try await refreshOperation(currentRefreshToken, scopes)
            } else {
                let config = try MicrosoftGraphConfiguration.load()
                responseToken = try await refresh(
                    config: config,
                    refreshToken: currentRefreshToken,
                    scopes: scopes
                )
            }
            return try commitRefreshedTokenIfCurrent(
                responseToken,
                expectedEpoch: expectedEpoch,
                expectedSession: currentSession
            )
        }
        guard token(refreshed, satisfies: scopes) else {
            throw MicrosoftGraphAuthError.invalidScope
        }
        return refreshed
    }

    /// Linearizes a refresh completion against both the shared account epoch and
    /// the exact protected session used to dispatch it. Concurrent refreshes may
    /// both return an explicit rotated refresh token, so a completion based on an
    /// older session must be rejected rather than merged over a newer commit.
    func commitRefreshedTokenIfCurrent(
        _ responseToken: NativeMicrosoftOAuthTokenSet,
        expectedEpoch: MicrosoftGraphAuthEpoch.Snapshot,
        expectedSession: NativeMicrosoftOAuthSession
    ) throws -> NativeMicrosoftOAuthTokenSet {
        let accountID = expectedSession.account.id
        try authEpoch.requireCurrent(expectedEpoch, accountID: accountID)
        guard let latestSession = keychainStore.load(),
              latestSession == expectedSession else {
            throw MicrosoftGraphAuthEpochError.staleCompletion
        }
        let mergedToken = Self.preservingRefreshState(
            responseToken,
            from: latestSession.token
        )
        try keychainStore.save(
            NativeMicrosoftOAuthSession(
                account: latestSession.account,
                token: mergedToken
            )
        )
        return mergedToken
    }

    func signOut() throws {
        try keychainStore.clear()
    }

    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        presentationAnchor ?? ASPresentationAnchor()
    }

    private func authenticate(url: URL, callbackScheme: String, presentationViewController: UIViewController) async throws -> URL {
        presentationAnchor = presentationViewController.view.window
        return try await withCheckedThrowingContinuation { continuation in
            let session = ASWebAuthenticationSession(url: url, callbackURLScheme: callbackScheme) { callbackURL, error in
                self.activeSession = nil
                if let callbackURL {
                    continuation.resume(returning: callbackURL)
                    return
                }
                if let error = error as? ASWebAuthenticationSessionError, error.code == .canceledLogin {
                    continuation.resume(throwing: MicrosoftGraphAuthError.signInCancelled)
                    return
                }
                continuation.resume(throwing: error ?? MicrosoftGraphAuthError.interactionRequired)
            }
            session.presentationContextProvider = self
            session.prefersEphemeralWebBrowserSession = false
            self.activeSession = session
            if !session.start() {
                self.activeSession = nil
                continuation.resume(throwing: MicrosoftGraphAuthError.presentationAnchorUnavailable)
            }
        }
    }

    private func authorizationURL(config: MicrosoftGraphConfiguration, scopes: [String], redirectURI: String, state: String, codeChallenge: String) throws -> URL {
        let authority = config.authorityURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        var components = URLComponents(string: "\(authority)/oauth2/v2.0/authorize")
        components?.queryItems = [
            URLQueryItem(name: "client_id", value: config.clientID),
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "redirect_uri", value: redirectURI),
            URLQueryItem(name: "response_mode", value: "query"),
            URLQueryItem(name: "scope", value: normalizedScopes(scopes)),
            URLQueryItem(name: "state", value: state),
            URLQueryItem(name: "prompt", value: "select_account"),
            URLQueryItem(name: "code_challenge", value: codeChallenge),
            URLQueryItem(name: "code_challenge_method", value: "S256")
        ]
        guard let url = components?.url else { throw GraphHTTPError.missingURL }
        return url
    }

    private func exchangeCode(config: MicrosoftGraphConfiguration, code: String, redirectURI: String, verifier: String, scopes: [String]) async throws -> NativeMicrosoftOAuthTokenSet {
        try await tokenRequest(config: config, form: [
            "client_id": config.clientID,
            "scope": normalizedScopes(scopes),
            "code": code,
            "redirect_uri": redirectURI,
            "grant_type": "authorization_code",
            "code_verifier": verifier
        ])
    }

    private func refresh(config: MicrosoftGraphConfiguration, refreshToken: String, scopes: [String]) async throws -> NativeMicrosoftOAuthTokenSet {
        try await tokenRequest(config: config, form: [
            "client_id": config.clientID,
            "scope": normalizedScopes(scopes),
            "refresh_token": refreshToken,
            "grant_type": "refresh_token"
        ])
    }

    private func tokenRequest(config: MicrosoftGraphConfiguration, form: [String: String]) async throws -> NativeMicrosoftOAuthTokenSet {
        let authority = config.authorityURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: "\(authority)/oauth2/v2.0/token") else { throw GraphHTTPError.missingURL }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = form
            .map { key, value in "\(Self.percentEncode(key))=\(Self.percentEncode(value))" }
            .joined(separator: "&")
            .data(using: .utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        guard (200...299).contains(http.statusCode) else {
            let oauthError = try? JSONDecoder().decode(NativeOAuthErrorResponse.self, from: data)
            if let classified = Self.authErrorForTokenEndpointFailure(
                errorCode: oauthError?.error,
                suberror: oauthError?.suberror,
                errorDescription: oauthError?.errorDescription,
                httpStatus: http.statusCode
            ) {
                throw classified
            }
            throw GraphHTTPError.unexpectedStatus(http.statusCode)
        }
        let decoded = try JSONDecoder().decode(NativeOAuthTokenResponse.self, from: data)
        return NativeMicrosoftOAuthTokenSet(
            accessToken: decoded.accessToken,
            refreshToken: decoded.refreshToken,
            idToken: decoded.idToken,
            expiresOn: Date(timeIntervalSinceNow: TimeInterval(max(decoded.expiresIn - 60, 60))),
            scope: decoded.scope,
            tokenType: decoded.tokenType
        )
    }

    private func fetchAccount(accessToken: String) async throws -> MicrosoftGraphAccountSnapshot {
        var request = URLRequest(url: URL(string: "https://graph.microsoft.com/v1.0/me?$select=id,displayName,userPrincipalName,mail")!)
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await URLSession.shared.data(for: request)
        try MicrosoftGraphMailClient.validate(response: response, data: data)
        let profile = try JSONDecoder().decode(NativeMicrosoftOAuthProfile.self, from: data)
        return MicrosoftGraphAccountSnapshot(
            id: profile.id,
            username: profile.mail ?? profile.userPrincipalName,
            name: profile.displayName,
            environment: "native-oauth",
            tenantID: nil
        )
    }

    private func normalizedScopes(_ scopes: [String]) -> String {
        Array(Set(scopes + [MicrosoftGraphScope.offlineAccess.rawValue, MicrosoftGraphScope.userRead.rawValue])).sorted().joined(separator: " ")
    }

    private static func authorizationCode(from url: URL, expectedState: String) throws -> String {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { throw MicrosoftGraphAuthError.interactionRequired }
        let items = components.queryItems ?? []
        if let error = items.first(where: { $0.name == "error" })?.value {
            throw Self.authErrorForAuthorizationFailure(errorCode: error)
        }
        guard items.first(where: { $0.name == "state" })?.value == expectedState else {
            throw MicrosoftGraphAuthError.invalidConfiguration("Microsoft sign-in state validation failed.")
        }
        guard let code = items.first(where: { $0.name == "code" })?.value, !code.isEmpty else {
            throw MicrosoftGraphAuthError.interactionRequired
        }
        return code
    }

    private static func makeCodeVerifier() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw MicrosoftGraphAuthError.interactionRequired
        }
        return Data(bytes).base64URLEncodedString()
    }

    private func token(_ token: NativeMicrosoftOAuthTokenSet, satisfies scopes: [String]) -> Bool {
        Self.accessTokenScopesSatisfy(grantedScopes: token.scope, requestedScopes: scopes)
    }

    nonisolated static func accessTokenScopesSatisfy(grantedScopes: String?, requestedScopes: [String]) -> Bool {
        let grantOnlyScopes = Set([MicrosoftGraphScope.offlineAccess.rawValue.lowercased()])
        let requested = Set((requestedScopes + [MicrosoftGraphScope.userRead.rawValue])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty })
            .subtracting(grantOnlyScopes)
        let granted = Set((grantedScopes ?? "")
            .split(whereSeparator: { $0.isWhitespace })
            .map { $0.lowercased() })
        return requested.isSubset(of: granted)
    }

    nonisolated static func preservingRefreshState(
        _ refreshed: NativeMicrosoftOAuthTokenSet,
        from previous: NativeMicrosoftOAuthTokenSet
    ) -> NativeMicrosoftOAuthTokenSet {
        NativeMicrosoftOAuthTokenSet(
            accessToken: refreshed.accessToken,
            refreshToken: refreshed.refreshToken ?? previous.refreshToken,
            idToken: refreshed.idToken ?? previous.idToken,
            expiresOn: refreshed.expiresOn,
            scope: refreshed.scope ?? previous.scope,
            tokenType: refreshed.tokenType
        )
    }

    nonisolated static func authErrorForTokenEndpointFailure(
        errorCode: String?,
        suberror: String? = nil,
        errorDescription: String? = nil,
        httpStatus: Int
    ) -> MicrosoftGraphAuthError? {
        if httpStatus == 429 {
            return .tokenEndpointThrottled
        }
        if (500...599).contains(httpStatus) {
            return .tokenEndpointUnavailable
        }

        let code = errorCode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        let detail = [suberror, errorDescription]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .joined(separator: " ")
        if detail.contains("consent_required") || detail.contains("aadsts65001") || detail.contains("aadsts65004") {
            return .consentRequired
        }
        if detail.contains("interaction_required")
            || detail.contains("login_required")
            || detail.contains("aadsts50076")
            || detail.contains("aadsts50079") {
            return .interactionRequired
        }
        if detail.contains("invalid_scope") || detail.contains("aadsts70011") {
            return .invalidScope
        }
        switch code {
        case "invalid_grant":
            return .invalidGrant
        case "interaction_required", "login_required", "account_selection_required":
            return .interactionRequired
        case "consent_required":
            return .consentRequired
        case "invalid_scope", "insufficient_scope":
            return .invalidScope
        case "temporarily_unavailable", "server_error":
            return .tokenEndpointUnavailable
        case "too_many_requests", "throttled":
            return .tokenEndpointThrottled
        case "invalid_client", "unauthorized_client", "invalid_request", "unsupported_grant_type":
            return .invalidConfiguration("Microsoft OAuth client or token-request configuration is invalid for this build.")
        default:
            return nil
        }
    }

    nonisolated static func authErrorForAuthorizationFailure(errorCode: String) -> MicrosoftGraphAuthError {
        switch errorCode.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "interaction_required", "login_required", "account_selection_required":
            return .interactionRequired
        case "consent_required":
            return .consentRequired
        case "access_denied":
            return .consentRequired
        case "invalid_scope":
            return .invalidScope
        case "temporarily_unavailable", "server_error":
            return .tokenEndpointUnavailable
        case "invalid_client", "unauthorized_client", "invalid_request", "unsupported_response_type":
            return .invalidConfiguration("Microsoft OAuth authorization configuration is invalid for this build.")
        default:
            return .interactionRequired
        }
    }

    private static func makeCodeChallenge(verifier: String) -> String {
        let digest = SHA256.hash(data: Data(verifier.utf8))
        return Data(digest).base64URLEncodedString()
    }

    private static func percentEncode(_ string: String) -> String {
        var allowed = CharacterSet.urlQueryAllowed
        allowed.remove(charactersIn: "+&=")
        return string.addingPercentEncoding(withAllowedCharacters: allowed) ?? string
    }
}

private nonisolated struct NativeOAuthTokenResponse: Decodable {
    let tokenType: String
    let scope: String?
    let expiresIn: Int
    let accessToken: String
    let refreshToken: String?
    let idToken: String?

    enum CodingKeys: String, CodingKey {
        case tokenType = "token_type"
        case scope
        case expiresIn = "expires_in"
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case idToken = "id_token"
    }
}

private nonisolated struct NativeOAuthErrorResponse: Decodable {
    let error: String
    let errorDescription: String?
    let suberror: String?

    enum CodingKeys: String, CodingKey {
        case error
        case errorDescription = "error_description"
        case suberror
    }
}

nonisolated enum NativeMicrosoftOAuthKeychainStoreError: LocalizedError, Equatable, Sendable {
    case unexpectedStatus(OSStatus)

    var errorDescription: String? {
        switch self {
        case .unexpectedStatus:
            return "Lumen could not update the protected Microsoft authentication session."
        }
    }
}

nonisolated struct NativeMicrosoftOAuthKeychainStore: Sendable {
    private let service: String
    private let account: String

    init(
        service: String = "ai.lumen.microsoftgraph.native-oauth",
        account: String = "default"
    ) {
        self.service = service
        self.account = account
    }

    func load() -> NativeMicrosoftOAuthSession? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        return try? JSONDecoder().decode(NativeMicrosoftOAuthSession.self, from: data)
    }

    func save(_ session: NativeMicrosoftOAuthSession) throws {
        let data = try JSONEncoder().encode(session)
        let updateAttributes: [String: Any] = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(baseQuery() as CFDictionary, updateAttributes as CFDictionary)
        switch updateStatus {
        case errSecSuccess:
            return
        case errSecItemNotFound:
            var addQuery = baseQuery()
            addQuery[kSecValueData as String] = data
            addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
            if addStatus == errSecSuccess { return }
            if addStatus == errSecDuplicateItem {
                let retryStatus = SecItemUpdate(baseQuery() as CFDictionary, updateAttributes as CFDictionary)
                guard retryStatus == errSecSuccess else {
                    throw NativeMicrosoftOAuthKeychainStoreError.unexpectedStatus(retryStatus)
                }
                return
            }
            throw NativeMicrosoftOAuthKeychainStoreError.unexpectedStatus(addStatus)
        default:
            throw NativeMicrosoftOAuthKeychainStoreError.unexpectedStatus(updateStatus)
        }
    }

    func clear() throws {
        let status = SecItemDelete(baseQuery() as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw NativeMicrosoftOAuthKeychainStoreError.unexpectedStatus(status)
        }
    }

    private func baseQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
    }
}

private nonisolated extension Data {
    func base64URLEncodedString() -> String {
        base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
