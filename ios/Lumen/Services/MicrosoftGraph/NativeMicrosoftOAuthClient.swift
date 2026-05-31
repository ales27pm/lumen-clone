import AuthenticationServices
import CryptoKit
import Foundation
import OSLog
import Security

nonisolated struct NativeMicrosoftOAuthTokenSet: Codable, Sendable {
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

nonisolated struct NativeMicrosoftOAuthSession: Codable, Sendable {
    let account: MicrosoftGraphAccountSnapshot
    let token: NativeMicrosoftOAuthTokenSet
}

@MainActor
final class NativeMicrosoftOAuthClient: NSObject, ASWebAuthenticationPresentationContextProviding {
    private let logger = Logger(subsystem: "ai.lumen.microsoftgraph", category: "native-oauth")
    private let callbackScheme: String
    private var activeSession: ASWebAuthenticationSession?
    private weak var presentationAnchor: UIWindow?

    override init() {
        let bundleID = Bundle.main.bundleIdentifier ?? "com.27pm.lumen"
        self.callbackScheme = "msauth.\(bundleID)"
        super.init()
    }

    func loadCachedSession() -> NativeMicrosoftOAuthSession? {
        NativeMicrosoftOAuthKeychainStore.load()
    }

    func cachedAccounts() -> [MicrosoftGraphAccountSnapshot] {
        guard let session = loadCachedSession() else { return [] }
        return [session.account]
    }

    func signIn(scopes: [String], presentationViewController: UIViewController) async throws -> NativeMicrosoftOAuthSession {
        let config = try MicrosoftGraphConfiguration.load()
        let verifier = try Self.makeCodeVerifier()
        let challenge = Self.makeCodeChallenge(verifier: verifier)
        let state = UUID().uuidString
        let redirectURI = config.redirectURI ?? "msauth.\(Bundle.main.bundleIdentifier ?? "com.27pm.lumen")://auth"
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
        NativeMicrosoftOAuthKeychainStore.save(session)
        return session
    }

    func acquireToken(scopes: [String], forceRefresh: Bool) async throws -> NativeMicrosoftOAuthTokenSet {
        guard let session = loadCachedSession() else { throw MicrosoftGraphAuthError.noAccount }
        if !forceRefresh && !session.token.shouldRefreshProactively && token(session.token, satisfies: scopes) {
            return session.token
        }
        guard let refreshToken = session.token.refreshToken else { throw MicrosoftGraphAuthError.interactionRequired }
        let config = try MicrosoftGraphConfiguration.load()
        let refreshed = try await refresh(config: config, refreshToken: refreshToken, scopes: scopes)
        let updated = NativeMicrosoftOAuthSession(account: session.account, token: refreshed)
        NativeMicrosoftOAuthKeychainStore.save(updated)
        return refreshed
    }

    func signOut() {
        NativeMicrosoftOAuthKeychainStore.clear()
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
            if let oauthError = try? JSONDecoder().decode(NativeOAuthErrorResponse.self, from: data) {
                throw MicrosoftGraphAuthError.invalidConfiguration(oauthError.errorDescription ?? oauthError.error)
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
            let description = items.first(where: { $0.name == "error_description" })?.value ?? error
            throw MicrosoftGraphAuthError.invalidConfiguration(description)
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
        let requested = Set(normalizedScopes(scopes).split(separator: " ").map(String.init))
        let granted = Set((token.scope ?? "").split(separator: " ").map(String.init))
        return requested.isSubset(of: granted)
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

    enum CodingKeys: String, CodingKey {
        case error
        case errorDescription = "error_description"
    }
}

private nonisolated enum NativeMicrosoftOAuthKeychainStore {
    private static let service = "ai.lumen.microsoftgraph.native-oauth"
    private static let account = "default"

    static func load() -> NativeMicrosoftOAuthSession? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        return try? JSONDecoder().decode(NativeMicrosoftOAuthSession.self, from: data)
    }

    static func save(_ session: NativeMicrosoftOAuthSession) {
        guard let data = try? JSONEncoder().encode(session) else { return }
        clear()
        var query = baseQuery()
        query[kSecValueData as String] = data
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(query as CFDictionary, nil)
    }

    static func clear() {
        SecItemDelete(baseQuery() as CFDictionary)
    }

    private static func baseQuery() -> [String: Any] {
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
