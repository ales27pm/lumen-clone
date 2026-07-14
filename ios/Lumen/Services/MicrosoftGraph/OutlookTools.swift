import Foundation

nonisolated struct OutlookToolFolder: Codable, Sendable {
    let id: String
    let displayName: String
    let totalItemCount: Int?
    let unreadItemCount: Int?
    let childFolderCount: Int?
}

nonisolated struct OutlookToolFolderPage: Codable, Sendable {
    let value: [OutlookToolFolder]
}

nonisolated struct OutlookToolAttachment: Codable, Sendable {
    let id: String
    let name: String?
    let contentType: String?
    let size: Int?
    let isInline: Bool?
    let lastModifiedDateTime: String?
}

nonisolated struct OutlookToolAttachmentPage: Codable, Sendable {
    let value: [OutlookToolAttachment]
}

nonisolated struct OutlookMessageReference: Codable, Hashable, Sendable {
    let ordinal: Int
    let id: String
    let subject: String
    let sender: String
    let receivedDateTime: String?
    let source: String
    let cachedAt: Date

    var displayLine: String {
        "#\(ordinal) — \(subject) — from \(sender) — id: \(id)"
    }
}

nonisolated enum OutlookToolAvailabilityState: String, Codable, Equatable, Sendable {
    case configured
    case notConfigured = "not_configured"
    case permissionDenied = "permission_denied"
    case authUnavailable = "auth_unavailable"
    case networkUnavailable = "network_unavailable"
    case providerError = "provider_error"
    case validEmptyResult = "valid_empty_result"
}

nonisolated struct OutlookToolOutcome: Equatable, Sendable {
    let text: String
    let status: ToolResultStatus
    let availability: OutlookToolAvailabilityState
    let errorCode: String?
    let diagnostics: [String: String]

    var structuredPayload: [String: String] {
        var payload = diagnostics
        payload["availability"] = availability.rawValue
        if let errorCode { payload["errorCode"] = errorCode }
        return payload
    }

    func metricsSummary(base: String) -> String {
        status == .success ? "\(base)_\(availability.rawValue)" : "\(base)_\(availability.rawValue)_\(status.rawValue)"
    }

    static func success(_ text: String, diagnostics: [String: String] = [:]) -> OutlookToolOutcome {
        OutlookToolOutcome(text: text, status: .success, availability: .configured, errorCode: nil, diagnostics: diagnostics)
    }

    static func validEmpty(_ text: String, diagnostics: [String: String] = [:]) -> OutlookToolOutcome {
        OutlookToolOutcome(text: text, status: .success, availability: .validEmptyResult, errorCode: nil, diagnostics: diagnostics)
    }

    static func invalidArguments(_ text: String, diagnostics: [String: String] = [:]) -> OutlookToolOutcome {
        OutlookToolOutcome(text: text, status: .failed, availability: .configured, errorCode: "outlook_invalid_arguments", diagnostics: diagnostics)
    }

    static func authUnavailable(
        _ text: String,
        errorCode: String = "outlook_auth_unavailable",
        diagnostics: [String: String] = [:]
    ) -> OutlookToolOutcome {
        OutlookToolOutcome(text: text, status: .unavailable, availability: .authUnavailable, errorCode: errorCode, diagnostics: diagnostics)
    }

    static func failure(from error: Error, diagnostics: [String: String] = [:]) -> OutlookToolOutcome {
        switch error {
        case MicrosoftGraphAuthError.missingClientID:
            return OutlookToolOutcome(
                text: "Outlook is not configured for this build. Verify Microsoft Graph MSAL client ID, redirect URI, and bundle identifier.",
                status: .unavailable,
                availability: .notConfigured,
                errorCode: "outlook_not_configured",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.invalidConfiguration:
            return OutlookToolOutcome(
                text: "Outlook is not configured correctly for this build. Verify Microsoft Graph MSAL client ID, redirect URI, and bundle identifier.",
                status: .unavailable,
                availability: .notConfigured,
                errorCode: "outlook_not_configured",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.noAccount,
             MicrosoftGraphAuthError.signInCancelled:
            return authUnavailable("Outlook is not signed in. Open Outlook in Lumen and sign in first.", diagnostics: diagnostics)
        case MicrosoftGraphAuthError.invalidGrant:
            return authUnavailable(
                "Outlook authorization expired or was revoked. Reconnect Outlook and sign in again.",
                errorCode: "outlook_reauthentication_required",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.interactionRequired:
            return authUnavailable(
                "Outlook requires an interactive sign-in. Reconnect Outlook to continue.",
                errorCode: "outlook_interaction_required",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.consentRequired:
            return OutlookToolOutcome(
                text: "Outlook needs consent for the requested Mail permissions. Reconnect Outlook and grant access.",
                status: .denied,
                availability: .permissionDenied,
                errorCode: "outlook_consent_required",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.invalidScope:
            return OutlookToolOutcome(
                text: "Outlook did not grant the Mail permission required for this action. Reconnect Outlook and grant access.",
                status: .denied,
                availability: .permissionDenied,
                errorCode: "outlook_scope_not_granted",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.tokenEndpointThrottled:
            return OutlookToolOutcome(
                text: "Microsoft is throttling Outlook authentication requests. Try again later.",
                status: .failed,
                availability: .providerError,
                errorCode: "outlook_provider_throttled",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.tokenEndpointUnavailable:
            return OutlookToolOutcome(
                text: "Outlook authentication is temporarily unavailable. Try again later.",
                status: .failed,
                availability: .providerError,
                errorCode: "outlook_auth_provider_unavailable",
                diagnostics: diagnostics
            )
        case MicrosoftGraphAuthError.msalNotLinked,
             MicrosoftGraphAuthError.presentationAnchorUnavailable:
            return authUnavailable("Outlook authentication is unavailable in this build.", diagnostics: diagnostics)
        case let error as URLError:
            return networkFailure(error, diagnostics: diagnostics)
        case let error as GraphHTTPError:
            return graphHTTPFailure(error, diagnostics: diagnostics)
        case let error as GraphAPIErrorEnvelope:
            return graphAPIFailure(error, diagnostics: diagnostics)
        default:
            return OutlookToolOutcome(
                text: "Outlook provider request failed. Try again later or reconnect Outlook.",
                status: .failed,
                availability: .providerError,
                errorCode: "outlook_provider_error",
                diagnostics: diagnostics
            )
        }
    }

    private static func networkFailure(_ error: URLError, diagnostics: [String: String]) -> OutlookToolOutcome {
        let networkCodes: Set<URLError.Code> = [
            .notConnectedToInternet,
            .networkConnectionLost,
            .timedOut,
            .cannotFindHost,
            .cannotConnectToHost,
            .dnsLookupFailed,
            .internationalRoamingOff,
            .dataNotAllowed,
            .secureConnectionFailed
        ]
        let state: OutlookToolAvailabilityState = networkCodes.contains(error.code) ? .networkUnavailable : .providerError
        return OutlookToolOutcome(
            text: state == .networkUnavailable
                ? "Network access is unavailable for Outlook. Check your connection and try again."
                : "Outlook provider request failed. Try again later or reconnect Outlook.",
            status: state == .networkUnavailable ? .unavailable : .failed,
            availability: state,
            errorCode: state == .networkUnavailable ? "outlook_network_unavailable" : "outlook_provider_error",
            diagnostics: diagnostics.merging(["urlErrorCode": String(error.errorCode)]) { current, _ in current }
        )
    }

    private static func graphHTTPFailure(_ error: GraphHTTPError, diagnostics: [String: String]) -> OutlookToolOutcome {
        switch error {
        case .unexpectedStatus(401):
            return authUnavailable("Outlook is not signed in. Open Outlook in Lumen and sign in first.", diagnostics: diagnostics.merging(["httpStatus": "401"]) { current, _ in current })
        case .unexpectedStatus(403):
            return OutlookToolOutcome(
                text: "Outlook permission is denied for this action. Reconnect Outlook and grant the required Mail permissions.",
                status: .denied,
                availability: .permissionDenied,
                errorCode: "outlook_permission_denied",
                diagnostics: diagnostics.merging(["httpStatus": "403"]) { current, _ in current }
            )
        case .unexpectedStatus(let status):
            return OutlookToolOutcome(
                text: "Outlook provider request failed. Try again later or reconnect Outlook.",
                status: .failed,
                availability: .providerError,
                errorCode: "outlook_provider_error",
                diagnostics: diagnostics.merging(["httpStatus": String(status)]) { current, _ in current }
            )
        case .missingURL:
            return OutlookToolOutcome(
                text: "Outlook is not configured correctly for this build. Verify Microsoft Graph URL configuration.",
                status: .unavailable,
                availability: .notConfigured,
                errorCode: "outlook_not_configured",
                diagnostics: diagnostics
            )
        case .throttled:
            return OutlookToolOutcome(
                text: "Outlook provider is throttling requests. Try again later.",
                status: .failed,
                availability: .providerError,
                errorCode: "outlook_provider_throttled",
                diagnostics: diagnostics.merging(["providerCode": "TooManyRequests"]) { current, _ in current }
            )
        }
    }

    private static func graphAPIFailure(_ error: GraphAPIErrorEnvelope, diagnostics: [String: String]) -> OutlookToolOutcome {
        let code = error.error.code.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = code.lowercased()
        var safeDiagnostics = diagnostics
        if !code.isEmpty {
            safeDiagnostics["providerCode"] = code
        }
        if lower.contains("accessdenied") || lower.contains("authorization_requestdenied") || lower.contains("forbidden") || lower.contains("insufficient") {
            return OutlookToolOutcome(
                text: "Outlook permission is denied for this action. Reconnect Outlook and grant the required Mail permissions.",
                status: .denied,
                availability: .permissionDenied,
                errorCode: "outlook_permission_denied",
                diagnostics: safeDiagnostics
            )
        }
        if lower.contains("invalidauthenticationtoken") || lower.contains("auth") || lower.contains("token") {
            return authUnavailable("Outlook is not signed in. Open Outlook in Lumen and sign in first.", diagnostics: safeDiagnostics)
        }
        return OutlookToolOutcome(
            text: "Outlook provider request failed. Try again later or reconnect Outlook.",
            status: .failed,
            availability: .providerError,
            errorCode: "outlook_provider_error",
            diagnostics: safeDiagnostics
        )
    }
}

actor OutlookGraphToolClient {
    private let baseURL = URL(string: "https://graph.microsoft.com/v1.0")!
    private let session: URLSession
    private let decoder = JSONDecoder()

    init(session: URLSession = .shared) {
        self.session = session
    }

    func listFolders(accessToken: String, includeHidden: Bool = false) async throws -> [OutlookToolFolder] {
        var components = URLComponents(url: baseURL.appendingPathComponent("me/mailFolders"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "$select", value: "id,displayName,totalItemCount,unreadItemCount,childFolderCount"),
            URLQueryItem(name: "$top", value: "100"),
            URLQueryItem(name: "includeHiddenFolders", value: includeHidden ? "true" : "false")
        ]
        let page: OutlookToolFolderPage = try await get(components.url, accessToken: accessToken)
        return page.value
    }

    func listMessages(folderID: String?, pageSize: Int, unreadOnly: Bool, accessToken: String) async throws -> [GraphMailMessage] {
        let safeTop = String(min(max(pageSize, 1), 50))
        let path = folderID.flatMap { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "me/mailFolders/\($0)/messages" : nil } ?? "me/messages"
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        var queryItems = [
            URLQueryItem(name: "$select", value: messageSelectFields),
            URLQueryItem(name: "$orderby", value: "receivedDateTime desc"),
            URLQueryItem(name: "$top", value: safeTop)
        ]
        if unreadOnly {
            queryItems.append(URLQueryItem(name: "$filter", value: "isRead eq false"))
        }
        components.queryItems = queryItems
        let page: GraphMailPage = try await get(components.url, accessToken: accessToken)
        return page.value
    }

    func searchMessages(query: String, folderID: String?, pageSize: Int, accessToken: String) async throws -> [GraphMailMessage] {
        let safeQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !safeQuery.isEmpty else { return [] }
        let path = folderID.flatMap { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "me/mailFolders/\($0)/messages" : nil } ?? "me/messages"
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "$select", value: messageSelectFields),
            URLQueryItem(name: "$search", value: "\"\(safeQuery)\""),
            URLQueryItem(name: "$top", value: String(min(max(pageSize, 1), 50)))
        ]
        let page: GraphMailPage = try await get(components.url, accessToken: accessToken, extraHeaders: ["ConsistencyLevel": "eventual"])
        return page.value
    }

    func readMessage(messageID: String, accessToken: String) async throws -> GraphMailMessage {
        var components = URLComponents(url: baseURL.appendingPathComponent("me/messages").appendingPathComponent(messageID), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "$select", value: "\(messageSelectFields),body")]
        return try await get(components.url, accessToken: accessToken)
    }

    func listAttachments(messageID: String, accessToken: String) async throws -> [OutlookToolAttachment] {
        var components = URLComponents(url: baseURL.appendingPathComponent("me/messages").appendingPathComponent(messageID).appendingPathComponent("attachments"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "$select", value: "id,name,contentType,size,isInline,lastModifiedDateTime")]
        let page: OutlookToolAttachmentPage = try await get(components.url, accessToken: accessToken)
        return page.value
    }

    func markRead(messageID: String, isRead: Bool, accessToken: String) async throws -> GraphMailMessage {
        try await patchMessage(messageID: messageID, payload: ["isRead": isRead], accessToken: accessToken)
    }

    func moveMessage(messageID: String, destinationID: String, accessToken: String) async throws -> GraphMailMessage {
        var request = makeRequest(url: baseURL.appendingPathComponent("me/messages").appendingPathComponent(messageID).appendingPathComponent("move"), accessToken: accessToken)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: ["destinationId": destinationID])
        let (data, response) = try await session.data(for: request)
        try MicrosoftGraphMailClient.validate(response: response, data: data, allowedStatuses: 200...201)
        return try decoder.decode(GraphMailMessage.self, from: data)
    }

    func deleteMessage(messageID: String, accessToken: String) async throws {
        var request = makeRequest(url: baseURL.appendingPathComponent("me/messages").appendingPathComponent(messageID), accessToken: accessToken)
        request.httpMethod = "DELETE"
        let (data, response) = try await session.data(for: request)
        try MicrosoftGraphMailClient.validate(response: response, data: data, allowedStatuses: 200...204)
    }

    func reply(messageID: String, comment: String, accessToken: String) async throws {
        try await messageAction(messageID: messageID, action: "reply", payload: ["comment": comment], accessToken: accessToken)
    }

    func replyAll(messageID: String, comment: String, accessToken: String) async throws {
        try await messageAction(messageID: messageID, action: "replyAll", payload: ["comment": comment], accessToken: accessToken)
    }

    func forward(messageID: String, comment: String, recipients: [String], accessToken: String) async throws {
        let payload: [String: Any] = [
            "comment": comment,
            "toRecipients": recipients.map { ["emailAddress": ["address": $0]] }
        ]
        try await messageAction(messageID: messageID, action: "forward", payload: payload, accessToken: accessToken)
    }

    func createDraft(subject: String, body: String, recipients: [String], accessToken: String) async throws -> GraphMailMessage {
        let client = MicrosoftGraphMailClient(session: session)
        return try await client.createDraft(subject: subject, htmlBody: body, to: recipients, accessToken: accessToken)
    }

    func sendMail(subject: String, body: String, recipients: [String], accessToken: String) async throws {
        let request = GraphSendMailRequest(
            message: GraphSendMailRequest.MailMessage(
                subject: subject,
                body: GraphSendMailRequest.Body(contentType: "HTML", content: body),
                toRecipients: recipients.map { GraphSendMailRequest.Recipient(emailAddress: .init(address: $0, name: nil)) },
                ccRecipients: nil,
                bccRecipients: nil,
                attachments: nil
            ),
            saveToSentItems: true
        )
        let client = MicrosoftGraphMailClient(session: session)
        try await client.sendMail(request, accessToken: accessToken)
    }

    private func patchMessage(messageID: String, payload: [String: Any], accessToken: String) async throws -> GraphMailMessage {
        var request = makeRequest(url: baseURL.appendingPathComponent("me/messages").appendingPathComponent(messageID), accessToken: accessToken)
        request.httpMethod = "PATCH"
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await session.data(for: request)
        try MicrosoftGraphMailClient.validate(response: response, data: data, allowedStatuses: 200...200)
        return try decoder.decode(GraphMailMessage.self, from: data)
    }

    private func messageAction(messageID: String, action: String, payload: [String: Any], accessToken: String) async throws {
        var request = makeRequest(url: baseURL.appendingPathComponent("me/messages").appendingPathComponent(messageID).appendingPathComponent(action), accessToken: accessToken)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await session.data(for: request)
        try MicrosoftGraphMailClient.validate(response: response, data: data, allowedStatuses: 200...202)
    }

    private func get<T: Decodable>(_ url: URL?, accessToken: String, extraHeaders: [String: String] = [:]) async throws -> T {
        guard let url else { throw GraphHTTPError.missingURL }
        let request = makeRequest(url: url, accessToken: accessToken, extraHeaders: extraHeaders)
        let (data, response) = try await session.data(for: request)
        try MicrosoftGraphMailClient.validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func makeRequest(url: URL, accessToken: String, extraHeaders: [String: String] = [:]) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(UUID().uuidString, forHTTPHeaderField: "client-request-id")
        extraHeaders.forEach { request.setValue($0.value, forHTTPHeaderField: $0.key) }
        return request
    }

    private var messageSelectFields: String {
        "id,subject,from,toRecipients,ccRecipients,receivedDateTime,sentDateTime,bodyPreview,isRead,hasAttachments"
    }
}

nonisolated enum OutlookToolUserVisibleOutput {
    static let maxPreviewCharacters = 600
    static let maxBodyCharacters = 3_500
    static let maxFinalCharacters = 8_000

    static func plainText(_ raw: String, maxCharacters: Int) -> String {
        guard maxCharacters > 0 else { return "" }
        var text = raw
            .replacingOccurrences(of: "\u{0000}", with: "")
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")

        for tag in ["head", "style", "script", "noscript", "svg"] {
            text = text.replacingOccurrences(
                of: "(?is)<\(tag)\\b[^>]*>.*?</\(tag)\\s*>",
                with: " ",
                options: .regularExpression
            )
        }
        text = text.replacingOccurrences(of: "(?is)<!--.*?-->", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?i)<br\\s*/?>", with: "\n", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?i)<li\\b[^>]*>", with: "\n• ", options: .regularExpression)
        text = text.replacingOccurrences(
            of: "(?i)</(?:p|div|li|tr|td|table|h[1-6]|section|article|header|footer)\\s*>",
            with: "\n",
            options: .regularExpression
        )
        text = text.replacingOccurrences(of: "(?is)<[^>]+>", with: " ", options: .regularExpression)
        text = decodeHTMLEntities(text)
        text = text
            .replacingOccurrences(of: "\u{200B}", with: "")
            .replacingOccurrences(of: "\u{200C}", with: "")
            .replacingOccurrences(of: "\u{200D}", with: "")
            .replacingOccurrences(of: "\u{FEFF}", with: "")

        var lines: [String] = []
        var previousWasEmpty = true
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine
                .replacingOccurrences(of: #"[\t ]+"#, with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                if !previousWasEmpty { lines.append("") }
                previousWasEmpty = true
            } else {
                lines.append(line)
                previousWasEmpty = false
            }
        }
        while lines.last?.isEmpty == true { lines.removeLast() }
        return bounded(lines.joined(separator: "\n"), maxCharacters: maxCharacters)
    }

    /// Removes provider-only identifiers and tracking content after internal
    /// multi-step routing has consumed the raw observation.
    static func sanitizedFinalObservation(_ observation: String, toolID _: String) -> String {
        let bodyMarker = observation.range(of: #"(?im)^Body:\s*"#, options: .regularExpression)
        let rawMetadata = bodyMarker.map { String(observation[..<$0.lowerBound]) } ?? observation
        let metadata = sanitizedProviderMetadata(rawMetadata)

        guard let bodyMarker else { return metadata }
        let body = plainText(String(observation[bodyMarker.upperBound...]), maxCharacters: maxBodyCharacters)
        let sections = [
            metadata,
            body.isEmpty ? "" : "Body:\n\(body)"
        ].filter { !$0.isEmpty }
        return bounded(sections.joined(separator: "\n"), maxCharacters: maxFinalCharacters)
    }

    private static func sanitizedProviderMetadata(_ rawMetadata: String) -> String {
        var text = rawMetadata.replacingOccurrences(
            of: #"(?im)^\s*Cached references:.*(?:\r?\n\s*)?(?:---\s*)?"#,
            with: "",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"(?im)^\s*(?:message\s+)?id:\s*[^\r\n]*(?:\r?\n)?"#,
            with: "",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"(?i)\s+[—-]\s+id:\s*[^,\r\n]+,\s*"#,
            with: " — ",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"(?i)No attachments found for message\s+[^.\s]+\."#,
            with: "No attachments found for the selected Outlook message.",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"(?i)\b(?:AAMk|AQMk)[A-Za-z0-9_+=-]{20,}"#,
            with: "[internal reference omitted]",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"(?i)https?://[^\s<>\"']+"#,
            with: "[link omitted]",
            options: .regularExpression
        )

        return plainText(text, maxCharacters: maxFinalCharacters)
    }

    private static func bounded(_ text: String, maxCharacters: Int) -> String {
        guard text.count > maxCharacters else { return text }
        let prefix = text.prefix(maxCharacters)
        if let newline = prefix.lastIndex(of: "\n"), prefix.distance(from: newline, to: prefix.endIndex) < 240 {
            return String(prefix[..<newline]).trimmingCharacters(in: .whitespacesAndNewlines) + "\n…"
        }
        return String(prefix).trimmingCharacters(in: .whitespacesAndNewlines) + "…"
    }

    private static func decodeHTMLEntities(_ value: String) -> String {
        var text = decodeNumericHTMLEntities(value)
        let named: [(String, String)] = [
            ("&nbsp;", " "), ("&ensp;", " "), ("&emsp;", " "),
            ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""),
            ("&#39;", "'"), ("&apos;", "'"), ("&amp;", "&"),
            ("&lsquo;", "‘"), ("&rsquo;", "’"), ("&ldquo;", "“"), ("&rdquo;", "”"),
            ("&ndash;", "–"), ("&mdash;", "—"), ("&hellip;", "…"), ("&bull;", "•"),
            ("&agrave;", "à"), ("&Agrave;", "À"), ("&acirc;", "â"), ("&ccedil;", "ç"),
            ("&eacute;", "é"), ("&Eacute;", "É"), ("&egrave;", "è"), ("&ecirc;", "ê"),
            ("&icirc;", "î"), ("&ocirc;", "ô"), ("&ugrave;", "ù"), ("&ucirc;", "û")
        ]
        for (entity, replacement) in named {
            text = text.replacingOccurrences(of: entity, with: replacement)
        }
        return text.replacingOccurrences(
            of: #"&[A-Za-z][A-Za-z0-9]+;"#,
            with: " ",
            options: .regularExpression
        )
    }

    private static func decodeNumericHTMLEntities(_ value: String) -> String {
        let pattern = #"&#(?:x([0-9A-Fa-f]+)|([0-9]+));"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return value }
        let matches = regex.matches(in: value, range: NSRange(value.startIndex..., in: value))
        guard !matches.isEmpty else { return value }

        let source = value as NSString
        var result = value
        for match in matches.reversed() {
            let hexRange = match.range(at: 1)
            let decimalRange = match.range(at: 2)
            let scalarValue: UInt32?
            if hexRange.location != NSNotFound {
                scalarValue = UInt32(source.substring(with: hexRange), radix: 16)
            } else if decimalRange.location != NSNotFound {
                scalarValue = UInt32(source.substring(with: decimalRange), radix: 10)
            } else {
                scalarValue = nil
            }
            guard let scalarValue,
                  let scalar = UnicodeScalar(scalarValue),
                  let range = Range(match.range, in: result) else { continue }
            result.replaceSubrange(range, with: String(scalar))
        }
        return result
    }
}

@MainActor
enum OutlookTools {
    private static let client = OutlookGraphToolClient()
    private static let recentMessageReferencesKey = "OutlookTools.recentMessageReferences.v1"
    private static let recentMessageTTL: TimeInterval = 30 * 60

    static func status() async -> OutlookToolOutcome {
        do {
            _ = try MicrosoftGraphConfiguration.load()
        } catch {
            return OutlookToolOutcome.failure(from: error, diagnostics: buildDiagnostics())
        }
        let auth = MicrosoftGraphAuthManager()
        await auth.bootstrap()
        let diagnostics = buildDiagnostics(auth: auth)
        if let authError = auth.lastError {
            return OutlookToolOutcome.failure(from: authError, diagnostics: diagnostics)
        }
        guard auth.isSignedIn else {
            return .authUnavailable("Outlook is not signed in. Open Outlook in Lumen and sign in first.", diagnostics: diagnostics)
        }
        let username = auth.account?.username ?? auth.account?.name ?? "Microsoft account"
        let cached = loadRecentReferences()
        let contextLine = cached.isEmpty ? "No cached message context." : "Cached message context: \(cached.count) recent message(s)."
        return .success("Outlook signed in as \(username). Auth provider: \(auth.authProviderDescription). \(contextLine)", diagnostics: diagnostics)
    }

    static func listFolders(args: [String: String]) async -> OutlookToolOutcome {
        await perform(scopes: MicrosoftGraphScope.inboxRead) { token in
            let folders = try await client.listFolders(accessToken: token, includeHidden: bool(args["includeHidden"]))
            if folders.isEmpty { return .validEmpty("No Outlook mail folders found.") }
            return .success(folders.map { folder in
                "- \(folder.displayName) — id: \(folder.id), unread: \(folder.unreadItemCount ?? 0), total: \(folder.totalItemCount ?? 0)"
            }.joined(separator: "\n"))
        }
    }

    static func listMessages(args: [String: String]) async -> OutlookToolOutcome {
        await perform(scopes: MicrosoftGraphScope.inboxRead) { token in
            let messages = try await client.listMessages(
                folderID: folderID(from: args),
                pageSize: int(args["limit"] ?? args["top"], defaultValue: 10),
                unreadOnly: bool(args["unreadOnly"] ?? args["unread"]),
                accessToken: token
            )
            remember(messages: messages, source: "list")
            return messages.isEmpty
                ? .validEmpty(formatMessages(messages, includeBody: false))
                : .success(formatMessages(messages, includeBody: false))
        }
    }

    static func searchMessages(args: [String: String]) async -> OutlookToolOutcome {
        let query = args["query"] ?? args["q"] ?? args["search"] ?? ""
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .invalidArguments("Missing Outlook search query.") }
        return await perform(scopes: MicrosoftGraphScope.inboxRead) { token in
            let messages = try await client.searchMessages(
                query: query,
                folderID: folderID(from: args),
                pageSize: int(args["limit"] ?? args["top"], defaultValue: 10),
                accessToken: token
            )
            remember(messages: messages, source: "search: \(query)")
            return messages.isEmpty
                ? .validEmpty(formatMessages(messages, includeBody: false))
                : .success(formatMessages(messages, includeBody: false))
        }
    }

    static func readMessage(args: [String: String]) async -> OutlookToolOutcome {
        guard let id = messageID(from: args) else { return .invalidArguments(missingMessageContextMessage(action: "read")) }
        return await perform(scopes: MicrosoftGraphScope.inboxRead) { token in
            let message = try await client.readMessage(messageID: id, accessToken: token)
            remember(message: message, source: "read")
            return .success(formatMessage(message, includeBody: true))
        }
    }

    static func listAttachments(args: [String: String]) async -> OutlookToolOutcome {
        guard let id = messageID(from: args) else { return .invalidArguments(missingMessageContextMessage(action: "list attachments for")) }
        return await perform(scopes: MicrosoftGraphScope.inboxRead) { token in
            let attachments = try await client.listAttachments(messageID: id, accessToken: token)
            if attachments.isEmpty { return .validEmpty("No attachments found for message \(id).") }
            return .success(attachments.map { attachment in
                "- \(attachment.name ?? "Unnamed attachment") — id: \(attachment.id), type: \(attachment.contentType ?? "unknown"), size: \(attachment.size ?? 0) bytes, inline: \(attachment.isInline ?? false)"
            }.joined(separator: "\n"))
        }
    }

    static func createDraft(args: [String: String]) async -> OutlookToolOutcome {
        let recipients = recipients(from: args)
        guard !recipients.isEmpty else { return .invalidArguments("Missing Outlook draft recipient. Args: to, subject, body.") }
        let subject = args["subject"] ?? ""
        let body = args["body"] ?? args["message"] ?? args["text"] ?? ""
        guard !subject.isEmpty || !body.isEmpty else { return .invalidArguments("Missing Outlook draft subject/body.") }
        return await perform(scopes: MicrosoftGraphScope.readWriteMail) { token in
            let draft = try await client.createDraft(subject: subject, body: body, recipients: recipients, accessToken: token)
            remember(message: draft, source: "draft")
            return .success("Created Outlook draft: \(draft.subject ?? "(No subject)")\nMessage id: \(draft.id)")
        }
    }

    static func sendMail(args: [String: String]) async -> OutlookToolOutcome {
        let recipients = recipients(from: args)
        guard !recipients.isEmpty else { return .invalidArguments("Missing Outlook send recipient. Args: to, subject, body.") }
        let subject = args["subject"] ?? ""
        let body = args["body"] ?? args["message"] ?? args["text"] ?? ""
        guard !subject.isEmpty || !body.isEmpty else { return .invalidArguments("Missing Outlook send subject/body.") }
        return await perform(scopes: MicrosoftGraphScope.readWriteMail) { token in
            try await client.sendMail(subject: subject, body: body, recipients: recipients, accessToken: token)
            return .success("Sent Outlook email to \(recipients.joined(separator: ", ")).")
        }
    }

    static func markRead(args: [String: String], isRead: Bool) async -> OutlookToolOutcome {
        guard let id = messageID(from: args) else { return .invalidArguments(missingMessageContextMessage(action: isRead ? "mark read" : "mark unread")) }
        return await perform(scopes: MicrosoftGraphScope.readWriteMail) { token in
            let message = try await client.markRead(messageID: id, isRead: isRead, accessToken: token)
            remember(message: message, source: isRead ? "mark_read" : "mark_unread")
            return .success("Marked Outlook message as \(isRead ? "read" : "unread"): \(message.subject ?? id)")
        }
    }

    static func moveMessage(args: [String: String]) async -> OutlookToolOutcome {
        guard let id = messageID(from: args) else { return .invalidArguments(missingMessageContextMessage(action: "move")) }
        let destination = args["destinationId"] ?? args["destination"] ?? args["folderId"] ?? args["folder"] ?? ""
        guard !destination.isEmpty else { return .invalidArguments("Missing destination folder id/name. Use archive, deleteditems, junkemail, inbox, or a folder id.") }
        return await perform(scopes: MicrosoftGraphScope.readWriteMail) { token in
            let moved = try await client.moveMessage(messageID: id, destinationID: canonicalFolderID(destination), accessToken: token)
            remember(message: moved, source: "move: \(destination)")
            return .success("Moved Outlook message to \(destination). New message id: \(moved.id)")
        }
    }

    static func deleteMessage(args: [String: String]) async -> OutlookToolOutcome {
        guard let id = messageID(from: args) else { return .invalidArguments(missingMessageContextMessage(action: "delete")) }
        return await perform(scopes: MicrosoftGraphScope.readWriteMail) { token in
            try await client.deleteMessage(messageID: id, accessToken: token)
            removeFromRecentReferences(messageID: id)
            return .success("Deleted Outlook message: \(id)")
        }
    }

    static func reply(args: [String: String], replyAll: Bool) async -> OutlookToolOutcome {
        guard let id = messageID(from: args) else { return .invalidArguments(missingMessageContextMessage(action: replyAll ? "reply-all to" : "reply to")) }
        let comment = args["body"] ?? args["comment"] ?? args["message"] ?? args["text"] ?? ""
        guard !comment.isEmpty else { return .invalidArguments("Missing reply body/comment.") }
        return await perform(scopes: MicrosoftGraphScope.readWriteMail) { token in
            if replyAll { try await client.replyAll(messageID: id, comment: comment, accessToken: token) }
            else { try await client.reply(messageID: id, comment: comment, accessToken: token) }
            return .success(replyAll ? "Sent Outlook reply-all to cached message \(id)." : "Sent Outlook reply to cached message \(id).")
        }
    }

    static func forward(args: [String: String]) async -> OutlookToolOutcome {
        guard let id = messageID(from: args) else { return .invalidArguments(missingMessageContextMessage(action: "forward")) }
        let to = recipients(from: args)
        guard !to.isEmpty else { return .invalidArguments("Missing forward recipient.") }
        let comment = args["body"] ?? args["comment"] ?? args["message"] ?? args["text"] ?? ""
        return await perform(scopes: MicrosoftGraphScope.readWriteMail) { token in
            try await client.forward(messageID: id, comment: comment, recipients: to, accessToken: token)
            return .success("Forwarded Outlook message \(id) to \(to.joined(separator: ", ")).")
        }
    }

    static func recentContextSummary() -> String {
        let refs = loadRecentReferences()
        if refs.isEmpty { return "No cached Outlook message context. List or search messages first." }
        return refs.map(\.displayLine).joined(separator: "\n")
    }

    private static func perform(scopes: [String], operation: @escaping (String) async throws -> OutlookToolOutcome) async -> OutlookToolOutcome {
        do {
            _ = try MicrosoftGraphConfiguration.load()
            let auth = MicrosoftGraphAuthManager()
            await auth.bootstrap()
            let diagnostics = buildDiagnostics(auth: auth)
            if let authError = auth.lastError {
                return OutlookToolOutcome.failure(from: authError, diagnostics: diagnostics)
            }
            guard auth.isSignedIn else {
                return .authUnavailable("Outlook is not signed in. Open Outlook in Lumen and sign in first.", diagnostics: diagnostics)
            }
            let token = try await auth.acquireToken(scopes: scopes, preferredAccountID: auth.account?.id)
            var outcome = try await operation(token)
            if outcome.diagnostics.isEmpty {
                outcome = OutlookToolOutcome(
                    text: outcome.text,
                    status: outcome.status,
                    availability: outcome.availability,
                    errorCode: outcome.errorCode,
                    diagnostics: diagnostics
                )
            }
            return outcome
        } catch {
            return OutlookToolOutcome.failure(from: error, diagnostics: buildDiagnostics())
        }
    }

    private static func buildDiagnostics(auth: MicrosoftGraphAuthManager? = nil) -> [String: String] {
        let config = try? MicrosoftGraphConfiguration.load()
        let bundleID = Bundle.main.bundleIdentifier ?? "Unavailable"
        var diagnostics: [String: String] = [
            "bundleID": bundleID,
            "redirectURI": config?.redirectURI ?? "msauth.\(bundleID)://auth",
            "authProvider": auth?.authProviderDescription ?? ((config?.forceNativeOAuth == true) ? "Native OAuth PKCE" : "MSAL")
        ]
        if let config {
            diagnostics["clientID"] = config.clientID
            diagnostics["authorityHost"] = config.authorityURL.host ?? "unknown"
        }
        if let account = auth?.account {
            diagnostics["accountPresent"] = "true"
            if let environment = account.environment { diagnostics["accountEnvironment"] = environment }
        } else {
            diagnostics["accountPresent"] = "false"
        }
        return diagnostics
    }

    private static func formatMessages(_ messages: [GraphMailMessage], includeBody: Bool) -> String {
        if messages.isEmpty { return "No Outlook messages found." }
        let contextHint = "Cached references: use ordinal args like {\"message\":\"first\"}, {\"message\":\"#2\"}, {\"message\":\"latest\"}, or the raw Message ID for follow-up tools."
        return ([contextHint] + messages.enumerated().map { index, message in
            formatMessage(message, includeBody: includeBody, ordinal: index + 1)
        }).joined(separator: "\n\n---\n\n")
    }

    private static func formatMessage(_ message: GraphMailMessage, includeBody: Bool, ordinal: Int? = nil) -> String {
        let preview = OutlookToolUserVisibleOutput.plainText(
            message.previewLine,
            maxCharacters: OutlookToolUserVisibleOutput.maxPreviewCharacters
        )
        var lines: [String] = []
        if let ordinal { lines.append("Index: \(ordinal)") }
        lines.append(contentsOf: [
            "Subject: \(message.subject?.isEmpty == false ? message.subject! : "(No subject)")",
            "ID: \(message.id)",
            "From: \(message.senderLine)",
            "Received: \(message.receivedDateTime ?? "unknown")",
            "Unread: \((message.isRead ?? true) ? "false" : "true")",
            "Has attachments: \(message.hasAttachments ?? false)",
            "Preview: \(preview)"
        ])
        if includeBody, let body = message.body?.content, !body.isEmpty {
            let bodyText = OutlookToolUserVisibleOutput.plainText(
                body,
                maxCharacters: OutlookToolUserVisibleOutput.maxBodyCharacters
            )
            if !bodyText.isEmpty {
                lines.append("Body:\n\(bodyText)")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func remember(messages: [GraphMailMessage], source: String) {
        let refs = messages.prefix(50).enumerated().map { index, message in
            OutlookMessageReference(
                ordinal: index + 1,
                id: message.id,
                subject: message.subject?.isEmpty == false ? message.subject! : "(No subject)",
                sender: message.senderLine,
                receivedDateTime: message.receivedDateTime,
                source: source,
                cachedAt: Date()
            )
        }
        saveRecentReferences(Array(refs))
    }

    private static func remember(message: GraphMailMessage, source: String) {
        let current = loadRecentReferences().filter { $0.id != message.id }
        let newRef = OutlookMessageReference(
            ordinal: 1,
            id: message.id,
            subject: message.subject?.isEmpty == false ? message.subject! : "(No subject)",
            sender: message.senderLine,
            receivedDateTime: message.receivedDateTime,
            source: source,
            cachedAt: Date()
        )
        let merged = ([newRef] + current).prefix(50).enumerated().map { index, ref in
            OutlookMessageReference(
                ordinal: index + 1,
                id: ref.id,
                subject: ref.subject,
                sender: ref.sender,
                receivedDateTime: ref.receivedDateTime,
                source: ref.source,
                cachedAt: ref.cachedAt
            )
        }
        saveRecentReferences(Array(merged))
    }

    private static func removeFromRecentReferences(messageID: String) {
        let remaining = loadRecentReferences().filter { $0.id != messageID }.enumerated().map { index, ref in
            OutlookMessageReference(
                ordinal: index + 1,
                id: ref.id,
                subject: ref.subject,
                sender: ref.sender,
                receivedDateTime: ref.receivedDateTime,
                source: ref.source,
                cachedAt: ref.cachedAt
            )
        }
        saveRecentReferences(remaining)
    }

    private static func loadRecentReferences() -> [OutlookMessageReference] {
        guard let data = UserDefaults.standard.data(forKey: recentMessageReferencesKey),
              let refs = try? JSONDecoder().decode([OutlookMessageReference].self, from: data) else { return [] }
        let cutoff = Date().addingTimeInterval(-recentMessageTTL)
        let fresh = refs.filter { $0.cachedAt >= cutoff }
        if fresh.count != refs.count { saveRecentReferences(fresh) }
        return fresh
    }

    private static func saveRecentReferences(_ refs: [OutlookMessageReference]) {
        let normalized = refs.prefix(50).enumerated().map { index, ref in
            OutlookMessageReference(
                ordinal: index + 1,
                id: ref.id,
                subject: ref.subject,
                sender: ref.sender,
                receivedDateTime: ref.receivedDateTime,
                source: ref.source,
                cachedAt: ref.cachedAt
            )
        }
        if let data = try? JSONEncoder().encode(Array(normalized)) {
            UserDefaults.standard.set(data, forKey: recentMessageReferencesKey)
        }
    }

    private static func recipients(from args: [String: String]) -> [String] {
        let raw = args["to"] ?? args["recipient"] ?? args["recipients"] ?? args["email"] ?? ""
        return raw.split { $0 == "," || $0 == ";" || $0 == "\n" }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func messageID(from args: [String: String]) -> String? {
        let raw = args["messageId"] ?? args["messageID"] ?? args["id"] ?? args["message"] ?? args["reference"] ?? args["ordinal"]
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let resolved = resolveCachedMessageID(trimmed) { return resolved }

        let lowered = trimmed.lowercased()
        let contextWords: Set<String> = [
            "this", "that", "it", "latest", "last", "newest", "recent", "first", "second", "third", "fourth", "fifth",
            "selected", "current", "previous", "prior", "top", "#1", "#2", "#3", "#4", "#5"
        ]
        if contextWords.contains(lowered) { return nil }
        return trimmed
    }

    private static func resolveCachedMessageID(_ reference: String) -> String? {
        let refs = loadRecentReferences()
        guard !refs.isEmpty else { return nil }
        let normalized = reference.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            .replacingOccurrences(of: "message", with: "")
            .replacingOccurrences(of: "email", with: "")
            .replacingOccurrences(of: "mail", with: "")
            .replacingOccurrences(of: "number", with: "")
            .replacingOccurrences(of: "no.", with: "")
            .replacingOccurrences(of: "#", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if ["this", "that", "it", "selected", "current", "latest", "newest", "recent", "top", "first", "1", "one"].contains(normalized) {
            return refs.first?.id
        }
        if ["last", "oldest"].contains(normalized) { return refs.last?.id }

        let wordsToIndex: [String: Int] = [
            "second": 2, "two": 2,
            "third": 3, "three": 3,
            "fourth": 4, "four": 4,
            "fifth": 5, "five": 5,
            "sixth": 6, "six": 6,
            "seventh": 7, "seven": 7,
            "eighth": 8, "eight": 8,
            "ninth": 9, "nine": 9,
            "tenth": 10, "ten": 10
        ]
        if let index = wordsToIndex[normalized], refs.indices.contains(index - 1) { return refs[index - 1].id }
        if let index = Int(normalized), refs.indices.contains(index - 1) { return refs[index - 1].id }
        if let exact = refs.first(where: { $0.id == reference }) { return exact.id }
        return nil
    }

    private static func missingMessageContextMessage(action: String) -> String {
        let refs = loadRecentReferences()
        guard !refs.isEmpty else {
            return "Missing Outlook message context. Ask me to list or search Outlook messages first, then say which one to \(action)."
        }
        return "Which Outlook message should I \(action)? Available cached messages:\n\(refs.prefix(10).map(\.displayLine).joined(separator: "\n"))"
    }

    private static func folderID(from args: [String: String]) -> String? {
        let value = args["folderId"] ?? args["folderID"] ?? args["folder"]
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else { return nil }
        return canonicalFolderID(trimmed)
    }

    private static func canonicalFolderID(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "archive", "archives": return "archive"
        case "deleted", "delete", "trash", "deleted items", "deleteditems": return "deleteditems"
        case "junk", "spam", "junk email", "junkemail": return "junkemail"
        case "sent", "sent items", "sentitems": return "sentitems"
        case "draft", "drafts": return "drafts"
        case "inbox": return "inbox"
        default: return raw
        }
    }

    private static func int(_ value: String?, defaultValue: Int) -> Int {
        guard let value, let parsed = Int(value.trimmingCharacters(in: .whitespacesAndNewlines)) else { return defaultValue }
        return parsed
    }

    private static func bool(_ value: String?) -> Bool {
        guard let value else { return false }
        return ["1", "true", "yes", "y", "on", "unread"].contains(value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }
}
