import Foundation

nonisolated enum MicrosoftGraphScope: String, CaseIterable, Sendable {
    case userRead = "User.Read"
    case mailRead = "Mail.Read"
    case mailReadWrite = "Mail.ReadWrite"
    case mailSend = "Mail.Send"
    case offlineAccess = "offline_access"

    static let inboxRead: [String] = [userRead.rawValue, mailRead.rawValue, offlineAccess.rawValue]
    static let mailSendScopes: [String] = [userRead.rawValue, mailSend.rawValue, offlineAccess.rawValue]
    static let readWriteMail: [String] = [userRead.rawValue, mailReadWrite.rawValue, mailSend.rawValue, offlineAccess.rawValue]
}

nonisolated enum MicrosoftGraphAuthError: LocalizedError, Equatable, Sendable {
    case missingClientID
    case msalNotLinked
    case noAccount
    case interactionRequired
    case presentationAnchorUnavailable
    case signInCancelled
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .missingClientID:
            return "Missing Microsoft Graph client ID. Set MSALClientID in Info.plist or pass it through the MSAL_CLIENT_ID build setting."
        case .msalNotLinked:
            return "MSAL is not linked. Add microsoft-authentication-library-for-objc with Swift Package Manager to enable Microsoft sign-in."
        case .noAccount:
            return "No Microsoft account is currently signed in."
        case .interactionRequired:
            return "Microsoft requires an interactive sign-in before this action can continue."
        case .presentationAnchorUnavailable:
            return "No active view controller is available for Microsoft sign-in."
        case .signInCancelled:
            return "Microsoft sign-in was cancelled."
        case .invalidConfiguration(let message):
            return message
        }
    }
}

nonisolated struct MicrosoftGraphAccountSnapshot: Identifiable, Codable, Hashable, Sendable {
    let id: String
    let username: String?
    let name: String?
    let environment: String?
    let tenantID: String?
}

nonisolated struct MicrosoftGraphTokenSnapshot: Codable, Hashable, Sendable {
    let accessToken: String
    let expiresOn: Date?
    let scopes: [String]

    var shouldRefreshProactively: Bool {
        guard let expiresOn else { return false }
        return expiresOn.timeIntervalSinceNow < 300
    }
}

nonisolated struct GraphEmailAddress: Codable, Hashable, Sendable {
    let name: String?
    let address: String
}

nonisolated struct GraphRecipient: Codable, Hashable, Sendable {
    let emailAddress: GraphEmailAddress
}

nonisolated struct GraphMessageBody: Codable, Hashable, Sendable {
    let contentType: String
    let content: String
}

nonisolated struct GraphRemovedMarker: Codable, Hashable, Sendable {
    let reason: String?
}

nonisolated struct GraphMailMessage: Codable, Hashable, Identifiable, Sendable {
    let id: String
    let subject: String?
    let bodyPreview: String?
    let receivedDateTime: String?
    let sentDateTime: String?
    let isRead: Bool?
    let hasAttachments: Bool?
    let from: GraphRecipient?
    let toRecipients: [GraphRecipient]?
    let ccRecipients: [GraphRecipient]?
    let body: GraphMessageBody?
    let removed: GraphRemovedMarker?

    enum CodingKeys: String, CodingKey {
        case id
        case subject
        case bodyPreview
        case receivedDateTime
        case sentDateTime
        case isRead
        case hasAttachments
        case from
        case toRecipients
        case ccRecipients
        case body
        case removed = "@removed"
    }

    var senderLine: String {
        if let from {
            let trimmedName = from.emailAddress.name?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let trimmedName, !trimmedName.isEmpty {
                return trimmedName
            }
            return from.emailAddress.address
        }
        return "Unknown sender"
    }

    var subjectText: String {
        let trimmedSubject = subject?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmedSubject, !trimmedSubject.isEmpty { return trimmedSubject }
        return "(No subject)"
    }

    var previewLine: String {
        let trimmedPreview = bodyPreview?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmedPreview, !trimmedPreview.isEmpty { return trimmedPreview }
        return "No preview available"
    }
}

nonisolated struct GraphMailPage: Codable, Hashable, Sendable {
    let value: [GraphMailMessage]
    let odataNextLink: String?
    let odataDeltaLink: String?

    enum CodingKeys: String, CodingKey {
        case value
        case odataNextLink = "@odata.nextLink"
        case odataDeltaLink = "@odata.deltaLink"
    }
}

nonisolated struct GraphSendMailRequest: Encodable, Sendable {
    let message: MailMessage
    let saveToSentItems: Bool

    nonisolated struct MailMessage: Encodable, Sendable {
        let subject: String
        let body: Body
        let toRecipients: [Recipient]
        let ccRecipients: [Recipient]?
        let bccRecipients: [Recipient]?
        let attachments: [FileAttachment]?
    }

    nonisolated struct Body: Encodable, Sendable {
        let contentType: String
        let content: String
    }

    nonisolated struct Recipient: Encodable, Sendable {
        let emailAddress: EmailAddress
    }

    nonisolated struct EmailAddress: Encodable, Sendable {
        let address: String
        let name: String?
    }

    nonisolated struct FileAttachment: Encodable, Sendable {
        let odataType: String = "#microsoft.graph.fileAttachment"
        let name: String
        let contentType: String
        let contentBytes: String

        enum CodingKeys: String, CodingKey {
            case odataType = "@odata.type"
            case name
            case contentType
            case contentBytes
        }
    }
}

nonisolated struct GraphUploadSession: Codable, Sendable {
    let uploadUrl: String
    let expirationDateTime: String?
    let nextExpectedRanges: [String]?
}

nonisolated struct GraphAPIErrorEnvelope: Codable, Error, Sendable {
    let error: GraphAPIErrorBody

    var isRetryable: Bool {
        ["TooManyRequests", "ServiceUnavailable", "temporarilyUnavailable"].contains(error.code)
    }

    nonisolated struct GraphAPIErrorBody: Codable, Sendable {
        let code: String
        let message: String
        let innerError: InnerError?

        nonisolated struct InnerError: Codable, Sendable {
            let requestId: String?
            let date: String?

            enum CodingKeys: String, CodingKey {
                case requestId = "request-id"
                case date
            }
        }
    }
}

nonisolated enum GraphHTTPError: LocalizedError, Equatable, Sendable {
    case unexpectedStatus(Int)
    case missingURL
    case throttled(retryAfter: TimeInterval?)

    var errorDescription: String? {
        switch self {
        case .unexpectedStatus(let status): return "Microsoft Graph returned unexpected HTTP status \(status)."
        case .missingURL: return "Could not construct a Microsoft Graph URL."
        case .throttled(let retryAfter):
            if let retryAfter { return "Microsoft Graph throttled the request. Retry after \(Int(retryAfter)) seconds." }
            return "Microsoft Graph throttled the request."
        }
    }
}


nonisolated enum MicrosoftGraphRuntimeConfig {
    static let clientIDDefaultsKey = "MSALClientIDOverride"

    static func saveClientIDOverride(_ clientID: String?) {
        let trimmed = clientID?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmed, !trimmed.isEmpty {
            UserDefaults.standard.set(trimmed, forKey: clientIDDefaultsKey)
        } else {
            UserDefaults.standard.removeObject(forKey: clientIDDefaultsKey)
        }
    }

    static func loadClientIDOverride() -> String? {
        let value = UserDefaults.standard.string(forKey: clientIDDefaultsKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value, !value.isEmpty else { return nil }
        return value
    }
}

nonisolated struct MicrosoftGraphConfiguration: Sendable {
    let clientID: String
    let authorityURL: URL
    let keychainSharingGroup: String
    let redirectURI: String?
    let forceNativeOAuth: Bool

    static func load(bundle: Bundle = .main) throws -> MicrosoftGraphConfiguration {
        let resourceConfig = bundle.url(forResource: "MicrosoftGraphConfig", withExtension: "plist")
            .flatMap { NSDictionary(contentsOf: $0) as? [String: Any] } ?? [:]

        func value(_ key: String) -> String? {
            let info = (bundle.object(forInfoDictionaryKey: key) as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            let resource = (resourceConfig[key] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            let candidate = info?.isEmpty == false ? info : resource
            guard let candidate, !candidate.isEmpty, !candidate.hasPrefix("$("), !candidate.hasPrefix("YOUR_") else { return nil }
            return candidate
        }

        let runtimeClientID = MicrosoftGraphRuntimeConfig.loadClientIDOverride()
        guard let clientID = runtimeClientID ?? value("MSALClientID") else { throw MicrosoftGraphAuthError.missingClientID }
        let authorityString = value("MSALAuthorityURL") ?? "https://login.microsoftonline.com/common"
        guard let authorityURL = URL(string: authorityString) else {
            throw MicrosoftGraphAuthError.invalidConfiguration("Invalid Microsoft identity authority URL: \(authorityString)")
        }
        let keychainGroup = value("MSALKeychainGroup") ?? "com.microsoft.adalcache"
        let redirect = value("MSALRedirectURI")
        let forceNativeOAuth = ((resourceConfig["MSALForceNativeOAuth"] as? NSNumber)?.boolValue ?? false)

        return MicrosoftGraphConfiguration(
            clientID: clientID,
            authorityURL: authorityURL,
            keychainSharingGroup: keychainGroup,
            redirectURI: redirect,
            forceNativeOAuth: forceNativeOAuth
        )
    }
}
