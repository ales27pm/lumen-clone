import Foundation
import OSLog

nonisolated enum MicrosoftGraphMailCacheError: LocalizedError, Equatable, Sendable {
    case saveFailed
    case purgeFailed

    var errorDescription: String? {
        switch self {
        case .saveFailed:
            return "The protected Outlook mail cache could not be saved."
        case .purgeFailed:
            return "The protected Outlook mail cache could not be removed during sign-out."
        }
    }
}

nonisolated enum MicrosoftGraphMutationCompletionError: LocalizedError, Equatable, Sendable {
    case indeterminate

    var errorDescription: String? {
        "The Outlook action may have completed after the account changed; verify in Outlook before taking another action."
    }
}

struct MicrosoftGraphMailCacheFileIO {
    let cacheDirectory: () throws -> URL
    let fileExists: (URL) -> Bool
    let readData: (URL) throws -> Data
    let createDirectory: (URL) throws -> Void
    let writeData: (Data, URL) throws -> Void
    let removeItem: (URL) throws -> Void

    static func live(fileManager: FileManager = .default) -> MicrosoftGraphMailCacheFileIO {
        MicrosoftGraphMailCacheFileIO(
            cacheDirectory: {
                let support = try fileManager.url(
                    for: .applicationSupportDirectory,
                    in: .userDomainMask,
                    appropriateFor: nil,
                    create: true
                )
                return support.appendingPathComponent("MicrosoftGraphMail", isDirectory: true)
            },
            fileExists: { fileManager.fileExists(atPath: $0.path) },
            readData: { try Data(contentsOf: $0) },
            createDirectory: {
                try fileManager.createDirectory(
                    at: $0,
                    withIntermediateDirectories: true,
                    attributes: nil
                )
            },
            writeData: {
                try $0.write(to: $1, options: [.atomic, .completeFileProtection])
            },
            removeItem: { try fileManager.removeItem(at: $0) }
        )
    }
}

@MainActor
final class MicrosoftGraphMailCacheStore {
    static let shared = MicrosoftGraphMailCacheStore()
    private let logger = Logger(subsystem: "ai.lumen.microsoftgraph", category: "cache")
    private let fileIO: MicrosoftGraphMailCacheFileIO
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let dateParserWithFractional = ISO8601DateFormatter()
    private let dateParser = ISO8601DateFormatter()

    init(fileIO: MicrosoftGraphMailCacheFileIO = .live()) {
        self.fileIO = fileIO
        encoder.outputFormatting = [.sortedKeys]
        dateParserWithFractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        dateParser.formatOptions = [.withInternetDateTime]
    }

    nonisolated struct Snapshot: Codable, Sendable {
        var messages: [GraphMailMessage]
        var deltaLink: String?
        var updatedAt: Date
    }

    func load(accountID: String) -> Snapshot {
        do {
            let url = try cacheURL(accountID: accountID)
            guard fileIO.fileExists(url) else {
                return Snapshot(messages: [], deltaLink: nil, updatedAt: .distantPast)
            }
            let data = try fileIO.readData(url)
            return try decoder.decode(Snapshot.self, from: data)
        } catch {
            logger.error("Failed to load Microsoft Graph mail cache: \(String(describing: error), privacy: .private)")
            return Snapshot(messages: [], deltaLink: nil, updatedAt: .distantPast)
        }
    }

    func save(_ snapshot: Snapshot, accountID: String) throws {
        do {
            let url = try cacheURL(accountID: accountID)
            try fileIO.createDirectory(url.deletingLastPathComponent())
            let data = try encoder.encode(snapshot)
            try fileIO.writeData(data, url)
        } catch {
            logger.error("Failed to save the protected Microsoft Graph mail cache.")
            throw MicrosoftGraphMailCacheError.saveFailed
        }
    }

    func merge(existing: Snapshot, incoming: [GraphMailMessage], deltaLink: String?) -> Snapshot {
        var byID = Dictionary(uniqueKeysWithValues: existing.messages.map { ($0.id, $0) })
        for message in incoming {
            if message.removed != nil {
                byID.removeValue(forKey: message.id)
            } else {
                byID[message.id] = message
            }
        }
        let sorted = byID.values.sorted { lhs, rhs in
            parsedDate(for: lhs) > parsedDate(for: rhs)
        }
        return Snapshot(messages: Array(sorted.prefix(250)), deltaLink: deltaLink ?? existing.deltaLink, updatedAt: Date())
    }

    func clearAll() throws {
        do {
            let directory = try cacheDirectory()
            if fileIO.fileExists(directory) {
                try fileIO.removeItem(directory)
            }
        } catch {
            logger.error("Failed to purge the protected Microsoft Graph mail cache.")
            throw MicrosoftGraphMailCacheError.purgeFailed
        }
    }

    private func cacheURL(accountID: String) throws -> URL {
        let safe = accountID.replacingOccurrences(of: "[^a-zA-Z0-9._-]", with: "_", options: .regularExpression)
        return try cacheDirectory().appendingPathComponent("\(safe).json")
    }

    private func cacheDirectory() throws -> URL {
        try fileIO.cacheDirectory()
    }

    private func parsedDate(for message: GraphMailMessage) -> Date {
        guard let timestamp = message.receivedDateTime ?? message.sentDateTime else { return .distantPast }
        return dateParserWithFractional.date(from: timestamp) ?? dateParser.date(from: timestamp) ?? .distantPast
    }
}

@MainActor
protocol MicrosoftGraphInboxAuthorizing: AnyObject {
    var account: MicrosoftGraphAccountSnapshot? { get }
    var token: MicrosoftGraphTokenSnapshot? { get }

    func acquireToken(scopes: [String], preferredAccountID: String?, forceRefresh: Bool) async throws -> String
    func captureAuthorization(for accountID: String) throws -> MicrosoftGraphAuthEpoch.Snapshot
    func requireCurrentAuthorization(
        _ snapshot: MicrosoftGraphAuthEpoch.Snapshot,
        for accountID: String
    ) throws
}

extension MicrosoftGraphAuthManager: MicrosoftGraphInboxAuthorizing {}

protocol MicrosoftGraphInboxClientProtocol: Sendable {
    func fetchInboxPage(
        accessToken: String,
        pageSize: Int,
        nextOrDeltaLink: String?
    ) async throws -> GraphMailPage
    func sendMail(_ mail: GraphSendMailRequest, accessToken: String) async throws
}

extension MicrosoftGraphMailClient: MicrosoftGraphInboxClientProtocol {}

protocol MicrosoftGraphMessageBodyClientProtocol: Sendable {
    func fetchMessageBody(messageID: String, accessToken: String) async throws -> GraphMailMessage
}

extension MicrosoftGraphMailClient: MicrosoftGraphMessageBodyClientProtocol {}

@MainActor
struct OutlookPresentationAccountGate: Equatable {
    let accountID: String
    let authorization: MicrosoftGraphAuthEpoch.Snapshot

    func isCurrent(_ currentAuthorization: MicrosoftGraphAuthEpoch.Snapshot) -> Bool {
        authorization == currentAuthorization
            && currentAuthorization.accountID == accountID
    }
}

@MainActor
enum MicrosoftGraphMessageBodyLoader {
    static func loadAndPublish(
        messageID: String,
        accountID: String,
        auth: any MicrosoftGraphInboxAuthorizing,
        client: any MicrosoftGraphMessageBodyClientProtocol,
        publish: (GraphMailMessage) -> Void
    ) async throws {
        let authorization = try auth.captureAuthorization(for: accountID)
        do {
            let accessToken = try await auth.acquireToken(
                scopes: MicrosoftGraphScope.inboxRead,
                preferredAccountID: accountID,
                forceRefresh: auth.token?.shouldRefreshProactively == true
            )
            let message = try await client.fetchMessageBody(
                messageID: messageID,
                accessToken: accessToken
            )
            // The check and publication are synchronous on the main actor. An
            // account transition cannot interleave and expose the old body.
            try auth.requireCurrentAuthorization(authorization, for: accountID)
            publish(message)
        } catch {
            do {
                try auth.requireCurrentAuthorization(authorization, for: accountID)
            } catch {
                throw MicrosoftGraphAuthEpochError.staleCompletion
            }
            throw error
        }
    }
}

@MainActor
@Observable
final class MicrosoftGraphInboxViewModel {
    private let auth: any MicrosoftGraphInboxAuthorizing
    private let client: any MicrosoftGraphInboxClientProtocol
    private let cache: MicrosoftGraphMailCacheStore

    private(set) var messages: [GraphMailMessage] = []
    private(set) var isLoading = false
    private(set) var lastSyncDate: Date?
    private(set) var error: Error?
    var unreadOnly = false

    init(
        auth: any MicrosoftGraphInboxAuthorizing,
        client: any MicrosoftGraphInboxClientProtocol = MicrosoftGraphMailClient(),
        cache: MicrosoftGraphMailCacheStore? = nil
    ) {
        self.auth = auth
        self.client = client
        self.cache = cache ?? MicrosoftGraphMailCacheStore.shared
    }

    func loadCached() {
        guard let account = auth.account else {
            messages = []
            lastSyncDate = nil
            return
        }
        do {
            let authorization = try auth.captureAuthorization(for: account.id)
            let snapshot = cache.load(accountID: account.id)
            try auth.requireCurrentAuthorization(authorization, for: account.id)
            messages = filteredMessages(from: snapshot)
            lastSyncDate = snapshot.updatedAt == .distantPast ? nil : snapshot.updatedAt
        } catch {
            messages = []
            lastSyncDate = nil
            self.error = error
        }
    }

    func refresh(resetDelta: Bool = false) async {
        guard let account = auth.account else {
            error = MicrosoftGraphAuthError.noAccount
            return
        }
        let authorization: MicrosoftGraphAuthEpoch.Snapshot
        do {
            authorization = try auth.captureAuthorization(for: account.id)
        } catch {
            self.error = error
            return
        }
        isLoading = true
        defer { isLoading = false }
        do {
            var snapshot = cache.load(accountID: account.id)
            let accessToken = try await auth.acquireToken(scopes: MicrosoftGraphScope.inboxRead, preferredAccountID: account.id, forceRefresh: auth.token?.shouldRefreshProactively == true)
            if resetDelta {
                snapshot = .init(messages: [], deltaLink: nil, updatedAt: .distantPast)
            }
            var nextLink: String? = resetDelta ? nil : snapshot.deltaLink
            var deltaLink: String?
            var changed: [GraphMailMessage] = []

            repeat {
                let page = try await client.fetchInboxPage(accessToken: accessToken, pageSize: 25, nextOrDeltaLink: nextLink)
                changed.append(contentsOf: page.value)
                if let pageDelta = page.odataDeltaLink {
                    deltaLink = pageDelta
                    nextLink = nil
                } else {
                    nextLink = page.odataNextLink
                }
            } while nextLink != nil

            snapshot = cache.merge(existing: snapshot, incoming: changed, deltaLink: deltaLink)
            // Both durable and visible mail state are guarded after the final
            // network suspension. A sign-out/account switch that occurred while
            // fetching therefore cannot repopulate disk or publish mailbox data.
            try auth.requireCurrentAuthorization(authorization, for: account.id)
            try cache.save(snapshot, accountID: account.id)
            try auth.requireCurrentAuthorization(authorization, for: account.id)
            messages = filteredMessages(from: snapshot)
            lastSyncDate = snapshot.updatedAt
            error = nil
        } catch {
            self.error = error
        }
    }

    func send(subject: String, body: String, recipients: [String], sendAsHTML: Bool = true) async throws {
        guard let account = auth.account else { throw MicrosoftGraphAuthError.noAccount }
        let authorization = try auth.captureAuthorization(for: account.id)
        let token = try await auth.acquireToken(
            scopes: MicrosoftGraphScope.mailSendScopes,
            preferredAccountID: account.id,
            forceRefresh: auth.token?.shouldRefreshProactively == true
        )
        let content = sendAsHTML ? Self.escapeHTML(body).replacingOccurrences(of: "\n", with: "<br>") : body
        let mail = GraphSendMailRequest(
            message: .init(
                subject: subject,
                body: .init(contentType: sendAsHTML ? "HTML" : "Text", content: content),
                toRecipients: recipients.map { .init(emailAddress: .init(address: $0, name: nil)) },
                ccRecipients: nil,
                bccRecipients: nil,
                attachments: nil
            ),
            saveToSentItems: true
        )
        do {
            try await client.sendMail(mail, accessToken: token)
            try auth.requireCurrentAuthorization(authorization, for: account.id)
        } catch {
            do {
                try auth.requireCurrentAuthorization(authorization, for: account.id)
            } catch is MicrosoftGraphAuthEpochError {
                throw MicrosoftGraphMutationCompletionError.indeterminate
            }
            throw error
        }
    }

    private func filteredMessages(from snapshot: MicrosoftGraphMailCacheStore.Snapshot) -> [GraphMailMessage] {
        guard unreadOnly else { return snapshot.messages }
        return snapshot.messages.filter { $0.isRead != true }
    }

    private nonisolated static func escapeHTML(_ input: String) -> String {
        input
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}
