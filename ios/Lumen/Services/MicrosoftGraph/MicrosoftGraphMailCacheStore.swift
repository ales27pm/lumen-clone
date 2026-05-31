import Foundation
import OSLog

@MainActor
final class MicrosoftGraphMailCacheStore {
    static let shared = MicrosoftGraphMailCacheStore()
    private let logger = Logger(subsystem: "ai.lumen.microsoftgraph", category: "cache")
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let dateParserWithFractional = ISO8601DateFormatter()
    private let dateParser = ISO8601DateFormatter()

    private init() {
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
            guard FileManager.default.fileExists(atPath: url.path) else {
                return Snapshot(messages: [], deltaLink: nil, updatedAt: .distantPast)
            }
            let data = try Data(contentsOf: url)
            return try decoder.decode(Snapshot.self, from: data)
        } catch {
            logger.error("Failed to load Microsoft Graph mail cache: \(String(describing: error), privacy: .private)")
            return Snapshot(messages: [], deltaLink: nil, updatedAt: .distantPast)
        }
    }

    func save(_ snapshot: Snapshot, accountID: String) {
        do {
            let url = try cacheURL(accountID: accountID)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: nil)
            let data = try encoder.encode(snapshot)
            try data.write(to: url, options: [.atomic, .completeFileProtection])
        } catch {
            logger.error("Failed to save Microsoft Graph mail cache: \(String(describing: error), privacy: .private)")
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

    func clearAll() {
        do {
            let directory = try cacheDirectory()
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
        } catch {
            logger.error("Failed to clear Microsoft Graph mail cache: \(String(describing: error), privacy: .private)")
        }
    }

    private func cacheURL(accountID: String) throws -> URL {
        let safe = accountID.replacingOccurrences(of: "[^a-zA-Z0-9._-]", with: "_", options: .regularExpression)
        return try cacheDirectory().appendingPathComponent("\(safe).json")
    }

    private func cacheDirectory() throws -> URL {
        let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        return support.appendingPathComponent("MicrosoftGraphMail", isDirectory: true)
    }

    private func parsedDate(for message: GraphMailMessage) -> Date {
        guard let timestamp = message.receivedDateTime ?? message.sentDateTime else { return .distantPast }
        return dateParserWithFractional.date(from: timestamp) ?? dateParser.date(from: timestamp) ?? .distantPast
    }
}

@MainActor
@Observable
final class MicrosoftGraphInboxViewModel {
    private let auth: MicrosoftGraphAuthManager
    private let client: MicrosoftGraphMailClient
    private let cache: MicrosoftGraphMailCacheStore

    private(set) var messages: [GraphMailMessage] = []
    private(set) var isLoading = false
    private(set) var lastSyncDate: Date?
    private(set) var error: Error?
    var unreadOnly = false

    init(auth: MicrosoftGraphAuthManager, client: MicrosoftGraphMailClient = MicrosoftGraphMailClient(), cache: MicrosoftGraphMailCacheStore? = nil) {
        self.auth = auth
        self.client = client
        self.cache = cache ?? MicrosoftGraphMailCacheStore.shared
    }

    func loadCached() {
        guard let account = auth.account else { return }
        let snapshot = cache.load(accountID: account.id)
        messages = filteredMessages(from: snapshot)
        lastSyncDate = snapshot.updatedAt == .distantPast ? nil : snapshot.updatedAt
    }

    func refresh(resetDelta: Bool = false) async {
        guard let account = auth.account else {
            error = MicrosoftGraphAuthError.noAccount
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
            cache.save(snapshot, accountID: account.id)
            messages = filteredMessages(from: snapshot)
            lastSyncDate = snapshot.updatedAt
            error = nil
        } catch {
            self.error = error
        }
    }

    func send(subject: String, body: String, recipients: [String], sendAsHTML: Bool = true) async throws {
        let token = try await auth.acquireToken(scopes: MicrosoftGraphScope.mailSendScopes, preferredAccountID: auth.account?.id)
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
        try await client.sendMail(mail, accessToken: token)
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
