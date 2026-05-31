import Foundation
import OSLog

actor MicrosoftGraphMailClient {
    private let baseURL = URL(string: "https://graph.microsoft.com/v1.0")!
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    private let logger = Logger(subsystem: "ai.lumen.microsoftgraph", category: "mail")

    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }

    func fetchInboxPage(accessToken: String, pageSize: Int = 25, nextOrDeltaLink: String? = nil) async throws -> GraphMailPage {
        let url: URL
        if
            let nextOrDeltaLink,
            let continuationURL = URL(string: nextOrDeltaLink),
            isTrustedContinuationURL(continuationURL)
        {
            url = continuationURL
        } else {
            var components = URLComponents(url: baseURL.appendingPathComponent("me/mailFolders/inbox/messages/delta"), resolvingAgainstBaseURL: false)!
            let queryItems = [
                URLQueryItem(name: "$select", value: "id,subject,from,toRecipients,ccRecipients,receivedDateTime,sentDateTime,bodyPreview,isRead,hasAttachments"),
                URLQueryItem(name: "$orderby", value: "receivedDateTime desc"),
                URLQueryItem(name: "$top", value: String(min(max(pageSize, 1), 100)))
            ]
            components.queryItems = queryItems
            guard let builtURL = components.url else { throw GraphHTTPError.missingURL }
            url = builtURL
        }

        let request = makeRequest(url: url, accessToken: accessToken)
        return try await retryWithBackoff { [session, decoder] in
            let (data, response) = try await session.data(for: request)
            try Self.validate(response: response, data: data)
            return try decoder.decode(GraphMailPage.self, from: data)
        }
    }

    func fetchMessageBody(messageID: String, accessToken: String) async throws -> GraphMailMessage {
        var components = URLComponents(url: baseURL.appendingPathComponent("me/messages/\(messageID)"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "$select", value: "id,subject,from,toRecipients,ccRecipients,receivedDateTime,sentDateTime,bodyPreview,isRead,hasAttachments,body")]
        guard let url = components.url else { throw GraphHTTPError.missingURL }
        let request = makeRequest(url: url, accessToken: accessToken)
        return try await retryWithBackoff { [session, decoder] in
            let (data, response) = try await session.data(for: request)
            try Self.validate(response: response, data: data)
            return try decoder.decode(GraphMailMessage.self, from: data)
        }
    }

    func sendMail(_ mail: GraphSendMailRequest, accessToken: String) async throws {
        var request = makeRequest(url: baseURL.appendingPathComponent("me/sendMail"), accessToken: accessToken)
        request.httpMethod = "POST"
        request.httpBody = try encoder.encode(mail)
        let (data, response) = try await session.data(for: request)
        try Self.validate(response: response, data: data, allowedStatuses: 200...202)
    }

    func createDraft(subject: String, htmlBody: String, to recipients: [String], accessToken: String) async throws -> GraphMailMessage {
        let payload: [String: Any] = [
            "subject": subject,
            "body": ["contentType": "HTML", "content": htmlBody],
            "toRecipients": recipients.map { ["emailAddress": ["address": $0]] }
        ]
        var request = makeRequest(url: baseURL.appendingPathComponent("me/messages"), accessToken: accessToken)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await session.data(for: request)
        try Self.validate(response: response, data: data, allowedStatuses: 200...201)
        return try decoder.decode(GraphMailMessage.self, from: data)
    }

    func uploadLargeAttachment(messageID: String, fileData: Data, fileName: String, contentType: String, accessToken: String) async throws {
        var sessionRequest = makeRequest(url: baseURL.appendingPathComponent("me/messages/\(messageID)/attachments/createUploadSession"), accessToken: accessToken)
        sessionRequest.httpMethod = "POST"
        sessionRequest.httpBody = try JSONSerialization.data(withJSONObject: [
            "AttachmentItem": [
                "attachmentType": "file",
                "name": fileName,
                "size": fileData.count,
                "contentType": contentType
            ]
        ])

        let (sessionData, sessionResponse) = try await session.data(for: sessionRequest)
        try Self.validate(response: sessionResponse, data: sessionData, allowedStatuses: 200...201)
        let uploadSession = try decoder.decode(GraphUploadSession.self, from: sessionData)

        guard let uploadURL = URL(string: uploadSession.uploadUrl) else { throw GraphHTTPError.missingURL }
        let chunkSize = 320 * 1024
        var offset = 0
        while offset < fileData.count {
            let end = min(offset + chunkSize, fileData.count) - 1
            let chunk = fileData.subdata(in: offset..<(end + 1))
            var chunkRequest = URLRequest(url: uploadURL)
            chunkRequest.httpMethod = "PUT"
            chunkRequest.setValue("bytes \(offset)-\(end)/\(fileData.count)", forHTTPHeaderField: "Content-Range")
            chunkRequest.setValue(String(chunk.count), forHTTPHeaderField: "Content-Length")
            chunkRequest.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            chunkRequest.httpBody = chunk
            let response = try await uploadChunkWithRetry(request: chunkRequest)
            if response.statusCode == 200 || response.statusCode == 201 { break }
            offset = end + 1
        }
    }

    private func makeRequest(url: URL, accessToken: String) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(UUID().uuidString, forHTTPHeaderField: "client-request-id")
        return request
    }

    private func isTrustedContinuationURL(_ url: URL) -> Bool {
        guard let baseComponents = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
              let urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return false }
        return baseComponents.scheme?.lowercased() == urlComponents.scheme?.lowercased()
            && baseComponents.host?.lowercased() == urlComponents.host?.lowercased()
    }

    private func uploadChunkWithRetry(request: URLRequest, maxAttempts: Int = 4) async throws -> HTTPURLResponse {
        var attempt = 0
        while true {
            do {
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
                if http.statusCode == 202 || http.statusCode == 200 || http.statusCode == 201 {
                    return http
                }
                if http.statusCode == 429, attempt < maxAttempts - 1 {
                    attempt += 1
                    let retryAfter = http.value(forHTTPHeaderField: "Retry-After").flatMap(TimeInterval.init)
                    let delay = retryAfter ?? (Double(1 << attempt) + Double.random(in: 0...0.75))
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }
                if (500...599).contains(http.statusCode), attempt < maxAttempts - 1 {
                    attempt += 1
                    let delay = Double(1 << attempt) + Double.random(in: 0...0.75)
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }
                if let graphError = try? JSONDecoder().decode(GraphAPIErrorEnvelope.self, from: data), graphError.isRetryable, attempt < maxAttempts - 1 {
                    attempt += 1
                    let delay = Double(1 << attempt) + Double.random(in: 0...0.75)
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }
                throw GraphHTTPError.unexpectedStatus(http.statusCode)
            } catch {
                if attempt < maxAttempts - 1 {
                    attempt += 1
                    let delay = Double(1 << attempt) + Double.random(in: 0...0.75)
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }
                throw error
            }
        }
    }

    private func retryWithBackoff<T>(maxAttempts: Int = 4, operation: @Sendable () async throws -> T) async throws -> T {
        var attempt = 0
        while true {
            do { return try await operation() }
            catch let error as GraphAPIErrorEnvelope where error.isRetryable && attempt < maxAttempts - 1 {
                attempt += 1
                let delay = Double(1 << attempt) + Double.random(in: 0...0.75)
                logger.warning("Graph retryable error \(error.error.code, privacy: .public); retrying attempt \(attempt, privacy: .public)")
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            } catch let error as GraphHTTPError {
                if case .throttled(let retryAfter) = error, attempt < maxAttempts - 1 {
                    attempt += 1
                    let delay = retryAfter ?? (Double(1 << attempt) + Double.random(in: 0...0.75))
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }
                throw error
            } catch {
                throw error
            }
        }
    }

    nonisolated static func validate(response: URLResponse, data: Data, allowedStatuses: ClosedRange<Int> = 200...204) throws {
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        guard allowedStatuses.contains(http.statusCode) else {
            if http.statusCode == 429 {
                let retryAfter = http.value(forHTTPHeaderField: "Retry-After").flatMap(TimeInterval.init)
                throw GraphHTTPError.throttled(retryAfter: retryAfter)
            }
            if let graphError = try? JSONDecoder().decode(GraphAPIErrorEnvelope.self, from: data) { throw graphError }
            throw GraphHTTPError.unexpectedStatus(http.statusCode)
        }
    }
}
