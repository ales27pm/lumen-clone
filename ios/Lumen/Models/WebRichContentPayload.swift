import Foundation

nonisolated struct WebRichContentPayload: Codable, Hashable {
    enum Kind: String, Codable {
        case searchResults
        case fetchedPage
    }

    static let startMarker = "<lumen_web_payload>"
    static let endMarker = "</lumen_web_payload>"

    let kind: Kind
    let query: String?
    let page: WebFetchedPagePayload?
    let results: [WebSearchResultPayload]
    let media: [WebMediaPayload]
    let generatedAt: Date

    init(
        kind: Kind,
        query: String? = nil,
        page: WebFetchedPagePayload? = nil,
        results: [WebSearchResultPayload] = [],
        media: [WebMediaPayload] = [],
        generatedAt: Date = Date()
    ) {
        self.kind = kind
        self.query = query
        self.page = page
        self.results = results
        self.media = media
        self.generatedAt = generatedAt
    }

    func encodedMarker() -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(self),
              let json = String(data: data, encoding: .utf8) else {
            return ""
        }
        return "\n\n\(Self.startMarker)\(json)\(Self.endMarker)"
    }

    static func decodeAll(from text: String) -> [WebRichContentPayload] {
        var payloads: [WebRichContentPayload] = []
        var searchRange = text.startIndex..<text.endIndex
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        while let start = text.range(of: startMarker, range: searchRange),
              let end = text.range(of: endMarker, range: start.upperBound..<text.endIndex) {
            let json = String(text[start.upperBound..<end.lowerBound])
            if let data = json.data(using: .utf8),
               let payload = try? decoder.decode(WebRichContentPayload.self, from: data) {
                payloads.append(payload)
            }
            searchRange = end.upperBound..<text.endIndex
        }
        return payloads
    }

    static func removingMarkers(from text: String) -> String {
        var output = text
        while let start = output.range(of: startMarker) {
            guard let end = output.range(of: endMarker, range: start.upperBound..<output.endIndex) else {
                output.removeSubrange(start.lowerBound..<output.endIndex)
                break
            }
            output.removeSubrange(start.lowerBound..<end.upperBound)
        }
        return output
            .replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

nonisolated struct WebSearchResultPayload: Codable, Hashable {
    let title: String
    let url: String?
    let snippet: String?
    let source: String?
    let mediaKind: WebMediaPayload.Kind?
}

nonisolated struct WebFetchedPagePayload: Codable, Hashable {
    let title: String?
    let url: String
    let excerpt: String
    let siteName: String?
    let description: String?
}

nonisolated struct WebMediaPayload: Codable, Hashable, Identifiable {
    enum Kind: String, Codable {
        case image
        case video
        case pdf
        case page
    }

    var id: String { url }
    let kind: Kind
    let url: String
    let title: String?
    let thumbnailURL: String?
    let sourcePageURL: String?
}
