import Foundation

nonisolated enum ModelOutputSanitizer {
    private static let hiddenReasoningTagPattern = "(?:think[\\w_-]*|thinking|reasoning|analysis|chain_of_thought)"

    static func stripHiddenBlocks(_ text: String) -> String {
        var output = text
        output = output.replacingOccurrences(
            of: "(?is)<\\s*(\(hiddenReasoningTagPattern))\\b[^>]*>.*?</\\s*\\1\\s*>",
            with: "",
            options: .regularExpression
        )
        output = output.replacingOccurrences(
            of: "(?im)^\\s*<\\s*\(hiddenReasoningTagPattern)\\b[^\\n]*(?:\\n|$)",
            with: "",
            options: .regularExpression
        )
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func stripHiddenBlocksPreservingPayloadMarkers(_ text: String) -> String {
        let payloads = WebRichContentPayload.decodeAll(from: text)
        var clean = stripHiddenBlocks(WebRichContentPayload.removingMarkers(from: text))
        let existingKeys = Set(WebRichContentPayload.decodeAll(from: clean).map(payloadKey))
        var seen = existingKeys
        for payload in payloads {
            let key = payloadKey(payload)
            guard seen.insert(key).inserted else { continue }
            clean += payload.encodedMarker()
        }
        return clean.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func boundedPrefix(_ text: String, limit: Int = 1200) -> String {
        String(stripHiddenBlocks(text).prefix(max(0, limit)))
    }

    private static func payloadKey(_ payload: WebRichContentPayload) -> String {
        switch payload.kind {
        case .searchResults:
            return "search:\(payload.query ?? ""):\(payload.results.map { $0.url ?? $0.title }.joined(separator: "|"))"
        case .fetchedPage:
            return "page:\(payload.page?.url ?? "")"
        }
    }
}
