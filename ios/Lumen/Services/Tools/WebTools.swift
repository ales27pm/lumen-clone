import Foundation
import WebKit

nonisolated enum WebTools {
    private static let searchPolicy = ToolRetryPolicy(maxAttempts: 3, baseDelay: 0.35, maxDelay: 1.5, jitterRatio: 0.25)
    private static let fetchPolicy = ToolRetryPolicy(maxAttempts: 2, baseDelay: 0.4, maxDelay: 1.2, jitterRatio: 0.2)

    static func webSearch(query: String) async -> String {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Need a query." }

        if let instant = await duckDuckGoInstantAnswer(query: trimmed), !instant.text.isEmpty {
            return instant.text + instant.payload.encodedMarker()
        }

        if let html = await duckDuckGoHTMLSearch(query: trimmed), !html.text.isEmpty {
            return html.text + html.payload.encodedMarker()
        }

        return "No search results found. Try a more specific query or provide a URL for web.fetch."
    }

    @MainActor
    static func webFetch(url: String) async -> String {
        let normalized = normalizeURL(url)
        guard let u = URL(string: normalized) else { return "Invalid URL." }
        guard let scheme = u.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            return "Invalid or unsupported URL scheme."
        }

        var req = URLRequest(url: u)
        req.setValue("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1", forHTTPHeaderField: "User-Agent")
        req.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")
        req.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")

        let direct = await executeTextRequest(endpoint: "web.fetch.urlsession", request: req, timeout: 14, retryPolicy: fetchPolicy, context: "Web page fetch")
        switch direct {
        case .success(let (html, _)):
            if let output = pageOutput(htmlOrText: html, url: u) { return output }
        case .failure:
            break
        }

        let result = await executeWebRequest(endpoint: "web.fetch.webview", request: req, timeout: 15, retryPolicy: fetchPolicy, context: "Web page fetch")
        switch result {
        case .failure(let error): return error.localizedDescription
        case .success(let (html, _)):
            return pageOutput(htmlOrText: html, url: u) ?? "Page was empty or unreadable."
        }
    }

    private static func pageOutput(htmlOrText: String, url: URL) -> String? {
        let title = firstHTMLCapture(htmlOrText, pattern: #"(?is)<title[^>]*>(.*?)</title>"#).map { decodeHTMLEntities(stripHTML($0)) }
        let description = metaContent(htmlOrText, names: ["description", "og:description", "twitter:description"])
        let siteName = metaContent(htmlOrText, names: ["og:site_name", "application-name"])
        let excerpt = summarizedText(fromHTMLOrText: htmlOrText, maxCharacters: 4000)
        guard !excerpt.isEmpty else { return nil }

        let media = extractMedia(fromHTML: htmlOrText, baseURL: url)
        let page = WebFetchedPagePayload(
            title: title?.isEmpty == false ? title : nil,
            url: url.absoluteString,
            excerpt: excerpt,
            siteName: siteName,
            description: description
        )
        let payload = WebRichContentPayload(kind: .fetchedPage, page: page, media: media)

        var lines: [String] = []
        if let title, !title.isEmpty { lines.append(title) }
        if let description, !description.isEmpty { lines.append(description) }
        lines.append(url.absoluteString)
        lines.append("")
        lines.append(excerpt)
        return uniqueNonEmptyLinesKeepingBreaks(lines).joined(separator: "\n") + payload.encodedMarker()
    }

    private struct SearchOutput {
        let text: String
        let payload: WebRichContentPayload
    }

    private static func duckDuckGoInstantAnswer(query: String) async -> SearchOutput? {
        let q = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        guard let url = URL(string: "https://api.duckduckgo.com/?q=\(q)&format=json&no_redirect=1&no_html=1&skip_disambig=1") else {
            return nil
        }

        var request = URLRequest(url: url)
        request.setValue("Mozilla/5.0 (iPhone; Lumen/2.0)", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let result = await executeAPIRequest(endpoint: "duckduckgo.instant", request: request, timeout: 8, retryPolicy: searchPolicy, context: "Web search")
        switch result {
        case .failure:
            return nil
        case .success(let (data, _)):
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            var lines: [String] = []
            var results: [WebSearchResultPayload] = []

            let heading = (obj["Heading"] as? String)?.nilIfEmpty
            let abstract = (obj["AbstractText"] as? String)?.nilIfEmpty
            let abstractURL = (obj["AbstractURL"] as? String)?.nilIfEmpty
            let answer = (obj["Answer"] as? String)?.nilIfEmpty

            if let heading { lines.append(heading) }
            if let abstract { lines.append(abstract) }
            if let abstractURL { lines.append(abstractURL) }
            if let answer { lines.append(answer) }

            if heading != nil || abstract != nil || abstractURL != nil {
                results.append(WebSearchResultPayload(
                    title: heading ?? query,
                    url: abstractURL,
                    snippet: abstract ?? answer,
                    source: abstractURL.flatMap { URL(string: $0)?.host },
                    mediaKind: abstractURL.flatMap { mediaKind(for: $0) }
                ))
            }

            if let related = obj["RelatedTopics"] as? [[String: Any]] {
                for item in related.prefix(5) {
                    collectRelatedTopic(item, lines: &lines, results: &results)
                }
            }

            let unique = uniqueNonEmptyLines(lines)
            guard !unique.isEmpty || !results.isEmpty else { return nil }
            let payload = WebRichContentPayload(kind: .searchResults, query: query, results: results, media: uniqueMedia(results.compactMap(mediaPayload(from:))))
            return SearchOutput(text: unique.joined(separator: "\n"), payload: payload)
        }
    }

    private static func duckDuckGoHTMLSearch(query: String) async -> SearchOutput? {
        let q = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        guard let url = URL(string: "https://duckduckgo.com/html/?q=\(q)") else { return nil }

        var request = URLRequest(url: url)
        request.setValue("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1", forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")
        request.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")

        let result = await executeTextRequest(endpoint: "duckduckgo.html", request: request, timeout: 12, retryPolicy: searchPolicy, context: "Web search")
        switch result {
        case .failure:
            return nil
        case .success(let (html, _)):
            let results = parseDuckDuckGoHTMLResults(html).prefix(8).map { result in
                WebSearchResultPayload(
                    title: result.title,
                    url: result.url,
                    snippet: result.snippet,
                    source: result.url.flatMap { URL(string: $0)?.host },
                    mediaKind: result.url.flatMap { mediaKind(for: $0) }
                )
            }
            guard !results.isEmpty else { return nil }
            var lines = ["Search results for: \(query)"]
            for (index, result) in results.prefix(6).enumerated() {
                lines.append("\n\(index + 1). \(result.title)")
                if let snippet = result.snippet, !snippet.isEmpty { lines.append(snippet) }
                if let url = result.url, !url.isEmpty { lines.append(url) }
            }
            let payload = WebRichContentPayload(kind: .searchResults, query: query, results: Array(results), media: uniqueMedia(results.compactMap(mediaPayload(from:))))
            return SearchOutput(text: lines.joined(separator: "\n"), payload: payload)
        }
    }

    private static func collectRelatedTopic(_ item: [String: Any], lines: inout [String], results: inout [WebSearchResultPayload]) {
        if let text = item["Text"] as? String, !text.isEmpty {
            let url = (item["FirstURL"] as? String)?.nilIfEmpty
            lines.append("• \(text)")
            if let url { lines.append(url) }
            results.append(WebSearchResultPayload(
                title: text,
                url: url,
                snippet: text,
                source: url.flatMap { URL(string: $0)?.host },
                mediaKind: url.flatMap { mediaKind(for: $0) }
            ))
        }
        if let nested = item["Topics"] as? [[String: Any]] {
            for topic in nested.prefix(2) {
                collectRelatedTopic(topic, lines: &lines, results: &results)
            }
        }
    }

    private struct SearchResult: Hashable {
        let title: String
        let url: String?
        let snippet: String?
    }

    private static func parseDuckDuckGoHTMLResults(_ html: String) -> [SearchResult] {
        var results: [SearchResult] = []
        let resultBlockPattern = #"(?is)<div[^>]+class=\"[^\"]*result[^\"]*\"[^>]*>(.*?)</div>\s*</div>"#
        for blockMatch in html.matches(pattern: resultBlockPattern).prefix(12) {
            let block = blockMatch[1]
            guard let titleHTML = block.firstMatch(pattern: #"(?is)<a[^>]+class=\"[^\"]*result__a[^\"]*\"[^>]*>(.*?)</a>"#)?[1] else { continue }
            let title = decodeHTMLEntities(stripHTML(titleHTML))
            guard !title.isEmpty else { continue }
            let href = block.firstMatch(pattern: #"(?is)<a[^>]+class=\"[^\"]*result__a[^\"]*\"[^>]+href=\"([^\"]+)\""#)?[1]
            let snippetHTML = block.firstMatch(pattern: #"(?is)<a[^>]+class=\"[^\"]*result__snippet[^\"]*\"[^>]*>(.*?)</a>"#)?[1]
                ?? block.firstMatch(pattern: #"(?is)<div[^>]+class=\"[^\"]*result__snippet[^\"]*\"[^>]*>(.*?)</div>"#)?[1]
            let snippet = snippetHTML.map { decodeHTMLEntities(stripHTML($0)) }.flatMap { $0.isEmpty ? nil : $0 }
            let cleanedURL = href.flatMap(cleanDuckDuckGoRedirectURL)
            let result = SearchResult(title: title, url: cleanedURL, snippet: snippet)
            if !results.contains(result) { results.append(result) }
        }

        if results.isEmpty {
            let anchorPattern = #"(?is)<a[^>]+class=\"[^\"]*result__a[^\"]*\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>"#
            for match in html.matches(pattern: anchorPattern).prefix(8) {
                let url = cleanDuckDuckGoRedirectURL(match[1])
                let title = decodeHTMLEntities(stripHTML(match[2]))
                guard !title.isEmpty else { continue }
                let result = SearchResult(title: title, url: url, snippet: nil)
                if !results.contains(result) { results.append(result) }
            }
        }
        return results
    }

    private static func cleanDuckDuckGoRedirectURL(_ raw: String) -> String? {
        let decoded = decodeHTMLEntities(raw)
        if decoded.hasPrefix("//") { return "https:\(decoded)" }
        guard let components = URLComponents(string: decoded) else { return decoded.isEmpty ? nil : decoded }
        if let uddg = components.queryItems?.first(where: { $0.name == "uddg" })?.value, !uddg.isEmpty {
            return uddg.removingPercentEncoding ?? uddg
        }
        return decoded
    }

    private static func normalizeURL(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.lowercased().hasPrefix("http://") || trimmed.lowercased().hasPrefix("https://") { return trimmed }
        if URL(string: trimmed)?.scheme != nil { return trimmed }
        return "https://\(trimmed)"
    }

    private static func executeAPIRequest(endpoint: String, request: URLRequest, timeout: TimeInterval, retryPolicy: ToolRetryPolicy, context: String) async -> Result<(Data, HTTPURLResponse?), any Error> {
        if !(await ToolNetworkResilience.circuitBreaker.allowRequest(endpoint: endpoint)) {
            return .failure(NSError(domain: "WebTools", code: 1, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: .circuitOpen, context: context)]))
        }
        var req = request
        req.timeoutInterval = timeout
        var retries = 0
        let started = Date()
        for attempt in 1...retryPolicy.maxAttempts {
            do {
                let (data, response) = try await URLSession.shared.data(for: req)
                let http = response as? HTTPURLResponse
                let errorClass = ToolNetworkResilience.classify(error: nil, response: http)
                if let status = http?.statusCode, !(200..<300).contains(status) {
                    if ToolNetworkResilience.shouldRetry(errorClass: errorClass), attempt < retryPolicy.maxAttempts {
                        retries += 1
                        try? await Task.sleep(nanoseconds: ToolNetworkResilience.backoffDelay(attempt: attempt, policy: retryPolicy))
                        continue
                    }
                    await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: false)
                    ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: false, errorClass: errorClass, retryCount: retries, statusCode: status))
                    return .failure(NSError(domain: "WebTools", code: 2, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: errorClass, context: context)]))
                }
                await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: true)
                ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: true, errorClass: nil, retryCount: retries, statusCode: http?.statusCode))
                return .success((data, http))
            } catch {
                let errorClass = ToolNetworkResilience.classify(error: error, response: nil)
                if ToolNetworkResilience.shouldRetry(errorClass: errorClass), attempt < retryPolicy.maxAttempts {
                    retries += 1
                    try? await Task.sleep(nanoseconds: ToolNetworkResilience.backoffDelay(attempt: attempt, policy: retryPolicy))
                    continue
                }
                await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: false)
                ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: false, errorClass: errorClass, retryCount: retries, statusCode: nil))
                return .failure(NSError(domain: "WebTools", code: 2, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: errorClass, context: context)]))
            }
        }
        return .failure(NSError(domain: "WebTools", code: 3, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: .unknown, context: context)]))
    }

    private static func executeTextRequest(endpoint: String, request: URLRequest, timeout: TimeInterval, retryPolicy: ToolRetryPolicy, context: String) async -> Result<(String, HTTPURLResponse?), any Error> {
        let dataResult = await executeAPIRequest(endpoint: endpoint, request: request, timeout: timeout, retryPolicy: retryPolicy, context: context)
        switch dataResult {
        case .failure(let error): return .failure(error)
        case .success(let (data, response)):
            if let text = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) ?? String(data: data, encoding: .ascii) {
                return .success((text, response))
            }
            return .failure(NSError(domain: "WebTools", code: 4, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: .parsing, context: context)]))
        }
    }

    private static func executeWebRequest(endpoint: String, request: URLRequest, timeout: TimeInterval, retryPolicy: ToolRetryPolicy, context: String) async -> Result<(String, HTTPURLResponse?), any Error> {
        if !(await ToolNetworkResilience.circuitBreaker.allowRequest(endpoint: endpoint)) {
            return .failure(NSError(domain: "WebTools", code: 1, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: .circuitOpen, context: context)]))
        }
        var retries = 0
        let started = Date()
        for attempt in 1...retryPolicy.maxAttempts {
            do {
                let (content, response) = try await WebViewRequestLoader.load(request: request, timeout: timeout)
                let errorClass = ToolNetworkResilience.classify(error: nil, response: response)
                if let status = response?.statusCode, !(200..<300).contains(status) {
                    if ToolNetworkResilience.shouldRetry(errorClass: errorClass), attempt < retryPolicy.maxAttempts {
                        retries += 1
                        try? await Task.sleep(nanoseconds: ToolNetworkResilience.backoffDelay(attempt: attempt, policy: retryPolicy))
                        continue
                    }
                    await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: false)
                    ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: false, errorClass: errorClass, retryCount: retries, statusCode: status))
                    return .failure(NSError(domain: "WebTools", code: 2, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: errorClass, context: context)]))
                }
                await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: true)
                ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: true, errorClass: nil, retryCount: retries, statusCode: response?.statusCode))
                return .success((content, response))
            } catch {
                let errorClass = ToolNetworkResilience.classify(error: error, response: nil)
                if ToolNetworkResilience.shouldRetry(errorClass: errorClass), attempt < retryPolicy.maxAttempts {
                    retries += 1
                    try? await Task.sleep(nanoseconds: ToolNetworkResilience.backoffDelay(attempt: attempt, policy: retryPolicy))
                    continue
                }
                await ToolNetworkResilience.circuitBreaker.record(endpoint: endpoint, success: false)
                ToolNetworkTelemetry.emit(.init(endpoint: endpoint, latencyMs: Date().timeIntervalSince(started) * 1000, success: false, errorClass: errorClass, retryCount: retries, statusCode: nil))
                return .failure(NSError(domain: "WebTools", code: 2, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: errorClass, context: context)]))
            }
        }
        return .failure(NSError(domain: "WebTools", code: 3, userInfo: [NSLocalizedDescriptionKey: ToolNetworkResilience.fallbackMessage(for: .unknown, context: context)]))
    }

    private static func summarizedText(fromHTMLOrText value: String, maxCharacters: Int) -> String {
        let stripped = decodeHTMLEntities(stripHTML(value))
        let normalized = stripped.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines)
        return String(normalized.prefix(maxCharacters))
    }

    private static func stripHTML(_ html: String) -> String {
        var s = html
        if let range = s.range(of: "<body", options: .caseInsensitive) { s = String(s[range.lowerBound...]) }
        let patterns = ["<script[\\s\\S]*?</script>", "<style[\\s\\S]*?</style>", "<noscript[\\s\\S]*?</noscript>", "<[^>]+>"]
        for p in patterns { s = s.replacingOccurrences(of: p, with: " ", options: .regularExpression) }
        return s
    }

    private static func decodeHTMLEntities(_ value: String) -> String {
        var s = value
        let entities: [(String, String)] = [("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#39;", "'"), ("&apos;", "'"), ("&amp;", "&")]
        for (entity, replacement) in entities { s = s.replacingOccurrences(of: entity, with: replacement) }
        s = s.regexReplace(#"&#(\d+);"#) { match in
            guard let scalarValue = UInt32(match.dropFirst(2).dropLast()), let scalar = UnicodeScalar(scalarValue) else { return String(match) }
            return String(Character(scalar))
        }
        return s
    }

    private static func metaContent(_ html: String, names: [String]) -> String? {
        for name in names {
            let escaped = NSRegularExpression.escapedPattern(for: name)
            let patterns = [
                #"(?is)<meta[^>]+(?:name|property)=['\"]"# + escaped + #"['\"][^>]+content=['\"]([^'\"]+)['\"][^>]*>"#,
                #"(?is)<meta[^>]+content=['\"]([^'\"]+)['\"][^>]+(?:name|property)=['\"]"# + escaped + #"['\"][^>]*>"#
            ]
            for pattern in patterns {
                if let value = firstHTMLCapture(html, pattern: pattern), !value.isEmpty { return decodeHTMLEntities(value) }
            }
        }
        return nil
    }

    private static func firstHTMLCapture(_ html: String, pattern: String) -> String? {
        html.firstMatch(pattern: pattern)?.dropFirst().first
    }

    private static func extractMedia(fromHTML html: String, baseURL: URL) -> [WebMediaPayload] {
        var media: [WebMediaPayload] = []
        let imageCandidates = [
            metaContent(html, names: ["og:image", "twitter:image"]),
            firstHTMLCapture(html, pattern: #"(?is)<img[^>]+src=['\"]([^'\"]+)['\"]"#)
        ].compactMap { $0 }
        for raw in imageCandidates.prefix(4) {
            if let url = absoluteURL(raw, baseURL: baseURL) {
                media.append(WebMediaPayload(kind: .image, url: url.absoluteString, title: nil, thumbnailURL: url.absoluteString, sourcePageURL: baseURL.absoluteString))
            }
        }
        let videoCandidates = [
            metaContent(html, names: ["og:video", "og:video:url", "twitter:player"]),
            firstHTMLCapture(html, pattern: #"(?is)<video[^>]+src=['\"]([^'\"]+)['\"]"#),
            firstHTMLCapture(html, pattern: #"(?is)<source[^>]+src=['\"]([^'\"]+)['\"][^>]+type=['\"]video/[^'\"]+['\"]"#)
        ].compactMap { $0 }
        for raw in videoCandidates.prefix(3) {
            if let url = absoluteURL(raw, baseURL: baseURL) {
                media.append(WebMediaPayload(kind: .video, url: url.absoluteString, title: nil, thumbnailURL: nil, sourcePageURL: baseURL.absoluteString))
            }
        }
        let pdfPattern = #"(?is)<a[^>]+href=['\"]([^'\"]+\.pdf(?:\?[^'\"]*)?)['\"]"#
        for match in html.matches(pattern: pdfPattern).prefix(3) {
            if let url = absoluteURL(match[1], baseURL: baseURL) {
                media.append(WebMediaPayload(kind: .pdf, url: url.absoluteString, title: nil, thumbnailURL: nil, sourcePageURL: baseURL.absoluteString))
            }
        }
        return uniqueMedia(media)
    }

    private static func mediaPayload(from result: WebSearchResultPayload) -> WebMediaPayload? {
        guard let kind = result.mediaKind, let url = result.url else { return nil }
        return WebMediaPayload(kind: kind, url: url, title: result.title, thumbnailURL: kind == .image ? url : nil, sourcePageURL: result.url)
    }

    private static func mediaKind(for urlString: String) -> WebMediaPayload.Kind? {
        guard let url = URL(string: urlString) else { return nil }
        let ext = url.pathExtension.lowercased()
        if ["jpg", "jpeg", "png", "gif", "webp", "heic", "bmp"].contains(ext) { return .image }
        if ["mp4", "mov", "m4v", "webm"].contains(ext) { return .video }
        if ext == "pdf" { return .pdf }
        let host = url.host?.lowercased() ?? ""
        if host.contains("youtube.com") || host.contains("youtu.be") || host.contains("vimeo.com") { return .video }
        return .page
    }

    private static func absoluteURL(_ raw: String, baseURL: URL) -> URL? {
        let decoded = decodeHTMLEntities(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        if decoded.hasPrefix("//") { return URL(string: "https:\(decoded)") }
        if let url = URL(string: decoded), url.scheme != nil { return url }
        return URL(string: decoded, relativeTo: baseURL)?.absoluteURL
    }

    private static func uniqueMedia(_ media: [WebMediaPayload]) -> [WebMediaPayload] {
        var seen: Set<String> = []
        var output: [WebMediaPayload] = []
        for item in media {
            let key = item.url.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard !key.isEmpty, !seen.contains(key) else { continue }
            seen.insert(key)
            output.append(item)
        }
        return output
    }

    private static func uniqueNonEmptyLines(_ lines: [String]) -> [String] {
        var seen: Set<String> = []
        var output: [String] = []
        for line in lines {
            let cleaned = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty, !seen.contains(cleaned) else { continue }
            seen.insert(cleaned)
            output.append(cleaned)
        }
        return output
    }

    private static func uniqueNonEmptyLinesKeepingBreaks(_ lines: [String]) -> [String] {
        var output: [String] = []
        for line in lines {
            if line.isEmpty { output.append(line); continue }
            let cleaned = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleaned.isEmpty { output.append(cleaned) }
        }
        return output
    }
}

nonisolated private extension String {
    var nilIfEmpty: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    func matches(pattern: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return [] }
        let nsRange = NSRange(startIndex..<endIndex, in: self)
        return regex.matches(in: self, options: [], range: nsRange).map { result in
            (0..<result.numberOfRanges).map { index in
                let range = result.range(at: index)
                guard range.location != NSNotFound, let swiftRange = Range(range, in: self) else { return "" }
                return String(self[swiftRange])
            }
        }
    }

    func firstMatch(pattern: String) -> [String]? {
        matches(pattern: pattern).first
    }

    func regexReplace(_ pattern: String, replacement: (Substring) -> String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return self }
        let nsRange = NSRange(startIndex..<endIndex, in: self)
        var result = ""
        var lastIndex = startIndex
        for match in regex.matches(in: self, range: nsRange) {
            guard let range = Range(match.range, in: self) else { continue }
            result += self[lastIndex..<range.lowerBound]
            result += replacement(self[range])
            lastIndex = range.upperBound
        }
        result += self[lastIndex..<endIndex]
        return result
    }
}

@MainActor
private final class WebViewRequestLoader: NSObject, WKNavigationDelegate {
    private var continuation: CheckedContinuation<(String, HTTPURLResponse?), Error>?
    private var response: HTTPURLResponse?
    private lazy var webView: WKWebView = {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        let view = WKWebView(frame: .zero, configuration: config)
        view.navigationDelegate = self
        return view
    }()

    static func load(request: URLRequest, timeout: TimeInterval) async throws -> (String, HTTPURLResponse?) {
        let loader = WebViewRequestLoader()
        return try await loader.load(request: request, timeout: timeout)
    }

    private func load(request: URLRequest, timeout: TimeInterval) async throws -> (String, HTTPURLResponse?) {
        try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            _ = webView.load(request)
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                guard let self, let continuation = self.continuation else { return }
                self.continuation = nil
                self.webView.stopLoading()
                continuation.resume(throwing: URLError(.timedOut))
            }
        }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard let continuation = continuation else { return }
        self.continuation = nil
        webView.evaluateJavaScript("document.documentElement.outerHTML.toString()") { result, error in
            if let error { continuation.resume(throwing: error); return }
            let html = result as? String ?? ""
            continuation.resume(returning: (html, self.response))
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        guard let continuation = continuation else { return }
        self.continuation = nil
        continuation.resume(throwing: error)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        guard let continuation = continuation else { return }
        self.continuation = nil
        continuation.resume(throwing: error)
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse) async -> WKNavigationResponsePolicy {
        response = navigationResponse.response as? HTTPURLResponse
        return .allow
    }
}
