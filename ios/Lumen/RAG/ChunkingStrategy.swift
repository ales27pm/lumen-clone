import Foundation

enum ChunkingContentType { case plain, markdown, code }
struct ChunkingConfig { let maxChars: Int; let overlap: Int }
struct ChunkPiece: Sendable { let text: String; let start: Int; let end: Int }

enum ChunkingStrategy {
    static func chunk(_ text: String, type: ChunkingContentType, config: ChunkingConfig = .init(maxChars: 700, overlap: 80)) -> [ChunkPiece] {
        guard config.maxChars > 0, config.overlap >= 0, config.overlap < config.maxChars else { return [ChunkPiece]() }
        let separators: [String]
        switch type {
        case .markdown: separators = ["\n#"]
        case .code: separators = ["\nfunc "]
        case .plain: separators = ["\n\n"]
        }
        var out: [ChunkPiece] = []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            let nextSeparator = separators.compactMap { sep in text[cursor...].range(of: sep)?.lowerBound }.min()
            let unitEnd = nextSeparator ?? text.endIndex
            let unit = String(text[cursor..<unitEnd])
            let leadingTrim = unit.prefix { $0.isWhitespace || $0.isNewline }.count
            let trailingTrim = unit.reversed().prefix { $0.isWhitespace || $0.isNewline }.count
            let trimmedStartOffset = text.distance(from: text.startIndex, to: cursor) + leadingTrim
            let trimmedLength = max(0, unit.count - leadingTrim - trailingTrim)
            if trimmedLength > 0 {
                var i = 0
                while i < trimmedLength {
                    let end = min(trimmedLength, i + config.maxChars)
                    let segStart = text.index(text.startIndex, offsetBy: trimmedStartOffset + i)
                    let segEnd = text.index(text.startIndex, offsetBy: trimmedStartOffset + end)
                    out.append(.init(text: String(text[segStart..<segEnd]), start: trimmedStartOffset + i, end: trimmedStartOffset + end))
                    if end == trimmedLength { break }
                    let next = end - config.overlap
                    guard next > i else { break }
                    i = next
                }
            }
            cursor = unitEnd
            if cursor < text.endIndex {
                cursor = text.index(after: cursor)
            }
        }
        return out
    }
}
