import Foundation
import PDFKit

nonisolated struct ChatAttachment: Sendable, Hashable, Identifiable {
    enum Kind: String, Sendable {
        case text
        case pdf

        var icon: String {
            switch self {
            case .text: "doc.text"
            case .pdf: "doc.richtext"
            }
        }
    }

    let id: UUID
    let name: String
    let kind: Kind
    /// Absolute path inside the app's Imports directory.
    let path: String
    /// Approximate byte size of the imported file (for UI).
    let byteSize: Int

    init(id: UUID = UUID(), name: String, kind: Kind, path: String, byteSize: Int) {
        self.id = id
        self.name = name
        self.kind = kind
        self.path = path
        self.byteSize = byteSize
    }
}

nonisolated enum AttachmentResolver {
    /// Hard ceiling on extraction regardless of prompt budget. Guards against
    /// pathological files (100MB dumps) from exploding memory during load.
    /// `PromptAssembler` applies the real, per-request budget on top of this.
    static let hardExtractionCeiling = PromptBudgetConstants.hardAttachmentCeiling

    static func make(from url: URL) -> ChatAttachment? {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
        let ext = url.pathExtension.lowercased()
        let kind: ChatAttachment.Kind = (ext == "pdf") ? .pdf : .text
        return ChatAttachment(name: url.lastPathComponent, kind: kind, path: url.path, byteSize: size)
    }

    /// Extracts readable text bounded only by the hard ceiling. The prompt
    /// assembler applies the actual per-request share afterward.
    static func rawExtractText(_ a: ChatAttachment) -> String {
        let url = URL(fileURLWithPath: a.path)
        let limit = hardExtractionCeiling
        switch a.kind {
        case .pdf:
            guard let pdf = PDFDocument(url: url) else { return "" }
            var out = ""
            out.reserveCapacity(min(limit, 32_000))
            for i in 0..<pdf.pageCount {
                out += pdf.page(at: i)?.string ?? ""
                out += "\n"
                if out.count >= limit { break }
            }
            return String(out.prefix(limit))
        case .text:
            guard let data = try? Data(contentsOf: url) else { return "" }
            if let utf8 = String(data: data, encoding: .utf8) {
                return String(utf8.prefix(limit))
            }
            if let latin = String(data: data, encoding: .isoLatin1) {
                return String(latin.prefix(limit))
            }
            if let attr = try? NSAttributedString(data: data, options: [:], documentAttributes: nil) {
                return String(attr.string.prefix(limit))
            }
            return ""
        }
    }
}
