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
    struct AttachmentExtractionResult: Sendable, Hashable {
        let text: String?
        let mode: String
        let diagnostic: String?
    }

    enum AttachmentExtractionError: LocalizedError, Equatable {
        case pdfOpenFailed
        case textDecodeFailed

        var errorDescription: String? {
            switch self {
            case .pdfOpenFailed:
                return "The PDF could not be opened."
            case .textDecodeFailed:
                return "The attachment text could not be decoded."
            }
        }
    }

    /// Hard ceiling on extraction regardless of prompt budget. Guards against
    /// pathological files (100MB dumps) from exploding memory during load.
    /// `PromptAssembler` applies the real, per-request budget on top of this.
    static let hardExtractionCeiling = PromptBudgetConstants.hardAttachmentCeiling

    static func make(from url: URL) -> ChatAttachment? {
        guard
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
            let size = (attrs[.size] as? NSNumber)?.intValue
        else {
            return nil
        }
        let ext = url.pathExtension.lowercased()
        let kind: ChatAttachment.Kind = (ext == "pdf") ? .pdf : .text
        return ChatAttachment(name: url.lastPathComponent, kind: kind, path: url.path, byteSize: size)
    }

    /// Extracts readable text bounded only by the hard ceiling. The prompt
    /// assembler applies the actual per-request share afterward.
    static func extractTextWithDiagnostics(_ attachment: ChatAttachment) -> AttachmentExtractionResult {
        extractTextWithDiagnostics(
            attachment,
            readData: { try Data(contentsOf: $0) },
            pdfText: extractPDFText(url:),
            attributedText: { data in
                try NSAttributedString(data: data, options: [:], documentAttributes: nil).string
            }
        )
    }

    static func extractTextWithDiagnosticsForTests(
        _ attachment: ChatAttachment,
        readData: (URL) throws -> Data,
        pdfText: (URL) throws -> String,
        attributedText: (Data) throws -> String
    ) -> AttachmentExtractionResult {
        extractTextWithDiagnostics(
            attachment,
            readData: readData,
            pdfText: pdfText,
            attributedText: attributedText
        )
    }

    private static func extractTextWithDiagnostics(
        _ attachment: ChatAttachment,
        readData: (URL) throws -> Data,
        pdfText: (URL) throws -> String,
        attributedText: (Data) throws -> String
    ) -> AttachmentExtractionResult {
        let url = URL(fileURLWithPath: attachment.path)
        let limit = hardExtractionCeiling
        switch attachment.kind {
        case .pdf:
            do {
                return loaded(String(try pdfText(url).prefix(limit)))
            } catch {
                return failed("attachment_pdf_open_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
            }
        case .text:
            let data: Data
            do {
                data = try readData(url)
            } catch {
                return failed("attachment_read_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
            }
            if data.isEmpty {
                return loaded("")
            }
            let ext = url.pathExtension.lowercased()
            if ext == "rtf" || ext == "rtfd" {
                do {
                    return loaded(String(try attributedText(data).prefix(limit)))
                } catch {
                    return failed("attachment_attributed_decode_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
                }
            }
            if let utf8 = String(data: data, encoding: .utf8) {
                return loaded(String(utf8.prefix(limit)))
            }
            if let latin = String(data: data, encoding: .isoLatin1) {
                return loaded(String(latin.prefix(limit)))
            }
            do {
                return loaded(String(try attributedText(data).prefix(limit)))
            } catch {
                return failed("attachment_decode_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
            }
        }
    }

    private static func extractPDFText(url: URL) throws -> String {
        guard let pdf = PDFDocument(url: url) else {
            throw AttachmentExtractionError.pdfOpenFailed
        }
        var out = ""
        out.reserveCapacity(min(hardExtractionCeiling, 32_000))
        for i in 0..<pdf.pageCount {
            out += pdf.page(at: i)?.string ?? ""
            out += "\n"
            if out.count >= hardExtractionCeiling { break }
        }
        return out
    }

    private static func loaded(_ text: String) -> AttachmentExtractionResult {
        AttachmentExtractionResult(
            text: text,
            mode: "loaded",
            diagnostic: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "empty_attachment_text" : nil
        )
    }

    private static func failed(_ diagnostic: String) -> AttachmentExtractionResult {
        AttachmentExtractionResult(text: nil, mode: "failed", diagnostic: diagnostic)
    }
}
