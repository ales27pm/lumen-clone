import Foundation
import PDFKit

nonisolated enum FilesTools {
    struct FileReadResult: Equatable {
        let text: String?
        let mode: String
        let diagnostic: String?
    }

    private enum FileReadError: Error, Equatable {
        case pdfOpenFailed
    }

    static func readImportedFile(name: String) async -> String {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        let fm = FileManager.default
        let imported = FileStore.importedFilesWithDiagnostics(fileManager: fm)
        if imported.mode == "failed" {
            return "Imported files unavailable. Diagnostic: \(diagnosticText(imported.diagnostic))."
        }
        let files = imported.files
        if trimmed.isEmpty {
            if files.isEmpty { return "No imported files. Tap the paperclip to add one." }
            return "Imported files:\n" + files.map { "• \($0.lastPathComponent)" }.joined(separator: "\n")
        }
        guard let url = files.first(where: { $0.lastPathComponent.localizedCaseInsensitiveContains(trimmed) }) else {
            let available = files.map(\.lastPathComponent).joined(separator: ", ")
            return "File not found. Available: \(available)"
        }
        let match = url.lastPathComponent
        let result = readMatchedFileWithDiagnostics(url: url)
        guard result.mode == "loaded", let text = result.text else {
            return "File read failed for \(match). Diagnostic: \(diagnosticText(result.diagnostic))."
        }
        return String(text.prefix(3000))
    }

    static func readMatchedFileWithDiagnosticsForTests(
        url: URL,
        readData: (URL) throws -> Data,
        pdfText: (URL) throws -> String
    ) -> FileReadResult {
        readMatchedFileWithDiagnostics(url: url, readData: readData, pdfText: pdfText)
    }

    private static func readMatchedFileWithDiagnostics(
        url: URL,
        readData: (URL) throws -> Data = { try Data(contentsOf: $0) },
        pdfText: (URL) throws -> String = extractPDFText
    ) -> FileReadResult {
        if url.pathExtension.lowercased() == "pdf" {
            do {
                let text = try pdfText(url)
                guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return FileReadResult(text: nil, mode: "failed", diagnostic: "empty_text")
                }
                return FileReadResult(text: text, mode: "loaded", diagnostic: nil)
            } catch {
                return FileReadResult(text: nil, mode: "failed", diagnostic: "pdf_open_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
            }
        }

        let data: Data
        do {
            data = try readData(url)
        } catch {
            return FileReadResult(text: nil, mode: "failed", diagnostic: "file_read_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
        }
        guard let text = String(data: data, encoding: .utf8) else {
            return FileReadResult(text: nil, mode: "failed", diagnostic: "text_decode_failed")
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return FileReadResult(text: nil, mode: "failed", diagnostic: "empty_text")
        }
        return FileReadResult(text: text, mode: "loaded", diagnostic: nil)
    }

    static func diagnosticText(_ diagnostic: String?) -> String {
        let trimmed = diagnostic?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? "unknown" : trimmed
    }

    private static func extractPDFText(url: URL) throws -> String {
        guard let pdf = PDFDocument(url: url) else {
            throw FileReadError.pdfOpenFailed
        }
        var text = ""
        for i in 0..<min(pdf.pageCount, 20) {
            text += pdf.page(at: i)?.string ?? ""
            text += "\n"
        }
        return text
    }
}
