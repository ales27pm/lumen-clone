import Foundation
import PDFKit

nonisolated enum FilesTools {
    static func readImportedFile(name: String) async -> String {
        let dir = FileStore.importsDirectory
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        let fm = FileManager.default
        let files = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        if trimmed.isEmpty {
            if files.isEmpty { return "No imported files. Tap the paperclip to add one." }
            return "Imported files:\n" + files.map { "• \($0)" }.joined(separator: "\n")
        }
        guard let match = files.first(where: { $0.localizedCaseInsensitiveContains(trimmed) }) else {
            return "File not found. Available: \(files.joined(separator: ", "))"
        }
        let url = dir.appendingPathComponent(match)
        if url.pathExtension.lowercased() == "pdf" {
            guard let pdf = PDFDocument(url: url) else { return "Couldn't open PDF." }
            var text = ""
            for i in 0..<min(pdf.pageCount, 20) {
                text += pdf.page(at: i)?.string ?? ""
                text += "\n"
            }
            return String(text.prefix(3000))
        }
        guard let data = try? Data(contentsOf: url),
              let s = String(data: data, encoding: .utf8) else {
            return "Couldn't read \(match)."
        }
        return String(s.prefix(3000))
    }
}
