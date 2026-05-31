import Foundation

struct LocalModelFile: Identifiable, Hashable {
    let id: String
    let url: URL
    let source: String

    var fileName: String { url.lastPathComponent }
    var displayName: String { url.deletingPathExtension().lastPathComponent }

    init(url: URL, source: String) {
        self.url = url
        self.source = source
        self.id = "\(source)::\(url.path)"
    }
}

enum LocalModelDiscovery {
    static func discoverGGUF() -> [LocalModelFile] {
        var files: [LocalModelFile] = []
        let fm = FileManager.default

        if let bundleRoot = Bundle.main.resourceURL,
           let enumerator = fm.enumerator(at: bundleRoot, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
            for case let url as URL in enumerator where url.pathExtension.lowercased() == "gguf" {
                files.append(LocalModelFile(url: url, source: "Bundle"))
            }
        }

        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first,
           let enumerator = fm.enumerator(at: docs, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
            for case let url as URL in enumerator where url.pathExtension.lowercased() == "gguf" {
                files.append(LocalModelFile(url: url, source: "Documents"))
            }
        }

        var dedup: [String: LocalModelFile] = [:]
        for file in files {
            dedup[file.url.path] = file
        }

        return dedup.values.sorted { lhs, rhs in
            lhs.fileName.localizedCaseInsensitiveCompare(rhs.fileName) == .orderedAscending
        }
    }
}
