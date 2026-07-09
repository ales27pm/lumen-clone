import Foundation
import SwiftData

@MainActor
enum SharedContainer {
    static var shared: ModelContainer?
}

nonisolated enum FileStore {
    struct ImportedFilesResult {
        let directory: URL?
        let files: [URL]
        let mode: String
        let diagnostic: String?
    }

    struct ImportFileResult {
        let url: URL?
        let mode: String
        let diagnostic: String?
    }

    enum FileStoreError: LocalizedError, Equatable {
        case documentDirectoryUnavailable
        case persistentDirectoryUnavailable
        case directoryCreationFailed(String)
        case contentsUnavailable(String)

        var errorDescription: String? {
            switch self {
            case .documentDirectoryUnavailable:
                return "The documents directory is unavailable."
            case .persistentDirectoryUnavailable:
                return "No persistent app data directory is available."
            case .directoryCreationFailed(let path):
                return "Could not create the imports directory at \(path)."
            case .contentsUnavailable:
                return "Could not list imported files."
            }
        }
    }

    static var importsDirectory: URL {
        (try? importsDirectoryOrThrow()) ?? unavailableImportsDirectoryURL(fileManager: .default)
    }

    static func importsDirectoryOrThrow(fileManager fm: FileManager = .default) throws -> URL {
        let base = try persistentBaseDirectoryURL(fileManager: fm)
        let dir = base.appendingPathComponent("Imports", isDirectory: true)
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            throw FileStoreError.directoryCreationFailed(dir.path)
        }
        return dir
    }

    static func documentsDirectoryURL(fileManager: FileManager = .default) throws -> URL {
        try documentsDirectoryURL(candidateDirectories: fileManager.urls(for: .documentDirectory, in: .userDomainMask))
    }

    static func documentsDirectoryURL(candidateDirectories: [URL]) throws -> URL {
        guard let base = candidateDirectories.first else {
            throw FileStoreError.documentDirectoryUnavailable
        }
        return base
    }

    static func persistentBaseDirectoryURL(fileManager: FileManager = .default) throws -> URL {
        try persistentBaseDirectoryURL(
            documentDirectories: fileManager.urls(for: .documentDirectory, in: .userDomainMask),
            applicationSupportDirectories: fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        )
    }

    static func persistentBaseDirectoryURL(documentDirectories: [URL], applicationSupportDirectories: [URL]) throws -> URL {
        if let documents = documentDirectories.first { return documents }
        if let appSupport = applicationSupportDirectories.first { return appSupport }
        throw FileStoreError.persistentDirectoryUnavailable
    }

    static func importedFiles() -> [URL] {
        importedFilesWithDiagnostics().files
    }

    static func importedFilesWithDiagnostics(fileManager fm: FileManager = .default) -> ImportedFilesResult {
        importedFilesWithDiagnostics(
            importsDirectory: { try importsDirectoryOrThrow(fileManager: fm) },
            contents: { directory in
                try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])
            }
        )
    }

    static func importedFilesWithDiagnosticsForTests(
        importsDirectory: () throws -> URL,
        contents: (URL) throws -> [URL]
    ) -> ImportedFilesResult {
        importedFilesWithDiagnostics(importsDirectory: importsDirectory, contents: contents)
    }

    private static func importedFilesWithDiagnostics(
        importsDirectory: () throws -> URL,
        contents: (URL) throws -> [URL]
    ) -> ImportedFilesResult {
        let directory: URL
        do {
            directory = try importsDirectory()
        } catch {
            return ImportedFilesResult(
                directory: nil,
                files: [],
                mode: "failed",
                diagnostic: "imports_directory_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
        do {
            let files = try contents(directory)
            return ImportedFilesResult(directory: directory, files: files, mode: "loaded", diagnostic: files.isEmpty ? "empty_imports" : nil)
        } catch {
            return ImportedFilesResult(
                directory: directory,
                files: [],
                mode: "failed",
                diagnostic: "imports_list_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
    }

    static func importFile(from source: URL) -> URL? {
        importFileWithDiagnostics(from: source).url
    }

    static func importFileWithDiagnostics(from source: URL, fileManager fm: FileManager = .default) -> ImportFileResult {
        let needsAccess = source.startAccessingSecurityScopedResource()
        defer { if needsAccess { source.stopAccessingSecurityScopedResource() } }
        return importFileWithDiagnostics(
            source: source,
            importsDirectory: { try importsDirectoryOrThrow(fileManager: fm) },
            destinationExists: { fm.fileExists(atPath: $0.path) },
            removeExisting: { try fm.removeItem(at: $0) },
            copyItem: { try fm.copyItem(at: $0, to: $1) }
        )
    }

    static func importFileWithDiagnosticsForTests(
        source: URL,
        importsDirectory: () throws -> URL,
        destinationExists: (URL) -> Bool,
        removeExisting: (URL) throws -> Void,
        copyItem: (URL, URL) throws -> Void
    ) -> ImportFileResult {
        importFileWithDiagnostics(
            source: source,
            importsDirectory: importsDirectory,
            destinationExists: destinationExists,
            removeExisting: removeExisting,
            copyItem: copyItem
        )
    }

    private static func importFileWithDiagnostics(
        source: URL,
        importsDirectory: () throws -> URL,
        destinationExists: (URL) -> Bool,
        removeExisting: (URL) throws -> Void,
        copyItem: (URL, URL) throws -> Void
    ) -> ImportFileResult {
        let directory: URL
        do {
            directory = try importsDirectory()
        } catch {
            return ImportFileResult(
                url: nil,
                mode: "failed",
                diagnostic: "imports_directory_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
        let dest = directory.appendingPathComponent(source.lastPathComponent)
        if destinationExists(dest) {
            do {
                try removeExisting(dest)
            } catch {
                return ImportFileResult(
                    url: nil,
                    mode: "failed",
                    diagnostic: "import_remove_existing_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
                )
            }
        }
        do {
            try copyItem(source, dest)
            return ImportFileResult(url: dest, mode: "imported", diagnostic: nil)
        } catch {
            return ImportFileResult(
                url: nil,
                mode: "failed",
                diagnostic: "import_copy_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
    }

    static func delete(_ url: URL) {
        let fm = FileManager.default
        try? fm.removeItem(at: url)
    }

    private static func unavailableImportsDirectoryURL(fileManager: FileManager) -> URL {
        fileManager.temporaryDirectory
            .appendingPathComponent("LumenPersistentDirectoryUnavailable", isDirectory: true)
            .appendingPathComponent("Imports", isDirectory: true)
    }
}
