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
        case stagedImportCleanupFailed

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
            case .stagedImportCleanupFailed:
                return "Could not clean up an interrupted file import."
            }
        }
    }

    private static let stagingFilePrefix = ".lumen-import-"
    private static let stagingFileSuffix = ".staged"
    private static let stagingProcessID = UUID().uuidString.lowercased()

    static var importsDirectory: URL {
        (try? importsDirectoryOrThrow()) ?? unavailableImportsDirectoryURL(fileManager: .default)
    }

    static func importsDirectoryOrThrow(fileManager fm: FileManager = .default) throws -> URL {
        let base = try persistentBaseDirectoryURL(fileManager: fm)
        let dir = base.appendingPathComponent("Imports", isDirectory: true)
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try purgeStaleImportStages(in: dir, fileManager: fm)
        } catch {
            if let fileStoreError = error as? FileStoreError {
                throw fileStoreError
            }
            throw FileStoreError.directoryCreationFailed(dir.path)
        }
        return dir
    }

    static func purgeStaleImportStages(in directory: URL, fileManager fm: FileManager = .default) throws {
        let candidates: [URL]
        do {
            candidates = try fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: []
            )
        } catch {
            throw FileStoreError.stagedImportCleanupFailed
        }

        for candidate in candidates where isInternalStagingFile(candidate) && !isCurrentProcessStagingFile(candidate) {
            do {
                let values = try candidate.resourceValues(forKeys: [.isRegularFileKey])
                guard values.isRegularFile == true else {
                    throw FileStoreError.stagedImportCleanupFailed
                }
                try fm.removeItem(at: candidate)
            } catch {
                let cocoaError = error as NSError
                if cocoaError.domain == NSCocoaErrorDomain && cocoaError.code == NSFileNoSuchFileError {
                    continue
                }
                throw FileStoreError.stagedImportCleanupFailed
            }
        }
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
            let visibleFiles = files.filter { !isInternalStagingFile($0) }
            return ImportedFilesResult(
                directory: directory,
                files: visibleFiles,
                mode: "loaded",
                diagnostic: visibleFiles.isEmpty ? "empty_imports" : nil
            )
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
            stageItem: { try fm.copyItem(at: $0, to: $1) },
            commitStagedItem: { staged, destination, destinationExists in
                if destinationExists {
                    _ = try fm.replaceItemAt(destination, withItemAt: staged)
                } else {
                    try fm.moveItem(at: staged, to: destination)
                }
            },
            cleanupStagedItem: { try? fm.removeItem(at: $0) }
        )
    }

    static func importFileWithDiagnosticsForTests(
        source: URL,
        importsDirectory: URL,
        fileManager fm: FileManager = .default
    ) -> ImportFileResult {
        importFileWithDiagnostics(
            source: source,
            importsDirectory: { importsDirectory },
            destinationExists: { fm.fileExists(atPath: $0.path) },
            stageItem: { try fm.copyItem(at: $0, to: $1) },
            commitStagedItem: { staged, destination, destinationExists in
                if destinationExists {
                    _ = try fm.replaceItemAt(destination, withItemAt: staged)
                } else {
                    try fm.moveItem(at: staged, to: destination)
                }
            },
            cleanupStagedItem: { try? fm.removeItem(at: $0) }
        )
    }

    static func importFileWithDiagnosticsForTests(
        source: URL,
        importsDirectory: () throws -> URL,
        destinationExists: (URL) -> Bool,
        stageItem: (URL, URL) throws -> Void,
        commitStagedItem: (URL, URL, Bool) throws -> Void,
        cleanupStagedItem: (URL) -> Void
    ) -> ImportFileResult {
        importFileWithDiagnostics(
            source: source,
            importsDirectory: importsDirectory,
            destinationExists: destinationExists,
            stageItem: stageItem,
            commitStagedItem: commitStagedItem,
            cleanupStagedItem: cleanupStagedItem
        )
    }

    private static func importFileWithDiagnostics(
        source: URL,
        importsDirectory: () throws -> URL,
        destinationExists: (URL) -> Bool,
        stageItem: (URL, URL) throws -> Void,
        commitStagedItem: (URL, URL, Bool) throws -> Void,
        cleanupStagedItem: (URL) -> Void
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
        let staged = stagingURL(in: directory)
        do {
            try stageItem(source, staged)
        } catch {
            cleanupStagedItem(staged)
            return ImportFileResult(
                url: nil,
                mode: "failed",
                diagnostic: "import_copy_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
        do {
            try commitStagedItem(staged, dest, destinationExists(dest))
            return ImportFileResult(url: dest, mode: "imported", diagnostic: nil)
        } catch {
            cleanupStagedItem(staged)
            return ImportFileResult(
                url: nil,
                mode: "failed",
                diagnostic: "import_commit_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
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

    private static func stagingURL(in directory: URL) -> URL {
        directory.appendingPathComponent(
            "\(stagingFilePrefix)\(stagingProcessID)-\(UUID().uuidString.lowercased())\(stagingFileSuffix)"
        )
    }

    private static func isInternalStagingFile(_ url: URL) -> Bool {
        stagingIdentity(in: url.lastPathComponent) != nil
    }

    private static func isCurrentProcessStagingFile(_ url: URL) -> Bool {
        guard let identity = stagingIdentity(in: url.lastPathComponent),
              let processID = identity.processID else {
            return false
        }
        return processID == stagingProcessID
    }

    private static func stagingIdentity(in filename: String) -> (processID: String?, itemID: String)? {
        guard filename.hasPrefix(stagingFilePrefix), filename.hasSuffix(stagingFileSuffix) else {
            return nil
        }

        let bodyStart = filename.index(filename.startIndex, offsetBy: stagingFilePrefix.count)
        let bodyEnd = filename.index(filename.endIndex, offsetBy: -stagingFileSuffix.count)
        let body = String(filename[bodyStart..<bodyEnd]).lowercased()

        if UUID(uuidString: body) != nil {
            return (processID: nil, itemID: body)
        }

        guard body.count == 73 else { return nil }
        let separator = body.index(body.startIndex, offsetBy: 36)
        guard body[separator] == "-" else { return nil }
        let itemStart = body.index(after: separator)
        let processID = String(body[..<separator])
        let itemID = String(body[itemStart...])
        guard UUID(uuidString: processID) != nil, UUID(uuidString: itemID) != nil else {
            return nil
        }
        return (processID: processID, itemID: itemID)
    }
}
