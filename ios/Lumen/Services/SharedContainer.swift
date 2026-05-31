import Foundation
import SwiftData

@MainActor
enum SharedContainer {
    static var shared: ModelContainer?
}

nonisolated enum FileStore {
    enum FileStoreError: Error, Equatable {
        case documentDirectoryUnavailable
        case persistentDirectoryUnavailable
    }

    static var importsDirectory: URL {
        let fm = FileManager.default
        guard let base = try? persistentBaseDirectoryURL(fileManager: fm) else {
            preconditionFailure("FileStore requires a persistent app data directory")
        }
        let dir = base.appendingPathComponent("Imports", isDirectory: true)
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
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
        (try? FileManager.default.contentsOfDirectory(at: importsDirectory, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])) ?? []
    }

    static func importFile(from source: URL) -> URL? {
        let fm = FileManager.default
        let needsAccess = source.startAccessingSecurityScopedResource()
        defer { if needsAccess { source.stopAccessingSecurityScopedResource() } }
        let dest = importsDirectory.appendingPathComponent(source.lastPathComponent)
        try? fm.removeItem(at: dest)
        do {
            try fm.copyItem(at: source, to: dest)
            return dest
        } catch {
            return nil
        }
    }

    static func delete(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}
