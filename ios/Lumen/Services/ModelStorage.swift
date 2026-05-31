import Foundation

nonisolated enum ModelStorage {
    enum StorageError: Error, Equatable {
        case documentDirectoryUnavailable
        case persistentDirectoryUnavailable
    }

    static func modelsDirectoryURL(fileManager: FileManager = .default) -> URL {
        guard let base = try? persistentBaseDirectoryURL(fileManager: fileManager) else {
            preconditionFailure("ModelStorage requires a persistent app data directory")
        }
        let directory = base.appendingPathComponent("Models", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
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
        throw StorageError.persistentDirectoryUnavailable
    }

    static func documentsDirectoryURL(fileManager: FileManager = .default) throws -> URL {
        try documentsDirectoryURL(candidateDirectories: fileManager.urls(for: .documentDirectory, in: .userDomainMask))
    }

    static func documentsDirectoryURL(candidateDirectories: [URL]) throws -> URL {
        guard let base = candidateDirectories.first else {
            throw StorageError.documentDirectoryUnavailable
        }
        return base
    }

    static func resumeDirectoryURL(fileManager: FileManager = .default) -> URL {
        let directory = modelsDirectoryURL(fileManager: fileManager).appendingPathComponent(".resume", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func resolvedModelURL(from storedPath: String, fileName: String, fileManager: FileManager = .default) -> URL {
        let storedURL = URL(fileURLWithPath: storedPath)
        if fileManager.fileExists(atPath: storedURL.path) {
            return storedURL
        }

        let preferred = modelsDirectoryURL(fileManager: fileManager).appendingPathComponent(fileName)
        if fileManager.fileExists(atPath: preferred.path) {
            return preferred
        }

        guard let base = try? documentsDirectoryURL(fileManager: fileManager) else {
            return storedURL
        }
        let previousNested = base
            .appendingPathComponent("Hybrid Coder", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(fileName)
        if fileManager.fileExists(atPath: previousNested.path) {
            return previousNested
        }

        return storedURL
    }
}
