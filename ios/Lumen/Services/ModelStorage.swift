import Foundation

nonisolated enum ModelStorage {
    struct ModelFilesResult {
        let directory: URL?
        let files: [URL]
        let mode: String
        let diagnostic: String?
    }

    enum StorageError: LocalizedError, Equatable {
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
                return "Could not create the model directory at \(path)."
            case .contentsUnavailable:
                return "Could not list model files."
            }
        }
    }

    static func modelsDirectoryURL(fileManager: FileManager = .default) -> URL {
        guard let directory = try? modelsDirectoryURLOrThrow(fileManager: fileManager) else {
            return unavailableModelsDirectoryURL(fileManager: fileManager)
        }
        return directory
    }

    static func modelsDirectoryURLOrThrow(fileManager: FileManager = .default) throws -> URL {
        let base = try persistentBaseDirectoryURL(fileManager: fileManager)
        return try modelsDirectoryURL(base: base, fileManager: fileManager)
    }

    static func modelFilesWithDiagnostics(fileManager fm: FileManager = .default) -> ModelFilesResult {
        modelFilesWithDiagnostics(
            modelsDirectory: { try modelsDirectoryURLOrThrow(fileManager: fm) },
            contents: { directory in
                try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            }
        )
    }

    static func modelFilesWithDiagnosticsForTests(
        modelsDirectory: () throws -> URL,
        contents: (URL) throws -> [URL]
    ) -> ModelFilesResult {
        modelFilesWithDiagnostics(modelsDirectory: modelsDirectory, contents: contents)
    }

    static func modelsDirectoryURLOrThrow(
        documentDirectories: [URL],
        applicationSupportDirectories: [URL],
        fileManager: FileManager = .default
    ) throws -> URL {
        let base = try persistentBaseDirectoryURL(
            documentDirectories: documentDirectories,
            applicationSupportDirectories: applicationSupportDirectories
        )
        return try modelsDirectoryURL(base: base, fileManager: fileManager)
    }

    private static func modelsDirectoryURL(base: URL, fileManager: FileManager) throws -> URL {
        let directory = base.appendingPathComponent("Models", isDirectory: true)
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw StorageError.directoryCreationFailed(directory.path)
        }
        return directory
    }

    private static func modelFilesWithDiagnostics(
        modelsDirectory: () throws -> URL,
        contents: (URL) throws -> [URL]
    ) -> ModelFilesResult {
        let directory: URL
        do {
            directory = try modelsDirectory()
        } catch {
            return ModelFilesResult(
                directory: nil,
                files: [],
                mode: "failed",
                diagnostic: "models_directory_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
        do {
            let files = try contents(directory)
            return ModelFilesResult(
                directory: directory,
                files: files,
                mode: "loaded",
                diagnostic: files.isEmpty ? "empty_models_directory" : nil
            )
        } catch {
            return ModelFilesResult(
                directory: directory,
                files: [],
                mode: "failed",
                diagnostic: "models_list_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
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
        guard let directory = try? resumeDirectoryURLOrThrow(fileManager: fileManager) else {
            return unavailableModelsDirectoryURL(fileManager: fileManager)
                .appendingPathComponent(".resume", isDirectory: true)
        }
        return directory
    }

    static func resumeDirectoryURLOrThrow(fileManager: FileManager = .default) throws -> URL {
        let directory = try modelsDirectoryURLOrThrow(fileManager: fileManager)
            .appendingPathComponent(".resume", isDirectory: true)
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw StorageError.directoryCreationFailed(directory.path)
        }
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

    private static func unavailableModelsDirectoryURL(fileManager: FileManager) -> URL {
        fileManager.temporaryDirectory
            .appendingPathComponent("LumenPersistentDirectoryUnavailable", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}
