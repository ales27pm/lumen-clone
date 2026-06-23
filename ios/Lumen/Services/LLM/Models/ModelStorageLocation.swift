import Foundation

struct ModelStorageLocation: Sendable, Codable, Equatable {
    let rootDirectory: URL
    let modelsDirectory: URL
    let metadataDirectory: URL
    let temporaryDirectory: URL
}

enum ModelStorageDirectoryResolver {
    static func resolve(fileManager: FileManager = .default) throws -> ModelStorageLocation {
        try resolve(
            documentDirectories: fileManager.urls(for: .documentDirectory, in: .userDomainMask),
            applicationSupportDirectories: fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask),
            fileManager: fileManager
        )
    }

    static func resolve(
        documentDirectories: [URL],
        applicationSupportDirectories: [URL],
        fileManager: FileManager = .default
    ) throws -> ModelStorageLocation {
        guard let documents = documentDirectories.first ?? applicationSupportDirectories.first else {
            throw ModelStorageError.applicationSupportUnavailable
        }

        let rootDirectory = documents.appendingPathComponent("Lumen", isDirectory: true)
        let modelsDirectory = rootDirectory.appendingPathComponent("Models", isDirectory: true)
        let metadataDirectory = modelsDirectory.appendingPathComponent("Metadata", isDirectory: true)
        let temporaryDirectory = modelsDirectory.appendingPathComponent("Tmp", isDirectory: true)

        for directory in [rootDirectory, modelsDirectory, metadataDirectory, temporaryDirectory] {
            do {
                try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            } catch {
                throw ModelStorageError.failedToCreateDirectory(directory, error.localizedDescription)
            }
        }

        do {
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var mutableModelsDirectory = modelsDirectory
            try mutableModelsDirectory.setResourceValues(values)
        } catch {
            throw ModelStorageError.failedToSetResourceValues(modelsDirectory, error.localizedDescription)
        }

        if let applicationSupport = applicationSupportDirectories.first, documentDirectories.first != nil {
            let previousRootDirectory = applicationSupport.appendingPathComponent("Lumen", isDirectory: true)
            if previousRootDirectory.standardizedFileURL != rootDirectory.standardizedFileURL {
                migratePreviousStorageRoot(
                    from: previousRootDirectory,
                    to: rootDirectory,
                    fileManager: fileManager
                )
            }
        }

        return ModelStorageLocation(
            rootDirectory: rootDirectory,
            modelsDirectory: modelsDirectory,
            metadataDirectory: metadataDirectory,
            temporaryDirectory: temporaryDirectory
        )
    }

    private static func migratePreviousStorageRoot(
        from previousRootDirectory: URL,
        to rootDirectory: URL,
        fileManager: FileManager
    ) {
        guard fileManager.fileExists(atPath: previousRootDirectory.path) else { return }

        let previousModels = previousRootDirectory.appendingPathComponent("Models", isDirectory: true)
        let currentModels = rootDirectory.appendingPathComponent("Models", isDirectory: true)
        guard let children = try? fileManager.contentsOfDirectory(
            at: previousModels,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        for source in children {
            let destination = currentModels.appendingPathComponent(source.lastPathComponent, isDirectory: source.hasDirectoryPath)
            guard fileManager.fileExists(atPath: destination.path) == false else { continue }
            do {
                try fileManager.moveItem(at: source, to: destination)
            } catch {
                try? fileManager.copyItem(at: source, to: destination)
            }
        }
    }
}
