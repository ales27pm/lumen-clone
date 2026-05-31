import Foundation

struct ModelStorageLocation: Sendable, Codable, Equatable {
    let rootDirectory: URL
    let modelsDirectory: URL
    let metadataDirectory: URL
    let temporaryDirectory: URL
}

enum ModelStorageDirectoryResolver {
    static func resolve(fileManager: FileManager = .default) throws -> ModelStorageLocation {
        guard let applicationSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            throw ModelStorageError.applicationSupportUnavailable
        }

        let rootDirectory = applicationSupport.appendingPathComponent("Lumen", isDirectory: true)
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

        return ModelStorageLocation(
            rootDirectory: rootDirectory,
            modelsDirectory: modelsDirectory,
            metadataDirectory: metadataDirectory,
            temporaryDirectory: temporaryDirectory
        )
    }
}
