import Foundation

actor LLMModelStorage {
    private var cachedLocation: ModelStorageLocation?
    private let fileManager: FileManager

    init(
        location: ModelStorageLocation? = nil,
        fileManager: FileManager = .default
    ) {
        self.cachedLocation = location
        self.fileManager = fileManager
    }

    func listInstalledModels() async throws -> [InstalledModelRecord] {
        let location = try storageLocation()
        let files: [URL]
        do {
            files = try fileManager.contentsOfDirectory(
                at: location.metadataDirectory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        } catch {
            throw ModelStorageError.metadataReadFailed(error.localizedDescription)
        }

        let records = try files
            .filter { $0.pathExtension.lowercased() == "json" }
            .map { try readRecord(from: $0) }

        return records.sorted { lhs, rhs in
            if lhs.installedAt == rhs.installedAt {
                return lhs.id < rhs.id
            }
            return lhs.installedAt < rhs.installedAt
        }
    }

    func record(for id: String) async throws -> InstalledModelRecord? {
        let location = try storageLocation()
        let fileURL = metadataFileURL(for: id, location: location)
        guard fileManager.fileExists(atPath: fileURL.path) else {
            return nil
        }
        return try readRecord(from: fileURL)
    }

    func saveRecord(_ record: InstalledModelRecord) async throws {
        let location = try storageLocation()
        let fileURL = metadataFileURL(for: record.id, location: location)

        let data: Data
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            data = try encoder.encode(record)
        } catch {
            throw ModelStorageError.metadataEncodingFailed(error.localizedDescription)
        }

        do {
            try data.write(to: fileURL, options: [.atomic])
        } catch {
            throw ModelStorageError.metadataWriteFailed(error.localizedDescription)
        }
    }

    func deleteModel(id: String, deleteFile: Bool) async throws {
        let location = try storageLocation()
        guard let record = try await record(for: id) else {
            return
        }

        let metadataURL = metadataFileURL(for: id, location: location)
        if fileManager.fileExists(atPath: metadataURL.path) {
            do {
                try fileManager.removeItem(at: metadataURL)
            } catch {
                throw ModelStorageError.deleteFailed(metadataURL, error.localizedDescription)
            }
        }

        guard deleteFile, let fileURL = record.fileURL else {
            return
        }

        guard isDescendant(fileURL, of: location.rootDirectory), fileManager.fileExists(atPath: fileURL.path) else {
            return
        }

        do {
            try fileManager.removeItem(at: fileURL)
        } catch {
            throw ModelStorageError.deleteFailed(fileURL, error.localizedDescription)
        }
    }

    func importExistingModelFile(
        fileURL: URL,
        catalogEntry: ModelCatalogEntry?,
        backend: LLMBackendKind,
        displayName: String,
        expectedSHA256: String?
    ) async throws -> InstalledModelRecord {
        let location = try storageLocation()
        try ModelFileValidator.validateExtension(for: fileURL, backend: backend)
        try ModelFileValidator.validateReadableFile(fileURL)

        let destinationURL = try uniqueDestinationURL(
            sourceURL: fileURL,
            preferredFileName: catalogEntry?.expectedFileName,
            location: location
        )

        do {
            try fileManager.copyItem(at: fileURL, to: destinationURL)
        } catch {
            throw ModelStorageError.importFailed(error.localizedDescription)
        }

        let resolvedExpectedSHA256 = expectedSHA256 ?? catalogEntry?.expectedSHA256
        let verifiedHash: String?
        do {
            verifiedHash = try ModelFileValidator.verifyHashIfNeeded(
                fileURL: destinationURL,
                expectedSHA256: resolvedExpectedSHA256
            )
        } catch {
            try? fileManager.removeItem(at: destinationURL)
            throw error
        }

        let sizeBytes = try fileSizeBytes(for: destinationURL)
        let recordID = installedRecordID(catalogEntry: catalogEntry, fileURL: destinationURL)
        let model = LocalLLMModel(
            id: recordID,
            displayName: displayName,
            backend: backend,
            localURL: destinationURL,
            expectedSHA256: resolvedExpectedSHA256,
            parameterCountBillion: catalogEntry?.parameterCountBillion,
            quantization: catalogEntry?.quantization,
            contextLength: catalogEntry?.contextLength ?? 4_096,
            fileSizeBytes: sizeBytes,
            createdAt: Date()
        )
        let record = InstalledModelRecord(
            id: recordID,
            catalogID: catalogEntry?.id,
            model: model,
            fileURL: destinationURL,
            relativePath: relativePath(for: destinationURL, root: location.rootDirectory),
            sha256: verifiedHash,
            sizeBytes: sizeBytes,
            installedAt: Date(),
            lastVerifiedAt: verifiedHash == nil ? nil : Date(),
            verificationStatus: verifiedHash == nil ? .unverified : .verified
        )

        do {
            try await saveRecord(record)
        } catch {
            try? fileManager.removeItem(at: destinationURL)
            throw error
        }
        return record
    }

    func registerTinyIntentModel() async throws -> InstalledModelRecord {
        let catalogEntry = BuiltInModelCatalog.entry(id: "builtin.tiny-intent") ?? ModelCatalogEntry(
            id: "builtin.tiny-intent",
            displayName: "Tiny Intent",
            backend: .tinyIntent,
            recommendedUse: .tinyIntent,
            source: .bundled,
            contextLength: 512,
            minimumRecommendedTier: .constrained
        )
        let model = catalogEntry.asLocalModel(localURL: nil)
        let record = InstalledModelRecord(
            id: model.id,
            catalogID: catalogEntry.id,
            model: model,
            fileURL: nil,
            relativePath: nil,
            sha256: nil,
            sizeBytes: nil,
            installedAt: Date(),
            lastVerifiedAt: Date(),
            verificationStatus: .verified
        )

        try await saveRecord(record)
        return record
    }

    private func storageLocation() throws -> ModelStorageLocation {
        if let cachedLocation {
            return cachedLocation
        }
        let resolved = try ModelStorageDirectoryResolver.resolve(fileManager: fileManager)
        cachedLocation = resolved
        return resolved
    }

    private func readRecord(from fileURL: URL) throws -> InstalledModelRecord {
        let data: Data
        do {
            data = try Data(contentsOf: fileURL)
        } catch {
            throw ModelStorageError.metadataReadFailed(error.localizedDescription)
        }

        do {
            return try JSONDecoder().decode(InstalledModelRecord.self, from: data)
        } catch {
            throw ModelStorageError.metadataDecodingFailed(error.localizedDescription)
        }
    }

    private func metadataFileURL(for id: String, location: ModelStorageLocation) -> URL {
        location.metadataDirectory.appendingPathComponent(safeFileName(id), isDirectory: false)
            .appendingPathExtension("json")
    }

    private func uniqueDestinationURL(
        sourceURL: URL,
        preferredFileName: String?,
        location: ModelStorageLocation
    ) throws -> URL {
        let candidateName = safeImportFileName(preferredFileName ?? sourceURL.lastPathComponent)
        let baseName = URL(fileURLWithPath: candidateName).deletingPathExtension().lastPathComponent
        let pathExtension = URL(fileURLWithPath: candidateName).pathExtension
        let firstCandidate = location.modelsDirectory.appendingPathComponent(candidateName, isDirectory: false)
        if fileManager.fileExists(atPath: firstCandidate.path) == false {
            return firstCandidate
        }

        for _ in 0..<100 {
            let suffix = UUID().uuidString.prefix(8).lowercased()
            let name = pathExtension.isEmpty ? "\(baseName)-\(suffix)" : "\(baseName)-\(suffix).\(pathExtension)"
            let candidate = location.modelsDirectory.appendingPathComponent(name, isDirectory: false)
            if fileManager.fileExists(atPath: candidate.path) == false {
                return candidate
            }
        }

        throw ModelStorageError.importFailed("Could not allocate a unique destination filename.")
    }

    private func fileSizeBytes(for fileURL: URL) throws -> Int64? {
        do {
            let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
            if let size = attributes[.size] as? NSNumber {
                return size.int64Value
            }
            return nil
        } catch {
            throw ModelStorageError.importFailed(error.localizedDescription)
        }
    }

    private func installedRecordID(catalogEntry: ModelCatalogEntry?, fileURL: URL) -> String {
        let base = catalogEntry?.id ?? fileURL.deletingPathExtension().lastPathComponent
        return "installed.\(safeIdentifier(base)).\(UUID().uuidString.lowercased())"
    }

    private func relativePath(for fileURL: URL, root: URL) -> String? {
        let rootPath = normalizedDirectoryPath(root)
        let filePath = fileURL.standardizedFileURL.resolvingSymlinksInPath().path
        guard filePath.hasPrefix(rootPath) else {
            return nil
        }
        return String(filePath.dropFirst(rootPath.count))
    }

    private func isDescendant(_ fileURL: URL, of rootURL: URL) -> Bool {
        fileURL.standardizedFileURL.resolvingSymlinksInPath().path.hasPrefix(normalizedDirectoryPath(rootURL))
    }

    private func normalizedDirectoryPath(_ url: URL) -> String {
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        return path.hasSuffix("/") ? path : "\(path)/"
    }

    private func safeFileName(_ value: String) -> String {
        let safe = safeIdentifier(value)
        return safe.isEmpty ? UUID().uuidString.lowercased() : safe
    }

    private func safeImportFileName(_ value: String) -> String {
        let url = URL(fileURLWithPath: value)
        let base = safeIdentifier(url.deletingPathExtension().lastPathComponent)
        let pathExtension = safeIdentifier(url.pathExtension.lowercased())
        let safeBase = base.isEmpty ? "model" : base
        return pathExtension.isEmpty ? safeBase : "\(safeBase).\(pathExtension)"
    }

    private func safeIdentifier(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        return value.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        .reduce(into: "") { $0.append($1) }
        .trimmingCharacters(in: CharacterSet(charactersIn: ".-_"))
    }
}
