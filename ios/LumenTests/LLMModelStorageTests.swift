import Foundation
import Testing
@testable import Lumen

struct LLMModelStorageTests {
    @Test func sha256FileHasherReturnsKnownHashForSmallFile() throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let fileURL = temp.baseDirectory.appendingPathComponent("hello.txt")
        try Data("hello".utf8).write(to: fileURL)

        let hash = try SHA256FileHasher.sha256Hex(for: fileURL)

        #expect(hash == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    }

    @Test func sha256ArtifactHasherRecursivelyHashesCompiledModelBundleContents() throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let firstBundle = temp.baseDirectory.appendingPathComponent("first.mlmodelc", isDirectory: true)
        let secondBundle = temp.baseDirectory.appendingPathComponent("second.mlmodelc", isDirectory: true)
        for bundle in [firstBundle, secondBundle] {
            let weights = bundle.appendingPathComponent("weights", isDirectory: true)
            try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
            try Data("manifest".utf8).write(to: bundle.appendingPathComponent("model.mil"))
            try Data("weights-a".utf8).write(to: weights.appendingPathComponent("weight.bin"))
        }

        let firstDigest = try SHA256FileHasher.sha256Hex(forArtifactAt: firstBundle)
        let secondDigest = try SHA256FileHasher.sha256Hex(forArtifactAt: secondBundle)
        #expect(firstDigest == secondDigest)

        try Data("weights-b".utf8).write(
            to: secondBundle.appendingPathComponent("weights", isDirectory: true).appendingPathComponent("weight.bin")
        )
        #expect(firstDigest != (try SHA256FileHasher.sha256Hex(forArtifactAt: secondBundle)))
    }

    @Test func modelFileValidatorAcceptsGGUFExtension() throws {
        let url = URL(fileURLWithPath: "/tmp/model.gguf")

        try ModelFileValidator.validateExtension(for: url, backend: .gguf)
    }

    @Test func modelFileValidatorRejectsWrongGGUFExtension() {
        let url = URL(fileURLWithPath: "/tmp/model.bin")

        do {
            try ModelFileValidator.validateExtension(for: url, backend: .gguf)
            #expect(Bool(false))
        } catch ModelStorageError.invalidModelFileExtension(let fileName) {
            #expect(fileName == "model.bin")
        } catch {
            #expect(Bool(false))
        }
    }

    @Test func modelFileIntegrityMissingFileFailureIsSanitizedAndDiagnostic() {
        let rawPath = "/private/raw/lumen/models/missing.gguf"

        let result = ModelFileIntegrity.validateInstalledFileWithDiagnostics(
            localPath: rawPath,
            fileName: "missing.gguf",
            expectedSizeBytes: 1
        )

        guard case .failure(let failure) = result else {
            Issue.record("Expected missing model file to fail integrity validation")
            return
        }
        #expect(failure.errorDescription == "Model file is missing.")
        #expect(failure.diagnosticCode.hasPrefix("file_missing:path_sha256="))
        #expect(!failure.localizedDescription.contains(rawPath))
        #expect(!failure.diagnosticCode.contains(rawPath))
        #expect(ModelFileIntegrity.validateInstalledFile(localPath: rawPath, fileName: "missing.gguf", expectedSizeBytes: 1) == false)
    }

    @Test func modelFileIntegrityInvalidGGUFMagicIsDistinctAndSanitized() throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let modelURL = temp.baseDirectory.appendingPathComponent("not-a-model.gguf")
        var data = Data(count: 16 * 1024 * 1024)
        data.replaceSubrange(0..<4, with: Data([0x4e, 0x4f, 0x50, 0x45]))
        try data.write(to: modelURL)

        let result = ModelFileIntegrity.validateInstalledFileWithDiagnostics(
            localPath: modelURL.path,
            fileName: "not-a-model.gguf",
            expectedSizeBytes: 1
        )

        guard case .failure(let failure) = result else {
            Issue.record("Expected invalid GGUF magic to fail integrity validation")
            return
        }
        #expect(failure.errorDescription == "Downloaded file is not a GGUF model.")
        #expect(failure.diagnosticCode.hasPrefix("invalid_gguf_magic:path_sha256="))
        #expect(!failure.localizedDescription.contains(modelURL.path))
        #expect(!failure.diagnosticCode.contains(modelURL.path))
    }

    @Test func pinnedCatalogArtifactRequiresExactSizeAndHash() throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let modelURL = temp.baseDirectory.appendingPathComponent("verified.gguf")
        var data = Data(count: 16 * 1024 * 1024)
        data.replaceSubrange(0..<4, with: Data([0x47, 0x47, 0x55, 0x46]))
        try data.write(to: modelURL)
        let digest = try SHA256FileHasher.sha256Hex(for: modelURL)
        let catalog = testCatalogModel(fileName: modelURL.lastPathComponent, sizeBytes: Int64(data.count), sha256: digest)

        guard case .success(let actualSize) = ModelFileIntegrity.validateDownloadedCatalogFile(catalog, at: modelURL) else {
            Issue.record("Expected exact pinned artifact to validate")
            return
        }
        #expect(actualSize == Int64(data.count))

        data.append(0)
        try data.write(to: modelURL)
        guard case .failure(.sizeMismatch(let actual, let expected)) = ModelFileIntegrity.validateDownloadedCatalogFile(catalog, at: modelURL) else {
            Issue.record("Expected an oversized pinned artifact to fail exact-size validation")
            return
        }
        #expect(actual == Int64(data.count))
        #expect(expected == catalog.sizeBytes)
    }

    @Test func pinnedCatalogArtifactRejectsSameSizeTampering() throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let modelURL = temp.baseDirectory.appendingPathComponent("tampered.gguf")
        var data = Data(count: 16 * 1024 * 1024)
        data.replaceSubrange(0..<4, with: Data([0x47, 0x47, 0x55, 0x46]))
        try data.write(to: modelURL)
        let digest = try SHA256FileHasher.sha256Hex(for: modelURL)
        let catalog = testCatalogModel(fileName: modelURL.lastPathComponent, sizeBytes: Int64(data.count), sha256: digest)

        data[data.count - 1] = 1
        try data.write(to: modelURL)
        guard case .failure(.hashMismatch(let expected, let actual)) = ModelFileIntegrity.validateDownloadedCatalogFile(catalog, at: modelURL) else {
            Issue.record("Expected same-size tampering to fail SHA-256 validation")
            return
        }
        #expect(expected == digest)
        #expect(actual != digest)
    }

    @Test func modelStorageRegistersTinyIntentRecord() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)

        let record = try await storage.registerTinyIntentModel()
        let fetched = try await storage.record(for: record.id)

        #expect(record.id == "builtin.tiny-intent")
        #expect(record.model.backend == .tinyIntent)
        #expect(record.isUsable)
        #expect(fetched == record)
    }

    private func testCatalogModel(fileName: String, sizeBytes: Int64, sha256: String) -> CatalogModel {
        CatalogModel(
            id: "test-pinned-artifact",
            name: "Pinned test artifact",
            repoId: "test/models",
            fileName: fileName,
            parameters: "test",
            quantization: "test",
            sizeBytes: sizeBytes,
            role: .chat,
            description: "test",
            tags: [],
            sourceRevision: String(repeating: "a", count: 40),
            expectedSHA256: sha256
        )
    }

    @Test func modelStorageImportsSmallFakeGGUFIntoTempStorage() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        let sourceURL = temp.baseDirectory.appendingPathComponent("source.gguf")
        try Data("fake gguf".utf8).write(to: sourceURL)

        let record = try await storage.importExistingModelFile(
            fileURL: sourceURL,
            catalogEntry: BuiltInModelCatalog.entry(id: "qwen2.5-1.5b-instruct-q4-k-m-gguf"),
            backend: .gguf,
            displayName: "Imported Test GGUF",
            expectedSHA256: nil
        )

        #expect(record.model.backend == .gguf)
        #expect(record.verificationStatus == .unverified)
        #expect(record.fileURL?.deletingLastPathComponent() == temp.location.modelsDirectory)
        #expect(record.relativePath?.hasPrefix("Models/") == true)
        #expect(record.sizeBytes == 9)
        #expect(record.isUsable)
        #expect(try await storage.record(for: record.id) == record)
    }

    @Test func modelStorageResolverUsesDocumentsVisibleRootAndMigratesPreviousModels() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("LumenModelStorageResolverTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let documents = base.appendingPathComponent("Documents", isDirectory: true)
        let appSupport = base.appendingPathComponent("ApplicationSupport", isDirectory: true)
        let previousModels = appSupport
            .appendingPathComponent("Lumen", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: previousModels, withIntermediateDirectories: true)
        let previousModel = previousModels.appendingPathComponent("previous.gguf")
        try Data("previous".utf8).write(to: previousModel)
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)

        let location = try ModelStorageDirectoryResolver.resolve(
            documentDirectories: [documents],
            applicationSupportDirectories: [appSupport]
        )

        #expect(location.rootDirectory == documents.appendingPathComponent("Lumen", isDirectory: true))
        #expect(location.modelsDirectory == documents.appendingPathComponent("Lumen", isDirectory: true).appendingPathComponent("Models", isDirectory: true))
        #expect(FileManager.default.fileExists(atPath: location.modelsDirectory.appendingPathComponent("previous.gguf").path))
        #expect(FileManager.default.fileExists(atPath: previousModel.path) == false)
    }

    @Test func modelStorageRepairsMigratedMetadataFileURLs() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        let migratedModelURL = temp.location.modelsDirectory.appendingPathComponent("migrated.gguf")
        let staleModelURL = temp.baseDirectory
            .appendingPathComponent("ApplicationSupport", isDirectory: true)
            .appendingPathComponent("Lumen", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("migrated.gguf")
        try Data("migrated".utf8).write(to: migratedModelURL)
        let record = installedGGUFRecord(id: "migrated.model", fileURL: staleModelURL)
        let staleRecord = InstalledModelRecord(
            id: record.id,
            catalogID: record.catalogID,
            model: LocalLLMModel(
                id: record.model.id,
                displayName: record.model.displayName,
                backend: record.model.backend,
                localURL: staleModelURL,
                contextLength: record.model.contextLength
            ),
            fileURL: staleModelURL,
            relativePath: "Models/migrated.gguf",
            sha256: nil,
            sizeBytes: nil,
            installedAt: record.installedAt,
            lastVerifiedAt: nil,
            verificationStatus: .unverified
        )
        try await storage.saveRecord(staleRecord)

        let repaired = try await storage.record(for: staleRecord.id)

        #expect(repaired?.fileURL == migratedModelURL)
        #expect(repaired?.model.localURL == migratedModelURL)
        #expect(repaired?.isUsable == true)
    }

    @Test func modelStorageWritesAndReadsMetadataRecord() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        let record = InstalledModelRecord(
            id: "test.record",
            catalogID: nil,
            model: LocalLLMModel(
                id: "test.record",
                displayName: "Test Record",
                backend: .tinyIntent,
                contextLength: 128
            ),
            fileURL: nil,
            relativePath: nil,
            sha256: nil,
            sizeBytes: nil,
            installedAt: Date(timeIntervalSince1970: 1_700_000_000),
            lastVerifiedAt: nil,
            verificationStatus: .verified
        )

        try await storage.saveRecord(record)

        #expect(try await storage.record(for: "test.record") == record)
        #expect(try await storage.listInstalledModels() == [record])
    }

    @Test func deleteModelDeletesFileUnderStorageRoot() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        let modelURL = temp.location.modelsDirectory.appendingPathComponent("delete-me.gguf")
        try Data("delete".utf8).write(to: modelURL)
        let record = installedGGUFRecord(id: "delete.under.root", fileURL: modelURL)
        try await storage.saveRecord(record)

        try await storage.deleteModel(id: record.id, deleteFile: true)

        #expect(FileManager.default.fileExists(atPath: modelURL.path) == false)
        #expect(try await storage.record(for: record.id) == nil)
    }

    @Test func deleteModelDoesNotDeleteOutsideRootFile() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        let outsideURL = temp.baseDirectory.appendingPathComponent("outside.gguf")
        try Data("outside".utf8).write(to: outsideURL)
        let record = installedGGUFRecord(id: "outside.root", fileURL: outsideURL)
        try await storage.saveRecord(record)

        try await storage.deleteModel(id: record.id, deleteFile: true)

        #expect(FileManager.default.fileExists(atPath: outsideURL.path))
        #expect(try await storage.record(for: record.id) == nil)
    }

    @Test func deleteModelDoesNotDeleteSymlinkResolvedFileOutsideRoot() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let outsideDirectory = temp.baseDirectory.appendingPathComponent("Outside", isDirectory: true)
        try FileManager.default.createDirectory(at: outsideDirectory, withIntermediateDirectories: true)
        let outsideURL = outsideDirectory.appendingPathComponent("outside.gguf")
        try Data("outside".utf8).write(to: outsideURL)

        let symlinkURL = temp.location.modelsDirectory.appendingPathComponent("LinkedOutside", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: symlinkURL, withDestinationURL: outsideDirectory)

        let storage = LLMModelStorage(location: temp.location)
        let symlinkedFileURL = symlinkURL.appendingPathComponent("outside.gguf")
        let record = installedGGUFRecord(id: "symlink.outside.root", fileURL: symlinkedFileURL)
        try await storage.saveRecord(record)

        try await storage.deleteModel(id: record.id, deleteFile: true)

        #expect(FileManager.default.fileExists(atPath: outsideURL.path))
        #expect(try await storage.record(for: record.id) == nil)
    }

    @Test func importRemovesCopiedFileWhenMetadataSaveFails() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let missingMetadataDirectory = temp.location.modelsDirectory
            .appendingPathComponent("MissingMetadata", isDirectory: true)
        let brokenLocation = ModelStorageLocation(
            rootDirectory: temp.location.rootDirectory,
            modelsDirectory: temp.location.modelsDirectory,
            metadataDirectory: missingMetadataDirectory,
            temporaryDirectory: temp.location.temporaryDirectory
        )
        let storage = LLMModelStorage(location: brokenLocation)
        let sourceURL = temp.baseDirectory.appendingPathComponent("leaky.gguf")
        try Data("leaky".utf8).write(to: sourceURL)
        let copiedURL = temp.location.modelsDirectory.appendingPathComponent("leaky.gguf")

        do {
            _ = try await storage.importExistingModelFile(
                fileURL: sourceURL,
                catalogEntry: nil,
                backend: .gguf,
                displayName: "Leaky",
                expectedSHA256: nil
            )
            #expect(Bool(false))
        } catch ModelStorageError.metadataWriteFailed {
            #expect(FileManager.default.fileExists(atPath: copiedURL.path) == false)
        } catch {
            #expect(Bool(false))
        }
    }

    @Test func builtInModelCatalogContainsTinyIntent() {
        let entry = BuiltInModelCatalog.entry(id: "builtin.tiny-intent")

        #expect(entry?.backend == .tinyIntent)
        #expect(entry?.recommendedUse == .tinyIntent)
        #expect(entry?.tags.contains("fallback") == false)
        #expect(entry?.notes?.localizedCaseInsensitiveContains("fallback") == false)
    }

    @Test func builtInNomicDescriptorDoesNotAdvertiseEmbeddingExecution() {
        let entry = BuiltInModelCatalog.entry(id: "nomic-embed-text-local")

        #if DEBUG
        #expect(entry?.backend == .gguf)
        #expect(entry?.recommendedUse != .embedding)
        #expect(entry?.tags.contains("embedding") == false)
        #else
        #expect(entry == nil)
        #endif
    }

    @Test func modelCatalogSourceDecodesUnknownTypeAsUnknown() throws {
        let data = Data(#"{"type":"futureBackend","url":"https://example.com/model.gguf"}"#.utf8)

        let source = try JSONDecoder().decode(ModelCatalogSource.self, from: data)

        #expect(source == .unknown)
    }

    @Test func modelSelectionServiceReturnsTinyIntentFallbackWhenOnlyUsableModel() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        _ = try await storage.registerTinyIntentModel()
        let policy = DeviceModelPolicy(provider: TestDeviceCapabilityProvider())
        let selection = ModelSelectionService(storage: storage, policy: policy)

        let best = try await selection.bestModel(for: .standardChat, appIsForeground: true)

        #expect(best?.id == "builtin.tiny-intent")
        #expect(best?.model.backend == .tinyIntent)
    }

    @Test func modelSelectionServiceDoesNotSelectRemoteModelWithoutEscalationPolicy() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        _ = try await storage.registerTinyIntentModel()
        try await storage.saveRecord(installedRemoteRecord(id: "remote.standard-chat"))
        let policy = DeviceModelPolicy(provider: TestDeviceCapabilityProvider())
        let selection = ModelSelectionService(storage: storage, policy: policy)

        let best = try await selection.bestModel(for: .standardChat, appIsForeground: true)
        let usable = try await selection.installedUsableModels(appIsForeground: true)

        #expect(best?.id == "builtin.tiny-intent")
        #expect(best?.model.backend == .tinyIntent)
        #expect(usable.contains { $0.model.backend == .remote } == false)
    }

    @Test func modelSelectionServiceDoesNotSelectMockModelWithoutTestHarnessPolicy() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        _ = try await storage.registerTinyIntentModel()
        try await storage.saveRecord(installedMockRecord(id: "mock.standard-chat"))
        let policy = DeviceModelPolicy(provider: TestDeviceCapabilityProvider())
        let selection = ModelSelectionService(storage: storage, policy: policy)

        let best = try await selection.bestModel(for: .standardChat, appIsForeground: true)
        let usable = try await selection.installedUsableModels(appIsForeground: true)

        #expect(best?.id == "builtin.tiny-intent")
        #expect(best?.model.backend == .tinyIntent)
        #expect(usable.contains { $0.model.backend == .mock } == false)
    }

    @Test func hashMismatchThrowsModelStorageError() async throws {
        let temp = try makeTemporaryStorage()
        defer { try? FileManager.default.removeItem(at: temp.baseDirectory) }
        let storage = LLMModelStorage(location: temp.location)
        let sourceURL = temp.baseDirectory.appendingPathComponent("hash-mismatch.gguf")
        try Data("actual".utf8).write(to: sourceURL)

        do {
            _ = try await storage.importExistingModelFile(
                fileURL: sourceURL,
                catalogEntry: nil,
                backend: .gguf,
                displayName: "Hash Mismatch",
                expectedSHA256: "0000000000000000000000000000000000000000000000000000000000000000"
            )
            #expect(Bool(false))
        } catch ModelStorageError.hashMismatch(let expected, let actual) {
            #expect(expected == "0000000000000000000000000000000000000000000000000000000000000000")
            #expect(actual.count == 64)
        } catch {
            #expect(Bool(false))
        }
    }

    private func installedGGUFRecord(id: String, fileURL: URL) -> InstalledModelRecord {
        InstalledModelRecord(
            id: id,
            catalogID: nil,
            model: LocalLLMModel(
                id: id,
                displayName: "Installed GGUF",
                backend: .gguf,
                localURL: fileURL,
                contextLength: 512
            ),
            fileURL: fileURL,
            relativePath: nil,
            sha256: nil,
            sizeBytes: nil,
            installedAt: Date(timeIntervalSince1970: 1_700_000_000),
            lastVerifiedAt: nil,
            verificationStatus: .unverified
        )
    }

    private func installedRemoteRecord(id: String) -> InstalledModelRecord {
        InstalledModelRecord(
            id: id,
            catalogID: nil,
            model: LocalLLMModel(
                id: id,
                displayName: "Installed Remote",
                backend: .remote,
                contextLength: 8_192
            ),
            fileURL: nil,
            relativePath: nil,
            sha256: nil,
            sizeBytes: nil,
            installedAt: Date(timeIntervalSince1970: 1_700_000_000),
            lastVerifiedAt: nil,
            verificationStatus: .verified
        )
    }

    private func installedMockRecord(id: String) -> InstalledModelRecord {
        InstalledModelRecord(
            id: id,
            catalogID: nil,
            model: LocalLLMModel(
                id: id,
                displayName: "Installed Mock",
                backend: .mock,
                contextLength: 8_192
            ),
            fileURL: nil,
            relativePath: nil,
            sha256: nil,
            sizeBytes: nil,
            installedAt: Date(timeIntervalSince1970: 1_700_000_000),
            lastVerifiedAt: nil,
            verificationStatus: .verified
        )
    }

    private func makeTemporaryStorage() throws -> (baseDirectory: URL, location: ModelStorageLocation) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("LumenModelStorageTests-\(UUID().uuidString)", isDirectory: true)
        let root = base.appendingPathComponent("Lumen", isDirectory: true)
        let models = root.appendingPathComponent("Models", isDirectory: true)
        let metadata = models.appendingPathComponent("Metadata", isDirectory: true)
        let temporary = models.appendingPathComponent("Tmp", isDirectory: true)

        for directory in [base, root, models, metadata, temporary] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        return (
            base,
            ModelStorageLocation(
                rootDirectory: root,
                modelsDirectory: models,
                metadataDirectory: metadata,
                temporaryDirectory: temporary
            )
        )
    }
}
