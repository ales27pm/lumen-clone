import CryptoKit
import Darwin
import Foundation
import Testing
@testable import Lumen

struct CoreAIModelStoreRecoveryTests {
    @Test func embeddingWrongSizeFailsBeforeCreatingAnOwnedCopy() throws {
        let fixture = try Fixture(.embedding)
        defer { fixture.remove() }
        try Data("wrong size".utf8).write(to: fixture.source.appendingPathComponent(fixture.graphPath))
        #expect(throws: CoreAILocalPackageIO.Failure.integrity) { try fixture.install() }
        #expect(!FileManager.default.fileExists(atPath: fixture.owned.path))
    }

    @Test func embeddingSameSizeCorruptionStillChecksDigestAndPreservesActive() throws {
        let fixture = try Fixture(.embedding)
        defer { fixture.remove() }
        try fixture.install()
        let previous = try fixture.installedRoot()
        let graph = fixture.source.appendingPathComponent(fixture.graphPath)
        var data = try Data(contentsOf: graph)
        data[data.startIndex] ^= 1
        try data.write(to: graph)
        #expect(throws: CoreAILocalPackageIO.Failure.integrity) { try fixture.install() }
        #expect(try fixture.installedRoot() == previous)
    }

    @Test func gemmaAggregateQuotaFailsBeforeCreatingAnyOwnedCopy() throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        let total = try fixture.fileIdentities().reduce(Int64(0)) { $0 + $1.bytes }
        let storage = CoreAIGemmaModelStore.Storage(directory: fixture.owned, maximumPackageBytes: total - 1)
        #expect(throws: CoreAILocalPackageIO.Failure.invalidPackage) { try storage.install(source: fixture.source) }
        #expect(!FileManager.default.fileExists(atPath: fixture.owned.path))
    }

    @Test func embeddingAggregateQuotaFailsBeforeCreatingAnyOwnedCopy() throws {
        let fixture = try Fixture(.embedding)
        defer { fixture.remove() }
        let total = fixture.expectedFiles.reduce(Int64(0)) { $0 + $1.bytes }
        let storage = CoreAITextEmbeddingModelStore.Storage(directory: fixture.owned, expectedFiles: fixture.expectedFiles,
                                                            maximumPackageBytes: total - 1)
        #expect(throws: CoreAILocalPackageIO.Failure.invalidPackage) { try storage.install(source: fixture.source) }
        #expect(!FileManager.default.fileExists(atPath: fixture.owned.path))
    }

    @Test func identitySizeAdmissionDoesNotCreateOwnedStorage() throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        try fixture.addLongNamedFiles(count: 3_000)
        let paths = try CoreAILocalPackageIO().inventory(fixture.source)
        // The path strings alone exceed the JSON identity cap. The entry count
        // and every individual path remain admissible to the package reader.
        #expect(paths.count == 3_008)
        #expect(paths.reduce(0) { $0 + $1.utf8.count } > 1_024 * 1_024)
        #expect(throws: CoreAILocalPackageIO.Failure.invalidPackage) { try fixture.install() }
        #expect(!FileManager.default.fileExists(atPath: fixture.owned.path))
    }

    @Test func oversizedIdentityPreservesActivePointerAndInterruptedArtifacts() throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        try fixture.install()
        let active = try fixture.installedRoot()
        let pointer = fixture.owned.appendingPathComponent("active.json")
        let previousPointer = try Data(contentsOf: pointer)
        let staging = try fixture.interruptedDirectory(staged: true)
        let unpublished = try fixture.interruptedDirectory(staged: false)
        try fixture.addLongNamedFiles(count: 3_000)
        #expect(throws: CoreAILocalPackageIO.Failure.invalidPackage) { try fixture.install() }
        #expect(try Data(contentsOf: pointer) == previousPointer)
        #expect(try fixture.installedRoot() == active)
        // Admission must finish before recovery can delete either artifact.
        #expect(FileManager.default.fileExists(atPath: staging.path))
        #expect(FileManager.default.fileExists(atPath: unpublished.path))
    }

    @Test func largeAdmissibleIdentityPublishesActualHashes() throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        try fixture.addLongNamedFiles(count: 1_700)
        try fixture.install()
        let installed = try fixture.installedRoot()
        let data = try Data(contentsOf: fixture.owned.appendingPathComponent("active.json"))
        #expect(data.count > 900_000)
        #expect(data.count <= 1_024 * 1_024)
        let identity = try JSONDecoder().decode(CoreAIGemmaModelStore.Identity.self, from: data)
        #expect(identity.files.count == 1_708)
        let file = try #require(identity.files.first { $0.path.hasSuffix("/file-0.bin") })
        // SHA-256 of the single literal byte 0x78 ('x'), independently known.
        #expect(file.sha256 == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881")
        #expect(try Data(contentsOf: installed.appendingPathComponent(file.path)) == Data([0x78]))
    }

    @Test func interruptedImportsAreRecoveredByBothStoresAndActiveSurvivesFailedReplacement() throws {
        for kind in [Kind.gemma, .embedding] {
            let fixture = try Fixture(kind)
            defer { fixture.remove() }
            try fixture.install()
            let active = try fixture.installedRoot()
            let staging = try fixture.interruptedDirectory(staged: true)
            let unpublished = try fixture.interruptedDirectory(staged: false)
            let unrelated = fixture.owned.appendingPathComponent("AppleCompilationCache")
            try FileManager.default.createDirectory(at: unrelated, withIntermediateDirectories: true)
            let marker = unrelated.appendingPathComponent("retain.bin")
            try Data("cache marker".utf8).write(to: marker)
            // Wrong digest with the original file size reaches copy-time validation
            // after recovery. The prior active model must remain available.
            let hash = fixture.source.appendingPathComponent(fixture.hashPath)
            var bytes = try Data(contentsOf: hash)
            bytes[bytes.startIndex] ^= 1
            try bytes.write(to: hash)
            #expect(throws: CoreAILocalPackageIO.Failure.integrity) { try fixture.install() }
            #expect(try fixture.installedRoot() == active)
            #expect(!FileManager.default.fileExists(atPath: staging.path))
            #expect(!FileManager.default.fileExists(atPath: unpublished.path))
            #expect(try Data(contentsOf: marker) == Data("cache marker".utf8))
        }
    }

    @Test func missingPointerRecoversUnpublishedInstallationsForBothStores() throws {
        for kind in [Kind.gemma, .embedding] {
            let fixture = try Fixture(kind)
            defer { fixture.remove() }
            let staged = try fixture.interruptedDirectory(staged: true)
            let unpublished = try fixture.interruptedDirectory(staged: false)
            try fixture.install()
            #expect(!FileManager.default.fileExists(atPath: staged.path))
            #expect(!FileManager.default.fileExists(atPath: unpublished.path))
            #expect(FileManager.default.fileExists(atPath: try fixture.installedRoot().path))
        }
    }

    @Test func corruptPointerPreservesCanonicalDirectoriesDuringRecoveryForBothStores() throws {
        for kind in [Kind.gemma, .embedding] {
            let fixture = try Fixture(kind)
            defer { fixture.remove() }
            try fixture.install()
            let previous = try fixture.installedRoot()
            let unknown = try fixture.interruptedDirectory(staged: false)
            let staged = try fixture.interruptedDirectory(staged: true)
            try Data("corrupt active pointer".utf8).write(to: fixture.owned.appendingPathComponent("active.json"))
            try fixture.install()
            #expect(FileManager.default.fileExists(atPath: previous.path))
            #expect(FileManager.default.fileExists(atPath: unknown.path))
            #expect(!FileManager.default.fileExists(atPath: staged.path))
        }
    }

    @Test func recoveryDoesNotFollowInstallationOrNestedSymlinks() throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        try fixture.install()
        let active = try fixture.installedRoot()
        let external = fixture.root.appendingPathComponent("outside-model-store")
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        let marker = external.appendingPathComponent("retain.bin")
        try Data("external marker".utf8).write(to: marker)
        let masquerading = fixture.owned.appendingPathComponent(".import-" + UUID().uuidString.lowercased())
        try FileManager.default.createSymbolicLink(at: masquerading, withDestinationURL: external)
        let staged = try fixture.interruptedDirectory(staged: true)
        try FileManager.default.createSymbolicLink(at: staged.appendingPathComponent("external-link"), withDestinationURL: external)
        try FileManager.default.createSymbolicLink(at: staged.appendingPathComponent("active-link"), withDestinationURL: active)
        try fixture.install()
        #expect(try Data(contentsOf: marker) == Data("external marker".utf8))
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: masquerading.path) == external.path)
        #expect(!FileManager.default.fileExists(atPath: staged.path))
    }

    @Test func cancelledStorageRemovalPreservesBothInstalledPackages() async throws {
        for kind in [Kind.gemma, .embedding] {
            let fixture = try Fixture(kind)
            defer { fixture.remove() }
            try fixture.install()
            let active = try fixture.installedRoot()
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                try fixture.removeInstalled()
            }
            do { try await task.value; Issue.record("An already-cancelled removal must fail") }
            catch { #expect(error is CancellationError) }
            #expect(try fixture.installedRoot() == active)
        }
    }

    @Test func cancelledRecoveryLeavesCrashArtifactsAndActiveUntouched() async throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        try fixture.install()
        let active = try fixture.installedRoot()
        let staged = try fixture.interruptedDirectory(staged: true)
        let unpublished = try fixture.interruptedDirectory(staged: false)
        let activeInstallation = active.deletingLastPathComponent().lastPathComponent
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try CoreAIModelInstallationRecovery.cleanup(directory: fixture.owned, activeInstallation: activeInstallation)
        }
        do { try await task.value; Issue.record("Cancelled recovery must fail") }
        catch { #expect(error is CancellationError) }
        #expect(try fixture.installedRoot() == active)
        #expect(FileManager.default.fileExists(atPath: staged.path))
        #expect(FileManager.default.fileExists(atPath: unpublished.path))
    }

    @Test func recoveryEnumerationRejectsExcessiveSiblingsBeforeDeletingAnything() throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        let staged = try fixture.interruptedDirectory(staged: true)
        for index in 0..<4_096 {
            #expect(FileManager.default.createFile(atPath: fixture.owned.appendingPathComponent("unrelated-\(index)").path, contents: nil))
        }
        #expect(throws: CoreAILocalPackageIO.Failure.invalidPackage) {
            try CoreAIModelInstallationRecovery.cleanup(directory: fixture.owned, activeInstallation: nil)
        }
        #expect(FileManager.default.fileExists(atPath: staged.path))
    }

    @Test func recoveryHandlesDeepInterruptedPackagesWithBoundedDescriptors() throws {
        let fixture = try Fixture(.gemma)
        defer { fixture.remove() }
        let staged = try fixture.interruptedDirectory(staged: true)
        // Greater than the usual Darwin soft descriptor limit; the installer
        // accepts paths this deep within its 1024-byte relative-path bound.
        let io = CoreAILocalPackageIO()
        let descriptor = try io.openDirectory(staged)
        var current = dup(descriptor)
        close(descriptor)
        #expect(current >= 0)
        do {
            for _ in 0..<300 {
                guard mkdirat(current, "d", 0o700) == 0 else { throw CoreAILocalPackageIO.Failure.storage }
                let next = openat(current, "d", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard next >= 0 else { throw CoreAILocalPackageIO.Failure.storage }
                close(current)
                current = next
            }
            close(current)
        } catch { close(current); throw error }
        try CoreAIModelInstallationRecovery.cleanup(directory: fixture.owned, activeInstallation: nil)
        #expect(!FileManager.default.fileExists(atPath: staged.path))
    }

    private enum Kind: Sendable, Equatable { case gemma, embedding }

    private struct Fixture: Sendable {
        let root: URL
        let source: URL
        let owned: URL
        let kind: Kind
        let expectedFiles: [CoreAILocalPackageIO.FileIdentity]
        var graphPath: String { kind == .gemma ? "gemma.aimodel/main.mlirb" : "model/embeddinggemma-2-text-mm.aimodel/main.mlirb" }
        var hashPath: String { kind == .gemma ? "gemma.aimodel/main.hash" : "model/embeddinggemma-2-text-mm.aimodel/main.hash" }

        init(_ kind: Kind) throws {
            self.kind = kind
            let rootURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
            let sourceURL = rootURL.appendingPathComponent("source", isDirectory: true)
            root = rootURL
            source = sourceURL
            owned = rootURL.appendingPathComponent("owned", isDirectory: true)
            let graphPath = kind == .gemma ? "gemma.aimodel/main.mlirb" : "model/embeddinggemma-2-text-mm.aimodel/main.mlirb"
            let hashPath = kind == .gemma ? "gemma.aimodel/main.hash" : "model/embeddinggemma-2-text-mm.aimodel/main.hash"
            let paths = kind == .gemma
                ? ["metadata.json", graphPath, hashPath, "gemma.aimodel/metadata.json", "gemma_ple.safetensors",
                   "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja"].sorted()
                : CoreAITextEmbeddingModelStore.pinnedFiles.map(\.path)
            for path in paths {
                let url = sourceURL.appendingPathComponent(path)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data("synthetic \(path); not model weights".utf8).write(to: url)
            }
            try Data(SHA256.hash(data: Data(contentsOf: sourceURL.appendingPathComponent(graphPath)))).write(to: sourceURL.appendingPathComponent(hashPath))
            let configuration: [String: Any]
            let configurationPath: String
            if kind == .gemma {
                configurationPath = "metadata.json"
                configuration = ["kind": "llm", "assets": ["main": "gemma.aimodel"],
                                 "auxiliary_assets": ["per_layer_embeddings": "gemma_ple.safetensors"],
                                 "language": ["tokenizer": "google/gemma-4-E2B-it", "max_context_length": 8_192, "embedded_tokenizer": true]]
            } else {
                configurationPath = "model/config.json"
                configuration = ["base_model": "google/embeddinggemma-2", "format": "coreai", "asset": "embeddinggemma-2-text-mm.aimodel",
                                 "functions": CoreAITextEmbeddingPolicy.functions, "buckets": CoreAITextEmbeddingPolicy.buckets,
                                 "max_tokens": 2_048, "precision_mode": "ane", "hidden_size": 512, "embedding_dim": 768,
                                 "matryoshka_dims": [768, 512, 256, 128], "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
                                 "inputs": ["input_ids": "fixture", "attention_mask": "fixture", "mm_embeds": "fixture", "mm_mask": "fixture"],
                                 "output": ["embedding": "fixture"],
                                 "prompts": ["document": CoreAITextEmbeddingPolicy.documentPrefix, "search_query": CoreAITextEmbeddingPolicy.queryPrefix,
                                             "sentence_similarity": "task: sentence similarity | query: ", "code_retrieval": "task: code retrieval | query: "]]
            }
            try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys]).write(to: sourceURL.appendingPathComponent(configurationPath))
            expectedFiles = try paths.map { path in
                let data = try Data(contentsOf: sourceURL.appendingPathComponent(path))
                return .init(path: path, bytes: Int64(data.count), sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
            }
        }

        func fileIdentities() throws -> [CoreAILocalPackageIO.FileIdentity] {
            try expectedFiles.map { file in
                let data = try Data(contentsOf: source.appendingPathComponent(file.path))
                return .init(path: file.path, bytes: Int64(data.count), sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
            }
        }
        func addLongNamedFiles(count: Int) throws {
            let prefix = String(repeating: "a", count: 240) + "/" + String(repeating: "b", count: 240)
            let folder = source.appendingPathComponent(prefix)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for index in 0..<count {
                try Data([0x78]).write(to: folder.appendingPathComponent("file-\(index).bin"))
            }
        }
        func install() throws {
            switch kind {
            case .gemma: try CoreAIGemmaModelStore.Storage(directory: owned).install(source: source)
            case .embedding: try CoreAITextEmbeddingModelStore.Storage(directory: owned, expectedFiles: expectedFiles).install(source: source)
            }
        }
        func installedRoot() throws -> URL {
            switch kind {
            case .gemma: try CoreAIGemmaModelStore.Storage(directory: owned).configuredURL()
            case .embedding: try CoreAITextEmbeddingModelStore.Storage(directory: owned, expectedFiles: expectedFiles).configuredPackage().rootURL
            }
        }
        func removeInstalled() throws {
            switch kind {
            case .gemma: try CoreAIGemmaModelStore.Storage(directory: owned).remove()
            case .embedding: try CoreAITextEmbeddingModelStore.Storage(directory: owned, expectedFiles: expectedFiles).remove()
            }
        }
        func interruptedDirectory(staged: Bool) throws -> URL {
            let path = owned.appendingPathComponent((staged ? ".import-" : "") + UUID().uuidString.lowercased(), isDirectory: true)
            try FileManager.default.createDirectory(at: path.appendingPathComponent("bundle"), withIntermediateDirectories: true)
            try Data("interrupted partial copy".utf8).write(to: path.appendingPathComponent("bundle/partial.bin"))
            return path
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
    }
}
