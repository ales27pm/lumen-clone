import Foundation
import Darwin

/// Installs only the pinned text encoder and its tokenizer/notices. Import,
/// selection, and lazy inference are separate operations; no network is used.
nonisolated enum CoreAITextEmbeddingModelStore {
    static let selectionKey = "coreai.textEmbedding.selected"
    struct SelectionSnapshot: Sendable, Equatable {
        let selected: Bool
        let revision: UUID
    }

    /// Revisions are process-local: no selection operation survives app exit.
    /// The lock also keeps a preference read and its revision indivisible.
    final class SelectionState: @unchecked Sendable {
        // Preferences may notify observers synchronously from set(_:forKey:).
        // Permit an observer to read this same snapshot without deadlocking.
        private let lock = NSRecursiveLock()
        private let defaults: UserDefaults
        private var revision = UUID()

        init(defaults: UserDefaults) { self.defaults = defaults }

        var snapshot: SelectionSnapshot {
            lock.lock()
            defer { lock.unlock() }
            return SelectionSnapshot(selected: defaults.object(forKey: selectionKey) as? Bool ?? false, revision: revision)
        }

        func commit(_ selected: Bool) -> SelectionSnapshot {
            lock.lock()
            defer { lock.unlock() }
            let committed = SelectionSnapshot(selected: selected, revision: UUID())
            revision = committed.revision
            defaults.set(selected, forKey: selectionKey)
            return committed
        }
    }

    private static let selectionState = SelectionState(defaults: .standard)
    static var selectionSnapshot: SelectionSnapshot { selectionState.snapshot }
    static var isSelected: Bool { selectionSnapshot.selected }

    @discardableResult
    static func setSelected(_ selected: Bool) async throws -> SelectionSnapshot {
        try await withModelReplacement {
            // The service gate remains closed while the preference changes.
            // Capture this exact revision before actor hops can admit newer work.
            let snapshot = selectionState.commit(selected)
            if selected { await AppLlamaService.shared.unloadEmbed() }
            return snapshot
        }
    }

    /// Consistency rollback remains permitted when its caller was cancelled.
    /// Compare while acquiring the replacement gate, before draining any model.
    static func restoreSelection(_ selected: Bool, ifUnchanged expected: SelectionSnapshot) async throws -> Bool {
        guard try await CoreAITextEmbeddingService.shared.beginModelReplacement(if: {
            selectionState.snapshot == expected
        }) else { return false }
        _ = selectionState.commit(selected)
        if selected { await AppLlamaService.shared.unloadEmbed() }
        await CoreAITextEmbeddingService.shared.endModelReplacement()
        return true
    }

    /// Cancellation before admission or while draining must not publish a
    /// preference, replace files, or delete a model after the user cancelled.
    @MainActor
    static func withModelReplacement<Value: Sendable>(
        begin: @MainActor () async throws -> Void = { _ = try await CoreAITextEmbeddingService.shared.beginModelReplacement() },
        end: @MainActor () async -> Void = { await CoreAITextEmbeddingService.shared.endModelReplacement() },
        operation: @MainActor () async throws -> Value
    ) async throws -> Value {
        try Task.checkCancellation()
        try await begin()
        do {
            try Task.checkCancellation()
            let value = try await operation()
            await end()
            return value
        } catch {
            await end()
            throw error
        }
    }

    static var isConfigured: Bool { storage.isConfigured }

    /// Presentation-only bounded metadata read, never a model load or hash scan.
    static func configurationSummary() async throws -> ConfigurationSummary {
        do { return try await Task.detached(priority: .utility) { try storage.configurationSummary() }.value }
        catch { throw sanitized(error) }
    }

    /// Full streaming integrity verification, called from the inference actor.
    static func configuredPackage() throws -> PackageInfo {
        do { return try storage.configuredPackage() }
        catch { throw sanitized(error) }
    }

    static func importDirectory(_ source: URL) async throws {
        do {
            try await withModelReplacement {
                let worker = Task.detached(priority: .utility) {
                    let accessed = source.startAccessingSecurityScopedResource()
                    defer { if accessed { source.stopAccessingSecurityScopedResource() } }
                    try storage.install(source: source)
                }
                try await withTaskCancellationHandler { try await worker.value } onCancel: { worker.cancel() }
            }
        } catch {
            throw sanitized(error)
        }
    }

    static func remove() async throws {
        do {
            try await withModelReplacement {
                let worker = Task.detached(priority: .utility) { try storage.remove() }
                try await withTaskCancellationHandler { try await worker.value } onCancel: { worker.cancel() }
                // Keep selection: removal surfaces unavailable, not a new space.
            }
        } catch {
            throw sanitized(error)
        }
    }

    private static var storage: Storage {
        .init(directory: URL.applicationSupportDirectory.appendingPathComponent("CoreAITextEmbeddings", isDirectory: true))
    }

    private static func sanitized(_ error: Error) -> Error {
        if error is CancellationError { return error }
        if error is CoreAITextEmbeddingError { return error }
        return (error as? Failure) ?? Failure.storage
    }

    typealias Failure = CoreAILocalPackageIO.Failure
    typealias FileIdentity = CoreAILocalPackageIO.FileIdentity

    struct PackageInfo: Sendable {
        let rootURL: URL
        let modelURL: URL
        let tokenizerURL: URL
        let tokenizerConfigURL: URL
        let configurationURL: URL
        let identifier: String
        let artifactDigest: String
        let configuration: CoreAITextEmbeddingConfiguration
    }

    struct ConfigurationSummary: Sendable, Equatable {
        let identifier: String
        let artifactDigest: String
        let maximumTokens: Int
        let nativeDimensions: Int
    }

    /// The content hashes are pinned to EverPaper/embeddinggemma-2-coreai commit
    /// 305fc2a96790d9343f38038aeca211dfca56d9a5 (published CHECKSUMS.sha256).
    /// The installation computes every digest locally, including the model graph.
    static let pinnedFiles: [FileIdentity] = [
        .init(path: "LICENSE", bytes: 11_358, sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        .init(path: "NOTICE", bytes: 21_485, sha256: "b9ac552f2eb418f79dc23cc325a7b8c38ae05c1fcc8c1880014bae281a0963a6"),
        .init(path: "model/MODEL-ATTRIBUTION.json", bytes: 5_166, sha256: "7951556c4559c8609cf67715acc53e58d8c70f7f62242494d7858406b1bbe2f7"),
        .init(path: "model/config.json", bytes: 4_643, sha256: "0204c8d4532f6930709c3c8eb7f7b25b54e4be42aa7b8c6c397fc17e872fb7bf"),
        .init(path: "model/embeddinggemma-2-text-mm.aimodel/main.hash", bytes: 32, sha256: "ea089710e57b22972d6ab4d4b960b385394b81f333ba2f75b4fb4aac1d238e6f"),
        .init(path: "model/embeddinggemma-2-text-mm.aimodel/main.mlirb", bytes: 632_206_233, sha256: "0b51638f2d17794825de21def6ab485d968fdf412853ea209617623a51f1bbde"),
        .init(path: "model/embeddinggemma-2-text-mm.aimodel/metadata.json", bytes: 431, sha256: "8984123cfdb32ba8cf526a1f3a41571205475aff40b3f1d65494f476f115f8a6"),
        .init(path: "model/tokenizer/tokenizer.json", bytes: 32_170_510, sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
        .init(path: "model/tokenizer/tokenizer_config.json", bytes: 1_599, sha256: "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874")
    ]

    private struct Identity: Codable {
        let version: Int
        let installation: String
        let identifier: String
        let artifactDigest: String
        let files: [FileIdentity]
    }

    struct Storage: Sendable {
        let directory: URL
        /// Synthetic file identities are a narrow deterministic storage-test seam.
        /// All production callers use the immutable published inventory above.
        let expectedFiles: [FileIdentity]
        init(directory: URL, expectedFiles: [FileIdentity] = CoreAITextEmbeddingModelStore.pinnedFiles) {
            self.directory = directory
            self.expectedFiles = expectedFiles
        }

        private static let maximumIdentityBytes = 256 * 1_024
        private let io = CoreAILocalPackageIO(maximumPackageBytes: 2 * 1_024 * 1_024 * 1_024)
        private var identityURL: URL { directory.appendingPathComponent("active.json") }
        var isConfigured: Bool { FileManager.default.fileExists(atPath: identityURL.path) }

        func configurationSummary() throws -> ConfigurationSummary {
            let identity = try readIdentity()
            return .init(identifier: identity.identifier, artifactDigest: identity.artifactDigest,
                         maximumTokens: 2_048, nativeDimensions: 768)
        }

        func configuredPackage() throws -> PackageInfo {
            let identity = try readIdentity()
            let owned = try io.openDirectory(directory)
            defer { close(owned) }
            let installation = openat(owned, identity.installation, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard installation >= 0 else { throw Failure.integrity }
            defer { close(installation) }
            let descriptor = openat(installation, "bundle", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else { throw Failure.integrity }
            defer { close(descriptor) }
            let root = directory.appendingPathComponent(identity.installation, isDirectory: true)
                .appendingPathComponent("bundle", isDirectory: true)
            guard try io.inventory(root) == identity.files.map(\.path) else { throw Failure.integrity }
            for expected in identity.files {
                try Task.checkCancellation()
                guard try io.digest(relative: expected.path, descriptor: descriptor) == expected else { throw Failure.integrity }
            }
            let configuration = try readConfiguration(descriptor: descriptor)
            try verifyMainHash(descriptor: descriptor, files: identity.files)
            return .init(rootURL: root, modelURL: root.appendingPathComponent(Self.modelPath, isDirectory: true),
                         tokenizerURL: root.appendingPathComponent("model/tokenizer/tokenizer.json"),
                         tokenizerConfigURL: root.appendingPathComponent("model/tokenizer/tokenizer_config.json"),
                         configurationURL: root.appendingPathComponent("model/config.json"),
                         identifier: identity.identifier, artifactDigest: identity.artifactDigest, configuration: configuration)
        }

        func install(source: URL) throws {
            let sourceDescriptor = try io.openDirectory(source)
            defer { close(sourceDescriptor) }
            let paths = try io.inventory(source)
            // Hugging Face's local-dir receipt files and Finder metadata are not
            // model inputs and are never copied. Every payload file is pinned;
            // extra model assets, including vision/audio, remain rejected.
            let payloadPaths = paths.filter { !Self.isImportMetadata($0) }
            guard payloadPaths == expectedFiles.map(\.path), expectedFiles.map(\.path) == CoreAITextEmbeddingModelStore.pinnedFiles.map(\.path)
            else { throw Failure.invalidPackage }
            let sourceConfiguration = try readConfiguration(descriptor: sourceDescriptor)
            let manager = FileManager.default
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
            var root = directory
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try root.setResourceValues(values)
            let installation = UUID().uuidString.lowercased()
            let stage = directory.appendingPathComponent(".import-" + installation, isDirectory: true)
            let destination = directory.appendingPathComponent(installation, isDirectory: true)
            let bundle = stage.appendingPathComponent("bundle", isDirectory: true)
            var committed = false
            defer {
                try? manager.removeItem(at: stage)
                if !committed { try? manager.removeItem(at: destination) }
            }
            try manager.createDirectory(at: bundle, withIntermediateDirectories: true)
            var copied: [FileIdentity] = []
            var total: Int64 = 0
            for expected in expectedFiles {
                try Task.checkCancellation()
                let output = bundle.appendingPathComponent(expected.path)
                try manager.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                let actual = try io.digest(relative: expected.path, descriptor: sourceDescriptor, copyTo: output)
                guard actual == expected else { throw Failure.integrity }
                total += actual.bytes
                guard total <= io.maximumPackageBytes else { throw Failure.invalidPackage }
                copied.append(actual)
            }
            guard try io.inventory(source) == paths else { throw Failure.integrity }
            let descriptor = try io.openDirectory(bundle)
            do {
                guard try readConfiguration(descriptor: descriptor) == sourceConfiguration else { throw Failure.integrity }
                try verifyMainHash(descriptor: descriptor, files: copied)
                close(descriptor)
            } catch { close(descriptor); throw error }
            let graph = try graphIdentity(copied)
            let identity = Identity(version: 1, installation: installation, identifier: CoreAITextEmbeddingPolicy.baseIdentifier,
                                    artifactDigest: graph.sha256, files: copied)
            let data = try JSONEncoder().encode(identity)
            guard data.count <= Self.maximumIdentityBytes else { throw Failure.invalidPackage }
            try Task.checkCancellation()
            try manager.moveItem(at: stage, to: destination)
            let previous = try? readIdentity()
            try data.write(to: identityURL, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
            committed = true
            if let previous, previous.installation != installation {
                try? manager.removeItem(at: directory.appendingPathComponent(previous.installation, isDirectory: true))
            }
        }

        func remove() throws {
            try Task.checkCancellation()
            guard FileManager.default.fileExists(atPath: directory.path) else { return }
            try FileManager.default.removeItem(at: directory)
        }

        private func readIdentity() throws -> Identity {
            guard isConfigured else { throw Failure.notConfigured }
            let descriptor = try io.openDirectory(directory)
            defer { close(descriptor) }
            let data = try io.smallFile("active.json", descriptor: descriptor, limit: Self.maximumIdentityBytes)
            guard let identity = try? JSONDecoder().decode(Identity.self, from: data), identity.version == 1,
                  UUID(uuidString: identity.installation)?.uuidString.lowercased() == identity.installation,
                  identity.identifier == CoreAITextEmbeddingPolicy.baseIdentifier,
                  identity.files == expectedFiles,
                  identity.artifactDigest == (try graphIdentity(expectedFiles)).sha256
            else { throw Failure.integrity }
            return identity
        }

        private static let modelPath = "model/embeddinggemma-2-text-mm.aimodel"

        private static func isImportMetadata(_ path: String) -> Bool {
            path.hasPrefix(".cache/huggingface/") || path.split(separator: "/").last == ".DS_Store"
        }

        private func graphIdentity(_ files: [FileIdentity]) throws -> FileIdentity {
            guard let graph = files.first(where: { $0.path == Self.modelPath + "/main.mlirb" }) else { throw Failure.invalidPackage }
            return graph
        }

        private func readConfiguration(descriptor: Int32) throws -> CoreAITextEmbeddingConfiguration {
            let data = try io.smallFile("model/config.json", descriptor: descriptor, limit: 64 * 1_024)
            do { return try CoreAITextEmbeddingPolicy.validateConfiguration(data) }
            catch { throw Failure.invalidPackage }
        }

        private func verifyMainHash(descriptor: Int32, files: [FileIdentity]) throws {
            let data = try io.smallFile(Self.modelPath + "/main.hash", descriptor: descriptor, limit: 32)
            guard data.count == 32, io.hex(data) == (try graphIdentity(files)).sha256 else { throw Failure.integrity }
        }
    }
}
