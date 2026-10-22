import Foundation
import Darwin

/// Owns a user-imported, completely local Gemma package. Imported hashes establish
/// byte identity for later loads; they do not certify the model's provenance.
nonisolated enum CoreAIGemmaModelStore {
    static var isEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "coreai.gemma.enabled") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "coreai.gemma.enabled") }
    }

    static func setEnabled(_ enabled: Bool) async throws {
        try await withModelReplacement {
            AppCancellationBus.shared.cancel(.chatGeneration)
            // Route new requests to the gated engine before yielding to llama cleanup.
            isEnabled = enabled
            if enabled {
                await BundledIntentClassifier.shared.unload()
                await AppLlamaService.shared.cancelActiveGeneration(reason: "coreai-engine-change")
            }
        }
    }

    @MainActor
    static func withModelReplacement<Value: Sendable>(
        begin: @MainActor () async throws -> Void = { try await CoreAIGemmaService.shared.beginModelReplacement() },
        end: @MainActor () async -> Void = { await CoreAIGemmaService.shared.endModelReplacement() },
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

    /// Imported metadata for presentation/budgeting only. This bounded read does
    /// not verify model bytes, load the engine or establish runtime readiness.
    static func configuredContextLength() async throws -> Int {
        do {
            return try await Task.detached(priority: .utility) { try storage.configuredContextLength() }.value
        } catch {
            throw sanitized(error)
        }
    }

    /// Full integrity verification belongs to the inference actor's on-demand load.
    /// Never call this synchronous, streaming check from startup or the main actor.
    static func configuredURL() throws -> URL { try storage.configuredURL() }

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
            }
        } catch {
            throw sanitized(error)
        }
    }

    private static var storage: Storage {
        Storage(directory: URL.applicationSupportDirectory.appendingPathComponent("Gemma", isDirectory: true))
    }

    private static func sanitized(_ error: Error) -> Error {
        if error is CancellationError { return error }
        return (error as? Failure) ?? Failure.storage
    }

    typealias Failure = CoreAILocalPackageIO.Failure
    typealias FileIdentity = CoreAILocalPackageIO.FileIdentity

    struct Identity: Codable, Sendable {
        let version: Int
        let installation: String
        let contextLength: Int
        let files: [FileIdentity]
    }

    /// A narrow filesystem boundary also used by deterministic, synthetic tests.
    struct Storage: Sendable {
        let directory: URL
        private static let maximumEntries = 4_096
        private static let maximumIdentityBytes = 1_024 * 1_024
        private let maximumPackageBytes: Int64
        init(directory: URL, maximumPackageBytes: Int64 = 32 * 1_024 * 1_024 * 1_024) {
            self.directory = directory
            self.maximumPackageBytes = min(max(0, maximumPackageBytes), 32 * 1_024 * 1_024 * 1_024)
        }
        private var io: CoreAILocalPackageIO { .init(maximumPackageBytes: maximumPackageBytes) }
        private var identityURL: URL { directory.appendingPathComponent("active.json") }
        var isConfigured: Bool { FileManager.default.fileExists(atPath: identityURL.path) }

        func configuredContextLength() throws -> Int {
            guard isConfigured else { throw Failure.notConfigured }
            return try readIdentity().contextLength
        }

        func configuredURL() throws -> URL {
            guard isConfigured else { throw Failure.notConfigured }
            let identity = try readIdentity()
            let bundle = directory.appendingPathComponent(identity.installation, isDirectory: true)
                .appendingPathComponent("bundle", isDirectory: true)
            let owned = try io.openDirectory(directory)
            defer { close(owned) }
            let installation = openat(owned, identity.installation, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard installation >= 0 else { throw Failure.integrity }
            defer { close(installation) }
            let descriptor = openat(installation, "bundle", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else { throw Failure.integrity }
            defer { close(descriptor) }
            let paths = try io.inventory(bundle)
            guard paths == identity.files.map(\.path), !paths.isEmpty else { throw Failure.integrity }
            var verifiedBytes: Int64 = 0
            for expected in identity.files {
                try Task.checkCancellation()
                guard try io.digest(relative: expected.path, descriptor: descriptor, expectedBytes: expected.bytes,
                                    maximumBytes: maximumPackageBytes - verifiedBytes) == expected else { throw Failure.integrity }
                verifiedBytes += expected.bytes
            }
            let metadata = try readMetadata(descriptor: descriptor)
            guard metadata.language.max_context_length == identity.contextLength else { throw Failure.integrity }
            try validateRequiredFiles(metadata: metadata, paths: Set(paths))
            return bundle
        }

        func install(source: URL) throws {
            guard source.isFileURL else { throw Failure.localDirectory }
            let sourceDescriptor = try io.openDirectory(source)
            defer { close(sourceDescriptor) }
            let paths = try io.inventory(source)
            let metadata = try readMetadata(descriptor: sourceDescriptor)
            try validateRequiredFiles(metadata: metadata, paths: Set(paths))
            let installation = UUID().uuidString.lowercased()
            var sourceBytes: Int64 = 0
            var plannedFiles: [FileIdentity] = []
            // Hex digests always encode to the same 64 ASCII bytes. These zeros
            // are used only to measure the identity, never persisted or trusted.
            let measuredDigest = String(repeating: "0", count: 64)
            for path in paths {
                let bytes = try io.fileSize(relative: path, descriptor: sourceDescriptor)
                guard bytes <= maximumPackageBytes - sourceBytes else { throw Failure.invalidPackage }
                sourceBytes += bytes
                plannedFiles.append(FileIdentity(path: path, bytes: bytes, sha256: measuredDigest))
            }
            let projectedIdentity = Identity(version: 1, installation: installation,
                                             contextLength: metadata.language.max_context_length, files: plannedFiles)
            let projectedIdentityBytes = try autoreleasepool { try JSONEncoder().encode(projectedIdentity).count }
            try Task.checkCancellation()
            guard projectedIdentityBytes <= Self.maximumIdentityBytes else { throw Failure.invalidPackage }
            let manager = FileManager.default
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
            try CoreAIModelInstallationRecovery.cleanup(directory: directory, activeInstallation: (try? readIdentity())?.installation)
            var root = directory
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try root.setResourceValues(values)
            let stage = directory.appendingPathComponent(".import-" + installation, isDirectory: true)
            let destination = directory.appendingPathComponent(installation, isDirectory: true)
            let bundle = stage.appendingPathComponent("bundle", isDirectory: true)
            var committed = false
            defer {
                try? manager.removeItem(at: stage)
                if !committed { try? manager.removeItem(at: destination) }
            }
            try manager.createDirectory(at: bundle, withIntermediateDirectories: true)
            var files: [FileIdentity] = []
            var total: Int64 = 0
            for expected in plannedFiles {
                try Task.checkCancellation()
                let output = bundle.appendingPathComponent(expected.path)
                try manager.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                let file = try io.digest(relative: expected.path, descriptor: sourceDescriptor, copyTo: output,
                                         expectedBytes: expected.bytes, maximumBytes: maximumPackageBytes - total)
                total += file.bytes
                files.append(file)
            }
            // Detect source-directory changes during the copy. All bytes above were
            // opened relative to the original root descriptor with no-follow opens.
            guard try io.inventory(source) == paths else { throw Failure.integrity }
            let copiedDescriptor = try io.openDirectory(bundle)
            let copiedMetadata: Metadata
            do {
                copiedMetadata = try readMetadata(descriptor: copiedDescriptor)
                close(copiedDescriptor)
            } catch {
                close(copiedDescriptor)
                throw error
            }
            guard copiedMetadata == metadata else { throw Failure.integrity }
            try verifyStoredMainHash(metadata: copiedMetadata, root: bundle, files: files)
            let identity = Identity(version: 1, installation: installation,
                                    contextLength: metadata.language.max_context_length, files: files)
            let data = try JSONEncoder().encode(identity)
            guard data.count <= Self.maximumIdentityBytes else { throw Failure.invalidPackage }
            try Task.checkCancellation()
            try manager.moveItem(at: stage, to: destination)
            let previous = try? readIdentity()
            // Atomic pointer publication retains the previous installation until
            // the copied package and all identities are complete inputs.
            try data.write(to: identityURL, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
            committed = true
            if let previous, previous.installation != installation {
                try? manager.removeItem(at: directory.appendingPathComponent(previous.installation, isDirectory: true))
            }
        }

        func remove() throws {
            try Task.checkCancellation()
            guard FileManager.default.fileExists(atPath: directory.path) else { return }
            // The entire directory is app-owned. Clearing it also recovers from a
            // corrupt pointer or an interrupted import without trusting its paths.
            try FileManager.default.removeItem(at: directory)
        }

        private func readIdentity() throws -> Identity {
            let descriptor = try io.openDirectory(directory)
            defer { close(descriptor) }
            let data = try io.smallFile("active.json", descriptor: descriptor, limit: Self.maximumIdentityBytes)
            guard let identity = try? JSONDecoder().decode(Identity.self, from: data), identity.version == 1,
                  UUID(uuidString: identity.installation)?.uuidString.lowercased() == identity.installation,
                  [8_192, 32_768].contains(identity.contextLength), !identity.files.isEmpty,
                  identity.files.count <= Self.maximumEntries,
                  identity.files.map(\.path) == identity.files.map(\.path).sorted(),
                  Set(identity.files.map(\.path)).count == identity.files.count,
                  identity.files.allSatisfy({ io.validRelativePath($0.path) && $0.bytes >= 0 && $0.bytes <= maximumPackageBytes &&
                      $0.sha256.count == 64 && $0.sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) })
            else { throw Failure.integrity }
            guard identity.files.reduce(Int64(0), { $0 + $1.bytes }) <= maximumPackageBytes else { throw Failure.integrity }
            return identity
        }

        private struct Metadata: Decodable, Equatable {
            struct Language: Decodable, Equatable {
                let tokenizer: String
                let max_context_length: Int
                let embedded_tokenizer: Bool?
            }
            let kind: String
            let assets: [String: String]
            let auxiliary_assets: [String: String]
            let language: Language
        }

        private func readMetadata(descriptor: Int32) throws -> Metadata {
            let data = try io.smallFile("metadata.json", descriptor: descriptor, limit: 256 * 1_024)
            guard let metadata = try? JSONDecoder().decode(Metadata.self, from: data), metadata.kind == "llm",
                  metadata.language.tokenizer == "google/gemma-4-E2B-it", metadata.language.embedded_tokenizer != false,
                  [8_192, 32_768].contains(metadata.language.max_context_length),
                  Set(metadata.assets.keys) == ["main"], Set(metadata.auxiliary_assets.keys) == ["per_layer_embeddings"],
                  metadata.assets.values.allSatisfy(io.validRelativePath), metadata.auxiliary_assets.values.allSatisfy(io.validRelativePath),
                  metadata.assets["main"]?.hasSuffix(".aimodel") == true
            else { throw Failure.invalidPackage }
            return metadata
        }

        private func validateRequiredFiles(metadata: Metadata, paths: Set<String>) throws {
            guard let model = metadata.assets["main"], let ple = metadata.auxiliary_assets["per_layer_embeddings"],
                  Set(["metadata.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja",
                       model + "/metadata.json", model + "/main.mlirb", model + "/main.hash", ple]).isSubset(of: paths)
            else { throw Failure.invalidPackage }
        }

        private func verifyStoredMainHash(metadata: Metadata, root: URL, files: [FileIdentity]) throws {
            guard let model = metadata.assets["main"], let bytecode = files.first(where: { $0.path == model + "/main.mlirb" })
            else { throw Failure.invalidPackage }
            let descriptor = try io.openDirectory(root)
            defer { close(descriptor) }
            let hash = try io.smallFile(model + "/main.hash", descriptor: descriptor, limit: 32)
            guard hash.count == 32, io.hex(hash) == bytecode.sha256 else { throw Failure.integrity }
        }
    }
}
