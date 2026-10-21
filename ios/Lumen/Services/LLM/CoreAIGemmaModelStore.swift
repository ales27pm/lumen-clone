import CryptoKit
import Foundation
import Darwin
#if canImport(FileProvider)
import FileProvider
#endif

/// Owns a user-imported, completely local Gemma package. Imported hashes establish
/// byte identity for later loads; they do not certify the model's provenance.
nonisolated enum CoreAIGemmaModelStore {
    static var isEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "coreai.gemma.enabled") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "coreai.gemma.enabled") }
    }

    static func setEnabled(_ enabled: Bool) async throws {
        try await CoreAIGemmaService.shared.beginModelReplacement()
        AppCancellationBus.shared.cancel(.chatGeneration)
        // Route new requests to the gated engine before yielding to llama cleanup.
        isEnabled = enabled
        if enabled {
            await BundledIntentClassifier.shared.unload()
            await AppLlamaService.shared.cancelActiveGeneration(reason: "coreai-engine-change")
        }
        await CoreAIGemmaService.shared.endModelReplacement()
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
        try await CoreAIGemmaService.shared.beginModelReplacement()
        do {
            let worker = Task.detached(priority: .utility) {
                let accessed = source.startAccessingSecurityScopedResource()
                defer { if accessed { source.stopAccessingSecurityScopedResource() } }
                try storage.install(source: source)
            }
            try await withTaskCancellationHandler {
                try await worker.value
            } onCancel: {
                worker.cancel()
            }
            await CoreAIGemmaService.shared.endModelReplacement()
        } catch {
            await CoreAIGemmaService.shared.endModelReplacement()
            throw sanitized(error)
        }
    }

    static func remove() async throws {
        try await CoreAIGemmaService.shared.beginModelReplacement()
        do {
            try await Task.detached(priority: .utility) { try storage.remove() }.value
            await CoreAIGemmaService.shared.endModelReplacement()
        } catch {
            await CoreAIGemmaService.shared.endModelReplacement()
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

    enum Failure: Error, LocalizedError, Equatable {
        case localDirectory, invalidPackage, integrity, notConfigured, storage

        var errorDescription: String? {
            switch self {
            case .localDirectory: return "Choose a fully downloaded local folder without symbolic links."
            case .invalidPackage: return "A complete Gemma 4 E2B Core AI text export with its local tokenizer and 8K or 32K context is required."
            case .integrity: return "The imported model changed or is incomplete. Import the original package again."
            case .notConfigured: return "Import a local Gemma Core AI model in Settings before starting a conversation."
            case .storage: return "The local Gemma model could not be saved or removed. Check available storage and try again."
            }
        }
    }

    struct FileIdentity: Codable, Sendable, Equatable {
        let path: String
        let bytes: Int64
        let sha256: String
    }

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
        private static let maximumPackageBytes: Int64 = 32 * 1_024 * 1_024 * 1_024
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
            let owned = try openDirectory(directory)
            defer { close(owned) }
            let installation = openat(owned, identity.installation, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard installation >= 0 else { throw Failure.integrity }
            defer { close(installation) }
            let descriptor = openat(installation, "bundle", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else { throw Failure.integrity }
            defer { close(descriptor) }
            let paths = try inventory(bundle)
            guard paths == identity.files.map(\.path), !paths.isEmpty else { throw Failure.integrity }
            for expected in identity.files {
                try Task.checkCancellation()
                guard try digest(relative: expected.path, descriptor: descriptor) == expected else { throw Failure.integrity }
            }
            let metadata = try readMetadata(descriptor: descriptor)
            guard metadata.language.max_context_length == identity.contextLength else { throw Failure.integrity }
            try validateRequiredFiles(metadata: metadata, paths: Set(paths))
            return bundle
        }

        func install(source: URL) throws {
            guard source.isFileURL else { throw Failure.localDirectory }
            let sourceDescriptor = try openDirectory(source)
            defer { close(sourceDescriptor) }
            let paths = try inventory(source)
            let metadata = try readMetadata(descriptor: sourceDescriptor)
            try validateRequiredFiles(metadata: metadata, paths: Set(paths))
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
            var files: [FileIdentity] = []
            var total: Int64 = 0
            for path in paths {
                try Task.checkCancellation()
                let output = bundle.appendingPathComponent(path)
                try manager.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                let file = try digest(relative: path, descriptor: sourceDescriptor, copyTo: output)
                total += file.bytes
                guard total <= Self.maximumPackageBytes else { throw Failure.invalidPackage }
                files.append(file)
            }
            // Detect source-directory changes during the copy. All bytes above were
            // opened relative to the original root descriptor with no-follow opens.
            guard try inventory(source) == paths else { throw Failure.integrity }
            let copiedDescriptor = try openDirectory(bundle)
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
            guard FileManager.default.fileExists(atPath: directory.path) else { return }
            // The entire directory is app-owned. Clearing it also recovers from a
            // corrupt pointer or an interrupted import without trusting its paths.
            try FileManager.default.removeItem(at: directory)
        }

        private func readIdentity() throws -> Identity {
            let descriptor = try openDirectory(directory)
            defer { close(descriptor) }
            let data = try smallFile("active.json", descriptor: descriptor, limit: Self.maximumIdentityBytes)
            guard let identity = try? JSONDecoder().decode(Identity.self, from: data), identity.version == 1,
                  UUID(uuidString: identity.installation)?.uuidString.lowercased() == identity.installation,
                  [8_192, 32_768].contains(identity.contextLength), !identity.files.isEmpty,
                  identity.files.count <= Self.maximumEntries,
                  identity.files.map(\.path) == identity.files.map(\.path).sorted(),
                  Set(identity.files.map(\.path)).count == identity.files.count,
                  identity.files.allSatisfy({ validRelativePath($0.path) && $0.bytes >= 0 && $0.bytes <= Self.maximumPackageBytes &&
                      $0.sha256.count == 64 && $0.sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) })
            else { throw Failure.integrity }
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
            let data = try smallFile("metadata.json", descriptor: descriptor, limit: 256 * 1_024)
            guard let metadata = try? JSONDecoder().decode(Metadata.self, from: data), metadata.kind == "llm",
                  metadata.language.tokenizer == "google/gemma-4-E2B-it", metadata.language.embedded_tokenizer != false,
                  [8_192, 32_768].contains(metadata.language.max_context_length),
                  Set(metadata.assets.keys) == ["main"], Set(metadata.auxiliary_assets.keys) == ["per_layer_embeddings"],
                  metadata.assets.values.allSatisfy(validRelativePath), metadata.auxiliary_assets.values.allSatisfy(validRelativePath),
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
            let descriptor = try openDirectory(root)
            defer { close(descriptor) }
            let hash = try smallFile(model + "/main.hash", descriptor: descriptor, limit: 32)
            guard hash.count == 32, hex(hash) == bytecode.sha256 else { throw Failure.integrity }
        }

        private func inventory(_ root: URL) throws -> [String] {
            let keys: Set<URLResourceKey> = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
                                           .isReadableKey, .isUbiquitousItemKey, .ubiquitousItemDownloadingStatusKey,
                                           .ubiquitousItemIsDownloadingKey]
            var pending = [root]
            var result: [String] = []
            var visited = 0
            while let url = pending.popLast() {
                try Task.checkCancellation()
                visited += 1
                guard visited <= Self.maximumEntries else { throw Failure.invalidPackage }
                var status = stat()
                guard lstat(url.path, &status) == 0, status.st_flags & UInt32(SF_DATALESS) == 0 else {
                    throw Failure.localDirectory
                }
                #if canImport(FileProvider)
                guard !FileManager.default.fileExists(atPath: NSFileProviderManager.placeholderURL(for: url).path) else {
                    throw Failure.localDirectory
                }
                #endif
                let values = try url.resourceValues(forKeys: keys)
                guard values.isSymbolicLink == false, values.isReadable == true,
                      values.ubiquitousItemIsDownloading != true,
                      values.isUbiquitousItem != true,
                      values.ubiquitousItemDownloadingStatus == nil || values.ubiquitousItemDownloadingStatus == .current ||
                        values.ubiquitousItemDownloadingStatus == .downloaded
                else { throw Failure.localDirectory }
                if values.isDirectory == true {
                    pending.append(contentsOf: try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: Array(keys)))
                } else {
                    guard values.isRegularFile == true, url.path.hasPrefix(root.path + "/") else { throw Failure.localDirectory }
                    let relative = String(url.path.dropFirst(root.path.count + 1))
                    guard validRelativePath(relative) else { throw Failure.localDirectory }
                    result.append(relative)
                }
            }
            return result.sorted()
        }

        private func openDirectory(_ url: URL) throws -> Int32 {
            guard url.isFileURL else { throw Failure.localDirectory }
            var status = stat()
            guard lstat(url.path, &status) == 0, UInt32(status.st_mode) & UInt32(S_IFMT) == UInt32(S_IFDIR),
                  status.st_flags & UInt32(SF_DATALESS) == 0 else { throw Failure.localDirectory }
            let descriptor = Darwin.open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else { throw Failure.localDirectory }
            return descriptor
        }

        private func openFile(_ relative: String, descriptor: Int32) throws -> Int32 {
            guard validRelativePath(relative) else { throw Failure.invalidPackage }
            var parent = dup(descriptor)
            guard parent >= 0 else { throw Failure.localDirectory }
            defer { close(parent) }
            let pieces = relative.split(separator: "/").map(String.init)
            for component in pieces.dropLast() {
                let next = openat(parent, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard next >= 0 else { throw Failure.localDirectory }
                close(parent)
                parent = next
            }
            var entry = stat()
            guard fstatat(parent, pieces[pieces.count - 1], &entry, AT_SYMLINK_NOFOLLOW) == 0,
                  UInt32(entry.st_mode) & UInt32(S_IFMT) == UInt32(S_IFREG), entry.st_flags & UInt32(SF_DATALESS) == 0 else {
                throw Failure.localDirectory
            }
            let result = openat(parent, pieces[pieces.count - 1], O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
            guard result >= 0 else { throw Failure.localDirectory }
            var information = stat()
            guard fstat(result, &information) == 0, UInt32(information.st_mode) & UInt32(S_IFMT) == UInt32(S_IFREG),
                  information.st_flags & UInt32(SF_DATALESS) == 0,
                  information.st_size >= 0, information.st_size <= Self.maximumPackageBytes else {
                close(result)
                throw Failure.localDirectory
            }
            return result
        }

        private func smallFile(_ path: String, descriptor: Int32, limit: Int) throws -> Data {
            let file = try openFile(path, descriptor: descriptor)
            let handle = FileHandle(fileDescriptor: file, closeOnDealloc: true)
            defer { try? handle.close() }
            guard let data = try handle.read(upToCount: limit + 1), data.count <= limit else { throw Failure.invalidPackage }
            return data
        }

        private func digest(relative: String, descriptor: Int32, copyTo output: URL? = nil) throws -> FileIdentity {
            let file = try openFile(relative, descriptor: descriptor)
            let input = FileHandle(fileDescriptor: file, closeOnDealloc: true)
            defer { try? input.close() }
            var before = stat()
            guard fstat(file, &before) == 0 else { throw Failure.integrity }
            let destination: FileHandle?
            if let output {
                guard FileManager.default.createFile(atPath: output.path, contents: nil,
                    attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication]) else { throw Failure.storage }
                destination = try FileHandle(forWritingTo: output)
            } else { destination = nil }
            defer { try? destination?.close() }
            var hasher = SHA256()
            var count: Int64 = 0
            while let data = try input.read(upToCount: 1_024 * 1_024), !data.isEmpty {
                try Task.checkCancellation()
                count += Int64(data.count)
                guard count <= before.st_size else { throw Failure.integrity }
                hasher.update(data: data)
                try destination?.write(contentsOf: data)
            }
            try destination?.synchronize()
            var after = stat()
            guard fstat(file, &after) == 0, count == before.st_size, after.st_size == before.st_size,
                  after.st_ino == before.st_ino, after.st_mtimespec.tv_sec == before.st_mtimespec.tv_sec,
                  after.st_mtimespec.tv_nsec == before.st_mtimespec.tv_nsec,
                  after.st_ctimespec.tv_sec == before.st_ctimespec.tv_sec,
                  after.st_ctimespec.tv_nsec == before.st_ctimespec.tv_nsec
            else { throw Failure.integrity }
            return FileIdentity(path: relative, bytes: count, sha256: hex(Data(hasher.finalize())))
        }

        private func validRelativePath(_ path: String) -> Bool {
            !path.isEmpty && path.utf8.count <= 1_024 && !path.hasPrefix("/") && !path.contains("\0") &&
                path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        }

        private func hex(_ data: Data) -> String { data.map { String(format: "%02x", $0) }.joined() }
    }
}
