import Foundation

nonisolated enum ModelFileIntegrity {
    nonisolated enum Failure: LocalizedError, Sendable, Equatable {
        case fileMissing(String)
        case fileTooSmall(actual: Int64, minimum: Int64)
        case sizeMismatch(actual: Int64, expected: Int64)
        case invalidGGUFMagic(String)
        case missingExpectedSHA256(String)
        case hashMismatch(expected: String, actual: String)
        case cancelled
        case unreadable(String)

        var errorDescription: String? {
            switch self {
            case .fileMissing:
                return "Model file is missing."
            case .fileTooSmall(let actual, let minimum):
                return "Model file is too small: \(ByteCountFormatter.string(fromByteCount: actual, countStyle: .file)); expected at least \(ByteCountFormatter.string(fromByteCount: minimum, countStyle: .file))."
            case .sizeMismatch(let actual, let expected):
                return "Model file size does not match the verified artifact: \(ByteCountFormatter.string(fromByteCount: actual, countStyle: .file)); expected \(ByteCountFormatter.string(fromByteCount: expected, countStyle: .file))."
            case .invalidGGUFMagic:
                return "Downloaded file is not a GGUF model."
            case .missingExpectedSHA256:
                return "Model integrity metadata is missing."
            case .hashMismatch:
                return "Model file failed SHA-256 verification."
            case .cancelled:
                return "Model integrity verification was cancelled."
            case .unreadable:
                return "Model file is unreadable."
            }
        }

        var diagnosticCode: String {
            switch self {
            case .fileMissing(let path):
                return "file_missing:path_sha256=\(Self.pathHash(path))"
            case .fileTooSmall(let actual, let minimum):
                return "file_too_small:actual=\(actual);minimum=\(minimum)"
            case .sizeMismatch(let actual, let expected):
                return "file_size_mismatch:actual=\(actual);expected=\(expected)"
            case .invalidGGUFMagic(let path):
                return "invalid_gguf_magic:path_sha256=\(Self.pathHash(path))"
            case .missingExpectedSHA256(let modelID):
                return "missing_expected_sha256:model_id=\(modelID)"
            case .hashMismatch(let expected, let actual):
                return "sha256_mismatch:expected=\(expected.prefix(16));actual=\(actual.prefix(16))"
            case .cancelled:
                return "integrity_verification_cancelled"
            case .unreadable(let path):
                return "file_unreadable:path_sha256=\(Self.pathHash(path))"
            }
        }

        private static func pathHash(_ path: String) -> String {
            String(RuntimeFallbackLogger.promptHash(path).prefix(16))
        }
    }

    private static let absoluteMinimumBytes: Int64 = 16 * 1024 * 1024
    private static let verificationCache = VerificationCache()

    static func validateDownloadedFile(at url: URL, expectedFileName: String, expectedSizeBytes: Int64) -> Result<Int64, Failure> {
        validateFile(at: url, expectedFileName: expectedFileName, expectedSizeBytes: expectedSizeBytes, expectedSHA256: nil, strictSize: true)
    }

    static func validateInstalledFile(localPath: String, fileName: String, expectedSizeBytes: Int64) -> Bool {
        if case .success = validateInstalledFileWithDiagnostics(localPath: localPath, fileName: fileName, expectedSizeBytes: expectedSizeBytes) {
            return true
        }
        return false
    }

    static func validateInstalledFileWithDiagnostics(localPath: String, fileName: String, expectedSizeBytes: Int64) -> Result<Int64, Failure> {
        let url = ModelStorage.resolvedModelURL(from: localPath, fileName: fileName)
        return validateFile(at: url, expectedFileName: fileName, expectedSizeBytes: expectedSizeBytes, expectedSHA256: nil, strictSize: false)
    }

    static func validateInstalledFile(_ model: StoredModel) -> Bool {
        validateInstalledFile(localPath: model.localPath, fileName: model.fileName, expectedSizeBytes: model.sizeBytes)
    }

    static func validateInstalledFileWithDiagnostics(_ model: StoredModel) -> Result<Int64, Failure> {
        let catalog = ModelCatalog.catalogModel(repoId: model.repoId, fileName: model.fileName)
        let url = ModelStorage.resolvedModelURL(from: model.localPath, fileName: model.fileName)
        return validateFile(
            at: url,
            expectedFileName: model.fileName,
            expectedSizeBytes: catalog?.sizeBytes ?? model.sizeBytes,
            expectedSHA256: catalog?.expectedSHA256,
            strictSize: false
        )
    }

    static func validateDownloadedCatalogFile(_ model: CatalogModel, at url: URL) -> Result<Int64, Failure> {
        validateDownloadedCatalogFile(model, at: url, checkingCancellation: false)
    }

    static func validateDownloadedCatalogFileAsync(_ model: CatalogModel, at url: URL) async -> Result<Int64, Failure> {
        let worker = Task.detached(priority: .utility) {
            validateDownloadedCatalogFile(model, at: url, checkingCancellation: true)
        }
        return await withTaskCancellationHandler {
            await worker.value
        } onCancel: {
            worker.cancel()
        }
    }

    static func validateInstalledFileWithDiagnosticsAsync(
        localPath: String,
        fileName: String,
        expectedSizeBytes: Int64,
        expectedSHA256: String?
    ) async -> Result<Int64, Failure> {
        let url = ModelStorage.resolvedModelURL(from: localPath, fileName: fileName)
        let worker = Task.detached(priority: .utility) {
            validateFile(
                at: url,
                expectedFileName: fileName,
                expectedSizeBytes: expectedSizeBytes,
                expectedSHA256: expectedSHA256,
                strictSize: false,
                checkingCancellation: true
            )
        }
        return await withTaskCancellationHandler {
            await worker.value
        } onCancel: {
            worker.cancel()
        }
    }

    static func validateInstalledFileWithDiagnosticsAsync(
        _ model: StoredModelLoadItem
    ) async -> Result<Int64, Failure> {
        await validateInstalledFileWithDiagnosticsAsync(
            localPath: model.resolvedPath,
            fileName: model.fileName,
            expectedSizeBytes: model.sizeBytes,
            expectedSHA256: model.expectedSHA256
        )
    }

    private static func validateDownloadedCatalogFile(
        _ model: CatalogModel,
        at url: URL,
        checkingCancellation: Bool
    ) -> Result<Int64, Failure> {
        guard CatalogModel.isValidSHA256(model.expectedSHA256) else {
            return .failure(.missingExpectedSHA256(model.id))
        }
        return validateFile(
            at: url,
            expectedFileName: model.fileName,
            expectedSizeBytes: model.sizeBytes,
            expectedSHA256: model.expectedSHA256,
            strictSize: true,
            checkingCancellation: checkingCancellation
        )
    }

    private static func validateFile(
        at url: URL,
        expectedFileName: String,
        expectedSizeBytes: Int64,
        expectedSHA256: String?,
        strictSize: Bool,
        checkingCancellation: Bool = false
    ) -> Result<Int64, Failure> {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return .failure(.fileMissing(url.path)) }

        let attrs: [FileAttributeKey: Any]
        do {
            attrs = try fm.attributesOfItem(atPath: url.path)
        } catch {
            return .failure(.unreadable(url.path))
        }

        let actualSize = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        if expectedSHA256 != nil, expectedSizeBytes > 0 {
            guard actualSize == expectedSizeBytes else {
                return .failure(.sizeMismatch(actual: actualSize, expected: expectedSizeBytes))
            }
        } else {
            let minimum = minimumAcceptableSize(expectedSizeBytes: expectedSizeBytes, strict: strictSize)
            guard actualSize >= minimum else {
                return .failure(.fileTooSmall(actual: actualSize, minimum: minimum))
            }
        }

        if expectedFileName.lowercased().hasSuffix(".gguf") || url.lastPathComponent.lowercased().hasSuffix(".gguf") {
            switch validateGGUFMagic(url) {
            case .success:
                break
            case .failure(let failure):
                return .failure(failure)
            }
        }

        if let expectedSHA256 {
            let normalizedExpected = expectedSHA256.lowercased()
            let fingerprint = VerificationFingerprint(
                sizeBytes: actualSize,
                modificationTimestamp: (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0,
                fileSystemNumber: (attrs[.systemNumber] as? NSNumber)?.uint64Value ?? 0,
                fileNumber: (attrs[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0,
                expectedSHA256: normalizedExpected
            )
            if verificationCache.contains(fingerprint) {
                return .success(actualSize)
            }

            let actualSHA256: String
            do {
                actualSHA256 = try SHA256FileHasher.sha256Hex(
                    for: url,
                    checkingCancellation: checkingCancellation
                ).lowercased()
            } catch is CancellationError {
                return .failure(.cancelled)
            } catch {
                return .failure(.unreadable(url.path))
            }
            guard actualSHA256 == normalizedExpected else {
                return .failure(.hashMismatch(expected: normalizedExpected, actual: actualSHA256))
            }
            verificationCache.insert(fingerprint)
        }

        return .success(actualSize)
    }

    private static func minimumAcceptableSize(expectedSizeBytes: Int64, strict: Bool) -> Int64 {
        guard expectedSizeBytes > 0 else { return absoluteMinimumBytes }
        if strict {
            return max(absoluteMinimumBytes, Int64(Double(expectedSizeBytes) * 0.75))
        }
        return max(absoluteMinimumBytes, Int64(Double(expectedSizeBytes) * 0.25))
    }

    private static func validateGGUFMagic(_ url: URL) -> Result<Void, Failure> {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            return .failure(.unreadable(url.path))
        }
        defer { try? handle.close() }

        do {
            guard let data = try handle.read(upToCount: 4), data.count == 4 else {
                return .failure(.unreadable(url.path))
            }
            guard data == Data([0x47, 0x47, 0x55, 0x46]) else {
                return .failure(.invalidGGUFMagic(url.path))
            }
            return .success(())
        } catch {
            return .failure(.unreadable(url.path))
        }
    }
}

private struct VerificationFingerprint: Hashable {
    let sizeBytes: Int64
    let modificationTimestamp: TimeInterval
    let fileSystemNumber: UInt64
    let fileNumber: UInt64
    let expectedSHA256: String
}

private final class VerificationCache: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: Set<VerificationFingerprint> = []

    func contains(_ fingerprint: VerificationFingerprint) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return entries.contains(fingerprint)
    }

    func insert(_ fingerprint: VerificationFingerprint) {
        lock.lock()
        entries.insert(fingerprint)
        lock.unlock()
    }
}
