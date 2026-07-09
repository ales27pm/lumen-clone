import Foundation

nonisolated enum ModelFileIntegrity {
    nonisolated enum Failure: LocalizedError, Sendable, Equatable {
        case fileMissing(String)
        case fileTooSmall(actual: Int64, minimum: Int64)
        case invalidGGUFMagic(String)
        case unreadable(String)

        var errorDescription: String? {
            switch self {
            case .fileMissing:
                return "Model file is missing."
            case .fileTooSmall(let actual, let minimum):
                return "Model file is too small: \(ByteCountFormatter.string(fromByteCount: actual, countStyle: .file)); expected at least \(ByteCountFormatter.string(fromByteCount: minimum, countStyle: .file))."
            case .invalidGGUFMagic:
                return "Downloaded file is not a GGUF model."
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
            case .invalidGGUFMagic(let path):
                return "invalid_gguf_magic:path_sha256=\(Self.pathHash(path))"
            case .unreadable(let path):
                return "file_unreadable:path_sha256=\(Self.pathHash(path))"
            }
        }

        private static func pathHash(_ path: String) -> String {
            String(RuntimeFallbackLogger.promptHash(path).prefix(16))
        }
    }

    private static let absoluteMinimumBytes: Int64 = 16 * 1024 * 1024

    static func validateDownloadedFile(at url: URL, expectedFileName: String, expectedSizeBytes: Int64) -> Result<Int64, Failure> {
        validateFile(at: url, expectedFileName: expectedFileName, expectedSizeBytes: expectedSizeBytes, strictSize: true)
    }

    static func validateInstalledFile(localPath: String, fileName: String, expectedSizeBytes: Int64) -> Bool {
        if case .success = validateInstalledFileWithDiagnostics(localPath: localPath, fileName: fileName, expectedSizeBytes: expectedSizeBytes) {
            return true
        }
        return false
    }

    static func validateInstalledFileWithDiagnostics(localPath: String, fileName: String, expectedSizeBytes: Int64) -> Result<Int64, Failure> {
        let url = ModelStorage.resolvedModelURL(from: localPath, fileName: fileName)
        return validateFile(at: url, expectedFileName: fileName, expectedSizeBytes: expectedSizeBytes, strictSize: false)
    }

    static func validateInstalledFile(_ model: StoredModel) -> Bool {
        validateInstalledFile(localPath: model.localPath, fileName: model.fileName, expectedSizeBytes: model.sizeBytes)
    }

    static func validateInstalledFileWithDiagnostics(_ model: StoredModel) -> Result<Int64, Failure> {
        validateInstalledFileWithDiagnostics(localPath: model.localPath, fileName: model.fileName, expectedSizeBytes: model.sizeBytes)
    }

    static func validateDownloadedCatalogFile(_ model: CatalogModel, at url: URL) -> Result<Int64, Failure> {
        validateDownloadedFile(at: url, expectedFileName: model.fileName, expectedSizeBytes: model.sizeBytes)
    }

    private static func validateFile(at url: URL, expectedFileName: String, expectedSizeBytes: Int64, strictSize: Bool) -> Result<Int64, Failure> {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return .failure(.fileMissing(url.path)) }

        let attrs: [FileAttributeKey: Any]
        do {
            attrs = try fm.attributesOfItem(atPath: url.path)
        } catch {
            return .failure(.unreadable(url.path))
        }

        let actualSize = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        let minimum = minimumAcceptableSize(expectedSizeBytes: expectedSizeBytes, strict: strictSize)
        guard actualSize >= minimum else {
            return .failure(.fileTooSmall(actual: actualSize, minimum: minimum))
        }

        if expectedFileName.lowercased().hasSuffix(".gguf") || url.lastPathComponent.lowercased().hasSuffix(".gguf") {
            switch validateGGUFMagic(url) {
            case .success:
                break
            case .failure(let failure):
                return .failure(failure)
            }
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
