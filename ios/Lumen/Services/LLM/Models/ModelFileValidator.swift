import Foundation

enum ModelFileValidator {
    nonisolated static func validateExtension(for url: URL, backend: LLMBackendKind) throws {
        let fileExtension = url.pathExtension.lowercased()

        switch backend {
        case .gguf:
            guard fileExtension == "gguf" else {
                throw ModelStorageError.invalidModelFileExtension(url.lastPathComponent)
            }
        case .coreML:
            guard ["mlmodel", "mlmodelc", "mlpackage"].contains(fileExtension) else {
                throw ModelStorageError.invalidModelFileExtension(url.lastPathComponent)
            }
        case .tinyIntent, .mock, .remote:
            return
        }
    }

    nonisolated static func validateReadableFile(_ url: URL) throws {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: url.path) else {
            throw ModelStorageError.fileNotFound(url)
        }
        guard fileManager.isReadableFile(atPath: url.path) else {
            throw ModelStorageError.unreadableFile(url)
        }
    }

    static func verifyHashIfNeeded(fileURL: URL, expectedSHA256: String?) throws -> String? {
        try validateReadableFile(fileURL)

        guard let expectedSHA256, expectedSHA256.isEmpty == false else {
            return nil
        }

        let actualSHA256 = try SHA256FileHasher.sha256Hex(for: fileURL)
        guard actualSHA256.lowercased() == expectedSHA256.lowercased() else {
            throw ModelStorageError.hashMismatch(expected: expectedSHA256.lowercased(), actual: actualSHA256)
        }
        return actualSHA256
    }
}
