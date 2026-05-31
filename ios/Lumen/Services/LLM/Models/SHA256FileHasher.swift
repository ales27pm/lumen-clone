import CryptoKit
import Foundation

enum SHA256FileHasher {
    private static let chunkSize = 4 * 1_024 * 1_024

    static func sha256Hex(for fileURL: URL) throws -> String {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: fileURL.path) else {
            throw ModelStorageError.fileNotFound(fileURL)
        }
        guard fileManager.isReadableFile(atPath: fileURL.path) else {
            throw ModelStorageError.unreadableFile(fileURL)
        }

        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: fileURL)
        } catch {
            throw ModelStorageError.unreadableFile(fileURL)
        }
        defer {
            try? handle.close()
        }

        var hasher = SHA256()
        do {
            while true {
                guard let data = try handle.read(upToCount: chunkSize), !data.isEmpty else {
                    break
                }
                hasher.update(data: data)
            }
        } catch {
            throw ModelStorageError.unreadableFile(fileURL)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
