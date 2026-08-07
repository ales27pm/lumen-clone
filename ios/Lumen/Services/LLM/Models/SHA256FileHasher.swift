import CryptoKit
import Foundation

enum SHA256FileHasher {
    private static let chunkSize = 4 * 1_024 * 1_024

    static func sha256Hex(forArtifactAt artifactURL: URL) throws -> String {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: artifactURL.path, isDirectory: &isDirectory) else {
            throw ModelStorageError.fileNotFound(artifactURL)
        }
        guard isDirectory.boolValue else {
            return try sha256Hex(for: artifactURL)
        }

        let resourceKeys: [URLResourceKey] = [.isRegularFileKey]
        var unreadableEntryURL: URL?
        guard let enumerator = fileManager.enumerator(
            at: artifactURL,
            includingPropertiesForKeys: resourceKeys,
            options: [],
            errorHandler: { url, _ in
                unreadableEntryURL = url
                return false
            }
        ) else {
            throw ModelStorageError.unreadableFile(artifactURL)
        }

        var entries: [(relativePath: String, digest: String)] = []
        for case let fileURL as URL in enumerator {
            let values: URLResourceValues
            do {
                values = try fileURL.resourceValues(forKeys: Set(resourceKeys))
            } catch {
                throw ModelStorageError.unreadableFile(fileURL)
            }
            guard values.isRegularFile == true else { continue }
            let relativePath = String(fileURL.path.dropFirst(artifactURL.path.count + 1))
            entries.append((relativePath, try sha256Hex(for: fileURL)))
        }
        if let unreadableEntryURL {
            throw ModelStorageError.unreadableFile(unreadableEntryURL)
        }

        var hasher = SHA256()
        for entry in entries.sorted(by: { $0.relativePath < $1.relativePath }) {
            update(&hasher, withLengthPrefixedUTF8: entry.relativePath)
            update(&hasher, withLengthPrefixedUTF8: entry.digest)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Hex(for fileURL: URL, checkingCancellation: Bool = false) throws -> String {
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
                if checkingCancellation {
                    try Task.checkCancellation()
                }
                guard let data = try handle.read(upToCount: chunkSize), !data.isEmpty else {
                    break
                }
                hasher.update(data: data)
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw ModelStorageError.unreadableFile(fileURL)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func update(_ hasher: inout SHA256, withLengthPrefixedUTF8 value: String) {
        let data = Data(value.utf8)
        var length = UInt64(data.count).bigEndian
        withUnsafeBytes(of: &length) { hasher.update(bufferPointer: $0) }
        hasher.update(data: data)
    }
}
