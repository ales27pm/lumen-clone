import CryptoKit
import Foundation
import Darwin
#if canImport(FileProvider)
import FileProvider
#endif

/// Shared, bounded filesystem boundary for app-owned Core AI packages. Reads are
/// anchored to open directory descriptors; symbolic links and cloud placeholders
/// are rejected, and payload hashes are streamed without materializing weights.
nonisolated struct CoreAILocalPackageIO: Sendable {
    let maximumPackageBytes: Int64
    let maximumEntries = 4_096

    init(maximumPackageBytes: Int64 = 32 * 1_024 * 1_024 * 1_024) {
        self.maximumPackageBytes = maximumPackageBytes
    }

    enum Failure: Error, LocalizedError, Equatable {
        case localDirectory, invalidPackage, integrity, notConfigured, storage

        var errorDescription: String? {
            switch self {
            case .localDirectory: return "Choose a fully downloaded local folder without symbolic links."
            case .invalidPackage: return "The local Core AI package is incomplete or has an unsupported format."
            case .integrity: return "The imported model changed or is incomplete. Import the original package again."
            case .notConfigured: return "Import the selected local Core AI model in Settings before using it."
            case .storage: return "The local Core AI model could not be saved or removed. Check available storage and try again."
            }
        }
    }

    struct FileIdentity: Codable, Sendable, Equatable {
        let path: String
        let bytes: Int64
        let sha256: String
    }

    func inventory(_ root: URL) throws -> [String] {
        try Task.checkCancellation()
        let descriptor = try openDirectory(root)
        defer { close(descriptor) }
        let keys: Set<URLResourceKey> = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
                                       .isReadableKey, .isUbiquitousItemKey, .ubiquitousItemDownloadingStatusKey,
                                       .ubiquitousItemIsDownloadingKey]
        try autoreleasepool { try validateInventoryEntry(root, keys: keys) }
        var pending = [""]
        var result: [String] = []
        // Admit entries before allocating their paths or fetching Foundation
        // metadata. The pending directory paths and result paths share this cap.
        var visited = 1
        while let relativeDirectory = pending.popLast() {
            try Task.checkCancellation()
            let directory = try openInventoryDirectory(relativeDirectory, descriptor: descriptor)
            guard let stream = fdopendir(directory) else {
                close(directory)
                throw Failure.localDirectory
            }
            defer { closedir(stream) }
            while true {
                let hasEntry = try autoreleasepool {
                    try Task.checkCancellation()
                    errno = 0
                    guard let entry = readdir(stream) else {
                        guard errno == 0 else { throw Failure.localDirectory }
                        return false
                    }
                    if entry.pointee.d_name.0 == 46 &&
                        (entry.pointee.d_name.1 == 0 || (entry.pointee.d_name.1 == 46 && entry.pointee.d_name.2 == 0)) {
                        return true
                    }
                    guard visited < maximumEntries else { throw Failure.invalidPackage }
                    visited += 1
                    let capacity = MemoryLayout.size(ofValue: entry.pointee.d_name)
                    let name = withUnsafePointer(to: &entry.pointee.d_name) { pointer in
                        pointer.withMemoryRebound(to: CChar.self, capacity: capacity) { String(validatingUTF8: $0) }
                    }
                    guard let name else { throw Failure.localDirectory }
                    let relative = relativeDirectory.isEmpty ? name : relativeDirectory + "/" + name
                    guard validRelativePath(relative) else { throw Failure.localDirectory }
                    var status = stat()
                    guard fstatat(dirfd(stream), name, &status, AT_SYMLINK_NOFOLLOW) == 0,
                          status.st_flags & UInt32(SF_DATALESS) == 0 else { throw Failure.localDirectory }
                    let kind = UInt32(status.st_mode) & UInt32(S_IFMT)
                    guard kind == UInt32(S_IFDIR) || kind == UInt32(S_IFREG) else { throw Failure.localDirectory }
                    try validateInventoryEntry(root.appendingPathComponent(relative), keys: keys)
                    if kind == UInt32(S_IFDIR) { pending.append(relative) }
                    else { result.append(relative) }
                    return true
                }
                if !hasEntry { break }
            }
        }
        return result.sorted()
    }

    private func validateInventoryEntry(_ url: URL, keys: Set<URLResourceKey>) throws {
        #if canImport(FileProvider) && os(iOS)
        guard !FileManager.default.fileExists(atPath: NSFileProviderManager.placeholderURL(for: url).path) else {
            throw Failure.localDirectory
        }
        #endif
        let values = try url.resourceValues(forKeys: keys)
        guard values.isSymbolicLink == false, values.isReadable == true,
              values.isDirectory == true || values.isRegularFile == true,
              values.ubiquitousItemIsDownloading != true, values.isUbiquitousItem != true,
              values.ubiquitousItemDownloadingStatus == nil || values.ubiquitousItemDownloadingStatus == .current ||
                values.ubiquitousItemDownloadingStatus == .downloaded
        else { throw Failure.localDirectory }
    }

    private func openInventoryDirectory(_ relative: String, descriptor: Int32) throws -> Int32 {
        var current = dup(descriptor)
        guard current >= 0 else { throw Failure.localDirectory }
        do {
            for component in relative.split(separator: "/") {
                try Task.checkCancellation()
                let next = openat(current, String(component), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard next >= 0 else { throw Failure.localDirectory }
                close(current)
                current = next
            }
            var status = stat()
            guard fstat(current, &status) == 0, status.st_flags & UInt32(SF_DATALESS) == 0 else { throw Failure.localDirectory }
            return current
        } catch {
            close(current)
            throw error
        }
    }

    func openDirectory(_ url: URL) throws -> Int32 {
        guard url.isFileURL else { throw Failure.localDirectory }
        var status = stat()
        guard lstat(url.path, &status) == 0, UInt32(status.st_mode) & UInt32(S_IFMT) == UInt32(S_IFDIR),
              status.st_flags & UInt32(SF_DATALESS) == 0 else { throw Failure.localDirectory }
        let descriptor = Darwin.open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.localDirectory }
        return descriptor
    }

    func openFile(_ relative: String, descriptor: Int32) throws -> Int32 {
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
              information.st_size >= 0, information.st_size <= maximumPackageBytes else {
            close(result)
            throw Failure.localDirectory
        }
        return result
    }

    func smallFile(_ path: String, descriptor: Int32, limit: Int) throws -> Data {
        let file = try openFile(path, descriptor: descriptor)
        let handle = FileHandle(fileDescriptor: file, closeOnDealloc: true)
        defer { try? handle.close() }
        guard let data = try handle.read(upToCount: limit + 1), data.count <= limit else { throw Failure.invalidPackage }
        return data
    }

    func fileSize(relative: String, descriptor: Int32) throws -> Int64 {
        try Task.checkCancellation()
        let file = try openFile(relative, descriptor: descriptor)
        defer { close(file) }
        var information = stat()
        guard fstat(file, &information) == 0, information.st_size >= 0,
              information.st_size <= maximumPackageBytes else { throw Failure.integrity }
        return information.st_size
    }

    func digest(relative: String, descriptor: Int32, copyTo output: URL? = nil,
                expectedBytes: Int64? = nil, maximumBytes: Int64? = nil) throws -> FileIdentity {
        try Task.checkCancellation()
        let file = try openFile(relative, descriptor: descriptor)
        let input = FileHandle(fileDescriptor: file, closeOnDealloc: true)
        defer { try? input.close() }
        var before = stat()
        guard fstat(file, &before) == 0 else { throw Failure.integrity }
        if let expectedBytes, before.st_size != expectedBytes { throw Failure.integrity }
        guard before.st_size >= 0, before.st_size <= maximumPackageBytes else { throw Failure.invalidPackage }
        if let maximumBytes {
            guard maximumBytes >= 0, before.st_size <= maximumBytes else { throw Failure.invalidPackage }
        }
        let destination: FileHandle?
        if let output {
            guard FileManager.default.createFile(atPath: output.path, contents: nil,
                attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication]) else { throw Failure.storage }
            destination = try FileHandle(forWritingTo: output)
        } else { destination = nil }
        defer { try? destination?.close() }
        var hasher = SHA256()
        var count: Int64 = 0
        while true {
            // Foundation may autorelease read buffers or bridged write data.
            // Drain those temporaries per chunk, including on errors/cancellation,
            // rather than retaining a model-sized payload in the worker's pool.
            let hasData = try autoreleasepool {
                try Task.checkCancellation()
                guard let data = try input.read(upToCount: 1_024 * 1_024), !data.isEmpty else { return false }
                try Task.checkCancellation()
                count += Int64(data.count)
                guard count <= before.st_size else { throw Failure.integrity }
                hasher.update(data: data)
                try destination?.write(contentsOf: data)
                return true
            }
            if !hasData { break }
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

    func validRelativePath(_ path: String) -> Bool {
        !path.isEmpty && path.utf8.count <= 1_024 && !path.hasPrefix("/") && !path.contains("\0") &&
            path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
    }

    func hex(_ data: Data) -> String { data.map { String(format: "%02x", $0) }.joined() }
}
