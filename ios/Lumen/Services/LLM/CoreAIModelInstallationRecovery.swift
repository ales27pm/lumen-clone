import Darwin
import Foundation

/// Recovers only names created by the package installer. Production callers hold
/// their model replacement lease; active files and unrelated caches are excluded.
nonisolated enum CoreAIModelInstallationRecovery {
    private static let maximumDirectoryEntries = 4_096
    private static let maximumRemovalEntries = 16_384

    static func cleanup(directory: URL, activeInstallation: String?) throws {
        try Task.checkCancellation()
        let io = CoreAILocalPackageIO()
        let root = try io.openDirectory(directory)
        defer { close(root) }

        var pointer = stat()
        let pointerExists = fstatat(root, "active.json", &pointer, AT_SYMLINK_NOFOLLOW) == 0
        if !pointerExists, errno != ENOENT { throw CoreAILocalPackageIO.Failure.storage }
        let active = activeInstallation.flatMap { canonicalUUID($0) ? $0 : nil }
        // A corrupt/unreadable pointer may still be the only reference to the last
        // working installation. Recover staging only until a new pointer commits.
        let mayRemoveInactive = active != nil || !pointerExists
        let names = try directoryNames(root)
        var removalBudget = maximumRemovalEntries
        for name in names {
            try Task.checkCancellation()
            let staging = name.hasPrefix(".import-") && canonicalUUID(String(name.dropFirst(8)))
            let inactive = mayRemoveInactive && canonicalUUID(name) && name != active
            guard staging || inactive else { continue }
            var status = stat()
            guard fstatat(root, name, &status, AT_SYMLINK_NOFOLLOW) == 0 else {
                if errno == ENOENT { continue }
                throw CoreAILocalPackageIO.Failure.storage
            }
            // Do not follow or remove a link masquerading as an owned installation.
            guard status.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR) else { continue }
            try removeDirectory(name, parent: root, remaining: &removalBudget)
        }
    }

    private static func canonicalUUID(_ value: String) -> Bool {
        UUID(uuidString: value)?.uuidString.lowercased() == value
    }

    private static func directoryNames(_ descriptor: Int32) throws -> [String] {
        let duplicate = dup(descriptor)
        guard duplicate >= 0 else { throw CoreAILocalPackageIO.Failure.storage }
        guard let stream = fdopendir(duplicate) else {
            close(duplicate)
            throw CoreAILocalPackageIO.Failure.storage
        }
        defer { closedir(stream) }
        var names: [String] = []
        while true {
            try Task.checkCancellation()
            errno = 0
            guard let entry = readdir(stream) else {
                guard errno == 0 else { throw CoreAILocalPackageIO.Failure.storage }
                return names
            }
            let name = withUnsafePointer(to: entry.pointee.d_name) { pointer in
                pointer.withMemoryRebound(to: CChar.self, capacity: MemoryLayout.size(ofValue: entry.pointee.d_name)) {
                    String(cString: $0)
                }
            }
            guard name != ".", name != ".." else { continue }
            guard names.count < maximumDirectoryEntries else { throw CoreAILocalPackageIO.Failure.invalidPackage }
            names.append(name)
        }
    }

    private static func removeDirectory(_ name: String, parent: Int32, remaining: inout Int) throws {
        try Task.checkCancellation()
        guard remaining > 0 else { throw CoreAILocalPackageIO.Failure.invalidPackage }
        remaining -= 1
        let directory = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directory >= 0 else { throw CoreAILocalPackageIO.Failure.storage }
        defer { close(directory) }
        for child in try directoryNames(directory) {
            try Task.checkCancellation()
            guard remaining > 0 else { throw CoreAILocalPackageIO.Failure.invalidPackage }
            var status = stat()
            guard fstatat(directory, child, &status, AT_SYMLINK_NOFOLLOW) == 0 else {
                throw CoreAILocalPackageIO.Failure.storage
            }
            if status.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR) {
                try removeDirectory(child, parent: directory, remaining: &remaining)
            } else {
                remaining -= 1
                guard unlinkat(directory, child, 0) == 0 else { throw CoreAILocalPackageIO.Failure.storage }
            }
        }
        try Task.checkCancellation()
        guard unlinkat(parent, name, AT_REMOVEDIR) == 0 else { throw CoreAILocalPackageIO.Failure.storage }
    }
}
