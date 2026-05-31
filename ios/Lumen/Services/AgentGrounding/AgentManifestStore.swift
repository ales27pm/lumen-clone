import Foundation

public enum AgentManifestStore {
    public static let directoryName = "AgentManifest"
    public static let fileName = "AgentBehaviorManifest.json"
    public static let hashFileName = "AgentBehaviorManifest.sha256"
    public static let bundledRelativeDirectory = "AgentGrounding/agent_manifest"

    public static func manifestDirectory(fileManager: FileManager = .default) throws -> URL {
        let base = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = base.appendingPathComponent(directoryName, isDirectory: true)
        if !fileManager.fileExists(atPath: directory.path) {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        return directory
    }

    public static func manifestURL(fileManager: FileManager = .default) throws -> URL {
        try manifestDirectory(fileManager: fileManager).appendingPathComponent(fileName, isDirectory: false)
    }

    public static func manifestHashURL(fileManager: FileManager = .default) throws -> URL {
        try manifestDirectory(fileManager: fileManager).appendingPathComponent(hashFileName, isDirectory: false)
    }

    public static func bundledManifestURL(resourceName: String = "AgentBehaviorManifest", bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: resourceName, withExtension: "json", subdirectory: bundledRelativeDirectory)
    }

    public static func bundledHashURL(resourceName: String = "AgentBehaviorManifest", bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: resourceName, withExtension: "sha256", subdirectory: bundledRelativeDirectory)
    }

    public static func load(fileManager: FileManager = .default) throws -> AgentBehaviorManifest? {
        let url = try manifestURL(fileManager: fileManager)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(AgentBehaviorManifest.self, from: data)
    }

    @discardableResult
    public static func persist(_ manifest: AgentBehaviorManifest, fileManager: FileManager = .default) throws -> URL {
        let url = try manifestURL(fileManager: fileManager)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: url, options: [.atomic])
        return url
    }

    @discardableResult
    public static func synchronizeWithBundle(resourceName: String = "AgentBehaviorManifest", bundle: Bundle = .main, fileManager: FileManager = .default) throws -> String {
        guard let bundledManifestURL = bundledManifestURL(resourceName: resourceName, bundle: bundle) else {
            return "bundled-missing-runtime-fallback"
        }
        let destinationDirectory = try manifestDirectory(fileManager: fileManager)
        let destinationManifestURL = try manifestURL(fileManager: fileManager)
        let destinationHashURL = try manifestHashURL(fileManager: fileManager)

        let destinationExists = fileManager.fileExists(atPath: destinationManifestURL.path)
        let bundledManifestData = try Data(contentsOf: bundledManifestURL)
        let bundledManifest = try JSONDecoder().decode(AgentBehaviorManifest.self, from: bundledManifestData)

        if !destinationExists {
            try copyBundledManifestAndHash(
                bundledManifestURL: bundledManifestURL,
                bundledHashURL: bundledHashURL(resourceName: resourceName, bundle: bundle),
                destinationManifestURL: destinationManifestURL,
                destinationHashURL: destinationHashURL,
                fileManager: fileManager
            )
            return "application-support-stale-replaced"
        }

        let storedManifestData = try Data(contentsOf: destinationManifestURL)
        let storedManifest = try JSONDecoder().decode(AgentBehaviorManifest.self, from: storedManifestData)
        let bundledHash = try readHash(from: bundledHashURL(resourceName: resourceName, bundle: bundle))
        let storedHash = try readHash(from: destinationHashURL)
        let hashDiffers = bundledHash != nil && storedHash != nil && bundledHash != storedHash
        let commitDiffers = bundledManifest.sourceIntegrity?.commit != storedManifest.sourceIntegrity?.commit
        if hashDiffers || commitDiffers {
            try fileManager.removeItem(at: destinationDirectory)
            try fileManager.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)
            try copyBundledManifestAndHash(
                bundledManifestURL: bundledManifestURL,
                bundledHashURL: bundledHashURL(resourceName: resourceName, bundle: bundle),
                destinationManifestURL: destinationManifestURL,
                destinationHashURL: destinationHashURL,
                fileManager: fileManager
            )
            return "application-support-stale-replaced"
        }
        return "application-support-current:\(directoryName)/\(fileName)"
    }

    private static func copyBundledManifestAndHash(
        bundledManifestURL: URL,
        bundledHashURL: URL?,
        destinationManifestURL: URL,
        destinationHashURL: URL,
        fileManager: FileManager
    ) throws {
        try fileManager.copyItem(at: bundledManifestURL, to: destinationManifestURL)
        if let bundledHashURL {
            if fileManager.fileExists(atPath: destinationHashURL.path) {
                try fileManager.removeItem(at: destinationHashURL)
            }
            try fileManager.copyItem(at: bundledHashURL, to: destinationHashURL)
        }
    }

    private static func readHash(from url: URL?) throws -> String? {
        guard let url else { return nil }
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
