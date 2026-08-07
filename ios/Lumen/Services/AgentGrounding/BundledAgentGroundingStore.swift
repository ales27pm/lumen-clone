import Foundation

nonisolated public struct BundledFleetSystemPrompt: Codable, Hashable, Sendable {
    public let slotID: String?
    public let role: String?
    public let systemPrompt: String?
    public let system_prompt: String?

}

nonisolated public struct RuntimeGroundingBundle: Codable, Hashable, Sendable {
    public let schemaVersion: String
    public let artifactKind: String
    public let sourceFamilies: [String]
    public let manifestCommit: String?
    public let manifestToolCount: Int?
    public let manifestIntentCount: Int?
    public let codebaseHome: RuntimeCodebaseHome
    public let injectionPolicy: RuntimeGroundingInjectionPolicy
}

nonisolated public struct RuntimeCodebaseHome: Codable, Hashable, Sendable {
    public let recordCount: Int
    public let moduleCounts: [String: Int]
    public let languageCounts: [String: Int]
    public let selectedFiles: [RuntimeGroundingFile]
}

nonisolated public struct RuntimeGroundingFile: Codable, Hashable, Sendable {
    public let path: String
    public let module: String?
    public let language: String?
    public let sha256: String?
    public let responsibility: String?
    public let symbols: [String]
    public let imports: [String]
}

nonisolated public struct RuntimeGroundingInjectionPolicy: Codable, Hashable, Sendable {
    public let target: String
    public let purpose: String
    public let privacy: String
    public let maxPromptCharacters: Int
}

public enum BundledAgentGroundingStoreError: LocalizedError, Sendable {
    case missingResource(String)
    case invalidResource(URL)
    case missingPrompt(slotID: String)

    public var errorDescription: String? {
        switch self {
        case .missingResource(let path):
            return "Missing bundled agent grounding resource: \(path)"
        case .invalidResource(let url):
            return "Invalid bundled agent grounding resource: \(url.path)"
        case .missingPrompt(let slotID):
            return "No bundled fleet system prompt exists for slot: \(slotID)"
        }
    }
}

/// Background actor for CPU-intensive grounding resource operations
/// (JSON parsing of large manifest files). Keeps heavy work off @MainActor.
public actor GroundingResourceLoader {
    public static let shared = GroundingResourceLoader()
    private let store = BundledAgentGroundingStore()

    public func loadManifestAsync() throws -> AgentBehaviorManifest {
        try store.loadManifest()
    }

    public func loadFleetSystemPromptsAsync() throws -> [String: BundledFleetSystemPrompt] {
        try store.loadFleetSystemPrompts()
    }

    public func verifyRequiredResourcesAsync() throws {
        try store.verifyRequiredResources()
    }

    public func loadManifestValidationReportAsync() throws -> Data {
        try store.loadValidationReportData()
    }

    public func loadRuntimeGroundingBundleAsync() throws -> RuntimeGroundingBundle {
        try store.loadRuntimeGroundingBundle()
    }

    public func loadRuntimeGroundingPromptAsync() throws -> String {
        try store.loadRuntimeGroundingPrompt()
    }
}

public final class BundledAgentGroundingStore: @unchecked Sendable {
    public static let shared = BundledAgentGroundingStore()

    private let bundle: Bundle

    public nonisolated init(bundle: Bundle = .main) {
        self.bundle = bundle
    }

    public nonisolated var agentGroundingRootURL: URL {
        get throws {
            try directoryURL("AgentGrounding")
        }
    }

    public nonisolated var agentManifestDirectoryURL: URL {
        get throws {
            try directoryURL("AgentGrounding/agent_manifest")
        }
    }

    public nonisolated var crossModelTrainingDirectoryURL: URL {
        get throws {
            try directoryURL("AgentGrounding/cross_model_training")
        }
    }

    public nonisolated func loadManifest() throws -> AgentBehaviorManifest {
        let url = try fileURL("AgentGrounding/agent_manifest/AgentBehaviorManifest", extension: "json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(AgentBehaviorManifest.self, from: data)
    }

    public nonisolated func loadFleetSystemPrompts() throws -> [String: BundledFleetSystemPrompt] {
        let url = try fileURL("AgentGrounding/agent_manifest/fleet_system_prompts", extension: "json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: BundledFleetSystemPrompt].self, from: data)
    }

    public nonisolated func systemPrompt(for slotID: String) throws -> String {
        let prompts = try loadFleetSystemPrompts()
        guard let prompt = prompts[slotID] else {
            throw BundledAgentGroundingStoreError.missingPrompt(slotID: slotID)
        }
        let resolved = (prompt.system_prompt ?? prompt.systemPrompt ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !resolved.isEmpty else {
            throw BundledAgentGroundingStoreError.missingPrompt(slotID: slotID)
        }
        return resolved
    }

    public nonisolated func loadManifestMarkdown() throws -> String {
        let url = try fileURL("AgentGrounding/agent_manifest/AgentBehaviorManifest", extension: "md")
        return try String(contentsOf: url, encoding: .utf8)
    }

    public nonisolated func loadValidationReportData() throws -> Data {
        let url = try fileURL("AgentGrounding/agent_manifest/manifest_validation_report", extension: "json")
        return try Data(contentsOf: url)
    }

    public nonisolated func loadRuntimeGroundingBundle() throws -> RuntimeGroundingBundle {
        let url = try fileURL("AgentGrounding/agent_manifest/runtime_grounding_bundle", extension: "json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(RuntimeGroundingBundle.self, from: data)
    }

    public nonisolated func loadRuntimeGroundingPrompt(maxCharacters: Int? = nil) throws -> String {
        let url = try fileURL("AgentGrounding/agent_manifest/runtime_grounding_prompt", extension: "md")
        let text = try String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let maxCharacters, text.count > maxCharacters else {
            return text
        }
        return String(text.prefix(maxCharacters)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public nonisolated func crossModelTrainingFileURL(named fileName: String) throws -> URL {
        let base = try crossModelTrainingDirectoryURL
        let url = base.appendingPathComponent(fileName)
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw BundledAgentGroundingStoreError.missingResource("AgentGrounding/cross_model_training/\(fileName)")
        }
        guard !isDirectory.boolValue else {
            throw BundledAgentGroundingStoreError.invalidResource(url)
        }
        return url
    }

    public nonisolated func verifyRequiredResources() throws {
        _ = try agentGroundingRootURL
        _ = try agentManifestDirectoryURL
        _ = try fileURL("AgentGrounding/agent_manifest/AgentBehaviorManifest", extension: "json")
        _ = try fileURL("AgentGrounding/agent_manifest/fleet_system_prompts", extension: "json")
        _ = try fileURL("AgentGrounding/agent_manifest/manifest_validation_report", extension: "json")
        _ = try fileURL("AgentGrounding/agent_manifest/AgentBehaviorManifest", extension: "md")
        _ = try fileURL("AgentGrounding/agent_manifest/runtime_grounding_bundle", extension: "json")
        _ = try fileURL("AgentGrounding/agent_manifest/runtime_grounding_prompt", extension: "md")
    }

    private nonisolated func directoryURL(_ relativePath: String) throws -> URL {
        guard let url = bundle.url(forResource: relativePath, withExtension: nil) else {
            throw BundledAgentGroundingStoreError.missingResource(relativePath)
        }
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw BundledAgentGroundingStoreError.invalidResource(url)
        }
        return url
    }

    private nonisolated func fileURL(_ relativePathWithoutExtension: String, extension fileExtension: String) throws -> URL {
        guard let url = bundle.url(forResource: relativePathWithoutExtension, withExtension: fileExtension) else {
            throw BundledAgentGroundingStoreError.missingResource("\(relativePathWithoutExtension).\(fileExtension)")
        }
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw BundledAgentGroundingStoreError.invalidResource(url)
        }
        return url
    }
}
