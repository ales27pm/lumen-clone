import Foundation

public struct BundledFleetSystemPrompt: Codable, Hashable, Sendable {
    public let slotID: String?
    public let role: String?
    public let systemPrompt: String?
    public let system_prompt: String?

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

    public func loadManifest() throws -> AgentBehaviorManifest {
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
        _ = try crossModelTrainingDirectoryURL
        _ = try fileURL("AgentGrounding/agent_manifest/AgentBehaviorManifest", extension: "json")
        _ = try fileURL("AgentGrounding/agent_manifest/fleet_system_prompts", extension: "json")
        _ = try fileURL("AgentGrounding/agent_manifest/manifest_validation_report", extension: "json")
        _ = try fileURL("AgentGrounding/agent_manifest/AgentBehaviorManifest", extension: "md")
        _ = try crossModelTrainingFileURL(named: "cross_model_training.jsonl")
        _ = try crossModelTrainingFileURL(named: "train_sft_cross.jsonl")
        _ = try crossModelTrainingFileURL(named: "val_sft_cross.jsonl")
        _ = try crossModelTrainingFileURL(named: "dpo_train_cross.jsonl")
        _ = try crossModelTrainingFileURL(named: "dpo_val_cross.jsonl")
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
