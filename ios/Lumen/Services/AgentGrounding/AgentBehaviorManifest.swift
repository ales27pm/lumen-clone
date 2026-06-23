import Foundation

public struct AgentBehaviorManifest: Codable, Hashable, Sendable {
    public let schemaVersion: String
    public let app: ManifestAppInfo
    public let sourceIntegrity: ManifestSourceIntegrity?
    public let fleet: ManifestFleet
    public let tools: [RuntimeToolDefinition]
    public let intents: [ManifestIntent]
    public let routingMatrix: [ManifestRoutingEntry]
    public let memory: ManifestMemory?
    public let sentinels: ManifestSentinels
}

public struct ManifestAppInfo: Codable, Hashable, Sendable {
    public let name: String
    public let bundleIdentifier: String?
    public let buildVersion: String?
    public let generatedAt: String?
}

public struct ManifestSourceIntegrity: Codable, Hashable, Sendable {
    public let commit: String?
    public let files: [ManifestSourceFileHash]
}

public struct ManifestSourceFileHash: Codable, Hashable, Sendable {
    public let path: String
    public let sha256: String
}

public struct ManifestFleet: Codable, Hashable, Sendable {
    public let contractVersion: String
    public let slots: [ManifestModelSlot]
}

public struct ManifestModelSlot: Codable, Hashable, Sendable {
    public let id: String
    public let role: String
    public let modelFamily: String?
    public let responsibilities: [String]
}

public struct ManifestIntent: Codable, Hashable, Sendable {
    public let id: String
    public let allowedToolIDs: [String]
}

public struct ManifestRoutingEntry: Codable, Hashable, Sendable {
    public let intent: String
    public let allowedTools: [String]
    public let forbiddenTools: [String]
}

public struct ManifestMemory: Codable, Hashable, Sendable {
    public let scopes: [String]
    public let freshnessClasses: [ManifestFreshnessClass]
}

public struct ManifestFreshnessClass: Codable, Hashable, Sendable {
    public let id: String
    public let ttlSeconds: Int?
    public let durable: Bool

    private enum CodingKeys: String, CodingKey {
        case id
        case ttlSeconds
        case durable
    }

    public init(id: String, ttlSeconds: Int?, durable: Bool = false) {
        self.id = id
        self.ttlSeconds = ttlSeconds
        self.durable = durable
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        ttlSeconds = try container.decodeIfPresent(Int.self, forKey: .ttlSeconds)
        durable = try container.decodeIfPresent(Bool.self, forKey: .durable) ?? false
    }
}

public struct ManifestSentinels: Codable, Hashable, Sendable {
    public let forbiddenInUserOutput: [String]
}

public struct RuntimeToolDefinition: Codable, Hashable, Sendable {
    public let id: String
    public let displayName: String?
    public let description: String?
    public let requiresApproval: Bool
    public let permissionKey: String?
    public let arguments: [RuntimeToolArgument]

    public init(id: String, displayName: String? = nil, description: String? = nil, requiresApproval: Bool = false, permissionKey: String? = nil, arguments: [RuntimeToolArgument] = []) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.requiresApproval = requiresApproval
        self.permissionKey = permissionKey
        self.arguments = arguments
    }
}

public struct RuntimeToolArgument: Codable, Hashable, Sendable {
    public let name: String
    public let type: String
    public let required: Bool

    public init(name: String, type: String, required: Bool = true) {
        self.name = name
        self.type = type
        self.required = required
    }
}
