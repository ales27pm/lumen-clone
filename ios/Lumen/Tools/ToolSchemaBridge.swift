import Foundation

@MainActor
enum ToolSchemaBridge {
    nonisolated static func toCatalogToolDefinitions(_ secure: [SecureToolDefinition]) -> [ToolDefinition] {
        secure.map {
            if let canonical = canonicalCatalogToolDefinition(forSecureToolID: $0.id) {
                return canonical
            }
            return ToolDefinition(
                id: $0.id,
                name: $0.displayName,
                category: mapCategory($0.category),
                description: $0.description,
                icon: "wrench.and.screwdriver",
                tint: "blue",
                requiresApproval: $0.requiresUserApproval,
                permissionKey: mapPermission($0.requiredPermissions.first)
            )
        }
    }

    nonisolated static func canonicalCatalogToolDefinition(forSecureToolID id: String) -> ToolDefinition? {
        switch id {
        case "calendar.read":
            return ToolRegistry.find(id: "calendar.list")
        case "contacts.lookup":
            return ToolRegistry.find(id: "contacts.search")
        case "location.snapshot":
            return ToolRegistry.find(id: "location.current")
        case "memory.search":
            return ToolRegistry.find(id: "memory.recall")
        case "rag.search.secure":
            return ToolRegistry.find(id: "rag.search")
        default:
            return nil
        }
    }

    nonisolated static func toInvocation(toolID: String, arguments: [String: String], source: ToolInvocationSource, conversationID: UUID?, turnID: UUID?) -> ToolInvocation {
        ToolInvocation(id: UUID(), toolID: toolID, arguments: arguments, source: source, conversationID: conversationID, turnID: turnID, createdAt: Date())
    }

    private nonisolated static func mapCategory(_ c: SecureToolCategory) -> ToolCategory {
        switch c {
        case .readOnly, .permissionRead: return .knowledge
        case .userVisibleAction: return .productivity
        case .sensitiveAction, .destructiveAction: return .communication
        case .externalNetwork: return .knowledge
        }
    }

    private nonisolated static func mapPermission(_ p: PermissionDomain?) -> String? {
        switch p {
        case .calendars: return "NSCalendarsUsageDescription"
        case .contacts: return "NSContactsUsageDescription"
        case .locationWhenInUse: return "NSLocationWhenInUseUsageDescription"
        case .notifications: return "NSUserNotificationUsageDescription"
        case .photoLibrary: return "NSPhotoLibraryUsageDescription"
        case .camera: return "NSCameraUsageDescription"
        case .microphone: return "NSMicrophoneUsageDescription"
        case .speechRecognition: return "NSSpeechRecognitionUsageDescription"
        default: return nil
        }
    }
}
