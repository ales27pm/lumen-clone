import Foundation

@MainActor
enum LegacyToolSchemaBridge {
    static func toLegacyToolDefinitions(_ secure: [SecureToolDefinition]) -> [ToolDefinition] {
        secure.map {
            ToolDefinition(
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

    static func toInvocation(toolID: String, arguments: [String: String], source: ToolInvocationSource, conversationID: UUID?, turnID: UUID?) -> ToolInvocation {
        ToolInvocation(id: UUID(), toolID: toolID, arguments: arguments, source: source, conversationID: conversationID, turnID: turnID, createdAt: Date())
    }

    private static func mapCategory(_ c: SecureToolCategory) -> ToolCategory {
        switch c {
        case .readOnly, .permissionRead: return .knowledge
        case .userVisibleAction: return .productivity
        case .sensitiveAction, .destructiveAction: return .communication
        case .externalNetwork: return .knowledge
        }
    }

    private static func mapPermission(_ p: PermissionDomain?) -> String? {
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
