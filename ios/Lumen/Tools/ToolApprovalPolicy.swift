import Foundation

struct ToolPolicySettings: Sendable { let networkAccessEnabled: Bool; let userAllowlist: Set<ToolID> }
enum ToolApprovalDecision: Sendable, Equatable { case allow, deny(String), requiresApproval(String) }

enum ToolApprovalPolicy {
    static func decide(definition: SecureToolDefinition, invocation: ToolInvocation, isForeground: Bool, permissionStates: [PermissionDomain: AssistantPermissionState], settings: ToolPolicySettings) -> ToolApprovalDecision {
        if !settings.userAllowlist.isEmpty && !settings.userAllowlist.contains(definition.id) { return .deny("Tool not in user allowlist") }
        if definition.category == .externalNetwork && !settings.networkAccessEnabled { return .deny("Network tools are disabled") }
        if invocation.source == .backgroundTrigger {
            guard definition.supportsBackgroundExecution else {
                return .deny("Tool unavailable in background")
            }
            guard !definition.requiresUserApproval else {
                return .deny("Tool requires foreground approval")
            }
            if definition.category != .readOnly && definition.category != .permissionRead {
                return .deny("Tool unavailable in background")
            }
        }
        if definition.category == .sensitiveAction && !isForeground { return .deny("Sensitive tool denied in background") }
        if definition.category == .destructiveAction && !isForeground { return .deny("Destructive tool denied in background") }
        for p in definition.requiredPermissions {
            let state = permissionStates[p] ?? .unknown
            if state != .granted { return .deny("Missing \(p.rawValue) permission") }
        }
        if definition.requiresUserApproval && invocation.source != .userApproved { return .requiresApproval("User approval required") }
        if definition.category == .sensitiveAction && invocation.source != .userApproved { return .requiresApproval("Sensitive action requires approval") }
        if definition.category == .destructiveAction && invocation.source != .userApproved { return .requiresApproval("Destructive action requires approval") }
        return .allow
    }
}
