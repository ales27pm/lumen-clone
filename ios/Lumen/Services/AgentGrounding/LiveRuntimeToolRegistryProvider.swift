import Foundation

public struct LiveRuntimeToolRegistryProvider: RuntimeToolRegistryProviding {
    public init() {}

    public func currentToolDefinitions() -> [RuntimeToolDefinition] {
        ToolRegistry.all.map { tool in
            let contract = tool.capabilityContract
            return RuntimeToolDefinition(
                id: tool.id,
                displayName: tool.name,
                description: tool.description,
                requiresApproval: contract.requiresApproval,
                permissionKey: contract.permissionKey,
                permissionKind: contract.permissionKind?.rawValue,
                confirmationMode: contract.confirmationMode.rawValue,
                arguments: contract.runtimeArguments
            )
        }
    }
}
