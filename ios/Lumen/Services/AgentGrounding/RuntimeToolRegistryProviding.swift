import Foundation

public protocol RuntimeToolRegistryProviding {
    func currentToolDefinitions() -> [RuntimeToolDefinition]
}

public struct StaticRuntimeToolRegistryProvider: RuntimeToolRegistryProviding {
    private let tools: [RuntimeToolDefinition]

    public init(tools: [RuntimeToolDefinition]) {
        self.tools = tools
    }

    public func currentToolDefinitions() -> [RuntimeToolDefinition] {
        tools
    }
}
