import Foundation

struct DeviceStatusTool: LocalTool {
    let definition = SecureToolDefinition(id: "device.status", displayName: "Device Status", description: "Summarize local device runtime capability", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .low, maxOutputCharacters: 800)
    func validateArguments(_ arguments: [String : String]) throws {}
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let snap = await DeviceCapabilityProfiler().captureSnapshot()
        let availableMemory = ByteCountFormatter.string(fromByteCount: Int64(snap.availableMemoryBytes), countStyle: .memory)
        let text = "OS: \(snap.osVersion), Metal: \(snap.metalAvailable), CoreML embeddings: \(snap.coreMLStatus), FoundationModels: \(snap.foundationModelsStatus), available memory: \(availableMemory), LPM: \(snap.lowPowerModeEnabled), Thermal: \(snap.thermalState.rawValue)"
        return SafeToolOutputLimiter.limit(result: .init(invocationID: invocation.id, status: .success, displayText: text, modelText: text, structuredPayload: ["thermal": snap.thermalState.rawValue, "availableMemoryBytes": String(snap.availableMemoryBytes)], privacyLevel: .low, metricsSummary: "ok", errorCode: nil), maxOutput: definition.maxOutputCharacters)
    }
}
