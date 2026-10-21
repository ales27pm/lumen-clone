import Foundation
import Metal

nonisolated struct RuntimeAccelerationDiagnostics: Codable, Sendable, Hashable {
    let backendRequested: String
    let metalDeviceAvailable: Bool
    let metalDeviceName: String?
    let metalLowPower: Bool?
    let metalHeadless: Bool?
    let metalUnifiedMemory: Bool?
    let recommendedMaxWorkingSetSizeMB: Double?
    let requestedGpuLayers: Int?
    let requestedKQVOffload: Bool?
    let actualBackend: String?
    let actualOffloadedLayers: Int?
    let actualTotalLayers: Int?
    let metalDeviceUsed: String?
    let actualKQVOffload: Bool?
    let promptEvalTokensPerSecond: Double?
    let decodeTokensPerSecond: Double?
    let actualGpuMemoryMB: Double?
    let actualGpuUtilizationPercent: Double?
    let aneAvailable: Bool?
    let aneUsedByCurrentRuntime: Bool
    let aneUtilizationPercent: Double?
    let verificationLevel: String
    let notes: [String]

    static func forSelectedRuntime() async -> RuntimeAccelerationDiagnostics {
        if CoreAIGemmaModelStore.isEnabled {
            return forCurrentRuntime(
                requestedBackend: "coreai", requestedGpuLayers: nil, requestedKQVOffload: nil,
                notes: ["Core AI Gemma is selected; selection does not establish a loaded model or successful inference."]
            )
        }
        return await AppLlamaService.shared.currentAccelerationDiagnostics()
    }

    static func forCurrentRuntime(
        requestedBackend: String,
        requestedGpuLayers: Int?,
        requestedKQVOffload: Bool?,
        actualBackend: String? = nil,
        actualOffloadedLayers: Int? = nil,
        actualTotalLayers: Int? = nil,
        metalDeviceUsed: String? = nil,
        actualKQVOffload: Bool? = nil,
        promptEvalTokensPerSecond: Double? = nil,
        decodeTokensPerSecond: Double? = nil,
        notes extra: [String] = []
    ) -> RuntimeAccelerationDiagnostics {
        let device = MTLCreateSystemDefaultDevice()
        let deviceAvailable = device != nil
        #if os(iOS)
        let metalLowPower: Bool? = nil
        let metalHeadless: Bool? = nil
        #else
        let metalLowPower = device?.isLowPower
        let metalHeadless = device?.isHeadless
        #endif
        var notes = extra
        if requestedBackend == "coreai" {
            notes.append("Core AI accelerator placement is unverified. Neural Engine execution requires a correlated device trace; GPU and ANE utilization are not measured.")
        } else {
            notes.append("ANE is not used by the GGUF/llama.cpp runtime. ANE requires a Core ML / ANE-compatible runtime path.")
        }
        if actualBackend == nil && actualOffloadedLayers == nil && requestedBackend == "metal" {
            notes.append("llama.cpp Metal offload has not been confirmed by runtime logs yet.")
        }
        let verification: String
        if requestedBackend == "cpu" {
            verification = "cpu_only"
        } else if actualBackend != nil || actualOffloadedLayers != nil || actualKQVOffload != nil {
            verification = "confirmed"
        } else if requestedBackend == "metal" || requestedBackend == "coreai" {
            verification = "requested_unverified"
        } else {
            verification = "unavailable"
        }
        return RuntimeAccelerationDiagnostics(
            backendRequested: requestedBackend,
            metalDeviceAvailable: deviceAvailable,
            metalDeviceName: device?.name,
            metalLowPower: metalLowPower,
            metalHeadless: metalHeadless,
            metalUnifiedMemory: device?.hasUnifiedMemory,
            recommendedMaxWorkingSetSizeMB: device.map { Double($0.recommendedMaxWorkingSetSize) / (1024 * 1024) },
            requestedGpuLayers: requestedGpuLayers,
            requestedKQVOffload: requestedKQVOffload,
            actualBackend: actualBackend,
            actualOffloadedLayers: actualOffloadedLayers,
            actualTotalLayers: actualTotalLayers,
            metalDeviceUsed: metalDeviceUsed,
            actualKQVOffload: actualKQVOffload,
            promptEvalTokensPerSecond: promptEvalTokensPerSecond,
            decodeTokensPerSecond: decodeTokensPerSecond,
            actualGpuMemoryMB: nil,
            actualGpuUtilizationPercent: nil,
            aneAvailable: nil,
            aneUsedByCurrentRuntime: false,
            aneUtilizationPercent: nil,
            verificationLevel: verification,
            notes: notes
        )
    }

    func withPerformance(promptEvalTokensPerSecond: Double?, decodeTokensPerSecond: Double?) -> RuntimeAccelerationDiagnostics {
        RuntimeAccelerationDiagnostics.forCurrentRuntime(
            requestedBackend: backendRequested,
            requestedGpuLayers: requestedGpuLayers,
            requestedKQVOffload: requestedKQVOffload,
            actualBackend: actualBackend,
            actualOffloadedLayers: actualOffloadedLayers,
            actualTotalLayers: actualTotalLayers,
            metalDeviceUsed: metalDeviceUsed,
            actualKQVOffload: actualKQVOffload,
            promptEvalTokensPerSecond: promptEvalTokensPerSecond,
            decodeTokensPerSecond: decodeTokensPerSecond,
            notes: notes.filter { !$0.hasPrefix("ANE is not used") && !$0.hasPrefix("llama.cpp Metal offload") }
        )
    }
}
