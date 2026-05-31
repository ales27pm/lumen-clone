import Foundation

struct InferenceProfile: Sendable, Codable, Equatable {
    let name: String
    let contextTokens: Int
    let batchSize: Int
    let threadCount: Int
    let gpuLayerCount: Int
    let useMetal: Bool
    let useMemoryMapping: Bool
    let lowPowerMode: Bool

    init(
        name: String,
        contextTokens: Int,
        batchSize: Int,
        threadCount: Int,
        gpuLayerCount: Int,
        useMetal: Bool,
        useMemoryMapping: Bool,
        lowPowerMode: Bool
    ) {
        self.name = name
        self.contextTokens = max(1, contextTokens)
        self.batchSize = max(1, batchSize)
        self.threadCount = max(1, threadCount)
        self.gpuLayerCount = max(0, gpuLayerCount)
        self.useMetal = useMetal
        self.useMemoryMapping = useMemoryMapping
        self.lowPowerMode = lowPowerMode
    }

    static let simulatorSafe = InferenceProfile(
        name: "Simulator Safe",
        contextTokens: 2_048,
        batchSize: 64,
        threadCount: 2,
        gpuLayerCount: 0,
        useMetal: false,
        useMemoryMapping: true,
        lowPowerMode: true
    )

    static let iphoneBalanced = InferenceProfile(
        name: "iPhone Balanced",
        contextTokens: 4_096,
        batchSize: 256,
        threadCount: 4,
        gpuLayerCount: 999,
        useMetal: true,
        useMemoryMapping: true,
        lowPowerMode: false
    )

    static let iphoneDeepThink = InferenceProfile(
        name: "iPhone Deep Think",
        contextTokens: 8_192,
        batchSize: 256,
        threadCount: 6,
        gpuLayerCount: 999,
        useMetal: true,
        useMemoryMapping: true,
        lowPowerMode: false
    )

    static let ipadProMaximum = InferenceProfile(
        name: "iPad Pro Maximum",
        contextTokens: 16_384,
        batchSize: 512,
        threadCount: 8,
        gpuLayerCount: 999,
        useMetal: true,
        useMemoryMapping: true,
        lowPowerMode: false
    )

    func adjusted(
        name: String? = nil,
        contextTokens: Int? = nil,
        batchSize: Int? = nil,
        threadCount: Int? = nil,
        gpuLayerCount: Int? = nil,
        useMetal: Bool? = nil,
        useMemoryMapping: Bool? = nil,
        lowPowerMode: Bool? = nil
    ) -> InferenceProfile {
        InferenceProfile(
            name: name ?? self.name,
            contextTokens: contextTokens ?? self.contextTokens,
            batchSize: batchSize ?? self.batchSize,
            threadCount: threadCount ?? self.threadCount,
            gpuLayerCount: gpuLayerCount ?? self.gpuLayerCount,
            useMetal: useMetal ?? self.useMetal,
            useMemoryMapping: useMemoryMapping ?? self.useMemoryMapping,
            lowPowerMode: lowPowerMode ?? self.lowPowerMode
        )
    }
}
