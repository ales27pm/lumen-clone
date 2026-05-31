import Foundation

struct LLMSamplingConfig: Sendable, Codable, Equatable {
    let temperature: Double
    let topP: Double
    let topK: Int
    let repeatPenalty: Double
    let seed: UInt64?
    let maxTokens: Int
    let stopSequences: [String]

    init(
        temperature: Double,
        topP: Double,
        topK: Int,
        repeatPenalty: Double,
        seed: UInt64? = nil,
        maxTokens: Int,
        stopSequences: [String] = []
    ) {
        self.temperature = temperature.isFinite ? max(0, temperature) : 0
        self.topP = topP.isFinite ? min(max(topP, 0), 1) : 1
        self.topK = max(0, topK)
        self.repeatPenalty = repeatPenalty.isFinite && repeatPenalty > 0 ? repeatPenalty : 1
        self.seed = seed
        self.maxTokens = max(1, maxTokens)
        self.stopSequences = stopSequences.filter { !$0.isEmpty }
    }

    static let balanced = LLMSamplingConfig(
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        repeatPenalty: 1.1,
        maxTokens: 1_024,
        stopSequences: []
    )

    static let deterministic = LLMSamplingConfig(
        temperature: 0,
        topP: 1,
        topK: 1,
        repeatPenalty: 1.05,
        maxTokens: 512,
        stopSequences: []
    )

    static let creative = LLMSamplingConfig(
        temperature: 0.95,
        topP: 0.95,
        topK: 80,
        repeatPenalty: 1.05,
        maxTokens: 1_536,
        stopSequences: []
    )

    static let toolCalling = LLMSamplingConfig(
        temperature: 0.1,
        topP: 0.85,
        topK: 20,
        repeatPenalty: 1.1,
        maxTokens: 768,
        stopSequences: []
    )
}
