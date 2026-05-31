import Foundation

struct InferenceBudget: Sendable, Codable, Equatable {
    let maxPromptTokens: Int
    let maxCompletionTokens: Int
    let maxWallClockSeconds: Double
    let maxMemoryMB: Int?
    let allowGPU: Bool
    let allowBackgroundExecution: Bool

    init(
        maxPromptTokens: Int,
        maxCompletionTokens: Int,
        maxWallClockSeconds: Double,
        maxMemoryMB: Int? = nil,
        allowGPU: Bool,
        allowBackgroundExecution: Bool
    ) {
        self.maxPromptTokens = max(1, maxPromptTokens)
        self.maxCompletionTokens = max(1, maxCompletionTokens)
        self.maxWallClockSeconds = max(0.1, maxWallClockSeconds)
        self.maxMemoryMB = maxMemoryMB.map { max(1, $0) }
        self.allowGPU = allowGPU
        self.allowBackgroundExecution = allowBackgroundExecution
    }

    static let fast = InferenceBudget(
        maxPromptTokens: 2_048,
        maxCompletionTokens: 256,
        maxWallClockSeconds: 15,
        maxMemoryMB: 1_024,
        allowGPU: true,
        allowBackgroundExecution: false
    )

    static let standard = InferenceBudget(
        maxPromptTokens: 4_096,
        maxCompletionTokens: 1_024,
        maxWallClockSeconds: 60,
        maxMemoryMB: 2_048,
        allowGPU: true,
        allowBackgroundExecution: false
    )

    static let deepThink = InferenceBudget(
        maxPromptTokens: 8_192,
        maxCompletionTokens: 2_048,
        maxWallClockSeconds: 180,
        maxMemoryMB: 4_096,
        allowGPU: true,
        allowBackgroundExecution: false
    )

    static let maximumForeground = InferenceBudget(
        maxPromptTokens: 16_384,
        maxCompletionTokens: 4_096,
        maxWallClockSeconds: 300,
        maxMemoryMB: nil,
        allowGPU: true,
        allowBackgroundExecution: false
    )

    func adjusted(
        maxPromptTokens: Int? = nil,
        maxCompletionTokens: Int? = nil,
        maxWallClockSeconds: Double? = nil,
        maxMemoryMB: Int?? = nil,
        allowGPU: Bool? = nil,
        allowBackgroundExecution: Bool? = nil
    ) -> InferenceBudget {
        InferenceBudget(
            maxPromptTokens: maxPromptTokens ?? self.maxPromptTokens,
            maxCompletionTokens: maxCompletionTokens ?? self.maxCompletionTokens,
            maxWallClockSeconds: maxWallClockSeconds ?? self.maxWallClockSeconds,
            maxMemoryMB: maxMemoryMB ?? self.maxMemoryMB,
            allowGPU: allowGPU ?? self.allowGPU,
            allowBackgroundExecution: allowBackgroundExecution ?? self.allowBackgroundExecution
        )
    }
}
