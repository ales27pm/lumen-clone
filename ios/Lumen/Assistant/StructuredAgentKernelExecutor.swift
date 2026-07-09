import Foundation

/// Kernel-owned entrypoint for structured model-backed agent turns that still
/// depend on the mature AgentService event stream.
@MainActor
enum StructuredAgentKernelExecutor {
    static func runModelBackedAgent(
        _ request: AgentRequest,
        options: LegacyAgentRunOptions
    ) -> AsyncStream<AgentKernelEvent> {
        AsyncStream { continuation in
            let task = Task { @MainActor in
                for await event in AgentService.shared.run(request, options: options) {
                    continuation.yield(event.agentKernelEvent)
                }
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}

private extension AgentEvent {
    var agentKernelEvent: AgentKernelEvent {
        switch self {
        case .step(let step):
            return .step(step)
        case .stepDelta(let id, let text):
            return .stepDelta(id: id, text: text)
        case .finalDelta(let text):
            return .finalDelta(text)
        case .done(let finalText, let steps):
            return .done(finalText: finalText, steps: steps)
        case .error(let message):
            return .error(message)
        }
    }
}
