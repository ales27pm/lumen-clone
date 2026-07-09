#if DEBUG
import Foundation

/// Explicit Agent Kernel compatibility boundary for legacy agent services.
///
/// Removal condition: delete this bridge once diagnostics, deterministic
/// compatibility, and remaining tool-capable kernel turns no longer require
/// `AgentService` or `SlotAgentService` event streams.
@MainActor
enum LegacyAgentCompatibilityBridge {
    static func runLegacyAgentService(_ request: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentKernelEvent> {
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

    static func runSlotAgentKernelCompatibility(_ request: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentKernelEvent> {
        AsyncStream { continuation in
            let task = Task { @MainActor in
                for await event in SlotAgentService.shared.run(request, options: options) {
                    continuation.yield(event.agentKernelEvent)
                }
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    static func runSlotAgentCompatibility(_ request: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentEvent> {
        SlotAgentService.shared.run(request, options: options)
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
#endif
