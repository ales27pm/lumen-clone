import Foundation

extension AsyncStream.Continuation where Element == AgentEvent {
    @discardableResult
    func yield(_ event: AgentEvent) -> YieldResult {
        yield(event as Element)
    }
}
