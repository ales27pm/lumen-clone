import Foundation

enum LLMTokenEvent: Sendable, Equatable {
    case started(requestID: UUID)
    case token(String)
    case partialText(String)
    case toolCallCandidate(String)
    case completed(LLMCompletionSummary)
    case cancelled
    case failed(String)
}
