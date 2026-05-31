import Foundation

enum LLMBackendKind: String, Sendable, Codable, Equatable, CaseIterable {
    case gguf
    case coreML
    case tinyIntent
    case remote
    case mock
}
