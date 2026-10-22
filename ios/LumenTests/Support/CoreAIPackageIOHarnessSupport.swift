#if COREAI_PACKAGE_IO_MEMORY_TESTS
import Foundation
import SwiftUI

// Standalone filesystem-test collaborators only. This file is inactive in the
// Xcode app/test targets. The runner compiles the unchanged production stores,
// configuration validator and cancellation bus, without a model runtime. These
// doubles do not establish inference-service or full-app integration evidence.
actor CoreAIGemmaService {
    static let shared = CoreAIGemmaService()
    func beginModelReplacement() async throws {}
    func endModelReplacement() async {}
    func cancelAndWait() async {}
}

actor CoreAITextEmbeddingService {
    static let shared = CoreAITextEmbeddingService()
    func beginModelReplacement(if shouldBegin: @Sendable () -> Bool = { true }) async throws -> Bool {
        shouldBegin()
    }
    func endModelReplacement() async {}
    func cancelAndWait() async {}
    func isLoaded() -> Bool { false }
}

actor AppLlamaService {
    static let shared = AppLlamaService()
    func cancelActiveGeneration(reason: String) async {}
    func unloadEmbed() async {}
}

actor BundledIntentClassifier {
    static let shared = BundledIntentClassifier()
    func unload() async {}
}

// The storage runner validates config.json with the real policy. It does not
// test request formatting; the full app owns these request contracts.
enum EmbeddingPurpose: Sendable, Equatable {
    case legacyFormatted, retrievalQuery, retrievalDocument(title: String?)
    case semanticSimilarity, codeQuery
}

struct EmbeddingRequest: Sendable, Equatable {
    let text: String
    let purpose: EmbeddingPurpose
    let dimensions: Int?
}

// The real Gemma settings view also compiles in the standalone Mac target;
// styling is the only view collaborator replaced here. No view flow is run.
enum Theme {
    static let textSecondary = Color.secondary
}
#endif
