import Foundation
import Testing
@testable import Lumen

struct CoreAIGemmaPresentationPolicyTests {
    @Test func gemmaCatalogKeepsEmbeddingChoicesWithoutLegacyGenerativeArtifacts() {
        let policy = CoreAIGemmaPresentationPolicy(usesGemma: true)
        let source = ModelCatalog.featured + ModelCatalog.legacyFeatured + LumenModelFleetCatalog.allFleetModels
        let visible = policy.catalogModels(source + source)
        #expect(!visible.isEmpty)
        #expect(visible.allSatisfy { $0.role == .embedding })
        #expect(Set(visible.map { $0.repoId.lowercased() + "/" + $0.fileName.lowercased() }).count == visible.count)
        let embeddingArtifacts = Set(source.filter { $0.role == .embedding }.map { $0.repoId.lowercased() + "/" + $0.fileName.lowercased() })
        #expect(Set(visible.map { $0.repoId.lowercased() + "/" + $0.fileName.lowercased() }) == embeddingArtifacts)
        #expect(policy.allowsModelManagement(role: .embedding))
        #expect(!policy.allowsModelManagement(role: .chat))
        #expect(!policy.allowsModelManagement(role: .roleAdapter))
    }

    @Test func legacyModeRetainsModelRolesAndGemmaNameDoesNotFollowLegacySelection() {
        let legacy = CoreAIGemmaPresentationPolicy(usesGemma: false)
        #expect(ModelRole.allCases.allSatisfy { legacy.allowsModelManagement(role: $0) })
        #expect(legacy.activeAssistantName(legacyName: "Existing local model") == "Existing local model")
        let gemma = CoreAIGemmaPresentationPolicy(usesGemma: true)
        #expect(gemma.activeAssistantName(legacyName: "Existing local model") == "Gemma 4 E2B")
        #expect(gemma.activeAssistantName(legacyName: nil) == "Gemma 4 E2B")
    }

    @Test func missingAndFailedEmbeddingStatesDoNotClaimReadiness() {
        let ready = CoreAIGemmaPresentationPolicy.embeddingStatusMessage("ready")
        for status in ["not_selected", "selected_model_missing", "selected_file_missing", "resource_denied", "not_checked", "embedding_unavailable:synthetic"] {
            let message = CoreAIGemmaPresentationPolicy.embeddingStatusMessage(status)
            #expect(message != ready)
            #expect(!message.contains("synthetic"))
            #expect(!message.isEmpty)
        }
    }

    @Test func chatGeneratesOnlyThroughKernelAndKeepsToolsOptional() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let chat = try String(contentsOf: root.appendingPathComponent("ios/Lumen/Views/ChatView.swift"), encoding: .utf8)
        #expect(chat.contains("kernel.run(kernelRequest, modelContext: modelContext)"))
        #expect(chat.contains("allowsToolExecution: appState.agentModeEnabled"))
        #expect(!chat.contains("AppLlamaService.shared.stream"))
        #expect(!chat.contains("runPlain("))
    }
}
