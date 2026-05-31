import Foundation
import Testing
@testable import Lumen

struct LumenFleetTests {
    @Test func contractValidationFailsDeterministicallyWhenSlotMappingIsMissing() async throws {
        var mapping: [LumenModelSlot: LumenModelSlotContract] = [
            .cortex: .cortex,
            .executor: .executor,
            .mouth: .mouth,
            .mimicry: .mimicry,
            .rem: .rem,
            .embedding: .embedding,
        ]
        mapping.removeValue(forKey: .rem)

        do {
            try LumenModelSlotContract.validateCompleteness(using: mapping)
            Issue.record("Expected validation to throw for missing slot contract")
        } catch let error as LumenModelSlotContract.ContractError {
            guard case .incompleteMapping(let missingSlots, _, _) = error else {
                Issue.record("Expected incompleteMapping error")
                return
            }
            #expect(missingSlots == [.rem])
        }
    }

    @Test func requiredContractThrowsForMissingSlotWithoutFallback() async throws {
        do {
            _ = try LumenModelSlotContract.requiredContract(for: .rem, using: [.cortex: .cortex])
            Issue.record("Expected missingContract error")
        } catch let error as LumenModelSlotContract.ContractError {
            guard case .missingContract(let slot, _, let modelConfigVersion) = error else {
                Issue.record("Expected missingContract error")
                return
            }
            #expect(slot == .rem)
            #expect(modelConfigVersion == LumenModelSlotContract.fleetContractVersion)
        }
    }

    @Test @MainActor func resolverAssignsAllTextSlotsFromSingleSharedAdapterFirstBase() async throws {
        let chat = StoredModel(
            name: "Fleet v1 Adapter Base — Qwen 2.5 1.5B",
            repoId: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            fileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            sizeBytes: 1_117_000_000,
            quantization: "Q4_K_M",
            parameters: "1.5B",
            role: .chat,
            localPath: "/tmp/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        )
        let embedding = StoredModel(
            name: "Qwen3 Embedding",
            repoId: "Qwen/Qwen3-Embedding-0.6B-GGUF",
            fileName: "qwen3-embedding-0.6b-q4_k_m.gguf",
            sizeBytes: 450_000_000,
            quantization: "Q4_K_M",
            parameters: "0.6B",
            role: .embedding,
            localPath: "/tmp/qwen3-embedding-0.6b-q4_k_m.gguf"
        )

        let snapshot = LumenModelFleetResolver.resolveV1(
            activeChatModelID: chat.id.uuidString,
            activeEmbeddingModelID: embedding.id.uuidString,
            storedModels: [chat, embedding]
        )

        #expect(snapshot.mode == .v1MultiResident)
        #expect(snapshot.isRunnableV1)
        #expect(snapshot.missingSlots.isEmpty)
        #expect(snapshot.assignment(for: .cortex)?.modelID == chat.id)
        #expect(snapshot.assignment(for: .executor)?.modelID == chat.id)
        #expect(snapshot.assignment(for: .mouth)?.modelID == chat.id)
        #expect(snapshot.assignment(for: .mimicry)?.modelID == chat.id)
        #expect(snapshot.assignment(for: .rem)?.modelID == chat.id)
        #expect(snapshot.assignment(for: .embedding)?.modelID == embedding.id)
        #expect(snapshot.runtimeResidentSlots.contains(.cortex))
        #expect(snapshot.runtimeResidentSlots.contains(.embedding))
    }

    @Test @MainActor func fleetResolverKeepsEmbeddingAssignmentWhenHintsDoNotMatch() async throws {
        let chat = StoredModel(
            name: "Local Chat",
            repoId: "local/chat",
            fileName: "local-chat.gguf",
            sizeBytes: 1,
            quantization: "local",
            parameters: "local",
            role: .chat,
            localPath: "/tmp/local-chat.gguf"
        )
        let customEmbedding = StoredModel(
            name: "Vector Store Model",
            repoId: "local/vector-store-model",
            fileName: "vectors.gguf",
            sizeBytes: 1,
            quantization: "local",
            parameters: "local",
            role: .embedding,
            localPath: "/tmp/vectors.gguf"
        )

        let snapshot = LumenModelFleetResolver.resolveV1(
            activeChatModelID: chat.id.uuidString,
            activeEmbeddingModelID: nil,
            storedModels: [chat, customEmbedding]
        )

        #expect(snapshot.assignment(for: .embedding)?.modelID == customEmbedding.id)
        #expect(!snapshot.missingSlots.contains(.embedding))
    }

    @Test @MainActor func resolverPrefersReleaseBakedSlotModelWhenAvailable() async throws {
        let sharedBase = StoredModel(
            name: "Fleet v1 Adapter Base — Qwen 2.5 1.5B",
            repoId: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            fileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            sizeBytes: 1,
            quantization: "Q4_K_M",
            parameters: "1.5B",
            role: .chat,
            localPath: "/tmp/shared-base.gguf"
        )
        let cortexReleaseBake = StoredModel(
            name: "Fleet v1 Release Bake Cortex — Qwen 1.5B",
            repoId: "ales27pm/lumen-fleet-gguf",
            fileName: "lumen-cortex-release-bake-q4_k_m.gguf",
            sizeBytes: 1,
            quantization: "Q4_K_M",
            parameters: "1.5B",
            role: .chat,
            localPath: "/tmp/models/gguf_release_bake/cortex_merged_gguf/lumen-cortex-release-bake-q4_k_m.gguf"
        )

        let snapshot = LumenModelFleetResolver.resolveV1(
            activeChatModelID: sharedBase.id.uuidString,
            activeEmbeddingModelID: nil,
            storedModels: [sharedBase, cortexReleaseBake]
        )

        #expect(snapshot.assignment(for: .cortex)?.modelID == cortexReleaseBake.id)
        #expect(snapshot.assignment(for: .executor)?.modelID == sharedBase.id)
    }

    @Test @MainActor func resolverDoesNotLoadAdapterOnlyArtifactsAsStandaloneChatModels() async throws {
        let sharedBase = StoredModel(
            name: "Fleet v1 Adapter Base — Qwen 2.5 1.5B",
            repoId: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            fileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            sizeBytes: 1,
            quantization: "Q4_K_M",
            parameters: "1.5B",
            role: .chat,
            localPath: "/tmp/shared-base.gguf"
        )
        let cortexAdapter = StoredModel(
            name: "Lumen Cortex LoRA Adapter",
            repoId: "ales27pm/lumen-fleet-adapters",
            fileName: "cortex.lora",
            sizeBytes: 1,
            quantization: "lora",
            parameters: "adapter",
            role: .chat,
            localPath: "/tmp/models/lora/cortex/cortex.lora"
        )

        let snapshot = LumenModelFleetResolver.resolveV1(
            activeChatModelID: sharedBase.id.uuidString,
            activeEmbeddingModelID: nil,
            storedModels: [sharedBase, cortexAdapter]
        )

        #expect(snapshot.assignment(for: .cortex)?.modelID == sharedBase.id)
        #expect(snapshot.assignment(for: .executor)?.modelID == sharedBase.id)
        #expect(snapshot.assignments.values.allSatisfy { $0.modelID != cortexAdapter.id })
    }
}

struct Qwen3AdapterRuntimeCatalogTests {
    @Test func qwen3BootstrapCatalogUsesOneSharedChatBaseAndSixAdapters() async throws {
        let models = LumenModelFleetCatalog.qwen3BootstrapModels
        let chatModels = models.filter { $0.role == .chat }
        let adapters = models.filter { $0.role == .roleAdapter }
        #expect(chatModels.map(\.fileName) == ["lumen-qwen3-fast-shared-q4_k_m.gguf"])
        #expect(models.filter { $0.role == .embedding }.map(\.fileName) == ["Qwen3-Embedding-0.6B-Q8_0.gguf"])
        #expect(adapters.count == 6)
        #expect(adapters.contains { $0.fileName == "lumen-fleet-lora.gguf" && $0.tags.contains("fleet") })
        #expect(!models.contains { $0.fileName == "lumen-fleet-lora.gguf" && $0.role == .embedding })
        #expect(Set(adapters.map(\.fileName)) == [
            "lumen-cortex-lora.gguf",
            "lumen-executor-lora.gguf",
            "lumen-mouth-lora.gguf",
            "lumen-mimicry-lora.gguf",
            "lumen-rem-lora.gguf",
            "lumen-fleet-lora.gguf",
        ])
        #expect(models.allSatisfy { !$0.fileName.contains("release-bake") })
    }

    @Test func qwen25BootstrapCatalogStillProvidesBaselineChatAndEmbedding() async throws {
        let models = LumenModelFleetCatalog.qwen25BootstrapModels
        #expect(models.contains { $0.role == .chat && $0.fileName.contains("qwen2.5") })
        #expect(models.contains { $0.role == .embedding })
    }

    @Test func traceInitializerDefaultsAdapterMetadataForBackwardCompatibility() async throws {
        let trace = AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "cortex",
            stage: "unit",
            intent: nil,
            promptPrefix: "",
            rawOutputPrefix: "",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["calendar.create"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false
        )
        #expect(trace.allowedToolIDs == ["calendar.create"])
        #expect(trace.adapterApplied == nil)
        #expect(trace.adapterSlot == nil)
    }
}
