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

    @Test func slotContractsDeclareRuntimePolicyAndOutputContracts() {
        let executor = LumenModelSlotContract.executor
        #expect(executor.outputContract == .structuredJSON)
        #expect(executor.budgetPolicy == .foregroundInteractive)
        #expect(executor.acceptsRuntimePath("sharedAdapter"))
        #expect(executor.acceptsRuntimePath("legacySlotLoadedContinuation"))
        #expect(!executor.acceptsRuntimePath("deterministic-compatibility"))
        #expect(!executor.acceptsRuntimePath("coreML"))

        let embedding = LumenModelSlotContract.embedding
        #expect(embedding.outputContract == .embeddingVector)
        #expect(embedding.acceptsRuntimePath("embedding"))
        #expect(embedding.acceptsRuntimePath("coreML"))
    }

    @Test func releaseVisibleModelCatalogsDoNotAdvertiseFallbackSurfaces() {
        let forbidden = ["fallback", "unavailable", "mock", "staged", "not implemented", "unimplemented"]
        let catalogText = LumenModelFamily.allCases.map { family in
            [
                family.id,
                family.displayName,
                family.shortLabel,
                family.description,
            ].joined(separator: " ")
        } + LumenModelFleetCatalog.allFleetModels.map { model in
            ([
                model.id,
                model.name,
                model.description,
            ] + model.tags).joined(separator: " ")
        } + LumenModelFleetCatalog.selectableBootstrapModels.map { model in
            ([
                model.id,
                model.name,
                model.description,
            ] + model.tags).joined(separator: " ")
        }

        for text in catalogText {
            let lower = text.lowercased()
            for marker in forbidden {
                #expect(!lower.contains(marker), "Release catalog text contains \(marker): \(text)")
            }
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
        try materializeModelFiles(chat, embedding)

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
        #expect(snapshot.targetResidentSlots.contains(.cortex))
        #expect(snapshot.targetResidentSlots.contains(.embedding))
        #expect(snapshot.runtimeResidentSlots.isEmpty)
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
        try materializeModelFiles(chat, customEmbedding)

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
        try materializeModelFiles(sharedBase, cortexReleaseBake)

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
        try materializeModelFiles(sharedBase, cortexAdapter)

        let snapshot = LumenModelFleetResolver.resolveV1(
            activeChatModelID: sharedBase.id.uuidString,
            activeEmbeddingModelID: nil,
            storedModels: [sharedBase, cortexAdapter]
        )

        #expect(snapshot.assignment(for: .cortex)?.modelID == sharedBase.id)
        #expect(snapshot.assignment(for: .executor)?.modelID == sharedBase.id)
        #expect(snapshot.assignments.values.allSatisfy { $0.modelID != cortexAdapter.id })
    }

    @Test func qwen3RuntimeContractDescribesTrainedModelAndAdapters() async throws {
        let contract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
        #expect(contract.sharedBaseModelID == "Qwen/Qwen3-1.7B")
        #expect(contract.sharedBaseRepoID == "ales27pm/lumen-qwen3-bootstrap-gguf")
        #expect(contract.adapterRoles.count == 6)
        #expect(contract.runtimeSlots == [.cortex, .executor, .mouth, .mimicry, .rem])
        #expect(contract.adapterRole(roleID: "fleet")?.slot == nil)
        #expect(contract.mergeAdaptersByDefault == false)
        #expect(contract.releaseBakeManualOnly == true)
        #expect(contract.traceValues["trainedBaseModelID"] == "Qwen/Qwen3-1.7B")
    }

    @Test @MainActor func qwen3ResolverReportsMissingRoleAdapters() async throws {
        let previousFamily = LumenModelFamily.persistedSelected
        LumenModelFamily.persistedSelected = .qwen3
        defer { LumenModelFamily.persistedSelected = previousFamily }

        let contract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
        let sharedBase = StoredModel(
            name: "Qwen3 Fast Shared Chat Base",
            repoId: contract.sharedBaseRepoID,
            fileName: contract.sharedBaseFileName,
            sizeBytes: 1,
            quantization: "Q4_K_M",
            parameters: "1.7B",
            role: .chat,
            localPath: "/tmp/\(contract.sharedBaseFileName)"
        )
        try materializeModelFiles(sharedBase)

        let snapshot = LumenModelFleetResolver.resolveV1(
            activeChatModelID: sharedBase.id.uuidString,
            activeEmbeddingModelID: nil,
            storedModels: [sharedBase]
        )

        #expect(snapshot.mode == .qwen3AdapterRuntime)
        #expect(snapshot.isRunnableV1)
        #expect(snapshot.missingAdapterSlots == [.cortex, .executor, .mouth, .mimicry, .rem])
        #expect(!snapshot.isFullyAdapted)
        #expect(snapshot.assignment(for: .executor)?.hasConfiguredRoleAdapter == false)
        #expect(snapshot.assignment(for: .executor)?.usesRoleAdapter == false)
        #expect(snapshot.assignment(for: .executor)?.requiresRoleAdapterForRuntime == true)
        #expect(snapshot.assignment(for: .executor)?.expectedRoleAdapterRepoID == contract.adapterRepoID)
        #expect(snapshot.assignment(for: .executor)?.expectedRoleAdapterFileName == contract.adapterRole(for: .executor)?.adapterFileName)
    }

    @Test @MainActor func qwen3ResolverPrefersContractAdapterArtifactOverLooseHints() async throws {
        let previousFamily = LumenModelFamily.persistedSelected
        LumenModelFamily.persistedSelected = .qwen3
        defer { LumenModelFamily.persistedSelected = previousFamily }

        let contract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
        let executorRole = try #require(contract.adapterRole(for: .executor))
        let sharedBase = StoredModel(
            name: "Qwen3 Fast Shared Chat Base",
            repoId: contract.sharedBaseRepoID,
            fileName: contract.sharedBaseFileName,
            sizeBytes: 1,
            quantization: "Q4_K_M",
            parameters: "1.7B",
            role: .chat,
            localPath: "/tmp/\(contract.sharedBaseFileName)"
        )
        let exactExecutorAdapter = StoredModel(
            name: "Executor role adapter",
            repoId: executorRole.adapterRepoID,
            fileName: executorRole.adapterFileName,
            sizeBytes: 1,
            quantization: "GGUF",
            parameters: "LoRA",
            role: .roleAdapter,
            localPath: "/tmp/\(executorRole.adapterFileName)"
        )
        let looseHintAdapter = StoredModel(
            name: "Executor-looking wrong artifact",
            repoId: "local/misleading-adapters",
            fileName: "not-the-trained-executor-lora.gguf",
            sizeBytes: 1,
            quantization: "GGUF",
            parameters: "LoRA",
            role: .roleAdapter,
            localPath: "/tmp/not-the-trained-executor-lora.gguf"
        )
        try materializeModelFiles(sharedBase, exactExecutorAdapter, looseHintAdapter)

        let snapshot = LumenModelFleetResolver.resolveV1(
            activeChatModelID: sharedBase.id.uuidString,
            activeEmbeddingModelID: nil,
            storedModels: [sharedBase, looseHintAdapter, exactExecutorAdapter]
        )

        #expect(snapshot.assignment(for: .executor)?.adapterID == exactExecutorAdapter.id)
        #expect(snapshot.assignment(for: .executor)?.adapterFileName == executorRole.adapterFileName)
        #expect(snapshot.assignment(for: .executor)?.adapterPath == exactExecutorAdapter.localPath)
        #expect(snapshot.assignment(for: .executor)?.hasConfiguredRoleAdapter == true)
        #expect(snapshot.assignment(for: .executor)?.usesRoleAdapter == true)
        #expect(snapshot.assignment(for: .executor)?.requiresRoleAdapterForRuntime == true)
    }

    @Test @MainActor func modelLoaderRegistersQwen3AdaptersBeforeStartupLoadGate() async throws {
        let previousFamily = LumenModelFamily.persistedSelected
        LumenModelFamily.persistedSelected = .qwen3
        defer { LumenModelFamily.persistedSelected = previousFamily }

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("lumen-model-loader-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let contract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
        let executorRole = try #require(contract.adapterRole(for: .executor))
        let sharedBase = StoredModel(
            name: "Qwen3 Fast Shared Chat Base",
            repoId: contract.sharedBaseRepoID,
            fileName: contract.sharedBaseFileName,
            sizeBytes: 1,
            quantization: "Q4_K_M",
            parameters: "1.7B",
            role: .chat,
            localPath: root.appendingPathComponent(contract.sharedBaseFileName).path
        )
        let exactExecutorAdapter = StoredModel(
            name: "Executor role adapter",
            repoId: executorRole.adapterRepoID,
            fileName: executorRole.adapterFileName,
            sizeBytes: 1,
            quantization: "GGUF",
            parameters: "LoRA",
            role: .roleAdapter,
            localPath: root.appendingPathComponent(executorRole.adapterFileName).path
        )
        try materializeModelFiles(sharedBase, exactExecutorAdapter)

        let appState = AppState()
        appState.activeChatModelID = sharedBase.id.uuidString
        appState.contextSize = 2048
        await SlotModelRuntimeCoordinator.shared.configure(assignments: [:], contextSize: 2048, preferExclusiveChatRuntime: true)

        _ = await ModelLoader.ensureChatLoaded(
            snapshot: ModelLoadSnapshot(appState: appState, stored: [sharedBase, exactExecutorAdapter]),
            intent: .appStartup
        )

        let assignment = await SlotModelRuntimeCoordinator.shared.assignment(for: .executor)
        #expect(assignment?.adapterPath == exactExecutorAdapter.localPath)
        #expect(assignment?.hasConfiguredRoleAdapter == true)
        #expect(assignment?.requiresRoleAdapterForRuntime == true)

        await SlotModelRuntimeCoordinator.shared.configure(assignments: [:], contextSize: 2048, preferExclusiveChatRuntime: true)
    }
}

struct ImplicitLlamaGenerationSlotPolicyTests {
    @Test func implicitPlainTextGenerationUsesMouth() {
        let request = GenerateRequest(
            systemPrompt: "Keep the caller's system prompt.",
            history: [],
            userMessage: "Hello",
            temperature: 0.5,
            topP: 0.9,
            repetitionPenalty: 1.1,
            maxTokens: 128,
            modelName: "chat",
            relevantMemories: []
        )

        #expect(AppLlamaService.defaultGenerationSlot(for: request) == .mouth)
    }

    @Test func implicitRawStructuredGenerationUsesExecutor() {
        let request = GenerateRequest(
            systemPrompt: "Return one JSON object.",
            history: [],
            userMessage: "Choose the next action.",
            temperature: 0,
            topP: 0.1,
            repetitionPenalty: 1.1,
            maxTokens: 128,
            modelName: "chat",
            relevantMemories: [],
            responseFormat: .constrainedJSON(schema: #"{"type":"object"}"#)
        )

        #expect(request.preservesRawStructuredAgentOutput)
        #expect(AppLlamaService.defaultGenerationSlot(for: request) == .executor)
    }
}

private func materializeModelFiles(_ models: StoredModel...) throws {
    let fileManager = FileManager.default
    for model in models {
        let url = URL(fileURLWithPath: model.localPath)
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fileManager.fileExists(atPath: url.path) {
            fileManager.createFile(atPath: url.path, contents: Data(), attributes: nil)
        }
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
        #expect(adapters.contains { $0.fileName == "lumen-executor-lora.gguf" && $0.sourcePath == "runs/20260706T011546Z/lora_gguf/lumen-executor-lora.gguf" })
        #expect(adapters.allSatisfy { $0.sourcePath?.hasPrefix("runs/20260706T011546Z/lora_gguf/") == true })
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

    @MainActor
    @Test func liveRuntimePreparationTargetsSharedBaseAndSlotAdaptersOnly() async throws {
        let models = ModelLaunchBootstrap.liveRuntimeModelsForInstall(family: .qwen3)
        #expect(models.filter { $0.role == .chat }.map(\.fileName) == ["lumen-qwen3-fast-shared-q4_k_m.gguf"])
        #expect(models.contains { $0.fileName == "lumen-executor-lora.gguf" && $0.role == .roleAdapter })
        #expect(!models.contains { $0.role == .embedding })
        #expect(!models.contains { $0.fileName == "lumen-fleet-lora.gguf" })
        #expect(Set(models.map(\.fileName)) == [
            "lumen-qwen3-fast-shared-q4_k_m.gguf",
            "lumen-cortex-lora.gguf",
            "lumen-executor-lora.gguf",
            "lumen-mouth-lora.gguf",
            "lumen-mimicry-lora.gguf",
            "lumen-rem-lora.gguf",
        ])
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
