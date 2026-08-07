import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class ModelLoaderPolicyTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        ModelLoader.cancelActiveLoads()
        try await super.tearDown()
    }

    func testForegroundLaunchChatAndVoiceCanStartModelLoadWhenGateAllows() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertTrue(ModelLoader.canStartModelLoad(intent: .userChat))
        XCTAssertTrue(ModelLoader.canStartModelLoad(intent: .userVoice))
        XCTAssertTrue(ModelLoader.canStartModelLoad(intent: .appStartup))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .diagnostics))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .background))
    }

    func testUserChatAndVoiceDeniedWhenGateDenies() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .userChat))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .userVoice))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .appStartup))
    }

    func testQwen3ChatSelectionRequiresTheContractSharedBase() {
        let contract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
        let sharedBase = StoredModel(
            name: "Qwen3 Fast Shared Chat Base",
            repoId: contract.sharedBaseRepoID,
            fileName: contract.sharedBaseFileName,
            sizeBytes: contract.sharedBaseSizeBytes,
            quantization: "Q4_K_M",
            parameters: "1.7B",
            role: .chat,
            localPath: "/tmp/\(contract.sharedBaseFileName)"
        )
        let incompatible = StoredModel(
            name: "Different Chat",
            repoId: "local/different-chat",
            fileName: "different-chat.gguf",
            sizeBytes: 100_000_000,
            quantization: "local",
            parameters: "local",
            role: .chat,
            localPath: "/tmp/different-chat.gguf"
        )
        let wrongSizeSharedBase = StoredModel(
            name: "Qwen3 Shared Base With Stale Size",
            repoId: contract.sharedBaseRepoID,
            fileName: contract.sharedBaseFileName,
            sizeBytes: contract.sharedBaseSizeBytes - 1,
            quantization: "Q4_K_M",
            parameters: "1.7B",
            role: .chat,
            localPath: "/tmp/wrong-size-\(contract.sharedBaseFileName)"
        )
        let items = [
            StoredModelLoadItem(stored: sharedBase),
            StoredModelLoadItem(stored: incompatible),
            StoredModelLoadItem(stored: wrongSizeSharedBase),
        ]

        XCTAssertTrue(ModelLoader.chatSelectionIsCompatible(snapshot: ModelLoadSnapshot(
            activeChatModelID: sharedBase.id.uuidString,
            activeEmbeddingModelID: nil,
            contextSize: 2_048,
            selectedModelFamily: .qwen3,
            storedModels: items
        )))
        XCTAssertFalse(ModelLoader.chatSelectionIsCompatible(snapshot: ModelLoadSnapshot(
            activeChatModelID: incompatible.id.uuidString,
            activeEmbeddingModelID: nil,
            contextSize: 2_048,
            selectedModelFamily: .qwen3,
            storedModels: items
        )))
        XCTAssertFalse(ModelLoader.chatSelectionIsCompatible(snapshot: ModelLoadSnapshot(
            activeChatModelID: wrongSizeSharedBase.id.uuidString,
            activeEmbeddingModelID: nil,
            contextSize: 2_048,
            selectedModelFamily: .qwen3,
            storedModels: items
        )))
        XCTAssertTrue(ModelLoader.chatSelectionIsCompatible(snapshot: ModelLoadSnapshot(
            activeChatModelID: incompatible.id.uuidString,
            activeEmbeddingModelID: nil,
            contextSize: 2_048,
            selectedModelFamily: .qwen25,
            storedModels: items
        )))
    }

    func testChatSelectionCompatibilityRejectsMissingOrNonChatSelection() {
        let embedding = StoredModel(
            name: "Embedding",
            repoId: "local/embedding",
            fileName: "embedding.gguf",
            sizeBytes: 100_000_000,
            quantization: "local",
            parameters: "local",
            role: .embedding,
            localPath: "/tmp/embedding.gguf"
        )
        let items = [StoredModelLoadItem(stored: embedding)]

        XCTAssertFalse(ModelLoader.chatSelectionIsCompatible(snapshot: ModelLoadSnapshot(
            activeChatModelID: nil,
            activeEmbeddingModelID: embedding.id.uuidString,
            contextSize: 2_048,
            selectedModelFamily: .qwen25,
            storedModels: items
        )))
        XCTAssertFalse(ModelLoader.chatSelectionIsCompatible(snapshot: ModelLoadSnapshot(
            activeChatModelID: embedding.id.uuidString,
            activeEmbeddingModelID: embedding.id.uuidString,
            contextSize: 2_048,
            selectedModelFamily: .qwen25,
            storedModels: items
        )))
    }

    func testStaleChatWaiterLosesMutationEligibilityWhenReplacementLoadBegins() async {
        let firstGate = ControlledModelLoadGate()
        let replacementGate = ControlledModelLoadGate()
        let capturedGate = ControlledModelLoadGate()
        ModelLoader.installChatLoadTaskForTesting(Task { await firstGate.wait() })

        let staleWaiter = Task { @MainActor in
            await ModelLoader.awaitInstalledChatLoadMutationEligibilityForTesting {
                await capturedGate.resume(returning: true)
            }
        }
        _ = await capturedGate.wait()
        ModelLoader.installChatLoadTaskForTesting(Task { await replacementGate.wait() })
        await firstGate.resume(returning: true)

        let staleEligible = await staleWaiter.value
        XCTAssertFalse(staleEligible)
        XCTAssertTrue(ModelLoader.hasActiveChatLoadTaskForTesting)

        let currentWaiter = Task { @MainActor in
            await ModelLoader.awaitInstalledChatLoadMutationEligibilityForTesting(onCaptured: {})
        }
        await replacementGate.resume(returning: true)
        let currentEligible = await currentWaiter.value
        XCTAssertTrue(currentEligible)
        XCTAssertFalse(ModelLoader.hasActiveChatLoadTaskForTesting)
    }

    func testStaleEmbeddingWaiterLosesMutationEligibilityWhenReplacementLoadBegins() async {
        let firstGate = ControlledModelLoadGate()
        let replacementGate = ControlledModelLoadGate()
        let capturedGate = ControlledModelLoadGate()
        ModelLoader.installEmbeddingLoadTaskForTesting(Task { await firstGate.wait() })

        let staleWaiter = Task { @MainActor in
            await ModelLoader.awaitInstalledEmbeddingLoadMutationEligibilityForTesting {
                await capturedGate.resume(returning: true)
            }
        }
        _ = await capturedGate.wait()
        ModelLoader.installEmbeddingLoadTaskForTesting(Task { await replacementGate.wait() })
        await firstGate.resume(returning: true)

        let staleEligible = await staleWaiter.value
        XCTAssertFalse(staleEligible)
        XCTAssertTrue(ModelLoader.hasActiveEmbeddingLoadTaskForTesting)

        let currentWaiter = Task { @MainActor in
            await ModelLoader.awaitInstalledEmbeddingLoadMutationEligibilityForTesting(onCaptured: {})
        }
        await replacementGate.resume(returning: true)
        let currentEligible = await currentWaiter.value
        XCTAssertTrue(currentEligible)
        XCTAssertFalse(ModelLoader.hasActiveEmbeddingLoadTaskForTesting)
    }

    func testLoadResultOwnershipAcceptsUnchangedOrExactLoadedSelection() {
        XCTAssertTrue(ModelLoader.selectionRemainsOwned(currentID: "requested", requestedID: "requested", loadedID: nil))
        XCTAssertTrue(ModelLoader.selectionRemainsOwned(currentID: "requested", requestedID: "requested", loadedID: "requested"))
        XCTAssertTrue(ModelLoader.selectionRemainsOwned(currentID: nil, requestedID: nil, loadedID: nil))
    }

    func testLoadResultOwnershipRejectsSelectionMadeWhileLoading() {
        XCTAssertFalse(ModelLoader.selectionRemainsOwned(currentID: "new-selection", requestedID: "requested", loadedID: "fallback"))
        XCTAssertFalse(ModelLoader.selectionRemainsOwned(currentID: "new-selection", requestedID: nil, loadedID: "fallback"))
        XCTAssertFalse(ModelLoader.selectionRemainsOwned(currentID: "fallback", requestedID: nil, loadedID: "fallback"))
        XCTAssertFalse(ModelLoader.selectionRemainsOwned(currentID: "requested", requestedID: "requested", loadedID: "fallback"))
    }

    func testLoadRequestOwnershipRejectsFamilyChangedWhileLoading() {
        XCTAssertTrue(ModelLoader.loadRequestRemainsOwned(
            currentID: "requested",
            requestedID: "requested",
            loadedID: nil,
            currentFamily: .qwen3,
            requestedFamily: .qwen3
        ))
        XCTAssertFalse(ModelLoader.loadRequestRemainsOwned(
            currentID: "requested",
            requestedID: "requested",
            loadedID: nil,
            currentFamily: .qwen25,
            requestedFamily: .qwen3
        ))
    }

    func testAutoloadRetriesAfterInterruptedOrPartialForegroundAttempt() {
        XCTAssertTrue(ModelAutoloadState.loading.shouldRetryAfterLeavingActiveScene)
        XCTAssertTrue(ModelAutoloadState.finished(chatLoaded: false, embeddingLoaded: true).shouldRetryAfterLeavingActiveScene)
        XCTAssertTrue(ModelAutoloadState.finished(chatLoaded: true, embeddingLoaded: false).shouldRetryAfterLeavingActiveScene)
        XCTAssertFalse(ModelAutoloadState.finished(chatLoaded: true, embeddingLoaded: true).shouldRetryAfterLeavingActiveScene)
        XCTAssertFalse(ModelAutoloadState.idle.shouldRetryAfterLeavingActiveScene)
    }

    func testRuntimeStateAllowsAutoloadOnlyAfterCoreBootAndSplashDismissal() {
        let runtime = RuntimeState()

        XCTAssertFalse(runtime.modelAutoloadBootstrapReady)
        runtime.completeBootCore()
        XCTAssertFalse(runtime.modelAutoloadBootstrapReady)
        runtime.dismissBootSplash()
        XCTAssertTrue(runtime.modelAutoloadBootstrapReady)

        runtime.startBoot()
        XCTAssertFalse(runtime.modelAutoloadBootstrapReady)
    }

    func testAutoloadRequiresCompletedBootstrapAndActiveScene() {
        let ready = ModelAutoloadRequestKey(
            bootstrapReady: true,
            sceneIsActive: true,
            activeChatModelID: "chat-a",
            activeEmbeddingModelID: "embed-a",
            selectedModelFamily: .qwen3,
            requestGeneration: 0
        )
        let bootstrapPending = ModelAutoloadRequestKey(
            bootstrapReady: false,
            sceneIsActive: true,
            activeChatModelID: "chat-a",
            activeEmbeddingModelID: "embed-a",
            selectedModelFamily: .qwen3,
            requestGeneration: 0
        )
        let inactive = ModelAutoloadRequestKey(
            bootstrapReady: true,
            sceneIsActive: false,
            activeChatModelID: "chat-a",
            activeEmbeddingModelID: "embed-a",
            selectedModelFamily: .qwen3,
            requestGeneration: 0
        )

        XCTAssertTrue(ready.canStartAutoload)
        XCTAssertFalse(bootstrapPending.canStartAutoload)
        XCTAssertFalse(inactive.canStartAutoload)
        XCTAssertNotEqual(ready, bootstrapPending)
    }

    func testAutoloadRequestKeyChangesForEitherRoleFamilySceneOrExplicitGeneration() {
        let baseline = ModelAutoloadRequestKey(
            bootstrapReady: true,
            sceneIsActive: true,
            activeChatModelID: "chat-a",
            activeEmbeddingModelID: "embed-a",
            selectedModelFamily: .qwen3,
            requestGeneration: 0
        )

        XCTAssertNotEqual(baseline, ModelAutoloadRequestKey(bootstrapReady: true, sceneIsActive: true, activeChatModelID: "chat-b", activeEmbeddingModelID: "embed-a", selectedModelFamily: .qwen3, requestGeneration: 0))
        XCTAssertNotEqual(baseline, ModelAutoloadRequestKey(bootstrapReady: true, sceneIsActive: true, activeChatModelID: "chat-a", activeEmbeddingModelID: "embed-b", selectedModelFamily: .qwen3, requestGeneration: 0))
        XCTAssertNotEqual(baseline, ModelAutoloadRequestKey(bootstrapReady: true, sceneIsActive: true, activeChatModelID: "chat-a", activeEmbeddingModelID: "embed-a", selectedModelFamily: .qwen25, requestGeneration: 0))
        XCTAssertNotEqual(baseline, ModelAutoloadRequestKey(bootstrapReady: true, sceneIsActive: false, activeChatModelID: "chat-a", activeEmbeddingModelID: "embed-a", selectedModelFamily: .qwen3, requestGeneration: 0))
        XCTAssertNotEqual(baseline, ModelAutoloadRequestKey(bootstrapReady: true, sceneIsActive: true, activeChatModelID: "chat-a", activeEmbeddingModelID: "embed-a", selectedModelFamily: .qwen3, requestGeneration: 1))
    }

    func testAutoloadResourceRetryPolicyIsBoundedAndRequiresActiveScene() {
        XCTAssertTrue(ModelAutoloadRetryPolicy.shouldRetry(completedRetryCount: 0, suggestedDelaySeconds: 10, sceneIsActive: true))
        XCTAssertTrue(ModelAutoloadRetryPolicy.shouldRetry(completedRetryCount: 1, suggestedDelaySeconds: 10, sceneIsActive: true))
        XCTAssertFalse(ModelAutoloadRetryPolicy.shouldRetry(completedRetryCount: 2, suggestedDelaySeconds: 10, sceneIsActive: true))
        XCTAssertFalse(ModelAutoloadRetryPolicy.shouldRetry(completedRetryCount: 0, suggestedDelaySeconds: 10, sceneIsActive: false))
        XCTAssertFalse(ModelAutoloadRetryPolicy.shouldRetry(completedRetryCount: 0, suggestedDelaySeconds: nil, sceneIsActive: true))
        XCTAssertEqual(ModelAutoloadRetryPolicy.boundedDelaySeconds(0.1), 1)
        XCTAssertEqual(ModelAutoloadRetryPolicy.boundedDelaySeconds(500), 120)
    }

    func testStagingAFamilyChoiceDoesNotClearTheWorkingAutoloadPair() {
        let baseline = ModelAutoloadRequestKey(
            bootstrapReady: true,
            sceneIsActive: true,
            activeChatModelID: "chat-a",
            activeEmbeddingModelID: "embed-a",
            selectedModelFamily: .qwen3,
            requestGeneration: 0
        )
        let stagedPickerChoice = LumenModelFamily.qwen25

        XCTAssertEqual(stagedPickerChoice, .qwen25)
        XCTAssertEqual(baseline.activeChatModelID, "chat-a")
        XCTAssertEqual(baseline.activeEmbeddingModelID, "embed-a")
        XCTAssertEqual(baseline.selectedModelFamily, .qwen3)
    }
}

private actor ControlledModelLoadGate {
    private var continuation: CheckedContinuation<Bool, Never>?
    private var bufferedResult: Bool?

    func wait() async -> Bool {
        if let bufferedResult {
            self.bufferedResult = nil
            return bufferedResult
        }
        return await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func resume(returning result: Bool) {
        if let continuation {
            self.continuation = nil
            continuation.resume(returning: result)
        } else {
            bufferedResult = result
        }
    }
}
