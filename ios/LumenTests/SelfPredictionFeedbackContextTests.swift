import Foundation
import XCTest
@testable import Lumen

/// Actual host rendering and prompt assembly with a synthetic snapshot. The
/// standalone runner does not exercise the snapshot builder or trace adapter.
final class SelfPredictionFeedbackContextTests: XCTestCase {
    private func snapshot() -> SelfModelSnapshot {
        .init(
            schemaVersion: SelfModelSnapshot.schemaVersion, generatedAt: Date(timeIntervalSince1970: 0),
            app: .init(name: "Synthetic", buildNumber: "1", shortVersion: "1", platform: "ios", mode: "foreground", sourceRevision: nil),
            agent: .init(logicalIdentity: "lumen", activeSlot: "assistant", availableSlots: [],
                         declaredSlots: ["assistant"], slotAvailability: .unknown,
                         manifestCommit: "unknown", fleetContractVersion: "synthetic"),
            runtime: .init(selectedRuntimePathKind: "coreAI", selectedRuntime: "coreAI", selectionReason: "synthetic",
                           readiness: .unknown, availableBackendKinds: [], embeddingAvailable: nil,
                           embeddingReadiness: .unknown, thermalState: "nominal", powerState: "normal", networkState: "unknown"),
            contextBudget: .init(profile: "chat", maxInputTokens: 1024,
                                 sections: .init(ContextBudgetTokenSections(system: 100, history: 100, memories: 0, rag: 0, tools: 100, runtime: 100))),
            tools: .init(availabilityScope: "policy_filtered_definitions_not_execution_guarantee",
                         available: ["perturbation.test"], requiresApproval: [], backgroundSafe: ["perturbation.test"]),
            evidence: .init(manifestFreshness: "unknown", runtimeAuditPresent: false,
                            exportPolicy: .init(sourceLayer: "synthetic", ownsLiveE2EScenarios: false, includesDeterministicStaticScenarios: false)),
            policy: .init(mustNotInventToolIDs: true, mustNotBypassApproval: true, mustCiteRuntimeSourceWhenClaimingRuntimeState: true)
        )
    }

    private func assemble(_ host: HostSelfModelContext, policy: String, cap: Int, user: String = "hello") -> PromptAssembly {
        PromptAssembler.assemble(
            systemPrompt: host.injecting(into: policy), history: [], userMessage: user,
            memories: [], attachments: [],
            budget: .init(totalChars: 4000, attachmentsShare: 0, memoriesShare: 0, historyShare: 0, maxSystemPromptChars: cap),
            hostSelfModel: host
        )
    }

    func testTightBudgetOmitsWholeFeedbackBeforeCurrentHostFacts() {
        let host = HostSelfModelContext(snapshot: snapshot(), predictionFeedback: .unavailable)
        XCTAssertFalse(host.snapshotBlock.isEmpty)
        XCTAssertFalse(host.feedbackBlock.isEmpty)
        let policy = "Keep tool and approval rules.\n"
        let cap = host.snapshotBlock.count + policy.count
        let result = assemble(host, policy: policy, cap: cap)
        XCTAssertEqual(result.systemPrompt, policy + host.snapshotBlock)
        XCTAssertFalse(host.feedbackIncluded(inRenderedPrompt: result.systemPrompt))
        XCTAssertFalse(result.systemPrompt.contains("[HOST TOOL HISTORY"))
        XCTAssertLessThanOrEqual(result.systemPrompt.count, cap)
    }

    func testNormalAssemblyIncludesWholeFeedbackAndAblationKeepsSameSnapshot() {
        let withFeedback = HostSelfModelContext(snapshot: snapshot(), predictionFeedback: .unavailable)
        let withoutFeedback = HostSelfModelContext(snapshot: withFeedback.snapshot)
        let policy = "Keep tool and approval rules.\n"
        let included = assemble(withFeedback, policy: policy, cap: 2000)
        let omitted = assemble(withoutFeedback, policy: policy, cap: 2000)
        XCTAssertTrue(withFeedback.feedbackIncluded(inRenderedPrompt: included.systemPrompt))
        XCTAssertEqual(included.systemPrompt, policy + withFeedback.snapshotBlock + withFeedback.feedbackBlock)
        XCTAssertEqual(omitted.systemPrompt, policy + withFeedback.snapshotBlock)
        XCTAssertEqual(withFeedback.snapshotBlock, withoutFeedback.snapshotBlock)
        XCTAssertLessThanOrEqual(withFeedback.feedbackBlock.count, HostSelfModelContext.maximumFeedbackCharacters)
        XCTAssertEqual(withFeedback.removing(from: included.systemPrompt), policy)
    }

    func testFeedbackCannotDisplaceCallerInstructionsAtAgentJSONSystemCap() {
        let withFeedback = HostSelfModelContext(snapshot: snapshot(), predictionFeedback: .unavailable)
        let withoutFeedback = HostSelfModelContext(snapshot: withFeedback.snapshot)
        let policy = String(repeating: "Preserve caller policy and tool schemas. ", count: 100)
        let with = assemble(withFeedback, policy: policy, cap: 2200)
        let without = assemble(withoutFeedback, policy: policy, cap: 2200)
        XCTAssertEqual(with.systemPrompt, without.systemPrompt)
        XCTAssertTrue(with.systemPrompt.hasSuffix(withFeedback.snapshotBlock))
        XCTAssertFalse(withFeedback.feedbackIncluded(inRenderedPrompt: with.systemPrompt))
        XCTAssertLessThanOrEqual(with.systemPrompt.count, 2200)
    }

    func testUserOrHistoryMarkersCannotRestoreCompactedHostFeedback() {
        let host = HostSelfModelContext(snapshot: snapshot(), predictionFeedback: .unavailable)
        let policy = "Keep tool and approval rules.\n"
        let result = PromptAssembler.assemble(
            systemPrompt: policy, history: [(.user, host.block)], userMessage: host.block,
            memories: [], attachments: [],
            budget: .init(totalChars: 4000, attachmentsShare: 0, memoriesShare: 0, historyShare: 1000, maxSystemPromptChars: 2000),
            hostSelfModel: host
        )
        XCTAssertEqual(result.systemPrompt, policy)
        XCTAssertFalse(host.feedbackIncluded(inRenderedPrompt: result.systemPrompt))
        XCTAssertFalse(result.systemPrompt.contains("[HOST SELF MODEL"))
    }
}
