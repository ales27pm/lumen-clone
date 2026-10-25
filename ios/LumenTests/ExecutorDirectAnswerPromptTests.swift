import Foundation
import XCTest
@testable import Lumen

/// Real runtime contract constants and PromptAssembler, with a synthetic long
/// context/catalogue. This protects instructions, not model arithmetic ability.
final class ExecutorDirectAnswerPromptTests: XCTestCase {
    private var runtimeCores: [String] {
        [StructuredAgentKernelExecutor.executorRuntimeSystemPromptContract,
         StructuredAgentKernelExecutor.unifiedAssistantSystemPromptContract]
    }

    private func assertDirectAnswerPolicy(_ prompt: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(prompt.contains("Use final when no tool is needed"), file: file, line: line)
        XCTAssertTrue(prompt.contains("even if tools are available"), file: file, line: line)
        XCTAssertFalse(prompt.contains("Use final only after trusted observations"), file: file, line: line)
    }

    private func assertCompleteContract(_ prompt: String, core: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(prompt.hasPrefix(core), file: file, line: line)
        assertDirectAnswerPolicy(prompt, file: file, line: line)
        XCTAssertTrue(prompt.contains("\"prediction\":{\"successProbability\":0.0}"), file: file, line: line)
        XCTAssertTrue(prompt.contains("a finite number from 0 to 1"), file: file, line: line)
        XCTAssertTrue(prompt.contains("Final turns must not include prediction"), file: file, line: line)
        XCTAssertTrue(prompt.contains("The host, not the model, owns approvals"), file: file, line: line)
    }

    private func snapshot() -> SelfModelSnapshot {
        .init(
            schemaVersion: SelfModelSnapshot.schemaVersion, generatedAt: Date(timeIntervalSince1970: 0),
            app: .init(name: "Synthetic", buildNumber: "1", shortVersion: "1", platform: "ios", mode: "foreground", sourceRevision: nil),
            agent: .init(logicalIdentity: "lumen", activeSlot: "assistant", availableSlots: [],
                         declaredSlots: ["assistant"], slotAvailability: .unknown,
                         manifestCommit: "unknown", fleetContractVersion: "synthetic"),
            runtime: .init(selectedRuntimePathKind: "llama", selectedRuntime: "llama", selectionReason: "synthetic",
                           readiness: .unknown, availableBackendKinds: [], embeddingAvailable: nil,
                           embeddingReadiness: .unknown, thermalState: "nominal", powerState: "normal", networkState: "unknown"),
            contextBudget: .init(profile: "chat", maxInputTokens: 1024,
                                 sections: .init(ContextBudgetTokenSections(system: 100, history: 100, memories: 0, rag: 0, tools: 100, runtime: 100))),
            tools: .init(availabilityScope: "policy_filtered_definitions_not_execution_guarantee",
                         available: ["outlook.message.mark_unread"], requiresApproval: ["outlook.message.mark_unread"], backgroundSafe: []),
            evidence: .init(manifestFreshness: "unknown", runtimeAuditPresent: false,
                            exportPolicy: .init(sourceLayer: "synthetic", ownsLiveE2EScenarios: false, includesDeterministicStaticScenarios: false)),
            policy: .init(mustNotInventToolIDs: true, mustNotBypassApproval: true, mustCiteRuntimeSourceWhenClaimingRuntimeState: true)
        )
    }

    private var preferredToolLine: String {
        // Longest current catalogue ID and the production compact-description cap.
        "- outlook.message.mark_unread: " + String(repeating: "s", count: 88)
    }

    private func longSystem(core: String, suffix: String = "") -> String {
        core + "\nAvailable tools:\n" + preferredToolLine + "\n"
            + String(repeating: "Synthetic routing context. ", count: 200)
            + "\n" + StructuredAgentKernelExecutor.executorRuntimeOutputBoundary + suffix
    }

    func testBothRuntimeIdentitiesAllowDirectFinalEvenWithAvailableTools() {
        for core in runtimeCores {
            assertDirectAnswerPolicy(core)
            XCTAssertTrue(core.contains("Final turns must not include prediction"))
            XCTAssertTrue(core.contains("The host, not the model, owns approvals"))
        }
    }

    func testDirectAnswerPolicySurvivesFastStructuredPromptCompaction() throws {
        for core in runtimeCores {
            let boundary = StructuredAgentKernelExecutor.executorRuntimeOutputBoundary
            let system = core + "\nAvailable tools:\n- weather: Current local weather.\n"
                + String(repeating: "Synthetic routing context. ", count: 200)
                + "\n" + boundary
            let budget = PromptBudget.agentJSON(contextSize: 2_048, maxTokens: 384)
            let systemCap = try XCTUnwrap(budget.maxSystemPromptChars)
            XCTAssertGreaterThan(system.count, systemCap)
            let assembly = PromptAssembler.assemble(
                systemPrompt: system, history: [], userMessage: "Combien font 6 × 7 ?",
                memories: [], attachments: [], budget: budget,
                attachmentNormalization: .agentRouting, latencyClass: .fastInteractive
            )
            assertCompleteContract(assembly.systemPrompt, core: core)
            XCTAssertTrue(assembly.systemPrompt.contains(boundary))
            XCTAssertTrue(assembly.systemPrompt.contains("Fast interactive mode:"))
            XCTAssertEqual(assembly.userMessage, "Combien font 6 × 7 ?")
            XCTAssertLessThanOrEqual(assembly.systemPrompt.count, systemCap)
        }
    }

    func testAgentJSONBudgetPreservesWholeCoreSnapshotAndPreferredTool() throws {
        let host = HostSelfModelContext(snapshot: snapshot(), predictionFeedback: .unavailable)
        let budget = PromptBudget.agentJSON(contextSize: 2_048, maxTokens: 384)
        let cap = try XCTUnwrap(budget.maxSystemPromptChars)
        XCTAssertEqual(cap, 2_200)
        XCTAssertFalse(host.snapshotBlock.isEmpty)
        XCTAssertLessThanOrEqual(host.snapshotBlock.count, HostSelfModelContext.maximumBlockCharacters)
        for core in runtimeCores {
            for latency in [PromptLatencyClass.normalInteractive, .fastInteractive] {
                let assembly = PromptAssembler.assemble(
                    systemPrompt: host.injecting(into: longSystem(core: core)),
                    history: [], userMessage: "Combien font 6 × 7 ?", memories: [], attachments: [],
                    budget: budget, attachmentNormalization: .agentRouting,
                    latencyClass: latency, hostSelfModel: host
                )
                assertCompleteContract(assembly.systemPrompt, core: core)
                XCTAssertTrue(assembly.systemPrompt.contains("Available tools:\n" + preferredToolLine))
                XCTAssertTrue(assembly.systemPrompt.contains(StructuredAgentKernelExecutor.executorRuntimeOutputBoundary))
                XCTAssertTrue(assembly.systemPrompt.hasSuffix(host.snapshotBlock))
                XCTAssertFalse(host.feedbackIncluded(inRenderedPrompt: assembly.systemPrompt))
                XCTAssertEqual(assembly.userMessage, "Combien font 6 × 7 ?")
                XCTAssertLessThanOrEqual(assembly.systemPrompt.count, cap)
            }
        }
    }

    func testActiveTurnSuffixSurvivesCompactionWithoutDisplacingSnapshot() throws {
        let host = HostSelfModelContext(snapshot: snapshot(), predictionFeedback: .unavailable)
        let constraint = "Active turn constraint: trusted observation complete. This turn is final-only. Do not emit action, tool, or args."
        let boundaryAndConstraint = StructuredAgentKernelExecutor.executorRuntimeOutputBoundary + "\n\n" + constraint
        let budget = PromptBudget.agentJSON(contextSize: 2_048, maxTokens: 384)
        let cap = try XCTUnwrap(budget.maxSystemPromptChars)
        for core in runtimeCores {
            for latency in [PromptLatencyClass.normalInteractive, .fastInteractive] {
                let assembly = PromptAssembler.assemble(
                    systemPrompt: host.injecting(into: longSystem(core: core, suffix: "\n\n" + constraint)),
                    history: [], userMessage: "Use the trusted observation.", memories: [], attachments: [],
                    budget: budget, attachmentNormalization: .agentRouting,
                    latencyClass: latency, hostSelfModel: host
                )
                assertCompleteContract(assembly.systemPrompt, core: core)
                XCTAssertTrue(assembly.systemPrompt.contains(boundaryAndConstraint))
                XCTAssertTrue(assembly.systemPrompt.hasSuffix(host.snapshotBlock))
                XCTAssertFalse(host.feedbackIncluded(inRenderedPrompt: assembly.systemPrompt))
                XCTAssertLessThanOrEqual(assembly.systemPrompt.count, cap)
            }
        }
    }
}
