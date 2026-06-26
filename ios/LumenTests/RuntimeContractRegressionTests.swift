import Foundation
import Testing
import SwiftUI
@testable import Lumen

struct LocalRuntimeErrorDescriptionsTests {
    @Test func unavailableHasPreciseLocalizedDescription() {
        let error = LocalRuntimeError.unavailable("adapterUnavailable slot=.executor adapterPath=/tmp/executor.lora")
        #expect(error.errorDescription == "Local runtime unavailable: adapterUnavailable slot=.executor adapterPath=/tmp/executor.lora")
        #expect(!error.localizedDescription.contains("error 0"))
    }

    @Test func notImplementedNamesRuntimeKind() {
        let error = LocalRuntimeError.generationNotImplemented(.foundationModels)
        #expect(error.errorDescription?.contains("foundationModels") == true)
        #expect(!error.localizedDescription.contains("error 0"))
    }
}

@MainActor
struct ExecutorPreflightTests {
    @Test func budgetDenialReasonIsPrecise() {
        let snapshot = ResourceBudgetGate.Snapshot(
            scenePhase: .background,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        let reason = ResourceBudgetGate.heavyModelWorkDenialReason(snapshot: snapshot, reason: "strict-live-training.executor-preflight")
        #expect(reason == "strict-live-training.executor-preflight: scenePhase=background")
    }

    @Test func preflightResultExportsStructuredDiagnosticsMetadata() {
        let result = ExecutorRuntimePreflightResult(
            passed: false,
            reason: "executor preflight failed: adapter required but adapter path missing",
            slot: "executor",
            modelFamily: "qwen3",
            runtimeKind: "adapter-first",
            baseModelPath: "/tmp/lumen-qwen3.gguf",
            baseModelExists: true,
            adapterPath: nil,
            adapterExists: false,
            activeAdapterSlot: nil,
            resourceGateAllowed: true,
            budgetReason: nil,
            ensureReadySucceeded: false,
            smokeProbeSucceeded: false,
            failureKind: "adapterPathMissing"
        )

        #expect(result.diagnosticsMetadata["slot"] == "executor")
        #expect(result.diagnosticsMetadata["modelFamily"] == "qwen3")
        #expect(result.diagnosticsMetadata["runtimeKind"] == "adapter-first")
        #expect(result.diagnosticsMetadata["baseModelPath"] == "/tmp/lumen-qwen3.gguf")
        #expect(result.diagnosticsMetadata["baseModelExists"] == "true")
        #expect(result.diagnosticsMetadata["adapterPath"] == "none")
        #expect(result.diagnosticsMetadata["adapterExists"] == "false")
        #expect(result.diagnosticsMetadata["activeAdapterSlot"] == "none")
        #expect(result.diagnosticsMetadata["resourceGateAllowed"] == "true")
        #expect(result.diagnosticsMetadata["budgetReason"] == "none")
        #expect(result.diagnosticsMetadata["ensureReadySucceeded"] == "false")
        #expect(result.diagnosticsMetadata["smokeProbeSucceeded"] == "false")
        #expect(result.diagnosticsMetadata["failureKind"] == "adapterPathMissing")
        #expect(result.diagnosticsSummary.contains("failureKind=adapterPathMissing"))
    }
}

struct AgentJsonRuntimeClassificationTests {
    @Test func zeroTokenBudgetFailureIsNotReportedAsParseFailure() async {
        let trace = AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-0",
            scenarioID: "training-weather-grounded",
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID(),
            intent: "weather",
            promptPrefix: "weather",
            rawOutputPrefix: "",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: "noJSONObject",
            emittedFinalInActionTurn: false,
            outputTokenCount: 0,
            runtimePath: "agent-model",
            emptyOutputReason: "resource-budget-denied-before-prompt-eval",
            streamStarted: false,
            firstChunkReceived: false,
            textChunkCount: 0,
            finalChunkReceived: false,
            streamTerminationReason: "resource-budget-denied-before-prompt-eval"
        )
        let message = E2ETestRunner.modelRuntimeEvidenceFailureMessageForTests(
            matchingTraces: [trace],
            acceptsPolicyFirstEvidence: false,
            requiresPrimaryAgentJSON: true
        )
        #expect(message.contains("budget failure"))
        #expect(message.contains("parseError suppressed because no text/tokens were produced"))
    }

    @Test func zeroTokenExecutorPreflightFailureIsClassifiedAsRuntimeReadiness() async {
        let reason = "executor preflight failed: adapterUnavailable; slot=.executor; activeAdapterSlot=none"
        let trace = AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-0",
            scenarioID: "training-weather-grounded",
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID(),
            intent: "weather",
            promptPrefix: "weather",
            rawOutputPrefix: "",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: "empty",
            emittedFinalInActionTurn: false,
            outputTokenCount: 0,
            runtimePath: "agent-model",
            emptyOutputReason: reason,
            streamStarted: false,
            firstChunkReceived: false,
            textChunkCount: 0,
            finalChunkReceived: false,
            streamTerminationReason: reason
        )
        let message = E2ETestRunner.modelRuntimeEvidenceFailureMessageForTests(
            matchingTraces: [trace],
            acceptsPolicyFirstEvidence: false,
            requiresPrimaryAgentJSON: true
        )
        #expect(message.contains("runtime readiness failure"))
        #expect(message.contains("executor preflight failed"))
        #expect(message.contains("parseError suppressed because no text/tokens were produced"))
    }
}

struct ContactObservationFinalizerTests {
    @Test func contactSearchSuccessFinalizesAndValidates() {
        let observation = "• Julie Charlebois — +1 (514) 555-0101"
        let final = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .contactSearch,
            toolID: "contacts.search",
            observation: observation,
            originalPrompt: "Find Julie Charlebois"
        )
        #expect(final == "Contact found: Julie Charlebois — +1 (514) 555-0101")
        let routing = IntentRoutingDecision(intent: .contactSearch, allowedToolIDs: ["contacts.search"], requiresClarification: false, clarificationPrompt: nil)
        #expect(FinalIntentValidator.validate(final ?? "", routing: routing, fallback: "Contact search is unavailable in this build right now.") == final)
    }

    @Test func filesAndPhotosAreFinalizerCovered() {
        let fileFinal = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .files,
            toolID: "files.read",
            observation: "diagnostics.txt: runtime checks passed",
            originalPrompt: "Open local document diagnostics.txt"
        )
        #expect(fileFinal.accepted)
        #expect(fileFinal.rejectionReason == nil)
        #expect(fileFinal.text?.contains("File result") == true)

        let photoFinal = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .photos,
            toolID: "photos.search",
            observation: "Found 3 photos matching receipts.",
            originalPrompt: "Search my photos for receipts"
        )
        #expect(photoFinal.accepted)
        #expect(photoFinal.rejectionReason == nil)
        #expect(photoFinal.text?.contains("Photo search results") == true)
    }

    @Test func triggerCancelFinalizerIsCovered() {
        let final = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .trigger,
            toolID: "trigger.cancel",
            observation: "Cancelled scheduled run nightly summary.",
            originalPrompt: "Cancel trigger named nightly summary."
        )
        #expect(final.accepted)
        #expect(final.rejectionReason == nil)
        #expect(final.text?.contains("Trigger cancellation") == true)
    }
}

struct PhoneCallContinuationTests {
    @Test func singleContactPhoneCreatesApprovalBoundary() {
        let routing = IntentRoutingDecision(intent: .phoneCall, allowedToolIDs: ["contacts.search", "phone.call"], requiresClarification: false, clarificationPrompt: nil)
        let continuation = SlotAgentService.phoneCallContinuationForTests(
            observation: "• Julie Charlebois — +1 (514) 555-0101",
            availableToolIDs: ["contacts.search", "phone.call"],
            routing: routing
        )
        #expect(continuation?.step.kind == .approvalBoundary)
        #expect(continuation?.step.toolID == "phone.call")
        #expect(continuation?.step.toolArgs?["number"] == "+15145550101")
        #expect(continuation?.text.contains("Approval required for phone.call") == true)
    }

    @Test func missingPhoneCallToolPreservesFoundContactReason() {
        let routing = IntentRoutingDecision(intent: .phoneCall, allowedToolIDs: ["contacts.search"], requiresClarification: false, clarificationPrompt: nil)
        let continuation = SlotAgentService.phoneCallContinuationForTests(
            observation: "• Julie Charlebois — +1 (514) 555-0101",
            availableToolIDs: ["contacts.search"],
            routing: routing
        )
        #expect(continuation?.step.kind == .reflection)
        #expect(continuation?.text.contains("Contact found: Julie Charlebois") == true)
        #expect(continuation?.text.contains("phone.call is unavailable") == true)
    }

    @Test func liveAgentContactSearchObservationCreatesPhoneApprovalBoundary() {
        let continuation = AgentService.phoneCallContinuationAfterContactSearchForTests(
            actionTool: "contacts.search",
            observation: "• Julie Charlebois — +1 (514) 555-0101",
            prompt: "Call Julie Charlebois",
            availableToolIDs: ["contacts.search", "phone.call"]
        )

        #expect(continuation?.step.kind == .approvalBoundary)
        #expect(continuation?.step.toolID == "phone.call")
        #expect(continuation?.step.toolArgs?["number"] == "+15145550101")
        #expect(continuation?.text.contains("Approval required for phone.call") == true)
        #expect(continuation?.text.contains("Phone call tools unavailable") == false)
    }

    @Test func liveAgentPostprocessRejectsUnavailableFinalAfterAcceptedContactObservation() {
        let tools = ["contacts.search", "phone.call"].compactMap { ToolRegistry.find(id: $0) }
        let request = AgentRequest(
            systemPrompt: "",
            history: [],
            userMessage: "Call Julie Charlebois",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: tools,
            relevantMemories: []
        )
        let repaired = AgentService.postprocessStructuredFinalAnswerForTests(
            "Contact search is unavailable in this build right now. Phone call tools unavailable.",
            req: request,
            observations: [("contacts.search", "• Julie Charlebois — +1 (514) 555-0101")],
            steps: []
        )

        #expect(repaired.contains("Approval required for phone.call"))
        #expect(!repaired.contains("Contact search is unavailable"))
        #expect(!repaired.contains("Phone call tools unavailable"))
    }
}

struct ToolRegistryFinalizerCoverageTests {
    @Test func userVisibleToolsHaveFinalizerCoverageOrExplicitExemption() {
        let uncovered = ToolRegistry.all
            .filter { ToolObservationFinalizer.finalizerCoverageKind(for: $0) == nil }
            .map(\.id)
            .sorted()
        #expect(uncovered.isEmpty)
    }
}

struct StrictVsProductionFallbackTests {
    @Test func strictTrainingDisablesDeterministicCompatibility() {
        let training = LegacyAgentRunOptions(
            groundingMode: .slotAgent,
            allowDegradedGrounding: false,
            preventDoubleGrounding: true,
            diagnosticsEnabled: false,
            allowDeterministicCompatibility: false,
            allowParseFailureDeterministicRecovery: false,
            allowsMemoryPressureContinuation: true
        )
        let production = LegacyAgentRunOptions(
            groundingMode: .slotAgent,
            allowDegradedGrounding: true,
            preventDoubleGrounding: true,
            diagnosticsEnabled: false,
            allowDeterministicCompatibility: true,
            allowParseFailureDeterministicRecovery: true,
            allowsMemoryPressureContinuation: false
        )
        #expect(!training.allowDeterministicCompatibility)
        #expect(!training.allowParseFailureDeterministicRecovery)
        #expect(training.allowsMemoryPressureContinuation)
        #expect(production.allowDeterministicCompatibility)
        #expect(production.allowParseFailureDeterministicRecovery)
    }
}
