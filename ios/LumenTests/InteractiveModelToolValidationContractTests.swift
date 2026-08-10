import Foundation
import Testing
@testable import Lumen

@Suite(.serialized)
struct InteractiveModelToolValidationContractTests {
    @Test func scenarioIsOneStrictReadOnlyModelToolProof() async throws {
        #if DEBUG
        let scenario = try #require(E2ETestScenario.interactiveModelToolValidation.only)
        #expect(scenario.evidenceMode == .modelBackedRequired)
        #expect(scenario.requiresAgentRun)
        #expect(scenario.kind == .toolGuard)
        #expect(scenario.expectedToolID == "alarm.authorization_status")
        #expect(scenario.requiredAllowedToolIDs == ["alarm.authorization_status"])

        let routing = await IntentClassifierService.shared.route(scenario.prompt)
        #expect(routing.intent == .alarm)
        #expect(!routing.requiresClarification)
        #expect(E2ETestRunner.requiresStructuredModelBackedAgentRunForTests(scenario: scenario, routing: routing))

        let enabledToolIDs = Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
        #expect(E2ETestRunner.executionToolIDsForTests(
            scenario: scenario,
            routing: routing,
            enabledCanonicalToolIDs: enabledToolIDs
        ) == ["alarm.authorization_status"])

        let definition = await SecureToolRegistry.shared.definitions().first {
            $0.id == "alarm.authorization_status"
        }
        #expect(definition?.category == .readOnly)
        #expect(definition?.requiredPermissions.isEmpty == true)
        #expect(definition?.requiresUserApproval == false)
        #else
        #expect(true)
        #endif
    }

    @Test @MainActor func alarmStatusToolDoesNotPromptThroughGenericPermissionGate() async {
        #if DEBUG
        let failure = await ToolRouteGuard.ensurePermissionIfNeeded(
            for: "alarm.authorization_status",
            arguments: [:]
        )
        #expect(failure == nil)
        #else
        #expect(true)
        #endif
    }

    @Test func attributionRequiresActionFinalAndNativeObservation() throws {
        #if DEBUG
        let scenario = try #require(E2ETestScenario.interactiveModelToolValidation.only)
        let correlation = AgentTraceCorrelation(
            scenarioID: scenario.id,
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID()
        )
        let actionTrace = trace(
            correlation: correlation,
            selectedToolID: "alarm.authorization_status",
            rawOutput: #"{"action":{"tool":"alarm.authorization_status","args":{}}}"#,
            emittedFinal: false,
            successfulObservationCount: 0,
            finalizerAccepted: nil
        )
        let finalTrace = trace(
            correlation: correlation,
            selectedToolID: nil,
            rawOutput: #"{"final":"Alarm authorization status is available."}"#,
            emittedFinal: true,
            successfulObservationCount: 1,
            finalizerAccepted: true
        )
        let observation = AgentStep(
            kind: .observation,
            content: "Alarm authorization status: authorized.",
            toolID: "alarm.authorization_status"
        )
        let groundedFinal = "Alarm authorization status is authorized."

        let complete = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [actionTrace, finalTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: groundedFinal
        )
        #expect(complete.failures.isEmpty)
        #expect(complete.metadata["attributableModelToolEvidence"] == "true")
        #expect(complete.metadata["primaryAgentJSONActionTraceCount"] == "1")
        #expect(complete.metadata["modelFinalTraceCount"] == "1")
        #expect(complete.metadata["nativeToolObservationStepCount"] == "1")
        #expect(complete.metadata["nativeToolResultEvidenceCount"] == "1")
        #expect(complete.metadata["modelFinalMatchesNativeObservation"] == "true")

        let actionOnly = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [actionTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: groundedFinal
        )
        #expect(actionOnly.metadata["attributableModelToolEvidence"] == "false")
        #expect(actionOnly.failures.contains { $0.contains("correlated model final") })

        let tracesWithoutObservation = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [],
            traces: [actionTrace, finalTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: groundedFinal
        )
        #expect(tracesWithoutObservation.metadata["attributableModelToolEvidence"] == "false")
        #expect(tracesWithoutObservation.failures.contains { $0.contains("observation step") })

        let overbroadActionTrace = trace(
            correlation: correlation,
            selectedToolID: "alarm.authorization_status",
            rawOutput: #"{"action":{"tool":"alarm.authorization_status","args":{}}}"#,
            emittedFinal: false,
            successfulObservationCount: 0,
            finalizerAccepted: nil,
            allowedToolIDs: ["alarm.authorization_status", "alarm.list"],
            requiresApproval: false
        )
        let overbroadAction = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [overbroadActionTrace, finalTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: groundedFinal
        )
        #expect(overbroadAction.metadata["attributableModelToolEvidence"] == "false")
        #expect(overbroadAction.failures.contains { $0.contains("primary agent-json action evidence") })

        let wrongStageActionTrace = trace(
            correlation: correlation,
            selectedToolID: "alarm.authorization_status",
            rawOutput: #"{"action":{"tool":"alarm.authorization_status","args":{}}}"#,
            emittedFinal: false,
            successfulObservationCount: 0,
            finalizerAccepted: nil,
            stage: "agent-json-step-2"
        )
        let wrongStageAction = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [wrongStageActionTrace, finalTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: groundedFinal
        )
        #expect(wrongStageAction.metadata["attributableModelToolEvidence"] == "false")

        let overbroadFinalTrace = trace(
            correlation: correlation,
            selectedToolID: nil,
            rawOutput: #"{"final":"Alarm authorization status is available."}"#,
            emittedFinal: true,
            successfulObservationCount: 1,
            finalizerAccepted: true,
            allowedToolIDs: ["alarm.authorization_status", "alarm.list"]
        )
        let overbroadFinal = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [actionTrace, overbroadFinalTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: groundedFinal
        )
        #expect(overbroadFinal.metadata["attributableModelToolEvidence"] == "false")

        let approvalActionTrace = trace(
            correlation: correlation,
            selectedToolID: "alarm.authorization_status",
            rawOutput: #"{"action":{"tool":"alarm.authorization_status","args":{}}}"#,
            emittedFinal: false,
            successfulObservationCount: 0,
            finalizerAccepted: nil,
            allowedToolIDs: ["alarm.authorization_status"],
            requiresApproval: true
        )
        let approvalAction = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [approvalActionTrace, finalTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: groundedFinal
        )
        #expect(approvalAction.metadata["attributableModelToolEvidence"] == "false")
        #expect(approvalAction.failures.contains { $0.contains("primary agent-json action evidence") })

        let missingNativeResult = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [actionTrace, finalTrace],
            nativeToolResultEvidenceCount: 0,
            finalText: groundedFinal
        )
        #expect(missingNativeResult.metadata["attributableModelToolEvidence"] == "false")
        #expect(missingNativeResult.failures.contains { $0.contains("result evidence") })

        let hallucinatedFinal = E2ETestRunner.interactiveModelToolAttributionForTests(
            scenario: scenario,
            correlation: correlation,
            agentSteps: [observation],
            traces: [actionTrace, finalTrace],
            nativeToolResultEvidenceCount: 1,
            finalText: "Alarm authorization status is denied."
        )
        #expect(hallucinatedFinal.metadata["attributableModelToolEvidence"] == "false")
        #expect(hallucinatedFinal.metadata["modelFinalMatchesNativeObservation"] == "false")
        #expect(hallucinatedFinal.failures.contains { $0.contains("did not match") })
        #else
        #expect(true)
        #endif
    }

    @Test func successfulNativeToolResultEmitsOnlyCategoricalEvidence() throws {
        #if DEBUG
        let scenario = try #require(E2ETestScenario.interactiveModelToolValidation.only)
        let result = ToolResult(
            invocationID: UUID(),
            status: .success,
            displayText: "private display text",
            modelText: "private model text",
            structuredPayload: [
                "toolID": "alarm.authorization_status",
                "implementation": "ProductivityLocalTool",
                "availability": "available",
                "runtimeEvidence": "alarmkit-runtime-observed"
            ],
            privacyLevel: .moderate,
            metricsSummary: "native_alarm_tool",
            errorCode: nil
        )
        let evidence = try #require(E2ETestRunner.interactiveModelToolResultEvidenceForTests(
            scenario: scenario,
            result: result
        ))
        #expect(evidence.isAttributableSuccess)
        #expect(evidence.message == "toolID=alarm.authorization_status, status=success, implementation=ProductivityLocalTool, availability=available, runtimeEvidence=alarmkit-runtime-observed")
        #expect(!evidence.message.contains("private"))
        #else
        #expect(true)
        #endif
    }

    @Test func runModeRequiresPhysicalIPhoneAndSafeThermals() {
        #if DEBUG
        #expect(E2ETestRunnerView.blockedRunReason(
            runMode: .interactiveModelToolValidation,
            thermalState: .nominal,
            isPhysicalIPhone: false
        )?.contains("physical iPhone") == true)
        #expect(E2ETestRunnerView.blockedRunReason(
            runMode: .interactiveModelToolValidation,
            thermalState: .nominal,
            isPhysicalIPhone: true
        ) == nil)
        #expect(E2ETestRunnerView.blockedRunReason(
            runMode: .interactiveModelToolValidation,
            thermalState: .serious,
            isPhysicalIPhone: true
        ) == ResourceBudgetGate.seriousThermalRetryHint)
        #else
        #expect(true)
        #endif
    }

    @Test func evidenceExportRequiresTheCurrentViewSessionsCompletedReport() {
        #if DEBUG
        let currentID = UUID()
        let staleID = UUID()
        #expect(E2ETestRunnerView.isJustFinishedInteractiveModelToolValidationReport(
            reportID: currentID,
            resultScenarioIDs: ["interactive-model-tool-alarm-authorization"],
            completedReportID: currentID
        ))
        #expect(!E2ETestRunnerView.isJustFinishedInteractiveModelToolValidationReport(
            reportID: staleID,
            resultScenarioIDs: ["interactive-model-tool-alarm-authorization"],
            completedReportID: currentID
        ))
        #expect(!E2ETestRunnerView.isJustFinishedInteractiveModelToolValidationReport(
            reportID: currentID,
            resultScenarioIDs: ["different-scenario"],
            completedReportID: currentID
        ))
        #else
        #expect(true)
        #endif
    }

    @Test func evidencePackageSourceActionIsClosedAndCanonical() {
        #if DEBUG
        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "interactive-model-tool-validation-live-e2e",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 0,
            sourceAction: .interactiveModelToolValidation
        )
        #expect(package.testFlight.sourceAction == InAppDatasetPackageSourceAction.interactiveModelToolValidation.rawValue)
        #expect(InAppDatasetPackageSourceAction(rawValue: "caller supplied private text") == nil)
        #else
        #expect(true)
        #endif
    }

    @Test func attributionReceiptMetadataExportsAsSafeCategoriesAndCounts() throws {
        #if DEBUG
        let scenario = try #require(E2ETestScenario.interactiveModelToolValidation.only)
        let correlation = AgentTraceCorrelation(
            scenarioID: scenario.id,
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID()
        )
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: scenario.id,
            kind: scenario.kind.rawValue,
            title: scenario.title,
            prompt: scenario.prompt,
            expectedIntent: scenario.expectedIntent.rawValue,
            actualIntent: scenario.expectedIntent.rawValue,
            e2eRunID: correlation.e2eRunID,
            agentRunID: correlation.agentRunID,
            conversationID: correlation.conversationID,
            turnID: correlation.turnID,
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.modelBackedRequired.rawValue,
            passed: true,
            failures: [],
            finalText: "Alarm authorization status: authorized.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: "",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil,
            metadata: [
                "attributableModelToolEvidence": "true",
                "modelFinalMatchesNativeObservation": "true",
                "primaryAgentJSONActionTraceCount": "1",
                "modelFinalTraceCount": "1",
                "nativeToolObservationStepCount": "1",
                "nativeToolResultEvidenceCount": "1"
            ]
        )
        let exported = EvidenceLayerExporter.privacySafeE2EResultForExport(result)
        #expect(exported.metadata["attributableModelToolEvidence"] == "true")
        #expect(exported.metadata["modelFinalMatchesNativeObservation"] == "true")
        #expect(exported.metadata["primaryAgentJSONActionTraceCount"] == "1")
        #expect(exported.metadata["modelFinalTraceCount"] == "1")
        #expect(exported.metadata["nativeToolObservationStepCount"] == "1")
        #expect(exported.metadata["nativeToolResultEvidenceCount"] == "1")

        let unrelatedCorrelation = AgentTraceCorrelation(
            scenarioID: "unrelated-scenario",
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID()
        )
        let relatedTraces = [
            trace(
                correlation: correlation,
                selectedToolID: "alarm.authorization_status",
                rawOutput: #"{"action":{"tool":"alarm.authorization_status","args":{}}}"#,
                emittedFinal: false,
                successfulObservationCount: 0,
                finalizerAccepted: nil
            ),
            trace(
                correlation: correlation,
                selectedToolID: nil,
                rawOutput: #"{"final":"Alarm authorization status is authorized."}"#,
                emittedFinal: true,
                successfulObservationCount: 1,
                finalizerAccepted: true
            )
        ]
        let package = InAppDatasetPackageExporter.makePackageForTests(
            liveE2EReport: E2ETestReport(
                id: UUID(),
                startedAt: result.startedAt,
                finishedAt: result.finishedAt,
                passed: 1,
                failed: 0,
                results: [result]
            ),
            traces: relatedTraces + [
                trace(
                    correlation: unrelatedCorrelation,
                    selectedToolID: "alarm.authorization_status",
                    rawOutput: #"{"action":{"tool":"alarm.authorization_status","args":{}}}"#,
                    emittedFinal: false,
                    successfulObservationCount: 0,
                    finalizerAccepted: nil
                )
            ],
            sourceAction: .interactiveModelToolValidation
        )
        #expect(package.recentTraces.count == 2)
        #expect(package.liveE2EReport?.correlatedTraceCount == 2)
        #expect(package.exportQualityFailures?.isEmpty == true)
        #else
        #expect(true)
        #endif
    }

    #if DEBUG
    private func trace(
        correlation: AgentTraceCorrelation,
        selectedToolID: String?,
        rawOutput: String,
        emittedFinal: Bool,
        successfulObservationCount: Int,
        finalizerAccepted: Bool?,
        allowedToolIDs: [String] = ["alarm.authorization_status"],
        requiresApproval: Bool = false,
        stage: String? = nil
    ) -> AgentBehaviorTrace {
        AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: stage ?? (selectedToolID == nil ? "agent-json-step-1" : "agent-json-step-0"),
            scenarioID: correlation.scenarioID,
            e2eRunID: correlation.e2eRunID,
            agentRunID: correlation.agentRunID,
            conversationID: correlation.conversationID,
            turnID: correlation.turnID,
            intent: UserIntent.alarm.rawValue,
            promptPrefix: "prompt summary",
            rawOutputPrefix: rawOutput,
            selectedToolID: selectedToolID,
            toolArguments: [:],
            allowedToolIDs: allowedToolIDs,
            requiresApproval: requiresApproval,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: emittedFinal,
            outputTokenCount: 8,
            runtimePath: "agent-model",
            streamStarted: true,
            selectedRuntime: "llama",
            modelIdentifier: "test-model",
            modelLoaded: true,
            firstChunkReceived: true,
            textChunkCount: 1,
            finalChunkReceived: true,
            streamTerminationReason: "stop",
            successfulObservationCount: successfulObservationCount,
            finalizerAccepted: finalizerAccepted
        )
    }
    #endif
}

private extension Collection {
    var only: Element? {
        count == 1 ? first : nil
    }
}
