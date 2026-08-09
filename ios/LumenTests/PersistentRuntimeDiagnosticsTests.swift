import XCTest
import SwiftUI
@testable import Lumen

final class PersistentRuntimeDiagnosticsTests: XCTestCase {
    func testCampaignStorePersistsAndRestoresEnabledCampaign() async throws {
        let store = try makeStore()
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: true, maxRunsPerScenario: 3, delayBetweenRunsSeconds: 2, scenarios: [.plainFastPrompt, .agentFastPrompt])
        try await store.saveCampaign(campaign)
        let restored = await store.loadCampaign()
        let unwrapped = try XCTUnwrap(restored)
        XCTAssertEqual(unwrapped.id, campaign.id)
        XCTAssertEqual(unwrapped.enabled, campaign.enabled)
        XCTAssertEqual(unwrapped.runContinuously, campaign.runContinuously)
        XCTAssertEqual(unwrapped.maxRunsPerScenario, campaign.maxRunsPerScenario)
        XCTAssertEqual(unwrapped.delayBetweenRunsSeconds, campaign.delayBetweenRunsSeconds)
        XCTAssertEqual(unwrapped.scenarios, campaign.scenarios)
        XCTAssertLessThan(abs(unwrapped.createdAt.timeIntervalSince(campaign.createdAt)), 1)
        XCTAssertLessThan(abs(unwrapped.updatedAt.timeIntervalSince(campaign.updatedAt)), 1)

        let campaignURL = await store.campaignURL
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: campaignURL)) as? [String: Any]
        )
        XCTAssertEqual(
            Set(object.keys),
            Set([
                "id", "createdAt", "updatedAt", "enabled", "runContinuously",
                "maxRunsPerScenario", "delayBetweenRunsSeconds", "scenarios",
            ])
        )
    }

    func testRunnerSkipsModelScenarioWhenNoModelLoadedInsteadOfFailing() async throws {
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.plainFastPrompt])
        let record = await runner.runOnce(campaign)
        XCTAssertNotEqual(record?.status, .failed)
        XCTAssertEqual(record?.scenario, .plainFastPrompt)
    }

    func testPlainFastPromptExpectationFailsIfFinalCharsExceedFastCap() {
        let result = PersistentRuntimeDiagnosticsRunner.evaluatePlainFastPrompt(finalChars: PromptBudgetConstants.fastInteractiveTotalChars + 1, estimatedTokens: 10, latencyClass: .fastInteractive)
        XCTAssertEqual(result.status, .failed)
        XCTAssertEqual(result.code, "fast_prompt_too_large")
    }

    func testDeveloperTraceBypassIsMarkedExpected() {
        let selection = PromptLatencyClassifier.classify(userMessage: "Yo", attachments: [], developerTraceModeEnabled: true, reasoningCaptureEnabled: true, modelName: "chat")
        XCTAssertEqual(selection.latencyClass, .developerTrace)
    }

    @MainActor
    func testManualOnlyDiagnosticSkipIncludesLocalRemediationProposal() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.liveAgentStream])

        let record = await runner.runOnce(campaign)

        let proposal = try XCTUnwrap(record?.remediationProposals?.first)
        XCTAssertEqual(record?.status, .skipped)
        XCTAssertEqual(proposal.id, "manual-scenario-foreground")
        XCTAssertEqual(proposal.severity, .info)
        XCTAssertTrue(record?.events.contains { $0.code == "diagnostic_remediation_proposal" } ?? false)
        let state = await store.loadState()
        XCTAssertEqual(
            state?.status.lastRemediationSummary,
            PersistentRuntimeDiagnosticsRedactor.summary(
                label: "lastRemediationSummary",
                text: proposal.title
            )
        )
    }

    func testDiskWriteGateBuffersAndDefersDiagnosticsDuringGeneration() async throws {
        let store = try makeStore()
        let lease = DiskWriteBudget.shared.beginGeneration()
        await store.appendEvent(PersistentDiagnosticEvent(code: "diagnostic_write", message: "safe synthetic event"))
        let deferredData = await store.readLogDataForExport()
        let logURL = await store.logURL
        XCTAssertFalse(deferredData.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: logURL.path))
        lease.end()
        await store.flushBufferedIfPossible()
        let data = await store.readLogDataForExport()
        XCTAssertFalse(data.isEmpty)
    }

    func testPendingDiagnosticEntriesArePrivacyProjectedBeforeExport() async throws {
        let store = try makeStore()
        let pendingMessage = "Synthetic pending private message Person Canary"
        let pendingKey = "SyntheticPendingPrivateKeyCanary"
        let pendingValue = "Synthetic pending private calendar title"
        let lease = DiskWriteBudget.shared.beginGeneration()
        defer { lease.end() }

        await store.appendEvent(PersistentDiagnosticEvent(
            code: "sandboxed_tool_plan",
            message: pendingMessage,
            values: ["toolCount": "2", "maxSteps": "2", pendingKey: pendingValue]
        ))
        let logURL = await store.logURL
        XCTAssertFalse(FileManager.default.fileExists(atPath: logURL.path))

        let data = await store.readLogDataForExport(full: true)
        let text = String(decoding: data, as: UTF8.self)
        for canary in [pendingMessage, pendingKey, pendingKey.lowercased(), pendingValue] {
            XCTAssertFalse(text.localizedCaseInsensitiveContains(canary), "Pending export leaked \(canary)")
        }
        XCTAssertTrue(text.contains("metadata_"))
        XCTAssertTrue(text.contains("sha256="))
        XCTAssertTrue(text.contains("\"toolcount\":\"2\""))
    }

    func testCrashResumeDetectionMarksUnfinishedActiveRunInterrupted() async throws {
        let store = try makeStore()
        var state = PersistentDiagnosticState()
        let runID = UUID()
        let campaignID = UUID()
        state.activeRunID = runID
        state.activeCampaignID = campaignID
        state.activeScenario = .lifecycleCancellation
        state.activeStartedAt = Date(timeIntervalSince1970: 100)
        state.activeLaunchUUID = UUID()
        try await store.saveState(state)

        let record = try await store.markUnfinishedRunInterrupted(launchUUID: UUID(), startupAt: Date(timeIntervalSince1970: 200))
        XCTAssertEqual(record?.id, runID)
        XCTAssertEqual(record?.campaignID, campaignID)
        XCTAssertEqual(record?.status, .interrupted)
        XCTAssertEqual(record?.failureSummary, "interrupted_or_terminated")
        XCTAssertEqual(record?.remediationProposals?.first?.id, "inspect-lifecycle-interruption")
        XCTAssertTrue(record?.events.contains { $0.code == "diagnostic_remediation_proposal" } ?? false)
    }

    func testRedactionRemovesPromptMemoryAndFileContentsFromLogEvents() {
        let event = PersistentDiagnosticEvent(
            code: "unsafe event",
            message: "prompt=My private question memory=secret file=/private/var/mobile/doc.txt email user@example.com",
            values: ["prompt": "My private question", "path": "/private/var/mobile/doc.txt"]
        )
        XCTAssertFalse(event.message.contains("My private question"))
        XCTAssertFalse(event.message.contains("secret"))
        XCTAssertFalse(event.message.contains("/private"))
        XCTAssertFalse(event.message.contains("user@example.com"))
        XCTAssertEqual(event.values["prompt"], "[redacted]")
        XCTAssertFalse(event.values.values.contains { $0.contains("My private question") })
    }

    func testPersistentStoreProtectsEveryDiagnosticArtifact() async throws {
        let store = try makeStore()
        try await store.saveCampaign(PersistentDiagnosticCampaign(enabled: true))
        try await store.saveState(PersistentDiagnosticState())
        await store.appendEvent(PersistentDiagnosticEvent(
            code: "sandboxed_tool_plan",
            message: "Synthetic protected diagnostic event",
            values: ["toolCount": "1", "maxSteps": "1"]
        ))
        await store.flushBufferedIfPossible()

        let urls = await [store.campaignURL, store.stateURL, store.logURL]
        for url in urls {
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            XCTAssertEqual(
                attributes[.protectionKey] as? FileProtectionType,
                FileProtectionType.complete,
                "Expected complete file protection for \(url.lastPathComponent)"
            )
        }
    }

    func testPersistentStoreSummarizesFreeFormFieldsAndOpaqueUnknownLabels() async throws {
        let store = try makeStore()
        let privateText = "Synthetic private calendar title for Person Canary"
        let privateMessage = "Synthetic private diagnostic message for Person Canary"
        let unknownKey = "SyntheticPrivateTitleCanary"
        let emailKey = "synthetic.person@example.test"
        let normalizedEmailKey = PersistentRuntimeDiagnosticsRedactor.safeCode(emailKey)
        let ssnKey = "987-65-4321"
        let unknownCode = "SyntheticPrivateEventCodeCanary"
        let event = PersistentDiagnosticEvent(
            code: "sandboxed_tool_plan",
            message: privateMessage,
            values: [
                "toolCount": "2",
                "maxSteps": "2",
                unknownKey: privateText,
                emailKey: privateText,
                ssnKey: privateText,
            ]
        )
        let unknownEvent = PersistentDiagnosticEvent(
            code: unknownCode,
            message: privateMessage,
            values: [unknownKey: privateText]
        )
        var record = PersistentDiagnosticRunRecord(
            campaignID: UUID(),
            scenario: .sandboxedToolPlanOnly,
            status: .failed,
            events: [event, unknownEvent],
            failureSummary: privateText
        )
        record.remediationProposals = [
            PersistentDiagnosticRemediationProposal(
                id: unknownKey,
                title: privateText,
                rationale: privateText,
                action: privateText,
                severity: .warning
            )
        ]
        var state = PersistentDiagnosticState()
        state.records = [record]
        state.status.lastCancellationReason = privateText
        state.status.lastCrashResumeStatus = privateText
        state.status.lastRemediationSummary = privateText

        try await store.saveState(state)
        await store.appendRunUpdate(record)
        await store.flushBufferedIfPossible()

        let stateURL = await store.stateURL
        let logURL = await store.logURL
        let persistedData = try Data(contentsOf: stateURL) + Data(contentsOf: logURL)
        let persistedText = String(decoding: persistedData, as: UTF8.self)
        for canary in [
            privateText, privateMessage, unknownKey, unknownKey.lowercased(), emailKey,
            normalizedEmailKey, ssnKey, unknownCode, unknownCode.lowercased(),
        ] {
            XCTAssertFalse(persistedText.localizedCaseInsensitiveContains(canary), "Persisted canary: \(canary)")
        }
        XCTAssertTrue(persistedText.contains("metadata_"))
        XCTAssertTrue(persistedText.contains("sha256="))

        let loadedState = await store.loadState()
        let restored = try XCTUnwrap(loadedState)
        let restoredRecord = try XCTUnwrap(restored.records.first)
        XCTAssertEqual(restoredRecord.events.first?.code, "sandboxed_tool_plan")
        XCTAssertEqual(restoredRecord.events.last?.code, "other")
        XCTAssertEqual(restoredRecord.events.first?.values["toolcount"], "2")
        XCTAssertTrue(
            restoredRecord.events.first?.values.keys.contains { $0.hasPrefix("metadata_") } == true
        )
        XCTAssertTrue(restoredRecord.failureSummary?.contains("sha256=") == true)
        XCTAssertTrue(restoredRecord.remediationProposals?.first?.title.contains("sha256=") == true)
        XCTAssertTrue(restored.status.lastRemediationSummary?.contains("sha256=") == true)
    }

    @MainActor func testScenarioSelectionPausesWhenResourceBudgetDeniesHeavyWork() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.plainFastPrompt])
        let record = await runner.runOnce(campaign)
        XCTAssertEqual(record?.status, .skipped)
        #endif
    }


    @MainActor func testThermalResourceGateUsesSimulatedDeniedSnapshotWhenRealStateAllowsWork() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .fair, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.thermalResourceGate])

        let record = await runner.runOnce(campaign)

        XCTAssertNil(ResourceBudgetGate.testSnapshotOverride)
        XCTAssertEqual(record?.status, .passed)
        XCTAssertEqual(record?.events.last?.code, "resource_gate_policy_passed")
        XCTAssertEqual(record?.metrics.didFallback, true)
        XCTAssertEqual(record?.metrics.fallbackReason, "resource_gate_probe")
        XCTAssertEqual(record?.metrics.realScenePhase, "active")
        XCTAssertEqual(record?.metrics.realThermalState, DeviceThermalState.fair.rawValue)
        XCTAssertEqual(record?.metrics.realDenied, false)
        XCTAssertEqual(record?.metrics.simulatedScenePhase, "background")
        XCTAssertEqual(record?.metrics.simulatedThermalState, DeviceThermalState.serious.rawValue)
        XCTAssertEqual(record?.metrics.simulatedDenied, true)
        #endif
    }

    func testPersistentDiagnosticsStateCapsCompletedRunIDs() async throws {
        let store = try makeStore()
        var state = PersistentDiagnosticState()
        let overflowCount = PersistentDiagnosticState.maxCompletedRunIDs + 50
        let ids = (0..<overflowCount).map { _ in UUID() }
        state.completedRunIDs = ids

        try await store.saveState(state)

        let loadedState = await store.loadState()
        let restored = try XCTUnwrap(loadedState)
        XCTAssertEqual(restored.completedRunIDs.count, PersistentDiagnosticState.maxCompletedRunIDs)
        XCTAssertEqual(restored.completedRunIDs.first, ids[overflowCount - PersistentDiagnosticState.maxCompletedRunIDs])
        XCTAssertEqual(restored.completedRunIDs.last, ids[overflowCount - 1])
    }

    func testDefaultExporterBoundsNormalExportSizeAndLogLines() async throws {
        let store = try makeStore()
        for index in 0..<700 {
            await store.appendEvent(PersistentDiagnosticEvent(
                code: "bounded_export",
                message: "safe synthetic event \(index)",
                values: ["index": String(index)]
            ))
        }

        let exporter = PersistentRuntimeDiagnosticsExporter(store: store)
        let url = try await exporter.export()
        let data = try Data(contentsOf: url)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let ndjson = try XCTUnwrap(object["ndjson"] as? String)

        XCTAssertLessThanOrEqual(data.count, 1_100_000)
        XCTAssertFalse(ndjson.contains("safe synthetic event"))
        XCTAssertLessThanOrEqual(ndjson.split(separator: "\n").count, 500)
    }

    func testAgentFastPromptScenarioUsesFastPathAndBoundedGroundingMetrics() {
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Yo", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 1, availableTools: ToolRegistry.all, relevantMemories: [])
        let result = SlotAgentService.fastGroundingResult(for: request, options: .default)
        XCTAssertTrue(SlotAgentService.shouldUseFastAgentPath(request))
        XCTAssertTrue(result.bridgedTools.isEmpty)
        XCTAssertLessThanOrEqual(result.userMessage.count + result.systemPrompt.count, PromptBudgetConstants.fastInteractiveTotalChars)
    }

    func testExporterExcludesRawSensitiveContent() async throws {
        let store = try makeStore()
        await store.appendEvent(PersistentDiagnosticEvent(code: "redaction", message: "prompt=Sensitive prompt memory=Private memory file=/tmp/secret.txt"))
        let exporter = PersistentRuntimeDiagnosticsExporter(store: store)
        let url = try await exporter.export()
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertFalse(text.contains("Sensitive prompt"))
        XCTAssertFalse(text.contains("Private memory"))
        XCTAssertFalse(text.contains("/tmp/secret"))
    }

    func testExporterRotatesAllIdentifiersAndUnknownMetadataAcrossShareableExports() async throws {
        let store = try makeStore()
        let campaignID = UUID(uuidString: "11111111-1111-4111-8111-111111111111")!
        let runID = UUID(uuidString: "22222222-2222-4222-8222-222222222222")!
        let launchID = UUID(uuidString: "33333333-3333-4333-8333-333333333333")!
        let eventID = UUID(uuidString: "44444444-4444-4444-8444-444444444444")!
        let unknownEventID = UUID(uuidString: "55555555-5555-4555-8555-555555555555")!
        let emailKey = "private.person@example.com"
        let normalizedEmailKey = PersistentRuntimeDiagnosticsRedactor.safeCode(emailKey)
        let ssnKey = "987-65-4321"
        let privateKey = "privatePayload"
        let privateValue = "Private calendar label for Alexis"
        let privateMessage = "Private diagnostic message for Alexis"
        let callerCorrelationToken = "caller-controlled-correlation-token"
        let arbitraryEventCode = "caller-event-\(callerCorrelationToken)"
        let oversizedCount = "15551234567"
        let epochLikeIndex = "1800000000"
        let promptDigest = String(repeating: "a", count: 64)
        let metricFileName = "mxmetric-2026-08-09-\(runID.uuidString).summary.json"
        let metricJSON = #"{"private":"metric payload for Alexis"}"#

        let campaign = PersistentDiagnosticCampaign(
            id: campaignID,
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            updatedAt: Date(timeIntervalSince1970: 1_800_000_100),
            enabled: true,
            runContinuously: false,
            maxRunsPerScenario: 2,
            delayBetweenRunsSeconds: 1,
            scenarios: [.plainFastPrompt]
        )
        try await store.saveCampaign(campaign)

        var metrics = PersistentDiagnosticMetrics()
        metrics.scenePhase = "active"
        metrics.thermalState = "nominal"
        metrics.promptSHA256 = promptDigest
        metrics.cancellationReason = callerCorrelationToken
        metrics.firstTokenLatencyMs = 42

        let event = PersistentDiagnosticEvent(
            id: eventID,
            at: Date(timeIntervalSince1970: 1_800_000_010),
            code: "sandboxed_tool_plan",
            message: privateMessage,
            values: [
                emailKey: "email-key-canary",
                ssnKey: "ssn-key-canary",
                privateKey: privateValue,
                "correlationToken": callerCorrelationToken,
                "toolCount": "2",
                "maxSteps": "2",
                "count": oversizedCount,
                "index": epochLikeIndex
            ]
        )
        let unknownEvent = PersistentDiagnosticEvent(
            id: unknownEventID,
            at: Date(timeIntervalSince1970: 1_800_000_011),
            code: arbitraryEventCode,
            message: privateMessage,
            values: ["correlationToken": callerCorrelationToken]
        )
        var record = PersistentDiagnosticRunRecord(
            id: runID,
            campaignID: campaignID,
            scenario: .plainFastPrompt,
            startedAt: Date(timeIntervalSince1970: 1_800_000_000),
            status: .failed,
            metrics: metrics,
            events: [event, unknownEvent],
            failureSummary: privateValue
        )
        record.finishedAt = Date(timeIntervalSince1970: 1_800_000_020)
        record.remediationProposals = [
            PersistentDiagnosticRemediationProposal(
                id: "private-remediation-identifier",
                title: privateValue,
                rationale: privateValue,
                action: privateValue,
                severity: .warning
            )
        ]

        var state = PersistentDiagnosticState()
        state.activeRunID = runID
        state.activeCampaignID = campaignID
        state.activeScenario = .plainFastPrompt
        state.activeStartedAt = record.startedAt
        state.activeLaunchUUID = launchID
        state.completedRunIDs = [runID]
        state.records = [record]
        state.status.lastCancellationReason = callerCorrelationToken
        state.status.lastCrashResumeStatus = privateValue
        state.status.lastRemediationSummary = privateValue
        try await store.saveState(state)
        await store.appendRunUpdate(record)

        let metricPayloads = [PersistentMetricKitSourcePayload(fileName: metricFileName, json: metricJSON)]
        let exporter = PersistentRuntimeDiagnosticsExporter(
            store: store,
            metricKitPayloadProvider: { metricPayloads }
        )

        let firstURL = try await exporter.export()
        let firstData = try Data(contentsOf: firstURL)
        let firstObject = try XCTUnwrap(JSONSerialization.jsonObject(with: firstData) as? [String: Any])
        XCTAssertTrue(PersistentRuntimeDiagnosticsExporter.isPrivacySafeShareURL(firstURL))
        XCTAssertFalse(PersistentRuntimeDiagnosticsExporter.isPrivacySafeShareURL(
            firstURL.deletingLastPathComponent().appendingPathComponent("persistent-runtime-diagnostics-export.json")
        ))

        let secondURL = try await exporter.export()
        let secondData = try Data(contentsOf: secondURL)
        let secondObject = try XCTUnwrap(JSONSerialization.jsonObject(with: secondData) as? [String: Any])
        XCTAssertTrue(PersistentRuntimeDiagnosticsExporter.isPrivacySafeShareURL(secondURL))
        XCTAssertNotEqual(firstURL.lastPathComponent, secondURL.lastPathComponent)
        XCTAssertTrue(FileManager.default.fileExists(atPath: firstURL.path))

        let canaries = [
            campaignID.uuidString, runID.uuidString, launchID.uuidString, eventID.uuidString,
            unknownEventID.uuidString,
            emailKey, normalizedEmailKey, ssnKey, privateKey, privateKey.lowercased(), privateValue,
            privateMessage, callerCorrelationToken, promptDigest, metricFileName, metricJSON,
            arbitraryEventCode, PersistentRuntimeDiagnosticsRedactor.safeCode(arbitraryEventCode),
            oversizedCount, epochLikeIndex, "private-remediation-identifier"
        ]
        for data in [firstData, secondData] {
            let text = String(decoding: data, as: UTF8.self)
            for canary in canaries {
                XCTAssertFalse(text.localizedCaseInsensitiveContains(canary), "Export leaked canary: \(canary)")
            }
        }

        let firstProjection = try exportProjection(from: firstObject)
        let secondProjection = try exportProjection(from: secondObject)

        XCTAssertEqual(firstProjection.campaignID, firstProjection.activeCampaignID)
        XCTAssertEqual(firstProjection.campaignID, firstProjection.recordCampaignID)
        XCTAssertEqual(firstProjection.runID, firstProjection.activeRunID)
        XCTAssertEqual(firstProjection.runID, firstProjection.completedRunID)
        XCTAssertEqual(firstProjection.runID, firstProjection.ndjsonRecordID)
        XCTAssertEqual(firstProjection.runID, firstProjection.ndjsonNestedRecordID)
        XCTAssertEqual(firstProjection.eventID, firstProjection.ndjsonEventID)
        XCTAssertEqual(firstProjection.eventCode, "sandboxed_tool_plan")
        XCTAssertEqual(firstProjection.unknownEventCode, "other")
        XCTAssertEqual(firstProjection.ndjsonUnknownEventCode, "other")
        XCTAssertEqual(firstProjection.toolCount, "2")
        XCTAssertEqual(firstProjection.maxSteps, "2")
        XCTAssertFalse(firstProjection.correlationTokenPresent)
        XCTAssertTrue(firstProjection.unknownMetadataKeys.allSatisfy { key in
            key.range(of: #"^metadata_[0-9a-f]{16}$"#, options: .regularExpression) != nil
        })

        XCTAssertNotEqual(firstProjection.exportScope, secondProjection.exportScope)
        XCTAssertNotEqual(firstProjection.campaignID, secondProjection.campaignID)
        XCTAssertNotEqual(firstProjection.runID, secondProjection.runID)
        XCTAssertNotEqual(firstProjection.launchID, secondProjection.launchID)
        XCTAssertNotEqual(firstProjection.eventID, secondProjection.eventID)
        XCTAssertNotEqual(firstProjection.metricFileToken, secondProjection.metricFileToken)
        XCTAssertNotEqual(Set(firstProjection.unknownMetadataKeys), Set(secondProjection.unknownMetadataKeys))
        XCTAssertNotEqual(Set(firstProjection.unknownMetadataValues), Set(secondProjection.unknownMetadataValues))
    }

    @MainActor func testAgentToolPromptDiagnosticUsesDryRunWithoutLiveSlotStream() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.dryRunPromptBudgetOnly])

        let record = await runner.runOnce(campaign)

        XCTAssertEqual(record?.status, .passed)
        XCTAssertEqual(record?.scenario, .dryRunPromptBudgetOnly)
        XCTAssertEqual(record?.metrics.didUseFastPath, false)
        XCTAssertLessThanOrEqual(record?.metrics.groundingChars ?? Int.max, 4_000)
        XCTAssertLessThanOrEqual(record?.metrics.groundingSectionCount ?? Int.max, 6)
        XCTAssertEqual(record?.metrics.inputToolCount, 2)
        XCTAssertEqual(record?.metrics.bridgedToolCount, 2)
        XCTAssertFalse(record?.events.contains { $0.code == PersistentRuntimeDiagnosticSignalKind.slotAgentStart.rawValue } ?? true)
        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testAgentToolPromptDryRunProducesBoundedGroundingWithoutCancellationRegistration() async {
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())

        let result = await SlotAgentService.shared.prepareGroundedRequestForDiagnostics(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true))

        XCTAssertFalse(SlotAgentService.shouldUseFastAgentPath(request))
        XCTAssertLessThanOrEqual(result.userMessage.count + result.systemPrompt.count, 4_000)
        XCTAssertLessThanOrEqual(result.sections.count, 6)
        XCTAssertEqual(result.bridgedTools.count, request.availableTools.count)
        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testLiveSlotAgentStreamUnregistersCancellationBusOnNormalCompletion() async {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())

        for await _ in SlotAgentService.shared.run(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true, allowDeterministicCompatibility: false)) {}
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testLiveSlotAgentStreamUnregistersCancellationBusOnCancellation() async {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())
        let stream = SlotAgentService.shared.run(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true))
        XCTAssertGreaterThanOrEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
        let task = Task {
            for await _ in stream {}
        }

        task.cancel()
        AppCancellationBus.shared.cancel(.chatGeneration)
        _ = await task.result
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testLiveSlotAgentDiagnosticSequenceIncludesPostGroundingMilestones() async {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        let capturedKinds = DiagnosticSignalKindCapture()
        let observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            capturedKinds.append(signal.kind)
        }
        defer { PersistentRuntimeDiagnosticsObserver.shared.removeObserver(observerID) }
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())

        for await _ in SlotAgentService.shared.run(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true, allowDeterministicCompatibility: false)) {}

        let kinds = capturedKinds.snapshot()
        XCTAssertTrue(kinds.contains(.slotAgentGroundingComplete))
        XCTAssertTrue(kinds.contains(.slotAgentEffectiveRequestBuilt))
        XCTAssertTrue(kinds.contains(.slotAgentDeterministicAnswerBuilt))
        XCTAssertTrue(kinds.contains(.slotAgentDoneYielded))
        XCTAssertTrue(kinds.contains(.slotAgentEndEmitted))
        XCTAssertTrue(kinds.contains(.slotAgentContinuationFinished))
    }



    func testAutomaticCampaignNeverSchedulesManualOnlyScenarios() {
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: true, scenarios: [.plainFastPrompt, .lifecycleCancellation, .liveAgentStream, .agentToolPrompt, .sandboxedToolPlanOnly])
        let automatic = campaign.automaticOnly()
        XCTAssertEqual(automatic.scenarios, [.plainFastPrompt, .sandboxedToolPlanOnly])
        XCTAssertFalse(automatic.scenarios.contains { $0.automationPolicy != .automatic })
    }

    func testLiveAgentStreamCannotRunWithoutExplicitUserRequest() async throws {
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let record = await runner.runLiveAgentStream(explicitUserRequested: false)
        XCTAssertNil(record)
    }

    @MainActor func testResourceGateAllowsNominalAndFairActiveState() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .fair, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = nil
    }

    @MainActor func testResourceGateDeniesSeriousCriticalAndBackground() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .serious, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .critical, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = nil
    }

    func testLifecycleProbePassesAfterInactiveBackgroundActiveCycle() async {
        let controller = LifecycleProbeController()
        let record = PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .lifecycleCancellation, status: .running)
        _ = await controller.arm(record: record)
        _ = await controller.record(phase: .inactive)
        _ = await controller.record(phase: .background)
        let result = await controller.record(phase: .active)
        XCTAssertEqual(result?.record.status, .passed)
        XCTAssertEqual(result?.record.metrics.appBecameInactiveOrBackgroundDuringRun, true)
    }

    func testLifecycleProbeSkipsWhenNoTransitionOccurs() async {
        let controller = LifecycleProbeController()
        let record = PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .lifecycleCancellation, status: .running)
        _ = await controller.arm(record: record)
        let finalized = await controller.finalizeWithoutTransition()
        XCTAssertEqual(finalized?.status, .skipped)
        XCTAssertEqual(finalized?.metrics.appBecameInactiveOrBackgroundDuringRun, false)
    }

    func testDiagnosticsWriteBufferCapsRecordCountAndBatchesWrites() async throws {
        let store = try makeStore()
        for index in 0..<60 {
            await store.appendEvent(PersistentDiagnosticEvent(code: "batched", message: "safe synthetic event \(index)"))
        }
        let data = await store.readLogDataForExport(full: true)
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertFalse(text.contains("safe synthetic event"))
        XCTAssertTrue(text.contains("sha256="))

        var state = PersistentDiagnosticState()
        state.records = (0..<520).map { _ in PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .plainFastPrompt, status: .passed) }
        try await store.saveState(state)
        let loadedState = await store.loadState()
        let restored = try XCTUnwrap(loadedState)
        XCTAssertEqual(restored.records.count, 500)
    }

    func testAgentCancellationPersistsCancelledStateNotInterrupted() async throws {
        let store = try makeStore()
        let coordinator = AgentRunCoordinator(store: store)
        let record = PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .agentCancellation, status: .running)
        let task = Task {
            await coordinator.run(record: record, cancellationReason: "unit-test-cancel") { starting in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                return starting
            }
        }
        try? await Task.sleep(nanoseconds: 20_000_000)
        await coordinator.cancelActive(reason: "unit-test-cancel")
        let result = await task.value
        XCTAssertEqual(result.status, .cancelled)
        XCTAssertEqual(result.metrics.didCancel, true)
        XCTAssertEqual(result.metrics.cancellationReason, "unit-test-cancel")
        XCTAssertNotEqual(result.status, .interrupted)
    }


    func testFullDiagnosticsExportDoesNotDuplicateFlushedRingEntries() async throws {
        let store = try makeStore()
        for index in 0..<50 {
            await store.appendEvent(PersistentDiagnosticEvent(code: "no_duplicate", message: "no duplicate event \(index)"))
        }
        let data = await store.readLogDataForExport(full: true)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let entries = data.split(separator: 0x0A).compactMap { line in
            try? decoder.decode(PersistentDiagnosticLogEntry.self, from: Data(line))
        }
        XCTAssertEqual(entries.count, 50)
    }


    func testDiagnosticsStorePurgesLegacyUnredactedJSONLLines() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = PersistentRuntimeDiagnosticsStore(directoryURL: directory)
        let legacyEntry = PersistentDiagnosticLogEntry(kind: "event", recordID: nil, campaignID: nil, event: PersistentDiagnosticEvent(code: "legacy_line", message: "legacy event"), record: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let line = String(data: try encoder.encode(legacyEntry), encoding: .utf8) ?? ""
        try (line + "\n").write(to: directory.appendingPathComponent("persistent-runtime-diagnostics.jsonl"), atomically: true, encoding: .utf8)

        let data = await store.readLogDataForExport(full: true)
        XCTAssertTrue(data.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("persistent-runtime-diagnostics.jsonl").path
        ))
    }

    func testMetricKitExportUsesSummariesAndRetentionCountsOnlyRawPayloads() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let store = MetricKitDiagnosticsStore(directoryURL: directory)
        let payload = Data("{}".utf8)
        for _ in 0..<51 {
            await store.persistMetricPayload(payload)
            try? await Task.sleep(nanoseconds: 1_000_000)
        }

        let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let rawPayloads = urls.filter { $0.lastPathComponent.hasPrefix("mxmetric-") && !$0.lastPathComponent.hasSuffix(".summary.json") }
        let summaries = await store.exportSummaryPayloadURLs()

        XCTAssertEqual(rawPayloads.count, 50)
        XCTAssertEqual(summaries.count, 50)
        XCTAssertTrue(summaries.allSatisfy { $0.lastPathComponent.hasSuffix(".summary.json") })
    }

    private struct ExportProjectionSnapshot {
        let exportScope: String
        let campaignID: String
        let activeCampaignID: String
        let recordCampaignID: String
        let runID: String
        let activeRunID: String
        let completedRunID: String
        let ndjsonRecordID: String
        let ndjsonNestedRecordID: String
        let launchID: String
        let eventID: String
        let ndjsonEventID: String
        let eventCode: String
        let unknownEventCode: String
        let ndjsonUnknownEventCode: String
        let toolCount: String
        let maxSteps: String
        let correlationTokenPresent: Bool
        let unknownMetadataKeys: [String]
        let unknownMetadataValues: [String]
        let metricFileToken: String
    }

    private func exportProjection(from object: [String: Any]) throws -> ExportProjectionSnapshot {
        let campaign = try XCTUnwrap(object["campaign"] as? [String: Any])
        let state = try XCTUnwrap(object["state"] as? [String: Any])
        let records = try XCTUnwrap(state["records"] as? [[String: Any]])
        let record = try XCTUnwrap(records.first)
        let events = try XCTUnwrap(record["events"] as? [[String: Any]])
        let event = try XCTUnwrap(events.first)
        let unknownEvent = try XCTUnwrap(events.dropFirst().first)
        let values = try XCTUnwrap(event["values"] as? [String: String])
        let completedRunIDs = try XCTUnwrap(state["completedRunIDs"] as? [String])

        let ndjson = try XCTUnwrap(object["ndjson"] as? String)
        let ndjsonLine = try XCTUnwrap(ndjson.split(separator: "\n").first)
        let ndjsonObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(ndjsonLine.utf8)) as? [String: Any]
        )
        let ndjsonRecord = try XCTUnwrap(ndjsonObject["record"] as? [String: Any])
        let ndjsonEvents = try XCTUnwrap(ndjsonRecord["events"] as? [[String: Any]])
        let ndjsonEvent = try XCTUnwrap(ndjsonEvents.first)
        let ndjsonUnknownEvent = try XCTUnwrap(ndjsonEvents.dropFirst().first)

        let metricPayloads = try XCTUnwrap(object["metricKitPayloads"] as? [[String: Any]])
        let metricPayload = try XCTUnwrap(metricPayloads.first)
        let unknownMetadataKeys = values.keys.filter { $0.hasPrefix("metadata_") }.sorted()
        let unknownMetadataValues = unknownMetadataKeys.compactMap { values[$0] }.sorted()
        XCTAssertEqual(unknownMetadataKeys.count, 5)

        return ExportProjectionSnapshot(
            exportScope: try XCTUnwrap(object["exportScope"] as? String),
            campaignID: try XCTUnwrap(campaign["id"] as? String),
            activeCampaignID: try XCTUnwrap(state["activeCampaignID"] as? String),
            recordCampaignID: try XCTUnwrap(record["campaignID"] as? String),
            runID: try XCTUnwrap(record["id"] as? String),
            activeRunID: try XCTUnwrap(state["activeRunID"] as? String),
            completedRunID: try XCTUnwrap(completedRunIDs.first),
            ndjsonRecordID: try XCTUnwrap(ndjsonObject["recordID"] as? String),
            ndjsonNestedRecordID: try XCTUnwrap(ndjsonRecord["id"] as? String),
            launchID: try XCTUnwrap(state["activeLaunchID"] as? String),
            eventID: try XCTUnwrap(event["id"] as? String),
            ndjsonEventID: try XCTUnwrap(ndjsonEvent["id"] as? String),
            eventCode: try XCTUnwrap(event["code"] as? String),
            unknownEventCode: try XCTUnwrap(unknownEvent["code"] as? String),
            ndjsonUnknownEventCode: try XCTUnwrap(ndjsonUnknownEvent["code"] as? String),
            toolCount: try XCTUnwrap(values["toolcount"]),
            maxSteps: try XCTUnwrap(values["maxsteps"]),
            correlationTokenPresent: values.keys.contains("correlationtoken"),
            unknownMetadataKeys: unknownMetadataKeys,
            unknownMetadataValues: unknownMetadataValues,
            metricFileToken: try XCTUnwrap(metricPayload["fileToken"] as? String)
        )
    }


    private func makeStore() throws -> PersistentRuntimeDiagnosticsStore {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        return PersistentRuntimeDiagnosticsStore(directoryURL: url)
    }
}

private final class DiagnosticSignalKindCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var kinds: [PersistentRuntimeDiagnosticSignalKind] = []

    func append(_ kind: PersistentRuntimeDiagnosticSignalKind) {
        lock.lock()
        kinds.append(kind)
        lock.unlock()
    }

    func snapshot() -> [PersistentRuntimeDiagnosticSignalKind] {
        lock.lock()
        defer { lock.unlock() }
        return kinds
    }
}
