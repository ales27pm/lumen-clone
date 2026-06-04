import XCTest
import SwiftUI
@testable import Lumen

final class PersistentRuntimeDiagnosticsTests: XCTestCase {
    func testCampaignStorePersistsAndRestoresEnabledCampaign() async throws {
        let store = try makeStore()
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: true, maxRunsPerScenario: 3, delayBetweenRunsSeconds: 2, scenarios: [.plainFastPrompt, .agentFastPrompt])
        try await store.saveCampaign(campaign)
        let restored = await store.loadCampaign()
        XCTAssertEqual(restored, campaign)
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

    func testDiskWriteGateBuffersAndDefersDiagnosticsDuringGeneration() async throws {
        let store = try makeStore()
        let lease = DiskWriteBudget.shared.beginGeneration()
        await store.appendEvent(PersistentDiagnosticEvent(code: "diagnostic_write", message: "safe synthetic event"))
        XCTAssertEqual(await store.readLogDataForExport(), Data())
        lease.end()
        await store.flushBufferedIfPossible()
        let data = await store.readLogDataForExport()
        XCTAssertFalse(data.isEmpty)
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
        XCTAssertFalse(event.values.description.contains("My private question"))
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
        let text = try String(contentsOf: url)
        XCTAssertFalse(text.contains("Sensitive prompt"))
        XCTAssertFalse(text.contains("Private memory"))
        XCTAssertFalse(text.contains("/tmp/secret"))
    }

    private func makeStore() throws -> PersistentRuntimeDiagnosticsStore {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        return PersistentRuntimeDiagnosticsStore(directoryURL: url)
    }
}
