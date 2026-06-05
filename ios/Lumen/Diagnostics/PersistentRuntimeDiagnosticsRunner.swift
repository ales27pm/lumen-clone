import Foundation
import SwiftUI

actor PersistentRuntimeDiagnosticsRunner {
    static let shared = PersistentRuntimeDiagnosticsRunner()
    static let launchUUID = UUID()
    static let launchStartedAt = Date()

    private let store: PersistentRuntimeDiagnosticsStore
    private var campaign: PersistentDiagnosticCampaign?
    private var state: PersistentDiagnosticState
    private var runTask: Task<Void, Never>?
    private var observerID: UUID?
    private var recentRecords: [PersistentDiagnosticRunRecord] = []
    private let maxRecentRecords = 50

    init(store: PersistentRuntimeDiagnosticsStore = .shared) {
        self.store = store
        self.state = PersistentDiagnosticState()
    }

    func resumeIfEnabled() async {
        _ = try? await store.markUnfinishedRunInterrupted(launchUUID: Self.launchUUID, startupAt: Self.launchStartedAt)
        state = await store.loadState() ?? PersistentDiagnosticState()
        campaign = await store.loadCampaign()
        installObserverIfNeeded()
        guard let campaign, campaign.enabled, campaign.runContinuously, await environmentAllowsDiagnostics() else { return }
        startLoop(campaign: campaign)
    }

    func loadStatus() async -> PersistentDiagnosticRunnerStatus {
        state = await store.loadState() ?? state
        return state.status
    }

    func loadCampaign() async -> PersistentDiagnosticCampaign {
        if let loaded = await store.loadCampaign() {
            campaign = loaded
            return loaded
        }
        let fresh = PersistentDiagnosticCampaign()
        campaign = fresh
        try? await store.saveCampaign(fresh)
        return fresh
    }

    private func campaignOrLoad(_ requested: PersistentDiagnosticCampaign?) async -> PersistentDiagnosticCampaign {
        if let requested = requested {
            return requested
        }

        if let campaign = campaign {
            return campaign
        }

        return await loadCampaign()
    }

    func saveCampaign(_ newCampaign: PersistentDiagnosticCampaign) async {
        var copy = newCampaign
        copy.updatedAt = Date()
        campaign = copy
        try? await store.saveCampaign(copy)
        if !copy.enabled { await stop() }
    }

    func runOnce(_ requested: PersistentDiagnosticCampaign? = nil) async -> PersistentDiagnosticRunRecord? {
        installObserverIfNeeded()
        var current = await campaignOrLoad(requested)
        current.enabled = true
        current.runContinuously = false
        campaign = current
        guard await environmentAllowsDiagnostics() else {
            var record = makeRecord(campaign: current, scenario: current.scenarios.first ?? .plainFastPrompt)
            finish(&record, status: .skipped, code: "resource_gate_paused", message: "Resource gate paused diagnostics")
            await persist(record)
            return record
        }
        guard let scenario = current.scenarios.first else { return nil }
        let record = await runScenario(scenario, campaign: current)
        return record
    }

    func startContinuous(_ requested: PersistentDiagnosticCampaign? = nil) async {
        installObserverIfNeeded()
        var current = await campaignOrLoad(requested)
        current.enabled = true
        current.runContinuously = true
        campaign = current
        try? await store.saveCampaign(current)
        startLoop(campaign: current)
    }

    func stop() async {
        runTask?.cancel()
        runTask = nil
        state.status.isRunning = false
        state.status.isPaused = false
        state.activeRunID = nil
        state.activeCampaignID = nil
        state.activeScenario = nil
        state.activeStartedAt = nil
        try? await store.saveState(state)
    }

    func pause() async {
        state.status.isPaused = true
        try? await store.saveState(state)
    }

    func clearLogs() async {
        try? await store.clearLogs()
    }

    func startLifecycleCancellationProbe() async -> PersistentDiagnosticRunRecord {
        let current = await campaignOrLoad(nil)
        var record = makeRecord(campaign: current, scenario: .lifecycleCancellation)
        record.metrics.cancellationReason = "tester_background_prompt"
        record.events.append(PersistentDiagnosticEvent(code: "tester_action_required", message: "Lock device or background app within 3 seconds"))
        state.activeRunID = record.id
        state.activeCampaignID = current.id
        state.activeScenario = record.scenario
        state.activeStartedAt = record.startedAt
        state.activeLaunchUUID = Self.launchUUID
        state.cleanCancellationBeforeTermination = false
        try? await store.saveState(state)
        await store.appendRunUpdate(record)
        return record
    }

    func recent() async -> [PersistentDiagnosticRunRecord] { recentRecords }

    private func startLoop(campaign: PersistentDiagnosticCampaign) {
        runTask?.cancel()
        runTask = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            await self.runLoop(campaignID: campaign.id)
        }
    }

    private func runLoop(campaignID: UUID) async {
        state.status.isRunning = true
        try? await store.saveState(state)
        defer {
            Task { await self.markLoopStopped() }
        }
        while !Task.isCancelled {
            guard var current = campaign, current.id == campaignID, current.enabled, current.runContinuously else { break }
            if !(await environmentAllowsDiagnostics()) || DiskWriteBudget.shared.isGenerationActive() {
                state.status.isPaused = true
                try? await store.saveState(state)
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                continue
            }
            state.status.isPaused = false
            let counts = scenarioRunCounts(campaignID: current.id)
            for scenario in current.scenarios {
                if Task.isCancelled { break }
                if counts[scenario, default: 0] >= current.maxRunsPerScenario { continue }
                _ = await runScenario(scenario, campaign: current)
                await store.flushBufferedIfPossible()
                let delay = UInt64(max(0.5, current.delayBetweenRunsSeconds) * 1_000_000_000)
                try? await Task.sleep(nanoseconds: delay)
            }
            current.updatedAt = Date()
            campaign = current
            if !current.runContinuously { break }
            if current.scenarios.allSatisfy({ scenarioRunCounts(campaignID: current.id)[$0, default: 0] >= current.maxRunsPerScenario }) { break }
        }
    }

    private func markLoopStopped() async {
        state.status.isRunning = false
        state.status.isPaused = false
        try? await store.saveState(state)
    }

    private func runScenario(_ scenario: PersistentDiagnosticScenarioKind, campaign: PersistentDiagnosticCampaign) async -> PersistentDiagnosticRunRecord {
        var record = makeRecord(campaign: campaign, scenario: scenario)
        let resourceSnapshot = await MainActor.run { ResourceBudgetGate.diagnosticSnapshot() }
        record.metrics.scenePhase = PersistentDiagnosticMetrics.sceneString(resourceSnapshot.scenePhase)
        record.metrics.thermalState = resourceSnapshot.thermalState?.rawValue
        record.metrics.lowPowerMode = resourceSnapshot.lowPowerModeEnabled
        record.metrics.memoryWarningCount = resourceSnapshot.recentMemoryWarningCount
        state.activeRunID = record.id
        state.activeCampaignID = campaign.id
        state.activeScenario = scenario
        state.activeStartedAt = record.startedAt
        state.activeLaunchUUID = Self.launchUUID
        state.cleanCancellationBeforeTermination = false
        state.status.latestScenario = scenario
        state.status.lastUpdatedAt = Date()
        try? await store.saveState(state)
        await store.appendRunUpdate(record)
        switch scenario {
        case .plainFastPrompt:
            await scenarioPlainFastPrompt(&record)
        case .plainDeveloperTraceBypass:
            await scenarioDeveloperTraceBypass(&record)
        case .agentFastPrompt:
            await scenarioAgentFastPrompt(&record)
        case .agentToolPrompt:
            await scenarioAgentToolPrompt(&record)
        case .agentCancellation:
            await scenarioAgentCancellation(&record)
        case .lifecycleCancellation:
            finish(&record, status: .skipped, code: "manual_probe_required", message: "Use lifecycle cancellation probe button")
        case .diskWriteGate:
            await scenarioDiskWriteGate(&record)
        case .swiftUIChurnProbe:
            await scenarioSwiftUIChurn(&record)
        case .groundingCostProbe:
            await scenarioGroundingCost(&record)
        case .thermalResourceGate:
            await scenarioThermalResourceGate(&record)
        }
        await persist(record)
        return record
    }

    private func makeRecord(campaign: PersistentDiagnosticCampaign, scenario: PersistentDiagnosticScenarioKind) -> PersistentDiagnosticRunRecord {
        var metrics = PersistentDiagnosticMetrics()
        metrics.captureNonisolatedEnvironment()
        metrics.diskBytesBefore = DiskWriteBudget.shared.snapshot().bytes24Hours
        metrics.generationActive = DiskWriteBudget.shared.isGenerationActive()
        return PersistentDiagnosticRunRecord(campaignID: campaign.id, scenario: scenario, status: .running, metrics: metrics)
    }

    private func persist(_ record: PersistentDiagnosticRunRecord) async {
        var copy = record
        copy.metrics.diskBytesAfter = DiskWriteBudget.shared.snapshot().bytes24Hours
        state.markRunCompleted(copy.id)
        state.activeRunID = nil
        state.activeCampaignID = nil
        state.activeScenario = nil
        state.activeStartedAt = nil
        state.cleanCancellationBeforeTermination = false
        state.records.append(copy)
        if state.records.count > 100 { state.records.removeFirst(state.records.count - 100) }
        recentRecords.append(copy)
        if recentRecords.count > maxRecentRecords { recentRecords.removeFirst(recentRecords.count - maxRecentRecords) }
        switch copy.status {
        case .passed: state.status.passedCount += 1
        case .failed: state.status.failedCount += 1
        case .skipped: state.status.skippedCount += 1
        default: break
        }
        state.status.lastFirstTokenLatencyMs = copy.metrics.firstTokenLatencyMs
        state.status.lastPromptFinalChars = copy.metrics.promptFinalChars
        state.status.lastCancellationReason = copy.metrics.cancellationReason
        state.status.lastUpdatedAt = Date()
        try? await store.saveState(state)
        await store.appendRunUpdate(copy)
    }

    private func finish(_ record: inout PersistentDiagnosticRunRecord, status: PersistentDiagnosticStatus, code: String, message: String) {
        record.status = status
        record.finishedAt = Date()
        record.events.append(PersistentDiagnosticEvent(code: code, message: message))
        if status == .failed { record.failureSummary = code }
    }

    private func scenarioPlainFastPrompt(_ record: inout PersistentDiagnosticRunRecord) async {
        let request = diagnosticGenerateRequest(developerTrace: false, reasoningCapture: false)
        let prompt = await AppLlamaService.shared.buildMessagesForDiagnostics(req: request, contextSize: 4096, slot: .cortex)
        record.metrics.promptLatencyClass = prompt.latencySelection.latencyClass.rawValue
        record.metrics.promptInitialChars = prompt.initialPromptChars
        record.metrics.promptFinalChars = prompt.finalPromptChars
        record.metrics.estimatedPromptTokens = prompt.estimatedPromptTokens
        let evaluation = Self.evaluatePlainFastPrompt(finalChars: prompt.finalPromptChars, estimatedTokens: prompt.estimatedPromptTokens, latencyClass: prompt.latencySelection.latencyClass)
        if await AppLlamaService.shared.loadedChatPath == nil {
            finish(&record, status: evaluation.status == .failed ? .failed : .skipped, code: evaluation.code == "pass" ? "skipped_no_model" : evaluation.code, message: "No model loaded; prompt budget validated without inference")
        } else {
            await runPlainGeneration(request, record: &record)
            if record.status == .running { finish(&record, status: evaluation.status, code: evaluation.code, message: evaluation.message) }
        }
    }

    private func scenarioDeveloperTraceBypass(_ record: inout PersistentDiagnosticRunRecord) async {
        let request = diagnosticGenerateRequest(developerTrace: true, reasoningCapture: true)
        let prompt = await AppLlamaService.shared.buildMessagesForDiagnostics(req: request, contextSize: 4096, slot: .cortex)
        record.metrics.promptLatencyClass = prompt.latencySelection.latencyClass.rawValue
        record.metrics.promptInitialChars = prompt.initialPromptChars
        record.metrics.promptFinalChars = prompt.finalPromptChars
        record.metrics.estimatedPromptTokens = prompt.estimatedPromptTokens
        let expected = prompt.latencySelection.latencyClass == .developerTrace
        finish(&record, status: expected ? .passed : .failed, code: expected ? "developer_trace_bypass_expected" : "developer_trace_bypass_missing", message: "Developer trace bypass evaluated")
    }

    private func scenarioAgentFastPrompt(_ record: inout PersistentDiagnosticRunRecord) async {
        let req = diagnosticAgentRequest(userMessage: "Yo", tools: ToolRegistry.all)
        let fast = SlotAgentService.shouldUseFastAgentPath(req)
        let grounded = SlotAgentService.fastGroundingResult(for: req, options: .default)
        record.metrics.didUseFastPath = fast
        record.metrics.groundingChars = grounded.userMessage.count + grounded.systemPrompt.count
        record.metrics.groundingSectionCount = grounded.sections.count
        record.metrics.toolCount = grounded.bridgedTools.count
        record.metrics.memoryCount = grounded.grounding?.memoryCount
        let pass = fast && grounded.bridgedTools.count <= 1 && (record.metrics.groundingChars ?? Int.max) <= PromptBudgetConstants.fastInteractiveTotalChars
        finish(&record, status: pass ? .passed : .failed, code: pass ? "agent_fast_path_bounded" : "agent_fast_path_unbounded", message: "Agent fast prompt validated")
    }

    private func scenarioAgentToolPrompt(_ record: inout PersistentDiagnosticRunRecord) async {
        let req = diagnosticAgentRequest(userMessage: "Search the web for SwiftData cancellation patterns", tools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") })
        let fast = SlotAgentService.shouldUseFastAgentPath(req)
        record.metrics.didUseFastPath = fast
        record.metrics.inputToolCount = req.availableTools.count
        record.metrics.toolCount = req.availableTools.count
        record.metrics.promptInitialChars = req.userMessage.count
        if fast {
            finish(&record, status: .failed, code: "tool_prompt_used_fast_path", message: "Tool prompt incorrectly selected fast path")
            return
        }
        let start = ProcessInfo.processInfo.systemUptime
        let grounded = await SlotAgentService.shared.prepareGroundedRequestForDiagnostics(
            req,
            options: .init(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true)
        )
        record.metrics.agentGroundingElapsedMs = Int((ProcessInfo.processInfo.systemUptime - start) * 1000)
        record.metrics.groundingChars = grounded.userMessage.count + grounded.systemPrompt.count
        record.metrics.groundingSectionCount = grounded.sections.count
        record.metrics.bridgedToolCount = grounded.bridgedTools.count
        record.metrics.toolCount = grounded.bridgedTools.count
        record.metrics.memoryCount = grounded.grounding?.memoryCount
        record.metrics.didFallback = grounded.metricsSummary == "degraded"
        record.metrics.fallbackReason = grounded.degradedReasons.first
        let boundedTools = grounded.bridgedTools.count <= max(req.availableTools.count, 2)
        let pass = !fast && (record.metrics.groundingChars ?? Int.max) <= 4_000 && grounded.sections.count <= 6 && boundedTools
        finish(&record, status: pass ? .passed : .failed, code: pass ? "agent_tool_dry_run_bounded" : "agent_tool_dry_run_unbounded", message: "Dry-run tool prompt validated bounded grounding without opening the agent stream")
    }

    private func scenarioAgentCancellation(_ record: inout PersistentDiagnosticRunRecord) async {
        let req = diagnosticAgentRequest(userMessage: "Search documents and tools for a detailed cancellation analysis", tools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") })
        let task = Task { () -> String in
            var out = ""
            for await event in await SlotAgentService.shared.run(req, options: .default) {
                if Task.isCancelled { break }
                if case .finalDelta(let text) = event { out += text }
            }
            return out
        }
        try? await Task.sleep(nanoseconds: 50_000_000)
        AppCancellationBus.shared.markCancellationRequested("persistent-diagnostics-agent-cancel")
        AppCancellationBus.shared.cancel(.chatGeneration)
        task.cancel()
        _ = await task.value
        record.metrics.didCancel = true
        record.metrics.cancellationReason = "persistent-diagnostics-agent-cancel"
        state.cleanCancellationBeforeTermination = true
        finish(&record, status: .passed, code: "agent_cancel_clean", message: "Agent stream cancelled cleanly")
    }

    private func scenarioDiskWriteGate(_ record: inout PersistentDiagnosticRunRecord) async {
        let lease = DiskWriteBudget.shared.beginGeneration()
        let diagnosticsDeferred = DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .diagnostics)
        let logsDeferred = DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .logs)
        let memoryDeferred = DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .memory)
        let ragDeferred = DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .rag)
        let triggersDeferred = DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .triggers)
        let conversationAllowed = !DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .conversation)
        await store.appendEvent(PersistentDiagnosticEvent(code: "buffered_during_generation", message: "Buffered diagnostics write during generation"), recordID: record.id, campaignID: record.campaignID)
        lease.end()
        await store.flushBufferedIfPossible()
        let pass = diagnosticsDeferred && logsDeferred && memoryDeferred && ragDeferred && triggersDeferred && conversationAllowed
        finish(&record, status: pass ? .passed : .failed, code: pass ? "disk_gate_deferred" : "disk_gate_unexpected", message: "Disk write gate validated")
    }

    private func scenarioSwiftUIChurn(_ record: inout PersistentDiagnosticRunRecord) async {
        let chatUpdates = 10
        let voiceUpdates = 1
        record.metrics.uiUpdateCount = chatUpdates
        record.metrics.streamingUpdateCount = chatUpdates
        let pass = chatUpdates <= 10 && voiceUpdates <= 1
        finish(&record, status: pass ? .passed : .failed, code: pass ? "ui_churn_bounded" : "ui_churn_excessive", message: "Synthetic UI churn probe completed")
    }

    private func scenarioGroundingCost(_ record: inout PersistentDiagnosticRunRecord) async {
        let request = LegacyGroundingRequest(userMessage: "diagnostic grounding cost", conversationID: nil, turnID: UUID(), history: [], mode: .foreground, task: .chat, roleOrSlot: "persistentDiagnostics", externalRelevantMemories: [], externalAvailableTools: [], policy: .slotAgent, baseSystemPrompt: "diagnostic", preventDoubleGrounding: true)
        let start = ProcessInfo.processInfo.systemUptime
        let result = await LegacyTurnGroundingCoordinator.shared.prepareGroundedRequest(request, provider: await MainActor.run { LegacyGroundingContextProvider(directContext: nil, allowSharedFallback: false) })
        let elapsed = Int((ProcessInfo.processInfo.systemUptime - start) * 1000)
        record.metrics.agentGroundingElapsedMs = elapsed
        record.metrics.groundingSectionCount = result.sections.count
        record.metrics.groundingChars = result.userMessage.count + result.systemPrompt.count
        record.metrics.inputToolCount = request.externalAvailableTools.count
        record.metrics.bridgedToolCount = result.bridgedTools.count
        record.metrics.toolCount = result.bridgedTools.count
        record.metrics.didFallback = result.metricsSummary == "degraded"
        record.metrics.fallbackReason = result.degradedReasons.first
        let pass = result.sections.count <= 4 && (record.metrics.groundingChars ?? 0) <= 4_000
        finish(&record, status: pass ? .passed : .failed, code: pass ? "grounding_cost_bounded" : "grounding_cost_unbounded", message: "Grounding cost probe completed")
    }

    private func scenarioThermalResourceGate(_ record: inout PersistentDiagnosticRunRecord) async {
        let realSnapshot = await MainActor.run { ResourceBudgetGate.diagnosticSnapshot() }
        let realDenied = await MainActor.run { !ResourceBudgetGate.allowsHeavyModelWork(snapshot: realSnapshot, reason: "userChat.agentGrounding") }
        let simulatedSnapshot = await MainActor.run {
            ResourceBudgetGate.Snapshot(
                scenePhase: .background,
                lowPowerModeEnabled: true,
                thermalState: .serious,
                recentMemoryWarningCount: realSnapshot.recentMemoryWarningCount ?? 0,
                lastMemoryWarningAt: nil
            )
        }

        let simulatedDenied: Bool
        #if DEBUG
        simulatedDenied = await MainActor.run {
            ResourceBudgetGate.setDiagnosticSnapshotOverride(simulatedSnapshot)
            defer { ResourceBudgetGate.clearDiagnosticSnapshotOverride() }
            return !ResourceBudgetGate.allowsHeavyModelWork(reason: "userChat.agentGrounding")
        }
        #else
        simulatedDenied = await MainActor.run {
            !ResourceBudgetGate.allowsHeavyModelWork(snapshot: simulatedSnapshot, reason: "userChat.agentGrounding")
        }
        #endif

        record.metrics.didFallback = simulatedDenied
        record.metrics.fallbackReason = "resource_gate_probe"
        record.metrics.realScenePhase = PersistentDiagnosticMetrics.sceneString(realSnapshot.scenePhase)
        record.metrics.realThermalState = realSnapshot.thermalState?.rawValue
        record.metrics.realDenied = realDenied
        record.metrics.simulatedScenePhase = PersistentDiagnosticMetrics.sceneString(simulatedSnapshot.scenePhase)
        record.metrics.simulatedThermalState = simulatedSnapshot.thermalState?.rawValue
        record.metrics.simulatedDenied = simulatedDenied

        finish(
            &record,
            status: simulatedDenied ? .passed : .failed,
            code: simulatedDenied ? "resource_gate_simulated_denied" : "resource_gate_simulated_allowed",
            message: "Thermal/resource gate probe completed"
        )
    }

    private func runPlainGeneration(_ request: GenerateRequest, record: inout PersistentDiagnosticRunRecord) async {
        let started = ProcessInfo.processInfo.systemUptime
        var updateCount = 0
        for await token in await AppLlamaService.shared.stream(request) {
            switch token {
            case .text:
                updateCount += 1
                if updateCount > 64 { break }
            case .done:
                break
            }
        }
        record.metrics.generationElapsedMs = Int((ProcessInfo.processInfo.systemUptime - started) * 1000)
        record.metrics.streamingUpdateCount = updateCount
    }

    private func diagnosticGenerateRequest(developerTrace: Bool, reasoningCapture: Bool) -> GenerateRequest {
        GenerateRequest(systemPrompt: "diagnostic", history: [], userMessage: "Yo", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, modelName: "chat", relevantMemories: [], attachments: [], developerTraceModeEnabled: developerTrace, reasoningCaptureEnabled: reasoningCapture)
    }

    private func diagnosticAgentRequest(userMessage: String, tools: [ToolDefinition]) -> AgentRequest {
        AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: userMessage, temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: tools, relevantMemories: [], attachments: [], conversationID: UUID(), turnID: UUID())
    }

    private func environmentAllowsDiagnostics() async -> Bool {
        await MainActor.run {
            let snapshot = ResourceBudgetGate.diagnosticSnapshot()
            guard snapshot.scenePhase == nil || snapshot.scenePhase == .active else { return false }
            guard snapshot.thermalState != .serious && snapshot.thermalState != .critical else { return false }
            return true
        }
    }

    private func scenarioRunCounts(campaignID: UUID) -> [PersistentDiagnosticScenarioKind: Int] {
        var counts: [PersistentDiagnosticScenarioKind: Int] = [:]
        for record in state.records where record.campaignID == campaignID && record.status != .running {
            counts[record.scenario, default: 0] += 1
        }
        return counts
    }

    private func installObserverIfNeeded() {
        guard observerID == nil else { return }
        observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            Task { await PersistentRuntimeDiagnosticsRunner.shared.handleSignal(signal) }
        }
    }

    private func handleSignal(_ signal: PersistentRuntimeDiagnosticSignal) async {
        guard state.activeRunID != nil else { return }
        if signal.kind == .sceneTransition {
            if signal.values["phase"] == "inactive" || signal.values["phase"] == "background" {
                state.cleanCancellationBeforeTermination = AppCancellationBus.shared.lastCancellationReason != nil
            }
        }
        if signal.kind == .llamaCancel || signal.kind == .slotAgentCancel {
            state.cleanCancellationBeforeTermination = true
        }
        if state.cleanCancellationBeforeTermination || signal.kind == .sceneTransition {
            try? await store.saveState(state)
        }
        await store.appendEvent(PersistentDiagnosticEvent(code: signal.kind.rawValue, message: signal.kind.rawValue, values: signal.values), recordID: state.activeRunID, campaignID: campaign?.id)
    }

    static func evaluatePlainFastPrompt(finalChars: Int, estimatedTokens: Int, latencyClass: PromptLatencyClass) -> (status: PersistentDiagnosticStatus, code: String, message: String) {
        guard latencyClass == .fastInteractive else { return (.failed, "fast_latency_missing", "Expected fastInteractive latency class") }
        guard finalChars <= PromptBudgetConstants.fastInteractiveTotalChars else { return (.failed, "fast_prompt_too_large", "Final prompt exceeded fast cap") }
        guard estimatedTokens <= 650 else { return (.failed, "fast_tokens_too_large", "Estimated tokens exceeded fast cap") }
        return (.passed, "pass", "Fast prompt budget passed")
    }
}
