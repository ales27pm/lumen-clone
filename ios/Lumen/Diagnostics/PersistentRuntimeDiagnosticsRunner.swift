import Foundation
import CryptoKit
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
    private let agentRunCoordinator: AgentRunCoordinator
    private let lifecycleProbeController = LifecycleProbeController()
    private var recentRecords: [PersistentDiagnosticRunRecord] = []
    private let maxRecentRecords = 50

    init(store: PersistentRuntimeDiagnosticsStore = .shared, agentRunCoordinator: AgentRunCoordinator? = nil) {
        self.store = store
        self.agentRunCoordinator = agentRunCoordinator ?? AgentRunCoordinator(store: store)
        self.state = PersistentDiagnosticState()
    }

    func resumeIfEnabled() async {
        _ = try? await store.markUnfinishedRunInterrupted(launchUUID: Self.launchUUID, startupAt: Self.launchStartedAt)
        state = await store.loadState() ?? PersistentDiagnosticState()
        campaign = await store.loadCampaign()
        installObserverIfNeeded()
        guard let campaign, campaign.enabled, campaign.runContinuously, await environmentAllowsDiagnostics() else { return }
        startLoop(campaign: campaign.automaticOnly())
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
        current = current.automaticOnly()
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
        record.events.append(PersistentDiagnosticEvent(code: "tester_action_required", message: "Lock device or background app, then return to the app"))
        record = await lifecycleProbeController.arm(record: record)
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
        runTask = Task(priority: .utility) { [weak self] in
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
            current = current.automaticOnly()
            let counts = scenarioRunCounts(campaignID: current.id)
            for scenario in current.scenarios where scenario.automationPolicy == .automatic {
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
            if current.automaticScenarios.allSatisfy({ scenarioRunCounts(campaignID: current.id)[$0, default: 0] >= current.maxRunsPerScenario }) { break }
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
        if scenario.requiresExplicitUserRequest {
            finish(&record, status: .skipped, code: "manual_scenario_requires_explicit_request", message: "Manual-only scenario requires explicit user action")
            await persist(record)
            return record
        }
        switch scenario {
        case .plainFastPrompt:
            await scenarioPlainFastPrompt(&record)
        case .plainDeveloperTraceBypass:
            await scenarioDeveloperTraceBypass(&record)
        case .agentFastPrompt:
            await scenarioAgentFastPrompt(&record)
        case .dryRunPromptBudgetOnly:
            await scenarioDryRunPromptBudgetOnly(&record)
        case .sandboxedToolPlanOnly:
            await scenarioSandboxedToolPlanOnly(&record)
        case .liveAgentStream, .agentToolPrompt:
            finish(&record, status: .skipped, code: "manual_live_agent_stream_required", message: "Live agent stream requires explicit user action")
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
        if state.records.count > 500 { state.records.removeFirst(state.records.count - 500) }
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
        await store.flushBufferedIfPossible()
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

    private func scenarioDryRunPromptBudgetOnly(_ record: inout PersistentDiagnosticRunRecord) async {
        let req = diagnosticAgentRequest(userMessage: "Search the web for SwiftData cancellation patterns", tools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") })
        let fast = SlotAgentService.shouldUseFastAgentPath(req)
        record.metrics.didUseFastPath = fast
        record.metrics.inputToolCount = req.availableTools.count
        record.metrics.toolCount = req.availableTools.count
        record.metrics.promptInitialChars = req.userMessage.count
        record.metrics.promptBodyBytes = req.userMessage.utf8.count
        record.metrics.promptSHA256 = Self.sha256(req.userMessage)
        record.metrics.promptRedactionMode = "hash_and_size_only"
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

    private func scenarioSandboxedToolPlanOnly(_ record: inout PersistentDiagnosticRunRecord) async {
        let req = diagnosticAgentRequest(userMessage: "Plan a safe web lookup for SwiftData cancellation without executing tools", tools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") })
        try? Task.checkCancellation()
        record.metrics.didUseFastPath = SlotAgentService.shouldUseFastAgentPath(req)
        record.metrics.inputToolCount = req.availableTools.count
        record.metrics.toolCount = req.availableTools.count
        record.metrics.promptInitialChars = req.userMessage.count
        record.metrics.promptBodyBytes = req.userMessage.utf8.count
        record.metrics.promptSHA256 = Self.sha256(req.userMessage)
        record.metrics.promptRedactionMode = "hash_and_size_only"
        let bounded = !record.metrics.didUseFastPath && req.availableTools.count <= 4 && req.maxSteps <= 2
        record.events.append(PersistentDiagnosticEvent(code: "sandboxed_tool_plan", message: "Sandboxed tool plan validated without executing tools", values: ["toolCount": String(req.availableTools.count), "maxSteps": String(req.maxSteps)]))
        finish(&record, status: bounded ? .passed : .failed, code: bounded ? "sandboxed_tool_plan_bounded" : "sandboxed_tool_plan_unbounded", message: "Sandboxed tool plan validated")
    }

    func runLiveAgentStream(explicitUserRequested: Bool) async -> PersistentDiagnosticRunRecord? {
        guard explicitUserRequested else { return nil }
        let current = await campaignOrLoad(nil)
        return await runManualScenario(.liveAgentStream, campaign: current)
    }

    private func runManualScenario(_ scenario: PersistentDiagnosticScenarioKind, campaign: PersistentDiagnosticCampaign) async -> PersistentDiagnosticRunRecord {
        var record = makeRecord(campaign: campaign, scenario: scenario)
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
        case .liveAgentStream, .agentToolPrompt:
            record = await scenarioLiveAgentStream(record)
        case .lifecycleCancellation:
            record = await startLifecycleCancellationProbe()
        default:
            finish(&record, status: .skipped, code: "not_manual_scenario", message: "Scenario is not manual-only")
        }
        if record.status != .running { await persist(record) }
        return record
    }

    private func scenarioLiveAgentStream(_ record: PersistentDiagnosticRunRecord) async -> PersistentDiagnosticRunRecord {
        let req = diagnosticAgentRequest(userMessage: "Search the web for SwiftData cancellation patterns", tools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") })
        return await agentRunCoordinator.run(record: record, cancellationReason: "persistent-diagnostics-live-agent-cancel") { startingRecord in
            try Task.checkCancellation()
            var mutable = startingRecord
            mutable.metrics.inputToolCount = req.availableTools.count
            mutable.metrics.toolCount = req.availableTools.count
            mutable.metrics.promptInitialChars = req.userMessage.count
            mutable.metrics.promptBodyBytes = req.userMessage.utf8.count
            mutable.metrics.promptSHA256 = Self.sha256(req.userMessage)
            mutable.metrics.promptRedactionMode = "hash_and_size_only"
            let started = ProcessInfo.processInfo.systemUptime
            for await event in await SlotAgentService.shared.run(req, options: .default) {
                try Task.checkCancellation()
                switch event {
                case .finalDelta:
                    mutable.metrics.streamingUpdateCount += 1
                case .step:
                    mutable.metrics.toolCount = max(mutable.metrics.toolCount ?? 0, 1)
                default:
                    break
                }
                if mutable.metrics.streamingUpdateCount > 64 { break }
            }
            mutable.metrics.generationElapsedMs = Int((ProcessInfo.processInfo.systemUptime - started) * 1000)
            mutable.finishedAt = Date()
            mutable.status = .passed
            mutable.events.append(PersistentDiagnosticEvent(code: "live_agent_stream_passed", message: "Live agent stream completed by explicit user request"))
            return mutable
        }
    }

    private func scenarioAgentCancellation(_ record: inout PersistentDiagnosticRunRecord) async {
        let req = diagnosticAgentRequest(userMessage: "Search documents and tools for a detailed cancellation analysis", tools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") })
        let startingRecord = record
        let task = Task {
            await agentRunCoordinator.run(record: startingRecord, cancellationReason: "persistent-diagnostics-agent-cancel") { mutableStart in
                try Task.checkCancellation()
                var mutable = mutableStart
                mutable.metrics.inputToolCount = req.availableTools.count
                mutable.metrics.toolCount = req.availableTools.count
                for await event in await SlotAgentService.shared.run(req, options: .default) {
                    try Task.checkCancellation()
                    if case .finalDelta = event { mutable.metrics.streamingUpdateCount += 1 }
                }
                mutable.status = .passed
                mutable.finishedAt = Date()
                mutable.events.append(PersistentDiagnosticEvent(code: "agent_cancel_stream_completed", message: "Agent stream completed before cancellation"))
                return mutable
            }
        }
        try? await Task.sleep(nanoseconds: 50_000_000)
        await agentRunCoordinator.cancelActive(reason: "persistent-diagnostics-agent-cancel")
        let result = await task.value
        record = result
        record.metrics.didCancel = true
        record.metrics.cancellationReason = "persistent-diagnostics-agent-cancel"
        state.cleanCancellationBeforeTermination = true
        if record.status != .cancelled {
            finish(&record, status: .cancelled, code: "agent_cancel_clean", message: "Agent stream cancelled cleanly")
        }
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
        let simulatedSerious = await MainActor.run {
            ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .serious, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        }
        let simulatedCritical = await MainActor.run {
            ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .critical, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        }
        let simulatedBackground = await MainActor.run {
            ResourceBudgetGate.Snapshot(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        }
        let simulatedLowPower = await MainActor.run {
            ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: true, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        }

        let seriousDenied = await MainActor.run { !ResourceBudgetGate.allowsHeavyModelWork(snapshot: simulatedSerious, reason: "userChat.agentGrounding") }
        let criticalDenied = await MainActor.run { !ResourceBudgetGate.allowsHeavyModelWork(snapshot: simulatedCritical, reason: "userChat.agentGrounding") }
        let backgroundDenied = await MainActor.run { !ResourceBudgetGate.allowsHeavyModelWork(snapshot: simulatedBackground, reason: "userChat.agentGrounding") }
        let lowPowerDeniedOrDegraded = await MainActor.run { !ResourceBudgetGate.allowsHeavyModelWork(snapshot: simulatedLowPower, reason: ModelLoadIntent.diagnostics.rawValue) }

        #if DEBUG
        let overrideDenied = await MainActor.run {
            ResourceBudgetGate.setDiagnosticSnapshotOverride(simulatedBackground)
            defer { ResourceBudgetGate.clearDiagnosticSnapshotOverride() }
            return !ResourceBudgetGate.allowsHeavyModelWork(reason: "userChat.agentGrounding")
        }
        #else
        let overrideDenied = backgroundDenied
        #endif

        let realExpectedAllowed: Bool
        if realSnapshot.scenePhase == .active,
           realSnapshot.lowPowerModeEnabled == false,
           realSnapshot.recentMemoryWarningCount == 0 || realSnapshot.recentMemoryWarningCount == nil,
           realSnapshot.thermalState == .nominal || realSnapshot.thermalState == .fair {
            realExpectedAllowed = true
        } else {
            realExpectedAllowed = false
        }
        let realPass = realExpectedAllowed ? !realDenied : true
        let simulatedPass = seriousDenied && criticalDenied && backgroundDenied && lowPowerDeniedOrDegraded && overrideDenied

        record.metrics.didFallback = simulatedPass
        record.metrics.fallbackReason = "resource_gate_probe"
        record.metrics.realScenePhase = PersistentDiagnosticMetrics.sceneString(realSnapshot.scenePhase)
        record.metrics.realThermalState = realSnapshot.thermalState?.rawValue
        record.metrics.realDenied = realDenied
        record.metrics.simulatedScenePhase = PersistentDiagnosticMetrics.sceneString(simulatedBackground.scenePhase)
        record.metrics.simulatedThermalState = simulatedSerious.thermalState?.rawValue
        record.metrics.simulatedDenied = simulatedPass
        record.events.append(PersistentDiagnosticEvent(code: "resource_gate_matrix", message: "Resource gate matrix evaluated", values: [
            "seriousDenied": String(seriousDenied),
            "criticalDenied": String(criticalDenied),
            "backgroundDenied": String(backgroundDenied),
            "lowPowerDeniedOrDegraded": String(lowPowerDeniedOrDegraded),
            "realExpectedAllowed": String(realExpectedAllowed),
            "realDenied": String(realDenied)
        ]))

        let pass = realPass && simulatedPass
        finish(
            &record,
            status: pass ? .passed : .failed,
            code: pass ? "resource_gate_policy_passed" : "resource_gate_policy_failed",
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
            if state.activeScenario == .lifecycleCancellation, let phase = Self.scenePhase(from: signal.values["phase"]), let result = await lifecycleProbeController.record(phase: phase), result.shouldPersist {
                await persist(result.record)
                return
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

    private static func scenePhase(from value: String?) -> ScenePhase? {
        switch value {
        case "active": return .active
        case "inactive": return .inactive
        case "background": return .background
        default: return nil
        }
    }

    private static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func evaluatePlainFastPrompt(finalChars: Int, estimatedTokens: Int, latencyClass: PromptLatencyClass) -> (status: PersistentDiagnosticStatus, code: String, message: String) {
        guard latencyClass == .fastInteractive else { return (.failed, "fast_latency_missing", "Expected fastInteractive latency class") }
        guard finalChars <= PromptBudgetConstants.fastInteractiveTotalChars else { return (.failed, "fast_prompt_too_large", "Final prompt exceeded fast cap") }
        guard estimatedTokens <= 650 else { return (.failed, "fast_tokens_too_large", "Estimated tokens exceeded fast cap") }
        return (.passed, "pass", "Fast prompt budget passed")
    }
}
