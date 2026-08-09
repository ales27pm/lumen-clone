import Foundation
import SwiftData

nonisolated struct HeadlessAgentRunResult: Sendable, Equatable {
    nonisolated enum Status: String, Sendable, Equatable {
        case completed
        case deferred
        case blocked
        case failed
        case cancelled
    }

    nonisolated enum IssueCategory: String, Sendable, Equatable {
        case transientUnavailable
        case userInteractionRequired
        case executionFailure
        case cancellation
    }

    let status: Status
    let text: String
    let steps: [AgentStep]
    let code: String?
    let issueCategory: IssueCategory?

    static func completed(text: String, steps: [AgentStep]) -> Self {
        Self(status: .completed, text: text, steps: steps, code: nil, issueCategory: nil)
    }

    static func deferred(text: String, steps: [AgentStep] = [], code: String) -> Self {
        Self(
            status: .deferred,
            text: text,
            steps: steps,
            code: code,
            issueCategory: .transientUnavailable
        )
    }

    static func blocked(text: String, steps: [AgentStep] = [], code: String) -> Self {
        Self(
            status: .blocked,
            text: text,
            steps: steps,
            code: code,
            issueCategory: .userInteractionRequired
        )
    }

    static func failed(text: String, steps: [AgentStep] = [], code: String) -> Self {
        Self(
            status: .failed,
            text: text,
            steps: steps,
            code: code,
            issueCategory: .executionFailure
        )
    }

    static func cancelled(steps: [AgentStep]) -> Self {
        Self(
            status: .cancelled,
            text: "Headless agent cancelled before completion.",
            steps: steps,
            code: "cancelled",
            issueCategory: .cancellation
        )
    }
}

/// Headless runner for AppIntent and scheduled trigger entrypoints.
///
/// This preserves the existing `(text, steps)` return shape while routing the
/// actual turn through `AssistantKernel.run(...)`. It should remain a thin adapter;
/// new orchestration logic belongs in Agent Kernel stages, not here.
@MainActor
enum HeadlessAgentKernelRunner {
    static func run(
        prompt: String,
        appState: AppState,
        context: ModelContext,
        maxSteps: Int? = nil,
        source: AgentKernelSource = .appIntent
    ) async -> (text: String, steps: [AgentStep]) {
        let stored: [StoredModel]
        do {
            stored = try fetchStoredModels(context: context)
        } catch {
            return (storedModelFetchFailureMessage(error: error), [])
        }
        let fleet = LumenModelFleetResolver.resolveV1(appState: appState, storedModels: stored)
        let result = await runWithOutcome(
            prompt: prompt,
            settings: appState.snapshot,
            context: context,
            maxSteps: maxSteps,
            source: source,
            fleetSnapshot: fleet
        )
        return (result.text, result.steps)
    }

    static func run(
        prompt: String,
        settings: SettingsSnapshot,
        context: ModelContext,
        maxSteps: Int? = nil,
        source: AgentKernelSource = .trigger
    ) async -> (text: String, steps: [AgentStep]) {
        let result = await runWithOutcome(
            prompt: prompt,
            settings: settings,
            context: context,
            maxSteps: maxSteps,
            source: source
        )
        return (result.text, result.steps)
    }

    static func runWithOutcome(
        prompt: String,
        settings: SettingsSnapshot,
        context: ModelContext,
        maxSteps: Int? = nil,
        source: AgentKernelSource = .trigger
    ) async -> HeadlessAgentRunResult {
        let stored: [StoredModel]
        do {
            stored = try fetchStoredModels(context: context)
        } catch {
            return .failed(
                text: storedModelFetchFailureMessage(error: error),
                code: "model_catalog_fetch_failed"
            )
        }
        let fleet = LumenModelFleetResolver.resolveV1(settings: settings, storedModels: stored)
        return await runWithOutcome(
            prompt: prompt,
            settings: settings,
            context: context,
            maxSteps: maxSteps,
            source: source,
            fleetSnapshot: fleet
        )
    }

    static func storedModelFetchFailureMessage(error: Error) -> String {
        let errorCode = RuntimeMetricErrorSanitizer.code(for: error)
        return "Headless agent skipped: model catalog fetch failed (\(errorCode))."
    }

    static func nonSuccessResult(
        for toolResult: ToolResult,
        steps: [AgentStep] = []
    ) -> HeadlessAgentRunResult? {
        switch toolResult.status {
        case .success:
            nil
        case .denied, .requiresApproval:
            .blocked(
                text: toolResult.displayText,
                steps: steps,
                code: toolResult.errorCode ?? "background_tool_requires_user_interaction"
            )
        case .unavailable:
            .deferred(
                text: toolResult.displayText,
                steps: steps,
                code: toolResult.errorCode ?? "background_tool_deferred"
            )
        case .failed:
            .failed(
                text: toolResult.displayText,
                steps: steps,
                code: toolResult.errorCode ?? "background_tool_failed"
            )
        }
    }

    static func nonRunnableBackgroundAssessmentResult(
        for assessment: BackgroundToolExecutionAssessment
    ) -> HeadlessAgentRunResult? {
        switch assessment.status {
        case .runnable, .notToolBacked:
            return nil
        case .clarificationRequired:
            return .blocked(
                text: assessment.skipMessage,
                code: "background_clarification_required"
            )
        case .noBackgroundSafeRoutedTools:
            return .blocked(
                text: assessment.skipMessage,
                code: "background_tool_not_safe"
            )
        case .blockedByCurrentPolicy:
            return .blocked(
                text: assessment.skipMessage,
                code: "background_tool_blocked_by_policy"
            )
        case .noRoutedTools:
            return .failed(
                text: assessment.skipMessage,
                code: "background_tool_route_missing"
            )
        case .toolMappingUnavailable:
            return .failed(
                text: assessment.skipMessage,
                code: "background_tool_mapping_unavailable"
            )
        }
    }

    private static func fetchStoredModels(context: ModelContext) throws -> [StoredModel] {
        try context.fetch(FetchDescriptor<StoredModel>())
    }

    private static func runWithOutcome(
        prompt: String,
        settings: SettingsSnapshot,
        context: ModelContext,
        maxSteps: Int?,
        source: AgentKernelSource,
        fleetSnapshot: LumenModelFleetSnapshot
    ) async -> HeadlessAgentRunResult {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return .failed(text: "Headless agent failed: prompt is empty.", code: "empty_prompt")
        }

        let cancellationToken = HeadlessAgentCancellationToken()
        let cancellationID = registerCancellation(token: cancellationToken)
        defer {
            AppCancellationBus.shared.unregister(cancellationID, category: .chatGeneration)
        }

        let backgroundTask = BackgroundRuntimeContinuation.begin(name: "Lumen Headless Agent")
        defer { backgroundTask?.end() }

        guard !cancellationToken.isCancelled, !Task.isCancelled else { return .cancelled(steps: []) }
        let cascade = await MemoryCascade.recall(query: trimmed, history: [], context: context)
        guard !cancellationToken.isCancelled, !Task.isCancelled else { return .cancelled(steps: []) }
        let resolution = ReferenceResolver.resolve(prompt: trimmed, history: [], relevantMemories: cascade.promptFragments)
        let executionPrompt = resolution.rewrittenPrompt
        let routing = await IntentClassifierService.shared.route(executionPrompt)
        guard !cancellationToken.isCancelled, !Task.isCancelled else { return .cancelled(steps: []) }
        let heavyModelAllowed = source != .trigger || ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.background.rawValue)
        let backgroundToolAssessment = source == .trigger
            ? await BackgroundToolExecutionPolicy.assess(
                prompt: executionPrompt,
                routing: routing,
                modelContext: context
            )
            : nil
        guard !cancellationToken.isCancelled, !Task.isCancelled else { return .cancelled(steps: []) }
        if let backgroundToolAssessment,
           let nonRunnableResult = Self.nonRunnableBackgroundAssessmentResult(for: backgroundToolAssessment) {
            return nonRunnableResult
        }
        let canRunBackgroundToolOnly = backgroundToolAssessment?.canRunWithoutLoadedTextRuntime ?? false
        let chatRuntimeLoaded = source == .trigger ? await AppLlamaService.shared.isChatLoaded : true
        guard !cancellationToken.isCancelled, !Task.isCancelled else { return .cancelled(steps: []) }
        if source == .trigger, !canRunBackgroundToolOnly {
            let toolSkipMessage = backgroundToolAssessment?.skipMessage
            if !heavyModelAllowed {
                return .deferred(
                    text: Self.backgroundSkipMessage(toolSkipMessage, fallback: "local model work is temporarily unavailable."),
                    code: "background_model_budget_unavailable"
                )
            }
            if !chatRuntimeLoaded {
                return .deferred(
                    text: Self.backgroundSkipMessage(toolSkipMessage, fallback: "local model not loaded."),
                    code: "background_model_not_loaded"
                )
            }
        }
        let memories = MemoryGate.filter(intent: routing.intent, items: cascade.promptFragments, userMessage: executionPrompt)
        let mimicry = MimicryProfiler.profile(userMessage: executionPrompt, settings: settings)
        let task: AssistantTaskKind = source == .trigger ? .backgroundTrigger : .chat
        let options = AgentKernelOptions(
            allowHeavyRuntime: source == .trigger ? (heavyModelAllowed && chatRuntimeLoaded) : true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: false,
            maxSteps: maxSteps ?? settings.maxAgentSteps,
            prefersFoundationModels: source == .appIntent,
            temperature: settings.temperature,
            topP: settings.topP,
            repetitionPenalty: settings.repetitionPenalty,
            maxTokens: source == .trigger ? min(settings.maxTokens, 256) : min(settings.maxTokens, 500)
        )
        let request = AgentKernelRequest(
            userMessage: executionPrompt,
            history: [],
            systemPrompt: composedSystemPrompt(basePrompt: settings.systemPrompt, fleetSnapshot: fleetSnapshot, mimicry: mimicry),
            relevantMemories: memories,
            task: task,
            source: source,
            options: options
        )

        var final = ""
        var steps: [AgentStep] = []
        var didComplete = false
        var deferredToolResult: ToolResult?
        var blockedToolResult: ToolResult?
        var failedToolResult: ToolResult?
        var kernelFailure: String?
        for await event in AssistantKernel.shared.run(request, modelContext: context) {
            if cancellationToken.isCancelled || Task.isCancelled {
                await AppLlamaService.shared.cancelActiveGeneration(reason: cancellationToken.reason ?? "headless-agent-cancelled")
                break
            }
            switch event {
            case .step(let step):
                if let idx = steps.firstIndex(where: { $0.id == step.id }) { steps[idx] = step }
                else { steps.append(step) }
            case .stepDelta(let id, let text):
                if let idx = steps.firstIndex(where: { $0.id == id }) { steps[idx].content = text }
            case .token(let chunk), .finalDelta(let chunk):
                final += chunk
            case .final(let text):
                final = text.isEmpty ? final : text
            case .done(let finalText, let allSteps):
                final = finalText.isEmpty ? final : finalText
                steps = allSteps.isEmpty ? steps : allSteps
                didComplete = true
            case .error(let message):
                final = message
                kernelFailure = message
            case .toolResult(let result):
                switch Self.nonSuccessResult(for: result)?.status {
                case .failed?:
                    failedToolResult = result
                case .blocked?:
                    blockedToolResult = result
                case .deferred?:
                    deferredToolResult = result
                case .completed?, .cancelled?, nil:
                    break
                }
            case .toolInvocation, .diagnostic:
                break
            }
        }
        guard !cancellationToken.isCancelled, !Task.isCancelled else {
            return .cancelled(steps: steps)
        }
        if let blockedToolResult {
            return Self.nonSuccessResult(for: blockedToolResult, steps: steps)!
        }
        if let kernelFailure {
            return .failed(
                text: kernelFailure,
                steps: steps,
                code: "agent_kernel_failed"
            )
        }
        if let failedToolResult {
            return Self.nonSuccessResult(for: failedToolResult, steps: steps)!
        }
        if let deferredToolResult {
            return Self.nonSuccessResult(for: deferredToolResult, steps: steps)!
        }
        let trimmedFinal = final.trimmingCharacters(in: .whitespacesAndNewlines)
        guard didComplete, !trimmedFinal.isEmpty else {
            return .failed(
                text: "Headless agent failed: no completed result was produced.",
                steps: steps,
                code: "missing_completed_result"
            )
        }
        return .completed(text: trimmedFinal, steps: steps)
    }

    private static func registerCancellation(token: HeadlessAgentCancellationToken) -> UUID {
        AppCancellationBus.shared.registerCancellation({
            let reason = AppCancellationBus.shared.lastCancellationReason ?? "headless-agent-cancelled"
            token.cancel(reason: reason)
            Task.detached(priority: .userInitiated) {
                await AppLlamaService.shared.cancelActiveGeneration(reason: reason)
            }
        }, category: .chatGeneration)
    }

    private static func composedSystemPrompt(basePrompt: String, fleetSnapshot: LumenModelFleetSnapshot, mimicry: MimicryProfile) -> String {
        let trimmedBasePrompt = basePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let contracts = LumenModelSlotContract.all
            .filter { $0.slot != .embedding }
            .map { contract in
                "- \(contract.slot.displayName): \(contract.systemContract)"
            }
            .joined(separator: "\n")

        let assignments = LumenModelSlot.allCases
            .map { slot -> String in
                if let assignment = fleetSnapshot.assignment(for: slot) {
                    let residency: String
                    if fleetSnapshot.runtimeResidentSlots.contains(slot) {
                        residency = "runtime resident"
                    } else if fleetSnapshot.targetResidentSlots.contains(slot) {
                        residency = "target resident · runtime pending"
                    } else {
                        residency = "not resident"
                    }
                    return "- \(slot.displayName): \(assignment.displayName) · \(assignment.parameters) · \(assignment.quantization) · \(residency)"
                }
                return "- \(slot.displayName): missing"
            }
            .joined(separator: "\n")

        let missingText = fleetSnapshot.missingSlots.isEmpty
            ? "none"
            : fleetSnapshot.missingSlots.map(\.displayName).joined(separator: ", ")

        let fleetPrompt = """
        Lumen model fleet v1 is enabled as an explicit role pipeline contract. The Agent Kernel is now the orchestration boundary; slot-specific behavior must be expressed through kernel stages and runtime adapters.

        Role contracts:
        \(contracts)

        Fleet runtime mode: \(fleetSnapshot.mode.displayName).

        Current v1 slot assignments:
        \(assignments)

        Missing slots: \(missingText).

        \(mimicry.promptFragment)
        """

        guard !trimmedBasePrompt.isEmpty else {
            return """
            You are Lumen, a concise on-device assistant.

            \(fleetPrompt)
            """
        }

        return """
        \(trimmedBasePrompt)

        \(fleetPrompt)
        """
    }

    private static func backgroundSkipMessage(_ toolReason: String?, fallback: String) -> String {
        guard let toolReason, !toolReason.isEmpty else {
            return "Background trigger skipped: \(fallback)"
        }
        return "\(toolReason) \(fallback)"
    }
}

private final class HeadlessAgentCancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelledReason: String?

    func cancel(reason: String) {
        lock.lock()
        if cancelledReason == nil {
            cancelledReason = reason
        }
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelledReason != nil
    }

    var reason: String? {
        lock.lock()
        defer { lock.unlock() }
        return cancelledReason
    }
}
