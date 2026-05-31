import Foundation
import SwiftData

@MainActor
enum AgentRunner {
    /// Foreground entry point. Uses the live `AppState` (reads its current snapshot).
    static func runHeadless(prompt: String, appState: AppState, context: ModelContext, maxSteps: Int? = nil) async -> (text: String, steps: [AgentStep]) {
        let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        let fleet = LumenModelFleetResolver.resolveV1(appState: appState, storedModels: stored)
        return await runHeadless(
            prompt: prompt,
            settings: appState.snapshot,
            context: context,
            maxSteps: maxSteps,
            fleetSnapshot: fleet
        )
    }

    /// Background-safe entry point. Takes a Sendable settings snapshot so background
    /// tasks never depend on live in-memory mutable state.
    static func runHeadless(prompt: String, settings: SettingsSnapshot, context: ModelContext, maxSteps: Int? = nil) async -> (text: String, steps: [AgentStep]) {
        let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        let fleet = LumenModelFleetResolver.resolveV1(settings: settings, storedModels: stored)
        return await runHeadless(
            prompt: prompt,
            settings: settings,
            context: context,
            maxSteps: maxSteps,
            fleetSnapshot: fleet
        )
    }

    private static func runHeadless(
        prompt: String,
        settings: SettingsSnapshot,
        context: ModelContext,
        maxSteps: Int?,
        fleetSnapshot: LumenModelFleetSnapshot
    ) async -> (text: String, steps: [AgentStep]) {
        let cascade = await MemoryCascade.recall(query: prompt, history: [], context: context)
        let resolution = ReferenceResolver.resolve(prompt: prompt, history: [], relevantMemories: cascade.promptFragments)
        let executionPrompt = resolution.rewrittenPrompt
        let routing = await IntentClassifierService.shared.route(executionPrompt)

        let grounding = await LegacyTurnGroundingCoordinator.shared.build(userMessage: executionPrompt, conversationID: nil, turnID: nil, history: [], modelContext: context, isBackground: true, task: .backgroundTrigger, role: "headless-trigger")
        let memories = MemoryGate.filter(intent: routing.intent, items: cascade.promptFragments, userMessage: executionPrompt)
        let tools = grounding.legacyTools.filter { settings.enabledToolIDs.contains($0.id) }
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: settings.systemPrompt, baseUserMessage: executionPrompt, sections: grounding.sections, policy: .headlessTrigger, roleMetadata: "headless-trigger")
        let mimicry = MimicryProfiler.profile(userMessage: executionPrompt, settings: settings)
        let req = AgentRequest(
            systemPrompt: composedSystemPrompt(basePrompt: assembled.systemPrompt, fleetSnapshot: fleetSnapshot, mimicry: mimicry),
            history: [],
            userMessage: assembled.userMessage,
            temperature: settings.temperature,
            topP: settings.topP,
            repetitionPenalty: settings.repetitionPenalty,
            maxTokens: settings.maxTokens,
            maxSteps: maxSteps ?? settings.maxAgentSteps,
            availableTools: tools,
            relevantMemories: memories
        )

        var final = ""
        var steps: [AgentStep] = []
        for await event in RolePipelineAgentService.shared.run(req) {
            switch event {
            case .step(let s):
                if let idx = steps.firstIndex(where: { $0.id == s.id }) { steps[idx] = s }
                else { steps.append(s) }
            case .stepDelta(let id, let text):
                if let idx = steps.firstIndex(where: { $0.id == id }) { steps[idx].content = text }
            case .finalDelta(let chunk):
                final += chunk
            case .done(let text, let all):
                final = text.isEmpty ? final : text
                steps = all.isEmpty ? steps : all
            case .error(let msg):
                final = msg
            }
        }
        return (final.trimmingCharacters(in: .whitespacesAndNewlines), steps)
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

        let runtimeMode = """
        Fleet runtime mode: \(fleetSnapshot.mode.displayName).
        v1 target: slot assignments are resolved before runtime. The active pipeline loads only the slot currently generating: Cortex, Executor, Mouth, Mimicry, then asynchronous REM.
        """

        let fleetPrompt = """
        Lumen model fleet v1 is enabled as an explicit role pipeline:
        User input → Cortex route/plan → Executor validate/repair action JSON → native tool execution → Cortex decides whether more evidence is needed → Mouth final answer → Mimicry style pass → user output. REM runs after the response as non-blocking audit/training signal.

        Role contracts:
        \(contracts)

        \(runtimeMode)

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
}
