import Foundation

nonisolated struct VoiceKernelEventState: Equatable {
    var finalText: String = ""
    var responseText: String = ""
    var steps: [AgentStep] = []
    var diagnostics: [AgentKernelDiagnosticEvent] = []
    var degraded: Bool = false
    var errorMessage: String?
    var isDone: Bool = false
    var isCancelled: Bool = false
    var cancellationReason: String?
}

nonisolated struct VoiceKernelEventMutation: Equatable {
    var textChanged: Bool = false
    var stepsChanged: Bool = false
    var diagnosticsChanged: Bool = false
    var shouldEmitUIUpdateDiagnostic: Bool = false
    var shouldEmitStepFeedback: Bool = false
    var shouldStartSpeaking: Bool = false
    var shouldSpeakPending: Bool = false
    var shouldUseFinalResponseText: Bool = false
}

nonisolated enum VoiceKernelEventReducer {
    static func reduce(
        _ event: AgentKernelEvent,
        state: inout VoiceKernelEventState,
        lastUserMessage: String,
        routing: IntentRoutingDecision
    ) -> VoiceKernelEventMutation {
        switch event {
        case .step(let step):
            upsert(step, in: &state.steps)
            return VoiceKernelEventMutation(stepsChanged: true, shouldEmitStepFeedback: true)

        case .stepDelta(let id, let text):
            guard let idx = state.steps.firstIndex(where: { $0.id == id }) else {
                return VoiceKernelEventMutation()
            }
            state.steps[idx].content = text
            return VoiceKernelEventMutation(stepsChanged: true)

        case .token(let chunk), .finalDelta(let chunk):
            appendVisibleText(chunk, state: &state)
            return VoiceKernelEventMutation(
                textChanged: true,
                shouldEmitUIUpdateDiagnostic: true,
                shouldStartSpeaking: true,
                shouldSpeakPending: true
            )

        case .final(let text):
            replaceVisibleTextIfPresent(text, state: &state, lastUserMessage: lastUserMessage, routing: routing)
            return VoiceKernelEventMutation(
                textChanged: !text.isEmpty,
                shouldEmitUIUpdateDiagnostic: !text.isEmpty,
                shouldStartSpeaking: !text.isEmpty,
                shouldSpeakPending: !text.isEmpty,
                shouldUseFinalResponseText: !text.isEmpty
            )

        case .toolInvocation(let invocation):
            let step = AgentStep(
                id: invocation.id,
                kind: .action,
                content: toolInvocationContent(invocation),
                toolID: invocation.toolID,
                toolArgs: toolArguments(invocation)
            )
            upsert(step, in: &state.steps)
            return VoiceKernelEventMutation(stepsChanged: true, shouldEmitStepFeedback: true)

        case .toolResult(let result):
            let step = AgentStep(
                kind: .observation,
                content: toolResultContent(result),
                toolID: nil,
                toolArgs: toolResultArguments(result)
            )
            state.steps.append(step)
            return VoiceKernelEventMutation(stepsChanged: true)

        case .diagnostic(let diagnostic):
            state.diagnostics.append(diagnostic)
            var mutation = VoiceKernelEventMutation(diagnosticsChanged: true)
            if isDegradedDiagnostic(diagnostic) {
                state.degraded = true
                upsertDegradedStep(from: diagnostic, in: &state.steps)
                mutation.stepsChanged = true
            }
            return mutation

        case .error(let message):
            state.errorMessage = message
            replaceVisibleTextIfPresent(message, state: &state, lastUserMessage: lastUserMessage, routing: routing)
            return VoiceKernelEventMutation(
                textChanged: true,
                shouldEmitUIUpdateDiagnostic: true,
                shouldStartSpeaking: true,
                shouldSpeakPending: true,
                shouldUseFinalResponseText: true
            )

        case .done(let finalText, let steps):
            replaceVisibleTextIfPresent(finalText, state: &state, lastUserMessage: lastUserMessage, routing: routing)
            if !steps.isEmpty {
                state.steps = steps
            }
            state.isDone = true
            return VoiceKernelEventMutation(
                textChanged: !finalText.isEmpty,
                stepsChanged: !steps.isEmpty,
                shouldStartSpeaking: !finalText.isEmpty,
                shouldSpeakPending: !finalText.isEmpty,
                shouldUseFinalResponseText: !finalText.isEmpty
            )
        }
    }

    static func cancel(state: inout VoiceKernelEventState, reason: String) -> VoiceKernelEventMutation {
        state.isCancelled = true
        state.cancellationReason = reason
        return VoiceKernelEventMutation()
    }

    static func streamingResponseText(
        from text: String,
        lastUserMessage: String
    ) -> String {
        let sanitized = AssistantOutputSanitizer.sanitize(text, lastUserMessage: lastUserMessage)
        return SchemaPlaceholderDetector.isPlaceholderPrefix(sanitized) ? "" : sanitized
    }

    static func finalResponseText(
        from text: String,
        lastUserMessage: String,
        routing: IntentRoutingDecision
    ) -> String {
        let sanitized = streamingResponseText(from: text, lastUserMessage: lastUserMessage)
        return FinalIntentValidator.validate(sanitized, routing: routing, fallback: nil)
    }

    private static func appendVisibleText(
        _ chunk: String,
        state: inout VoiceKernelEventState
    ) {
        state.finalText += chunk
        state.responseText = state.finalText
    }

    private static func replaceVisibleTextIfPresent(
        _ text: String,
        state: inout VoiceKernelEventState,
        lastUserMessage: String,
        routing: IntentRoutingDecision
    ) {
        guard !text.isEmpty else { return }
        state.finalText = text
        state.responseText = finalResponseText(from: text, lastUserMessage: lastUserMessage, routing: routing)
    }

    private static func upsert(_ step: AgentStep, in steps: inout [AgentStep]) {
        if let idx = steps.firstIndex(where: { $0.id == step.id }) {
            steps[idx] = step
        } else {
            steps.append(step)
        }
    }

    private static func toolInvocationContent(_ invocation: ToolInvocation) -> String {
        guard !invocation.arguments.isEmpty else {
            return "Invoking \(invocation.toolID)"
        }
        let arguments = invocation.arguments.keys.sorted()
            .map { key in "\(key)=\(invocation.arguments[key] ?? "")" }
            .joined(separator: ", ")
        return "Invoking \(invocation.toolID) with \(arguments)"
    }

    private static func toolArguments(_ invocation: ToolInvocation) -> [String: String] {
        var args = invocation.arguments
        args["invocationID"] = invocation.id.uuidString
        args["source"] = invocation.source.rawValue
        return args
    }

    private static func toolResultContent(_ result: ToolResult) -> String {
        let visible = result.displayText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !visible.isEmpty { return visible }

        let modelText = result.modelText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelText.isEmpty { return modelText }

        return "Tool result: \(result.status.rawValue)"
    }

    private static func toolResultArguments(_ result: ToolResult) -> [String: String] {
        var args: [String: String] = [
            "invocationID": result.invocationID.uuidString,
            "status": result.status.rawValue,
            "privacyLevel": result.privacyLevel.rawValue
        ]
        if !result.metricsSummary.isEmpty {
            args["metricsSummary"] = result.metricsSummary
        }
        if let errorCode = result.errorCode, !errorCode.isEmpty {
            args["errorCode"] = errorCode
        }
        return args
    }

    private static func isDegradedDiagnostic(_ diagnostic: AgentKernelDiagnosticEvent) -> Bool {
        if diagnostic.metadata["runtime"] == AssistantRuntimeKind.deterministicFallback.rawValue {
            return true
        }
        if diagnostic.metadata.values.contains(where: { value in
            let normalized = value.lowercased()
            return normalized.contains("fallback") || normalized.contains("degraded")
        }) {
            return true
        }
        let stage = diagnostic.stage.lowercased()
        let message = diagnostic.message.lowercased()
        return stage.contains("fallback")
            || stage.contains("degraded")
            || message.contains("fallback")
            || message.contains("degraded")
    }

    private static func upsertDegradedStep(
        from diagnostic: AgentKernelDiagnosticEvent,
        in steps: inout [AgentStep]
    ) {
        let content = diagnostic.message.isEmpty ? "Agent Kernel is running in degraded mode." : diagnostic.message
        if let idx = steps.firstIndex(where: { step in
            step.kind == .reflection && step.toolArgs?["diagnosticStage"] == diagnostic.stage
        }) {
            steps[idx].content = content
            return
        }

        steps.append(AgentStep(
            kind: .reflection,
            content: content,
            toolID: nil,
            toolArgs: ["diagnosticStage": diagnostic.stage]
        ))
    }
}
