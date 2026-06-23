import Foundation

nonisolated struct ChatKernelEventState: Equatable {
    var steps: [AgentStep] = []
    var finalText: String = ""
    var streamingText: String = ""
    var diagnostics: [AgentKernelDiagnosticEvent] = []
    var degraded: Bool = false
    var errorMessage: String?
    var isDone: Bool = false
}

nonisolated struct ChatKernelEventMutation: Equatable {
    var textChanged: Bool = false
    var stepsChanged: Bool = false
    var diagnosticsChanged: Bool = false
    var shouldEmitUIUpdateDiagnostic: Bool = false
    var shouldEmitStepFeedback: Bool = false
}

nonisolated enum ChatKernelEventReducer {
    static func reduce(
        _ event: AgentKernelEvent,
        state: inout ChatKernelEventState,
        lastUserMessage: String
    ) -> ChatKernelEventMutation {
        switch event {
        case .step(let step):
            upsert(step, in: &state.steps)
            return ChatKernelEventMutation(stepsChanged: true, shouldEmitStepFeedback: true)

        case .stepDelta(let id, let text):
            guard let idx = state.steps.firstIndex(where: { $0.id == id }) else {
                return ChatKernelEventMutation()
            }
            state.steps[idx].content = text
            return ChatKernelEventMutation(stepsChanged: true)

        case .token(let chunk), .finalDelta(let chunk):
            appendVisibleText(chunk, state: &state, lastUserMessage: lastUserMessage)
            return ChatKernelEventMutation(textChanged: true, shouldEmitUIUpdateDiagnostic: true)

        case .final(let text):
            replaceVisibleTextIfPresent(text, state: &state, lastUserMessage: lastUserMessage)
            return ChatKernelEventMutation(textChanged: !text.isEmpty, shouldEmitUIUpdateDiagnostic: !text.isEmpty)

        case .toolInvocation(let invocation):
            let step = AgentStep(
                id: invocation.id,
                kind: .action,
                content: toolInvocationContent(invocation),
                toolID: invocation.toolID,
                toolArgs: toolArguments(invocation)
            )
            upsert(step, in: &state.steps)
            return ChatKernelEventMutation(stepsChanged: true, shouldEmitStepFeedback: true)

        case .toolResult(let result):
            let step = AgentStep(
                kind: .observation,
                content: toolResultContent(result),
                toolID: nil,
                toolArgs: toolResultArguments(result)
            )
            state.steps.append(step)
            return ChatKernelEventMutation(stepsChanged: true)

        case .diagnostic(let diagnostic):
            state.diagnostics.append(diagnostic)
            var mutation = ChatKernelEventMutation(diagnosticsChanged: true)
            if isDegradedDiagnostic(diagnostic) {
                state.degraded = true
                upsertDegradedStep(from: diagnostic, in: &state.steps)
                mutation.stepsChanged = true
            }
            return mutation

        case .error(let message):
            state.errorMessage = message
            replaceVisibleTextIfPresent(message, state: &state, lastUserMessage: lastUserMessage)
            return ChatKernelEventMutation(textChanged: true, shouldEmitUIUpdateDiagnostic: true)

        case .done(let finalText, let steps):
            replaceVisibleTextIfPresent(finalText, state: &state, lastUserMessage: lastUserMessage)
            if !steps.isEmpty {
                state.steps = steps
            }
            state.isDone = true
            return ChatKernelEventMutation(textChanged: !finalText.isEmpty, stepsChanged: !steps.isEmpty)
        }
    }

    private static func appendVisibleText(
        _ chunk: String,
        state: inout ChatKernelEventState,
        lastUserMessage: String
    ) {
        state.finalText += chunk
        state.streamingText = sanitizedStreamingText(state.finalText, lastUserMessage: lastUserMessage)
    }

    private static func replaceVisibleTextIfPresent(
        _ text: String,
        state: inout ChatKernelEventState,
        lastUserMessage: String
    ) {
        guard !text.isEmpty else { return }
        state.finalText = text
        state.streamingText = sanitizedStreamingText(text, lastUserMessage: lastUserMessage)
    }

    private static func sanitizedStreamingText(_ text: String, lastUserMessage: String) -> String {
        let sanitized = AssistantOutputSanitizer.sanitize(text, lastUserMessage: lastUserMessage)
        return SchemaPlaceholderDetector.isPlaceholderPrefix(sanitized) ? "" : sanitized
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
