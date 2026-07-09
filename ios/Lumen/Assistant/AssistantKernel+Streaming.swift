import Foundation
import SwiftData

extension AssistantKernel: AgentKernelRunning {
    func run(_ request: AgentKernelRequest, modelContext: ModelContext? = nil) -> AsyncStream<AgentKernelEvent> {
        AsyncStream { continuation in
            let task = Task { @MainActor in
                let start = Date()
                var emittedSteps: [AgentStep] = []

                func emitStep(_ kind: AgentStep.Kind, _ content: String, toolID: String? = nil, toolArgs: [String: String]? = nil) {
                    let step = AgentStep(kind: kind, content: content, toolID: toolID, toolArgs: toolArgs)
                    emittedSteps.append(step)
                    continuation.yield(.step(step))
                }

                emitStep(.thought, "Agent Kernel accepted \(request.source.rawValue) turn for \(String(describing: request.task)).")
                let historyTuples = request.history.map { (role: $0.role.messageRole, content: $0.content) }
                let referenceResolution = ReferenceResolver.resolve(
                    prompt: request.userMessage,
                    history: historyTuples,
                    relevantMemories: request.relevantMemories
                )
                let effectiveUserMessage = referenceResolution.rewrittenPrompt

                if request.supportsDeterministicToolExecution {
                    let routing = IntentRouter.classify(effectiveUserMessage)
                    if IntentRouter.intentRequiresTool(routing) {
                        let backgroundAssessment: BackgroundToolExecutionAssessment?
                        let availableTools: [ToolDefinition]
                        if request.requiresBackgroundSafeToolExecution {
                            let assessment = await BackgroundToolExecutionPolicy.assess(
                                prompt: effectiveUserMessage,
                                routing: routing,
                                modelContext: modelContext,
                                toolRegistry: toolRegistry,
                                metricsStore: metricsStore
                            )
                            backgroundAssessment = assessment
                            availableTools = assessment.availableTools
                        } else {
                            backgroundAssessment = nil
                            availableTools = await toolExecutionAvailableTools(
                                for: request,
                                userMessage: effectiveUserMessage,
                                routing: routing,
                                modelContext: modelContext
                            )
                        }

                        if let backgroundAssessment, !backgroundAssessment.canRunWithoutLoadedTextRuntime {
                            if request.options.diagnosticsEnabled {
                                continuation.yield(.diagnostic(.init(
                                    stage: "background-tool-execution",
                                    message: backgroundAssessment.skipMessage,
                                    metadata: backgroundAssessment.diagnosticMetadata.merging([
                                        "source": request.source.rawValue,
                                        "task": String(describing: request.task)
                                    ], uniquingKeysWith: { current, _ in current })
                                )))
                            }
                            emitStep(.observation, backgroundAssessment.skipMessage)
                            continuation.yield(.final(backgroundAssessment.skipMessage))
                            continuation.yield(.done(finalText: backgroundAssessment.skipMessage, steps: emittedSteps))
                            continuation.finish()
                            return
                        }

                        let outcome = await runNativeToolTurn(
                            request: request,
                            userMessage: effectiveUserMessage,
                            routing: routing,
                            availableTools: availableTools,
                            referenceWasRewritten: referenceResolution.hasRewrite,
                            modelContext: modelContext
                        )
                        for event in outcome.events {
                            continuation.yield(event)
                        }
                        emittedSteps.append(contentsOf: outcome.steps)
                        continuation.yield(.done(finalText: outcome.finalText, steps: emittedSteps))
                        continuation.finish()
                        return
                    }
                }

                let thermalState = ProcessInfo.processInfo.thermalState
                let lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
                let turn = AssistantTurnContext(
                    task: request.task,
                    input: effectiveUserMessage,
                    systemPrompt: request.systemPrompt,
                    history: historyTuples,
                    relevantMemories: request.relevantMemories,
                    attachments: request.attachments,
                    isForeground: request.source.isForeground,
                    lowPowerMode: lowPowerMode,
                    thermalState: thermalState,
                    prefersFoundationModels: request.options.prefersFoundationModels,
                    allowHeavyRuntime: request.options.allowHeavyRuntime,
                    temperature: request.options.temperature,
                    topP: request.options.topP,
                    repetitionPenalty: request.options.repetitionPenalty,
                    maxTokens: request.options.maxTokens
                )

                let runtimeSelection = selectRuntimeSelection(for: turn)
                let selectedRuntime = runtimeSelection.runtime
                let computeDecision = ComputePolicy.decide(for: turn)
                if request.options.diagnosticsEnabled {
                    continuation.yield(.diagnostic(.init(
                        stage: "runtime-selection",
                        message: "Selected \(selectedRuntime.rawValue)",
                        metadata: [
                            "source": request.source.rawValue,
                            "task": String(describing: request.task),
                            "foreground": String(request.source.isForeground),
                            "allowHeavyRuntime": String(request.options.allowHeavyRuntime),
                            "policyAllowHeavyRuntime": String(computeDecision.allowHeavyRuntime),
                            "budgetPolicy": computeDecision.budgetPolicy.rawValue,
                            "budgetDenialReason": computeDecision.denialReason ?? "none",
                            "runtime": selectedRuntime.rawValue,
                            "selectionReason": runtimeSelection.reason,
                            "lowPowerMode": String(lowPowerMode),
                            "thermalState": "\(thermalState.rawValue)",
                            "maxTokens": String(min(request.options.maxTokens, computeDecision.maxTokens))
                        ]
                    )))
                }

                if let modelContext, request.options.diagnosticsEnabled {
                    let grounding = await buildGroundingContext(turn: turn, modelContext: modelContext)
                    continuation.yield(.diagnostic(.init(
                        stage: "grounding",
                        message: "Grounding context prepared",
                        metadata: [
                            "memoryCount": String(grounding.memoryCount),
                            "ragCount": String(grounding.ragCount),
                            "toolCount": String(grounding.toolCount),
                            "estimatedChars": String(grounding.estimatedChars),
                            "estimatedTokens": String(grounding.estimatedTokens),
                            "contextProfile": grounding.contextProfile ?? "unknown",
                            "maxInputTokens": grounding.maxInputTokens.map(String.init) ?? "unknown",
                            "ragConfidence": grounding.ragConfidence.map { String(format: "%.3f", $0) } ?? "unknown",
                            "selfModelIncluded": String(grounding.selfModelIncluded ?? false),
                            "selfModelSchemaVersion": grounding.selfModelSchemaVersion ?? "unknown",
                            "selfModelMode": grounding.selfModelMode ?? "unknown",
                            "selfModelEstimatedChars": grounding.selfModelEstimatedChars.map(String.init) ?? "unknown"
                        ]
                    )))
                }

                do {
                    let output = try await runTextTurn(turn)
                    let finalText = request.options.requireUserVisibleFinal
                        ? FinalOutputSanitizer.sanitizeUserVisibleText(output).text
                        : output
                    if !finalText.isEmpty {
                        continuation.yield(.finalDelta(finalText))
                        continuation.yield(.final(finalText))
                    }
                    let elapsed = Int(Date().timeIntervalSince(start) * 1000)
                    if request.options.diagnosticsEnabled {
                        continuation.yield(.diagnostic(.init(
                            stage: "complete",
                            message: "Agent Kernel turn completed",
                            metadata: [
                                "latencyMs": String(elapsed),
                                "runtime": selectedRuntime.rawValue
                            ]
                        )))
                    }
                    continuation.yield(.done(finalText: finalText, steps: emittedSteps))
                } catch {
                    let message = RuntimeMetricErrorSanitizer.code(for: error)
                    emitStep(.observation, "Agent Kernel failed: \(message)")
                    continuation.yield(.error(message))
                }

                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    #if DEBUG
    func runLegacyAgentBridge(_ request: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentKernelEvent> {
        LegacyAgentCompatibilityBridge.runLegacyAgentService(request, options: options)
    }
    #endif

    private struct NativeToolTurnOutcome {
        let finalText: String
        let steps: [AgentStep]
        let events: [AgentKernelEvent]
    }

    private func runNativeToolTurn(
        request: AgentKernelRequest,
        userMessage: String,
        routing: IntentRoutingDecision,
        availableTools: [ToolDefinition],
        referenceWasRewritten: Bool,
        modelContext: ModelContext?
    ) async -> NativeToolTurnOutcome {
        var steps: [AgentStep] = []
        var events: [AgentKernelEvent] = []
        func appendStep(_ step: AgentStep) {
            steps.append(step)
            events.append(.step(step))
        }

        if request.options.diagnosticsEnabled {
            events.append(.diagnostic(.init(
                stage: "native-tool-routing",
                message: "Routing tool-backed \(String(describing: request.task)) turn through native kernel tool execution",
                metadata: [
                    "intent": routing.intent.rawValue,
                    "allowedToolIDs": routing.allowedToolIDs.sorted().joined(separator: ","),
                    "availableToolIDs": availableTools.map(\.id).sorted().joined(separator: ","),
                    "mode": request.requiresBackgroundSafeToolExecution ? "background-safe" : "foreground",
                    "source": request.source.rawValue,
                    "referenceRewrite": String(referenceWasRewritten)
                ]
            )))
        }

        if routing.requiresClarification, let clarification = routing.clarificationPrompt {
            let step = AgentStep(kind: .observation, content: clarification)
            appendStep(step)
            events.append(.final(clarification))
            return NativeToolTurnOutcome(finalText: clarification, steps: steps, events: events)
        }

        guard !availableTools.isEmpty else {
            let message = "No approved tool is available for this \(routing.intent.rawValue) request."
            appendStep(AgentStep(kind: .observation, content: message))
            events.append(.final(message))
            return NativeToolTurnOutcome(finalText: message, steps: steps, events: events)
        }

        let availableToolIDs = Set(availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        let plannedActions = DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: userMessage,
            availableToolIDs: availableToolIDs
        )
        guard !plannedActions.isEmpty else {
            let message = "I could not determine a validated tool action for this \(routing.intent.rawValue) request."
            appendStep(AgentStep(kind: .observation, content: message))
            events.append(.final(message))
            return NativeToolTurnOutcome(finalText: message, steps: steps, events: events)
        }

        let maxActions = max(1, request.options.maxSteps)
        var finalText = ""
        var executedKeys: Set<String> = []
        for action in plannedActions.prefix(maxActions) {
            let validation = StructuredToolCallValidator.validate(action: action, availableTools: availableTools)
            let validatedCall: ValidatedStructuredToolCall
            switch validation {
            case .success(let call):
                validatedCall = call
            case .failure(let error):
                let canonical = ToolRouteGuard.canonicalToolID(action.tool)
                let message = "Tool call schema rejected: \(error.diagnostic)."
                appendStep(AgentStep(kind: .observation, content: message, toolID: canonical))
                finalText = "I could not run that tool because its generated arguments failed validation."
                events.append(.final(finalText))
                return NativeToolTurnOutcome(finalText: finalText, steps: steps, events: events)
            }

            let validatedAction = AgentAction(
                tool: validatedCall.canonicalToolID,
                args: AgentJSONArguments(stringDictionary: validatedCall.arguments)
            )
            guard !executedKeys.contains(validatedAction.dedupeKey) else {
                finalText = "Duplicate tool call blocked: \(validatedAction.tool)."
                appendStep(AgentStep(kind: .reflection, content: finalText, toolID: validatedAction.tool))
                events.append(.final(finalText))
                return NativeToolTurnOutcome(finalText: finalText, steps: steps, events: events)
            }
            executedKeys.insert(validatedAction.dedupeKey)

            let invocation = ToolInvocation(
                id: UUID(),
                toolID: validatedCall.canonicalToolID,
                arguments: validatedCall.arguments,
                source: request.toolInvocationSource,
                conversationID: request.conversationID,
                turnID: request.turnID,
                createdAt: Date()
            )
            if ToolRouteGuard.requiresUserApproval(validatedCall.canonicalToolID) {
                let approval = Self.approvalBoundaryFinal(for: validatedCall.canonicalToolID)
                appendStep(AgentStep(
                    kind: .approvalBoundary,
                    content: approval,
                    toolID: validatedCall.canonicalToolID,
                    toolArgs: validatedCall.arguments
                ))
                finalText = approval
                events.append(.final(finalText))
                return NativeToolTurnOutcome(finalText: finalText, steps: steps, events: events)
            }
            appendStep(AgentStep(
                kind: .action,
                content: "\(validatedCall.canonicalToolID)(validated)",
                toolID: validatedCall.canonicalToolID,
                toolArgs: validatedCall.arguments
            ))
            events.append(.toolInvocation(invocation))

            let context = ToolExecutionContext(
                isForeground: request.source.isForeground,
                appState: nil,
                modelContext: modelContext,
                permissionRegistry: .shared,
                metricsStore: metricsStore
            )
            let result = await toolRegistry.execute(invocation, context: context)
            events.append(.toolResult(result))
            let observationText = Self.userVisibleToolObservation(toolID: validatedCall.canonicalToolID, result: result)
            appendStep(AgentStep(kind: .observation, content: observationText, toolID: validatedCall.canonicalToolID))
            finalText = observationText

            if result.status != .success {
                break
            }
        }

        if finalText.isEmpty {
            finalText = "Native tool execution finished without a user-visible result."
        }
        events.append(.final(finalText))
        return NativeToolTurnOutcome(finalText: finalText, steps: steps, events: events)
    }

    private nonisolated static func approvalBoundaryFinal(for toolID: String) -> String {
        switch toolID {
        case "messages.draft":
            return "Approval required for messages.draft. I did not prepare or send the message yet."
        case "mail.draft":
            return "Approval required for mail.draft. I did not prepare or send the email yet."
        case "calendar.create":
            return "Approval required for calendar.create. I did not create an event yet."
        case "trigger.create":
            return "Approval required for trigger.create. I did not schedule an agent run yet."
        case "phone.call":
            return "Approval required for phone.call. I did not place the call yet."
        case "alarm.request_authorization":
            return "Approval required for alarm.request_authorization. I did not request alarm authorization yet."
        case "alarm.schedule", "alarm.countdown", "alarm.pause", "alarm.resume", "alarm.stop", "alarm.snooze", "alarm.cancel":
            return "Approval required for \(toolID). I did not change alarms yet."
        case let id where id.hasPrefix("outlook."):
            return "Approval required for \(toolID). I did not modify Outlook mail yet."
        default:
            return "Approval required for \(toolID). I did not run it yet."
        }
    }

    private static func userVisibleToolObservation(toolID: String, result: ToolResult) -> String {
        let text = result.displayText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty { return text }
        let modelText = result.modelText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelText.isEmpty { return modelText }
        if let errorCode = result.errorCode, !errorCode.isEmpty {
            return "\(toolID) finished with status \(result.status.rawValue): \(errorCode)."
        }
        return "\(toolID) finished with status \(result.status.rawValue) and no user-visible output."
    }

    private func toolExecutionAvailableTools(
        for request: AgentKernelRequest,
        userMessage: String,
        routing: IntentRoutingDecision,
        modelContext: ModelContext?
    ) async -> [ToolDefinition] {
        let routedIDs = Set(routing.allowedToolIDs.map { ToolRouteGuard.canonicalToolID($0) })
        guard request.requiresBackgroundSafeToolExecution else {
            return ToolRegistry.all.filter { routedIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
        }

        return await BackgroundToolExecutionPolicy.availableTools(
            for: userMessage,
            routing: routing,
            modelContext: modelContext,
            toolRegistry: toolRegistry,
            metricsStore: metricsStore
        )
    }
}

private extension AgentKernelRequest {
    var supportsDeterministicToolExecution: Bool {
        task == .chat || task == .backgroundTrigger
    }

    var requiresBackgroundSafeToolExecution: Bool {
        task == .backgroundTrigger || source == .trigger
    }

    var toolInvocationSource: ToolInvocationSource {
        switch source {
        case .trigger:
            return .backgroundTrigger
        case .appIntent:
            return .appIntent
        case .chat, .voice, .diagnostics, .benchmark:
            return .modelProposed
        }
    }
}
