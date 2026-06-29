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

                if request.supportsDeterministicToolBridge {
                    let routing = IntentRouter.classify(effectiveUserMessage)
                    if IntentRouter.intentRequiresTool(routing) {
                        let backgroundAssessment: BackgroundToolBridgeAssessment?
                        let availableTools: [ToolDefinition]
                        if request.requiresBackgroundSafeToolBridge {
                            let assessment = await BackgroundToolBridgePolicy.assess(
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
                            availableTools = await toolBridgeAvailableTools(
                                for: request,
                                userMessage: effectiveUserMessage,
                                routing: routing,
                                modelContext: modelContext
                            )
                        }

                        if let backgroundAssessment, !backgroundAssessment.canRunWithoutLoadedTextRuntime {
                            if request.options.diagnosticsEnabled {
                                continuation.yield(.diagnostic(.init(
                                    stage: "background-tool-bridge",
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

                        if request.options.diagnosticsEnabled {
                            continuation.yield(.diagnostic(.init(
                                stage: "tool-routing-bridge",
                                message: "Routing tool-backed \(String(describing: request.task)) turn through deterministic legacy bridge",
                                metadata: [
                                    "intent": routing.intent.rawValue,
                                    "allowedToolIDs": routing.allowedToolIDs.sorted().joined(separator: ","),
                                    "availableToolIDs": availableTools.map(\.id).sorted().joined(separator: ","),
                                    "mode": request.requiresBackgroundSafeToolBridge ? "background-safe" : "foreground",
                                    "source": request.source.rawValue,
                                    "referenceRewrite": String(referenceResolution.hasRewrite)
                                ]
                            )))
                        }

                        let legacyRequest = AgentRequest(
                            systemPrompt: request.systemPrompt,
                            history: historyTuples,
                            userMessage: effectiveUserMessage,
                            temperature: request.options.temperature,
                            topP: request.options.topP,
                            repetitionPenalty: request.options.repetitionPenalty,
                            maxTokens: request.options.maxTokens,
                            maxSteps: request.options.maxSteps,
                            availableTools: availableTools,
                            relevantMemories: request.relevantMemories,
                            attachments: request.attachments,
                            conversationID: request.conversationID,
                            turnID: request.turnID
                        )
                        let groundingMode: LegacyAgentRunOptions.GroundingMode = request.source == .trigger ? .headlessTrigger : .slotAgent
                        let legacyOptions = LegacyAgentRunOptions(
                            modelContext: modelContext,
                            conversationID: request.conversationID,
                            turnID: request.turnID,
                            groundingMode: groundingMode,
                            allowDegradedGrounding: request.options.allowDegradedMode,
                            preventDoubleGrounding: true,
                            diagnosticsEnabled: true
                        )

                        let bridgedEvents = request.requiresBackgroundSafeToolBridge
                            ? LegacyAgentCompatibilityBridge.runSlotAgentKernelCompatibility(legacyRequest, options: legacyOptions)
                            : runLegacyAgentBridge(legacyRequest, options: legacyOptions)
                        for await bridgedEvent in bridgedEvents {
                            continuation.yield(bridgedEvent)
                        }
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

    func runLegacyAgentBridge(_ request: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentKernelEvent> {
        LegacyAgentCompatibilityBridge.runLegacyAgentService(request, options: options)
    }

    private func toolBridgeAvailableTools(
        for request: AgentKernelRequest,
        userMessage: String,
        routing: IntentRoutingDecision,
        modelContext: ModelContext?
    ) async -> [ToolDefinition] {
        let routedIDs = Set(routing.allowedToolIDs.map { ToolRouteGuard.canonicalToolID($0) })
        guard request.requiresBackgroundSafeToolBridge else {
            return ToolRegistry.all.filter { routedIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
        }

        return await BackgroundToolBridgePolicy.availableTools(
            for: userMessage,
            routing: routing,
            modelContext: modelContext,
            toolRegistry: toolRegistry,
            metricsStore: metricsStore
        )
    }
}

private extension AgentKernelRequest {
    var supportsDeterministicToolBridge: Bool {
        task == .chat || task == .backgroundTrigger
    }

    var requiresBackgroundSafeToolBridge: Bool {
        task == .backgroundTrigger || source == .trigger
    }
}
