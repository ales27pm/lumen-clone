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

                if request.task == .chat {
                    let routing = IntentRouter.classify(request.userMessage)
                    if IntentRouter.intentRequiresTool(routing) {
                        if request.options.diagnosticsEnabled {
                            continuation.yield(.diagnostic(.init(
                                stage: "tool-routing-bridge",
                                message: "Routing tool-backed chat turn through deterministic legacy bridge",
                                metadata: [
                                    "intent": routing.intent.rawValue,
                                    "allowedToolIDs": routing.allowedToolIDs.sorted().joined(separator: ","),
                                    "source": request.source.rawValue
                                ]
                            )))
                        }

                        let availableTools = ToolRegistry.all.filter { tool in
                            routing.allowedToolIDs.contains(ToolRouteGuard.canonicalToolID(tool.id))
                        }
                        let legacyRequest = AgentRequest(
                            systemPrompt: request.systemPrompt,
                            history: request.history.map { (role: $0.role.messageRole, content: $0.content) },
                            userMessage: request.userMessage,
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

                        for await bridgedEvent in runLegacyAgentBridge(legacyRequest, options: legacyOptions) {
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
                    input: request.userMessage,
                    systemPrompt: request.systemPrompt,
                    history: request.history.map { (role: $0.role.messageRole, content: $0.content) },
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

                let selectedRuntime = selectRuntime(for: turn)
                if request.options.diagnosticsEnabled {
                    continuation.yield(.diagnostic(.init(
                        stage: "runtime-selection",
                        message: "Selected \(selectedRuntime.rawValue)",
                        metadata: [
                            "source": request.source.rawValue,
                            "task": String(describing: request.task),
                            "foreground": String(request.source.isForeground),
                            "allowHeavyRuntime": String(request.options.allowHeavyRuntime),
                            "runtime": selectedRuntime.rawValue,
                            "lowPowerMode": String(lowPowerMode),
                            "thermalState": "\(thermalState.rawValue)"
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
                            "estimatedChars": String(grounding.estimatedChars)
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
}
