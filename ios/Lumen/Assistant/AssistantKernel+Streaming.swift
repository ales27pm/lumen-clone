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

                let thermalState = ProcessInfo.processInfo.thermalState
                let lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
                let turn = AssistantTurnContext(
                    task: request.task,
                    input: request.userMessage,
                    isForeground: request.source.isForeground,
                    lowPowerMode: lowPowerMode,
                    thermalState: thermalState,
                    prefersFoundationModels: request.options.prefersFoundationModels,
                    allowHeavyRuntime: request.options.allowHeavyRuntime
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
                    continuation.yield(.diagnostic(.init(
                        stage: "complete",
                        message: "Agent Kernel turn completed",
                        metadata: [
                            "latencyMs": String(elapsed),
                            "runtime": selectedRuntime.rawValue
                        ]
                    )))
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
}
