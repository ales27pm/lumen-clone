import Foundation
import SwiftData

@MainActor
struct StructuredAgentKernelExecutor {
    private static let structuredTurnMaxTokenCap = 384
    private static let structuredTurnMinTokenCap = 128
    private static let structuredContextNoteCharCap = 280
    private static let structuredUserMessageCharCap = 900
    private static let structuredHistoryTurnCharCap = 180
    private static let structuredHistoryTotalCharCap = 540
    private static let structuredScratchpadCharCap = 900
    private static let structuredToolDescriptionCharCap = 88
    private static let structuredPromptPreflightSafetyTokens = 128
    private static let structuredAgentModelSlot: LumenModelSlot = .executor
    private static let contextWindowExceededRawOutputPrefix = "Prompt exceeded context window before generation"

    nonisolated static let structuredAgentResponseSchema = #"{"type":"object","oneOf":[{"required":["action"],"properties":{"thought":{"type":"string"},"action":{"type":"object","required":["tool","args"],"properties":{"tool":{"type":"string"},"args":{"type":"object"}}},"additionalProperties":false},{"required":["final"],"properties":{"thought":{"type":"string"},"final":{"type":"string"}},"additionalProperties":false}]}"#
    nonisolated static let structuredAgentActionResponseSchema = #"{"type":"object","required":["action"],"properties":{"thought":{"type":"string"},"action":{"type":"object","required":["tool","args"],"properties":{"tool":{"type":"string"},"args":{"type":"object"}}}},"additionalProperties":false}"#
    nonisolated static let structuredAgentFinalResponseSchema = #"{"type":"object","required":["final"],"properties":{"thought":{"type":"string"},"final":{"type":"string"}},"additionalProperties":false}"#

    private let kernel: AssistantKernel
    private let modelContext: ModelContext?

    init(kernel: AssistantKernel, modelContext: ModelContext?) {
        self.kernel = kernel
        self.modelContext = modelContext
    }

    func run(_ request: AgentKernelRequest) -> AsyncStream<AgentKernelEvent> {
        AsyncStream { continuation in
            let task = Task { @MainActor in
                await runLoop(request, continuation: continuation)
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func runLoop(_ request: AgentKernelRequest, continuation: AsyncStream<AgentKernelEvent>.Continuation) async {
        var steps: [AgentStep] = []
        var observations: [(tool: String, result: String)] = []
        var executedActionKeys: Set<String> = []
        var scratchpad = ""
        var finalAnswer = ""
        let userMessage = Self.sanitizedStructuredUserMessage(request.userMessage)
        let routing = IntentRouter.classify(userMessage)
        let availableTools = await availableTools(for: request, routing: routing)
        let memoryCommandPlan = MemoryCommandPlan.saveThenRecall(from: userMessage)
        let systemPrompt = Self.buildSystemPrompt(request: request, availableTools: availableTools)
        let availableToolIDs = Set(availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        let maxSteps = max(1, request.options.maxSteps)

        emit(.thought, "Structured Agent Kernel accepted agent-json turn.", steps: &steps, continuation: continuation)
        emitRoutingDiagnostic(request: request, routing: routing, availableToolIDs: availableToolIDs, maxSteps: maxSteps, continuation: continuation)

        if routing.requiresClarification, let clarification = routing.clarificationPrompt, !clarification.isEmpty {
            emit(.reflection, "Clarification required before structured tool execution.", steps: &steps, continuation: continuation)
            finalAnswer = clarification
            continuation.yield(.finalDelta(finalAnswer))
            continuation.yield(.final(finalAnswer))
            continuation.yield(.done(finalText: finalAnswer, steps: steps))
            continuation.finish()
            return
        }

        stepsLoop: for stepIndex in 0..<maxSteps {
            if Task.isCancelled { break }

            let userTurn = Self.buildAgentUserTurn(
                request: request,
                availableTools: availableTools,
                stepIndex: stepIndex,
                scratchpad: scratchpad
            )
            var generation = await generateStructuredTurn(
                request: request,
                availableTools: availableTools,
                systemPrompt: systemPrompt,
                userTurn: userTurn,
                stepIndex: stepIndex,
                hasObservations: !observations.isEmpty
            )
            var turn = Self.strictToolExecutableTurn(AgentTurnParser.parse(generation.raw))

            if !Task.isCancelled,
               generation.forcedParseError == nil,
               turn.parseError == .missingActionOrFinal,
               Self.shouldForceActionSchema(request: request, availableTools: availableTools, stepIndex: stepIndex, hasObservations: !observations.isEmpty) {
                recordParseFailure(parseError: .missingActionOrFinal, raw: generation.raw, systemPrompt: systemPrompt, userTurn: userTurn, stepIndex: stepIndex)
                generation = await generateStructuredTurn(
                    request: request,
                    availableTools: availableTools,
                    systemPrompt: systemPrompt,
                    userTurn: Self.agentJSONMissingDecisionRetryUserTurn(
                        from: userTurn,
                        rawOutput: generation.raw,
                        allowedToolIDs: availableTools.map(\.id)
                    ),
                    stepIndex: stepIndex,
                    hasObservations: false,
                    responseFormatOverride: .constrainedJSON(schema: Self.structuredAgentActionResponseSchema)
                )
                turn = Self.strictToolExecutableTurn(AgentTurnParser.parse(generation.raw))
            }

            if let forced = generation.forcedParseError {
                turn = AgentTurn(thought: nil, action: nil, final: nil, parseError: forced, hadNoise: false)
            } else if let runtimeFailure = Self.runtimeFailureParseError(from: generation.raw) {
                turn = AgentTurn(thought: nil, action: nil, final: nil, parseError: runtimeFailure, hadNoise: false)
            }

            recordModelTurnTrace(
                request: request,
                userTurn: userTurn,
                raw: generation.raw,
                turn: turn,
                availableTools: availableTools,
                stepIndex: stepIndex,
                diagnostics: generation.diagnostics
            )
            emitModelTurnDiagnostic(
                request: request,
                turn: turn,
                availableTools: availableTools,
                stepIndex: stepIndex,
                diagnostics: generation.diagnostics,
                continuation: continuation
            )

            if let thought = turn.thought?.trimmingCharacters(in: .whitespacesAndNewlines), !thought.isEmpty {
                emit(.thought, thought, steps: &steps, continuation: continuation)
                scratchpad += "\nThought: \(thought)"
            }

            var actionToExecute = turn.action
            if let parsedAction = actionToExecute {
                let repair = Self.repairedMemoryActionIfNeeded(
                    modelAction: parsedAction,
                    memoryPlan: memoryCommandPlan,
                    steps: steps,
                    availableToolIDs: availableToolIDs
                )
                if let reflection = repair.reflection {
                    steps.append(reflection)
                    continuation.yield(.step(reflection))
                }
                actionToExecute = repair.action
                if let mapsRepair = Self.repairedMapsSearchActionIfNeeded(
                    modelAction: actionToExecute,
                    routing: routing,
                    prompt: userMessage,
                    steps: steps,
                    availableToolIDs: availableToolIDs
                ) {
                    steps.append(mapsRepair.reflection)
                    continuation.yield(.step(mapsRepair.reflection))
                    actionToExecute = mapsRepair.action
                }
            } else if turn.final?.isEmpty == false,
                      let requiredMemoryAction = Self.nextRequiredMemoryAction(
                        memoryPlan: memoryCommandPlan,
                        steps: steps,
                        availableToolIDs: availableToolIDs
                      ) {
                emit(.reflection, "Memory save-then-recall invariant repaired a premature final before required memory actions completed.", steps: &steps, continuation: continuation)
                actionToExecute = requiredMemoryAction
            } else if turn.parseError != nil,
                      let requiredMemoryAction = Self.nextRequiredMemoryAction(
                        memoryPlan: memoryCommandPlan,
                        steps: steps,
                        availableToolIDs: availableToolIDs
                      ) {
                emit(.reflection, "Memory save-then-recall invariant repaired malformed structured output into the next required memory action.", steps: &steps, continuation: continuation)
                actionToExecute = requiredMemoryAction
            } else if turn.final?.isEmpty == false || turn.parseError != nil,
                      let requiredMapsAction = Self.nextRequiredMapsSearchAction(
                        routing: routing,
                        prompt: userMessage,
                        steps: steps,
                        availableToolIDs: availableToolIDs
                      ) {
                emit(.reflection, "Maps search continuation repaired degraded location-only output into maps.search.", steps: &steps, continuation: continuation)
                actionToExecute = requiredMapsAction
            }

            if let action = actionToExecute {
                let validation = StructuredToolCallValidator.validate(action: action, availableTools: availableTools)
                let validatedCall: ValidatedStructuredToolCall
                switch validation {
                case .success(let call):
                    validatedCall = call
                case .failure(let error):
                    let canonical = ToolRouteGuard.canonicalToolID(action.tool)
                    let obs = AgentStep(
                        kind: .observation,
                        content: "Tool call schema rejected: \(error.diagnostic). Emit a corrected action or final turn.",
                        toolID: canonical,
                        toolArgs: action.args.stringCoerced
                    )
                    steps.append(obs)
                    continuation.yield(.step(obs))
                    scratchpad += "\nAction: \(action.displayContent)\nObservation: \(Self.compactScratchpadObservation(obs.content))"
                    continue
                }

                let validatedAction = AgentAction(
                    tool: validatedCall.canonicalToolID,
                    args: AgentJSONArguments(stringDictionary: validatedCall.arguments)
                )
                if executedActionKeys.contains(validatedAction.dedupeKey) {
                    emit(.reflection, "Duplicate tool call blocked: \(validatedAction.displayContent). Synthesizing answer from observations.", toolID: validatedAction.tool, steps: &steps, continuation: continuation)
                    finalAnswer = Self.deterministicObservationFallback(observations: observations, intent: routing.intent)
                        ?? "I found tool observations, but could not synthesize a grounded final answer from them."
                    break stepsLoop
                }
                executedActionKeys.insert(validatedAction.dedupeKey)

                if ToolRouteGuard.requiresUserApproval(validatedCall.canonicalToolID) {
                    finalAnswer = Self.approvalBoundaryFinal(for: validatedCall.canonicalToolID, action: validatedAction)
                    let step = AgentStep(kind: .approvalBoundary, content: finalAnswer, toolID: validatedCall.canonicalToolID, toolArgs: validatedCall.arguments)
                    steps.append(step)
                    continuation.yield(.step(step))
                    break stepsLoop
                }

                let actionStep = AgentStep(kind: .action, content: validatedAction.displayContent, toolID: validatedCall.canonicalToolID, toolArgs: validatedCall.arguments)
                steps.append(actionStep)
                continuation.yield(.step(actionStep))

                let invocation = ToolInvocation(
                    id: UUID(),
                    toolID: validatedCall.canonicalToolID,
                    arguments: validatedCall.arguments,
                    source: request.toolInvocationSourceForStructuredAgent,
                    conversationID: request.conversationID,
                    turnID: request.turnID,
                    createdAt: Date()
                )
                continuation.yield(.toolInvocation(invocation))
                let context = ToolExecutionContext(
                    isForeground: request.source.isForegroundForStructuredAgent,
                    appState: nil,
                    modelContext: modelContext,
                    permissionRegistry: .shared,
                    metricsStore: kernel.metricsStore
                )
                let result = await kernel.toolRegistry.execute(invocation, context: context)
                continuation.yield(.toolResult(result))
                let observationText = Self.userVisibleToolObservation(toolID: validatedCall.canonicalToolID, result: result)
                let observationStep = AgentStep(kind: .observation, content: observationText, toolID: validatedCall.canonicalToolID)
                steps.append(observationStep)
                continuation.yield(.step(observationStep))
                if Self.shouldStopAfterToolResult(result.status) {
                    emit(.reflection, "Structured tool loop stopped after non-success \(validatedCall.canonicalToolID) result.", toolID: validatedCall.canonicalToolID, steps: &steps, continuation: continuation)
                    finalAnswer = observationText
                    break stepsLoop
                }
                observations.append((validatedCall.canonicalToolID, observationText))
                scratchpad += "\nAction: \(validatedAction.displayContent)\nObservation: \(Self.compactScratchpadObservation(observationText))"

                if let phoneContinuation = Self.phoneCallContinuationAfterContactObservation(
                    routing: routing,
                    actionTool: validatedCall.canonicalToolID,
                    observation: observationText,
                    availableToolIDs: availableToolIDs
                ) {
                    steps.append(phoneContinuation.step)
                    continuation.yield(.step(phoneContinuation.step))
                    finalAnswer = phoneContinuation.text
                    break stepsLoop
                }

                if Self.shouldStopAfterFirstWebObservation(request: request, actionTool: validatedCall.canonicalToolID, observations: observations),
                   let webFinal = Self.deterministicObservationFallback(observations: observations, intent: .webSearch) {
                    emit(.reflection, "deterministic web synthesis fallback used after observations", steps: &steps, continuation: continuation)
                    finalAnswer = webFinal
                    break stepsLoop
                }

                if stepIndex == maxSteps - 1 {
                    finalAnswer = Self.deterministicObservationFallback(observations: observations, intent: routing.intent)
                        ?? "I gathered tool observations but reached the structured step limit before a final answer."
                    break stepsLoop
                }
                continue
            }

            if let final = turn.final?.trimmingCharacters(in: .whitespacesAndNewlines), !final.isEmpty {
                if Self.toolRequiredFinalNeedsAction(final, request: request, availableTools: availableTools, observations: observations) {
                    emit(.reflection, "Tool-backed structured final rejected before any trusted observation.", steps: &steps, continuation: continuation)
                    finalAnswer = observations.isEmpty
                        ? "I couldn't complete the tool-backed request because the model emitted a placeholder final instead of a tool action."
                        : Self.deterministicObservationFallback(observations: observations, intent: routing.intent) ?? final
                    break stepsLoop
                }
                finalAnswer = final
                break stepsLoop
            }

            if let parseError = turn.parseError {
                recordParseFailure(parseError: parseError, raw: generation.raw, systemPrompt: systemPrompt, userTurn: userTurn, stepIndex: stepIndex)
                if Self.hasUsableObservation(for: routing.intent, observations: observations) {
                    emit(.reflection, "Malformed structured turn repaired by synthesizing from existing tool observations.", steps: &steps, continuation: continuation)
                    finalAnswer = Self.deterministicObservationFallback(observations: observations, intent: routing.intent)
                        ?? "I gathered tool observations, but could not synthesize a grounded final answer from them."
                } else if parseError == .contextWindowExceeded {
                    finalAnswer = "I couldn't run the structured agent turn because the prompt exceeded the local model context window."
                } else if parseError == .empty {
                    finalAnswer = "I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: \(generation.diagnostics.emptyOutputReason ?? "unknownEmptyStream")."
                } else {
                    finalAnswer = "I hit an internal structured-output formatting issue before a valid tool observation was available."
                }
                break stepsLoop
            }
        }

        if Task.isCancelled {
            continuation.finish()
            return
        }

        if finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            finalAnswer = observations.isEmpty
                ? "I couldn't complete the structured agent turn because no valid final answer or tool observation was produced."
                : Self.deterministicObservationFallback(observations: observations, intent: routing.intent) ?? "I gathered tool observations but could not produce a final answer."
        }

        finalAnswer = Self.postprocessStructuredFinalAnswer(finalAnswer, request: request, availableTools: availableTools, observations: observations, steps: steps)
        continuation.yield(.finalDelta(finalAnswer))
        continuation.yield(.final(finalAnswer))
        continuation.yield(.done(finalText: finalAnswer, steps: steps))
        continuation.finish()
    }

    private func availableTools(for request: AgentKernelRequest, routing: IntentRoutingDecision) async -> [ToolDefinition] {
        let context = ToolExecutionContext(
            isForeground: request.source.isForegroundForStructuredAgent,
            appState: nil,
            modelContext: modelContext,
            permissionRegistry: .shared,
            metricsStore: kernel.metricsStore
        )
        let secure = await kernel.toolRegistry.availableDefinitions(context: context, source: request.toolInvocationSourceForStructuredAgent)
        let secureCatalog = ToolSchemaBridge.toCatalogToolDefinitions(secure)
        let catalogByID = Dictionary(uniqueKeysWithValues: ToolRegistry.all.map { (ToolRouteGuard.canonicalToolID($0.id), $0) })
        let secureIDs = Set(secureCatalog.map { ToolRouteGuard.canonicalToolID($0.id) })
        let optionIDs = Set(request.options.structuredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID))
        let routingIDs = Set(routing.allowedToolIDs.map(ToolRouteGuard.canonicalToolID))
        let sourceIDs = Self.structuredToolSourceIDs(secureIDs: secureIDs, optionIDs: optionIDs, routingIDs: routingIDs)
        let tools = sourceIDs.compactMap { catalogByID[$0] }.sorted { $0.id < $1.id }
        if tools.isEmpty, !sourceIDs.isEmpty {
            return sourceIDs.compactMap { catalogByID[$0] }.sorted { $0.id < $1.id }
        }
        return tools
    }

    private func generateStructuredTurn(
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        systemPrompt: String,
        userTurn: String,
        stepIndex: Int,
        hasObservations: Bool,
        responseFormatOverride: LLMResponseFormat? = nil
    ) async -> StructuredGenerationResult {
        let startedAt = Date()
        let turnContext = AssistantTurnContext(
            task: request.task,
            input: userTurn,
            systemPrompt: systemPrompt,
            history: [],
            relevantMemories: request.relevantMemories,
            attachments: request.attachments,
            isForeground: request.source.isForegroundForStructuredAgent,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: ProcessInfo.processInfo.thermalState,
            prefersFoundationModels: false,
            allowHeavyRuntime: request.options.allowHeavyRuntime,
            temperature: Self.agentTemperature(from: request.options.temperature),
            topP: Self.agentTopP(from: request.options.topP),
            repetitionPenalty: max(request.options.repetitionPenalty, 1.05),
            maxTokens: Self.structuredTurnMaxTokens(from: request.options.maxTokens),
            traceCorrelation: request.traceCorrelation,
            allowedToolIDs: availableTools.map { ToolRouteGuard.canonicalToolID($0.id) }
        )
        let selection = kernel.selectRuntimeSelection(for: turnContext)
        guard selection.runtime == .llama else {
            let reason = selection.runtime == .unavailable ? selection.reason : "structured agent-json requires llama executor runtime; selected \(selection.runtime.rawValue)"
            return StructuredGenerationResult(
                raw: "Structured agent-json unavailable: \(reason)",
                forcedParseError: .empty,
                diagnostics: .init(
                    generationElapsedMs: Int(Date().timeIntervalSince(startedAt) * 1000),
                    firstTokenLatencyMs: nil,
                    outputTokenCount: 0,
                    estimatedPromptTokenCount: nil,
                    maxTokensRequested: request.options.maxTokens,
                    maxTokensEffective: Self.structuredTurnMaxTokens(from: request.options.maxTokens),
                    promptCharCount: userTurn.count,
                    emptyOutputReason: "structuredAgentJSONUnavailable:\(selection.runtime.rawValue)",
                    streamStarted: false,
                    selectedRuntime: selection.runtime.rawValue,
                    selectedAdapter: nil,
                    modelIdentifier: selection.runtime.rawValue,
                    modelLoaded: false,
                    stopSequences: [],
                    temperature: Self.agentTemperature(from: request.options.temperature),
                    topP: Self.agentTopP(from: request.options.topP),
                    cancellationStateBeforeStream: nil,
                    firstChunkReceived: false,
                    textChunkCount: 0,
                    finalChunkReceived: false,
                    streamTerminationReason: "structured-agent-unavailable"
                )
            )
        }

        var genReq = GenerateRequest(
            systemPrompt: systemPrompt,
            history: [],
            userMessage: userTurn,
            temperature: Self.agentTemperature(from: request.options.temperature),
            topP: Self.agentTopP(from: request.options.topP),
            repetitionPenalty: max(request.options.repetitionPenalty, 1.05),
            maxTokens: Self.structuredTurnMaxTokens(from: request.options.maxTokens),
            modelName: "agent-json",
            relevantMemories: request.relevantMemories,
            attachments: request.attachments,
            responseFormat: responseFormatOverride ?? Self.structuredAgentResponseFormat(
                request: request,
                availableTools: availableTools,
                stepIndex: stepIndex,
                hasObservations: hasObservations
            ),
            allowsMemoryPressureContinuation: request.options.allowDegradedMode
        )
        var preflight = await preflightAgentJSONPrompt(genReq)
        var forcedParseError: AgentTurnParseError?
        if !preflight.fits {
            genReq = Self.agentJSONContextCompactionRequest(from: genReq)
            preflight = await preflightAgentJSONPrompt(genReq)
            if !preflight.fits {
                forcedParseError = .contextWindowExceeded
            }
        }

        let runtimePreflight = await ExecutorRuntimePreflight.checkReadiness(
            allowsLoadedMemoryPressureContinuation: request.options.allowDegradedMode
        )
        if !runtimePreflight.passed, forcedParseError == nil {
            forcedParseError = .empty
        }

        var raw = ""
        var firstTokenLatencyMs: Int?
        var textChunkCount = 0
        var finalChunkReceived = false
        var streamStarted = false
        if forcedParseError == nil {
            streamStarted = true
            streamLoop: for await token in await AppLlamaService.shared.stream(genReq, slot: Self.structuredAgentModelSlot) {
                if Task.isCancelled { break streamLoop }
                switch token {
                case .text(let text):
                    if firstTokenLatencyMs == nil {
                        firstTokenLatencyMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                    }
                    if !text.isEmpty { textChunkCount += 1 }
                    raw += text
                case .done:
                    finalChunkReceived = true
                    break streamLoop
                }
            }
        } else if forcedParseError == .contextWindowExceeded {
            raw = Self.contextWindowExceededRawOutputPrefix
        }

        let payload = await AppLlamaService.shared.takeCompletedTracePayload(requestID: genReq.id)
        let trimmedRaw = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let emptyReason = payload?.emptyOutputReason
            ?? (trimmedRaw.isEmpty ? runtimePreflight.reason : nil)
            ?? (trimmedRaw.isEmpty ? Self.agentJSONEmptyStreamReason(streamStarted: streamStarted, textChunkCount: textChunkCount, finalChunkReceived: finalChunkReceived, taskCancelled: Task.isCancelled, maxTokensEffective: genReq.maxTokens) : nil)
        return StructuredGenerationResult(
            raw: raw,
            forcedParseError: forcedParseError,
            diagnostics: .init(
                generationElapsedMs: payload?.elapsedMs ?? Int(Date().timeIntervalSince(startedAt) * 1000),
                firstTokenLatencyMs: firstTokenLatencyMs,
                outputTokenCount: payload?.outputTokenCount ?? (trimmedRaw.isEmpty ? 0 : nil),
                estimatedPromptTokenCount: payload?.estimatedPromptTokenCount ?? preflight.estimatedPromptTokens,
                maxTokensRequested: payload?.maxTokensRequested ?? request.options.maxTokens,
                maxTokensEffective: payload?.maxTokensEffective ?? genReq.maxTokens,
                promptCharCount: payload?.promptCharCount ?? preflight.finalPromptChars,
                emptyOutputReason: emptyReason,
                streamStarted: payload?.streamStarted ?? streamStarted,
                selectedRuntime: payload?.selectedRuntime ?? selection.runtime.rawValue,
                selectedAdapter: payload?.selectedAdapter,
                modelIdentifier: payload?.modelIdentifier ?? selection.runtime.rawValue,
                modelLoaded: payload?.modelLoaded ?? streamStarted,
                stopSequences: payload?.stopSequences ?? [],
                temperature: payload?.temperature ?? genReq.temperature,
                topP: payload?.topP ?? genReq.topP,
                cancellationStateBeforeStream: payload?.cancellationStateBeforeStream,
                firstChunkReceived: payload?.firstChunkReceived ?? (textChunkCount > 0),
                textChunkCount: payload?.textChunkCount ?? textChunkCount,
                finalChunkReceived: payload?.finalChunkReceived ?? finalChunkReceived,
                streamTerminationReason: payload?.streamTerminationReason ?? (forcedParseError?.rawValue ?? (finalChunkReceived ? "stop" : nil))
            )
        )
    }

    private func preflightAgentJSONPrompt(_ request: GenerateRequest) async -> StructuredPromptPreflight {
        let contextSize = await AppLlamaService.shared.contextSizeForDiagnostics(slot: Self.structuredAgentModelSlot)
        let promptBuild = await AppLlamaService.shared.buildMessagesForDiagnostics(
            req: request,
            contextSize: contextSize,
            slot: Self.structuredAgentModelSlot
        )
        let tokenLimit = max(128, contextSize - request.maxTokens - Self.structuredPromptPreflightSafetyTokens)
        let fits = promptBuild.estimatedPromptTokens <= tokenLimit
            && promptBuild.finalPromptChars <= PromptBudget.agentJSON(contextSize: contextSize, maxTokens: request.maxTokens).totalChars + 256
        return StructuredPromptPreflight(
            contextSize: contextSize,
            finalPromptChars: promptBuild.finalPromptChars,
            estimatedPromptTokens: promptBuild.estimatedPromptTokens,
            fits: fits
        )
    }

    private func recordModelTurnTrace(
        request: AgentKernelRequest,
        userTurn: String,
        raw: String,
        turn: AgentTurn,
        availableTools: [ToolDefinition],
        stepIndex: Int,
        diagnostics: StructuredTurnGenerationDiagnostics
    ) {
        let routing = IntentRouter.classify(Self.sanitizedStructuredUserMessage(request.userMessage))
        AgentBehaviorTraceEmitter.recordModelTurn(
            correlation: request.traceCorrelation,
            slot: "agent",
            stage: "agent-json-step-\(stepIndex)",
            intent: routing.intent.rawValue,
            prompt: AgentDiagnosticFileRedactor.summary(label: "prompt", text: request.userMessage),
            rawOutput: Self.redactedForDiagnostics(raw),
            selectedToolID: turn.action.map { ToolRouteGuard.canonicalToolID($0.tool) },
            toolArguments: AgentDiagnosticFileRedactor.redactedMap(turn.action?.args.stringCoerced ?? [:]),
            allowedToolIDs: availableTools.map { ToolRouteGuard.canonicalToolID($0.id) }.sorted(),
            requiresApproval: turn.action.map { ToolRouteGuard.requiresUserApproval(ToolRouteGuard.canonicalToolID($0.tool)) },
            parseError: turn.parseError?.rawValue,
            emittedFinalInActionTurn: turn.final?.isEmpty == false,
            modelFamily: LumenModelFamily.persistedSelected.rawValue,
            adapterSlot: Self.structuredAgentModelSlot.rawValue,
            generationElapsedMs: diagnostics.generationElapsedMs,
            firstTokenLatencyMs: diagnostics.firstTokenLatencyMs,
            outputTokenCount: diagnostics.outputTokenCount,
            estimatedPromptTokenCount: diagnostics.estimatedPromptTokenCount,
            runtimePath: "agent-model",
            activeAdapterSlot: Self.structuredAgentModelSlot.rawValue,
            maxTokensRequested: diagnostics.maxTokensRequested,
            maxTokensEffective: diagnostics.maxTokensEffective,
            promptCharCount: diagnostics.promptCharCount ?? userTurn.count,
            emptyOutputReason: diagnostics.emptyOutputReason,
            streamStarted: diagnostics.streamStarted,
            selectedRuntime: diagnostics.selectedRuntime,
            selectedAdapter: diagnostics.selectedAdapter,
            modelIdentifier: diagnostics.modelIdentifier,
            modelLoaded: diagnostics.modelLoaded,
            stopSequences: diagnostics.stopSequences,
            temperature: diagnostics.temperature,
            topP: diagnostics.topP,
            cancellationStateBeforeStream: diagnostics.cancellationStateBeforeStream,
            firstChunkReceived: diagnostics.firstChunkReceived,
            textChunkCount: diagnostics.textChunkCount,
            finalChunkReceived: diagnostics.finalChunkReceived,
            streamTerminationReason: diagnostics.streamTerminationReason,
            selfModel: AgentBehaviorTrace.SelfModelDecisionSummary.fromPrompt(
                userTurn,
                selectedToolID: turn.action.map { ToolRouteGuard.canonicalToolID($0.tool) },
                requiresApproval: turn.action.map { ToolRouteGuard.requiresUserApproval(ToolRouteGuard.canonicalToolID($0.tool)) },
                approvalMode: nil
            )
        )
    }

    private func emitRoutingDiagnostic(
        request: AgentKernelRequest,
        routing: IntentRoutingDecision,
        availableToolIDs: Set<String>,
        maxSteps: Int,
        continuation: AsyncStream<AgentKernelEvent>.Continuation
    ) {
        guard request.options.diagnosticsEnabled else { return }
        continuation.yield(.diagnostic(.init(
            stage: "structured-agent-json-routing",
            message: "Structured agent-json routing prepared.",
            metadata: [
                "intent": routing.intent.rawValue,
                "source": request.source.rawValue,
                "availableToolIDs": availableToolIDs.sorted().joined(separator: ","),
                "maxSteps": String(maxSteps),
                "runtimePath": "agent-model"
            ]
        )))
    }

    private func emitModelTurnDiagnostic(
        request: AgentKernelRequest,
        turn: AgentTurn,
        availableTools: [ToolDefinition],
        stepIndex: Int,
        diagnostics: StructuredTurnGenerationDiagnostics,
        continuation: AsyncStream<AgentKernelEvent>.Continuation
    ) {
        guard request.options.diagnosticsEnabled else { return }
        var metadata: [String: String] = [
            "stepIndex": String(stepIndex),
            "runtimePath": "agent-model",
            "allowedToolIDs": availableTools.map { ToolRouteGuard.canonicalToolID($0.id) }.sorted().joined(separator: ",")
        ]
        if let selected = turn.action.map({ ToolRouteGuard.canonicalToolID($0.tool) }) {
            metadata["selectedToolID"] = selected
        }
        if let parseError = turn.parseError?.rawValue {
            metadata["parseError"] = parseError
        }
        if let modelLoaded = diagnostics.modelLoaded {
            metadata["modelLoaded"] = String(modelLoaded)
        }
        if let streamStarted = diagnostics.streamStarted {
            metadata["streamStarted"] = String(streamStarted)
        }
        if let firstChunkReceived = diagnostics.firstChunkReceived {
            metadata["firstChunkReceived"] = String(firstChunkReceived)
        }
        if let textChunkCount = diagnostics.textChunkCount {
            metadata["textChunkCount"] = String(textChunkCount)
        }
        if let finalChunkReceived = diagnostics.finalChunkReceived {
            metadata["finalChunkReceived"] = String(finalChunkReceived)
        }
        if let reason = diagnostics.streamTerminationReason {
            metadata["streamTerminationReason"] = reason
        }
        if let emptyReason = diagnostics.emptyOutputReason {
            metadata["emptyOutputReason"] = emptyReason
        }
        continuation.yield(.diagnostic(.init(
            stage: "structured-agent-json-model-turn",
            message: "Structured agent-json model turn completed.",
            metadata: metadata
        )))
    }

    private func recordParseFailure(
        parseError: AgentTurnParseError,
        raw: String,
        systemPrompt: String,
        userTurn: String,
        stepIndex: Int
    ) {
        let snapshot = AgentNoiseInspector.inspect(raw)
        let trace = AgentParseFailureTrace(
            id: UUID(),
            createdAt: Date(),
            parseError: parseError.rawValue,
            modelName: "agent-json",
            temperature: 0.05,
            topP: 0.6,
            maxTokens: Self.structuredTurnMaxTokenCap,
            stepIndex: stepIndex,
            systemPromptPrefix: AgentDiagnosticFileRedactor.summary(label: "systemPrompt", text: systemPrompt),
            userTurnPrefix: AgentDiagnosticFileRedactor.summary(label: "userTurn", text: userTurn),
            rawOutputPrefix: String(Self.redactedForDiagnostics(raw).prefix(4_000)),
            streamedThoughtPrefix: "",
            streamedFinalPrefix: "",
            selectedJSONPrefix: snapshot.selectedJSON.map { AgentDiagnosticFileRedactor.summary(label: "selectedJSON", text: $0) },
            prefixNoise: snapshot.prefixNoise.map { AgentDiagnosticFileRedactor.summary(label: "prefixNoise", text: $0) },
            suffixNoise: snapshot.suffixNoise.map { AgentDiagnosticFileRedactor.summary(label: "suffixNoise", text: $0) }
        )
        AgentParseFailureRecorder.record(trace)
    }

    private func emit(
        _ kind: AgentStep.Kind,
        _ content: String,
        toolID: String? = nil,
        toolArgs: [String: String]? = nil,
        steps: inout [AgentStep],
        continuation: AsyncStream<AgentKernelEvent>.Continuation
    ) {
        let step = AgentStep(kind: kind, content: content, toolID: toolID, toolArgs: toolArgs)
        steps.append(step)
        continuation.yield(.step(step))
    }
}

private extension StructuredAgentKernelExecutor {
    struct StructuredTurnGenerationDiagnostics: Sendable {
        let generationElapsedMs: Int
        let firstTokenLatencyMs: Int?
        let outputTokenCount: Int?
        let estimatedPromptTokenCount: Int?
        let maxTokensRequested: Int
        let maxTokensEffective: Int
        let promptCharCount: Int?
        let emptyOutputReason: String?
        let streamStarted: Bool?
        let selectedRuntime: String?
        let selectedAdapter: String?
        let modelIdentifier: String?
        let modelLoaded: Bool?
        let stopSequences: [String]
        let temperature: Double?
        let topP: Double?
        let cancellationStateBeforeStream: String?
        let firstChunkReceived: Bool?
        let textChunkCount: Int?
        let finalChunkReceived: Bool?
        let streamTerminationReason: String?
    }

    struct StructuredGenerationResult {
        let raw: String
        let forcedParseError: AgentTurnParseError?
        let diagnostics: StructuredTurnGenerationDiagnostics
    }

    struct StructuredPromptPreflight: Sendable {
        let contextSize: Int
        let finalPromptChars: Int
        let estimatedPromptTokens: Int
        let fits: Bool
    }

    static func buildSystemPrompt(request: AgentKernelRequest, availableTools: [ToolDefinition]) -> String {
        var sys = """
        You are Lumen's structured routing executor. Emit one raw JSON object only.

        The runtime attaches the active JSON schema for this turn. Follow that active schema exactly.
        Possible schemas:
        {"thought":"short","action":{"tool":"tool.id","args":{}}}
        {"thought":"short","final":"user-facing answer"}

        Rules:
        - Start with { and stop after the matching }.
        - No markdown, prose, code fences, XML, bullets, or hidden reasoning outside JSON.
        - Use double-quoted JSON. Use {} for empty args.
        - Choose exactly one of action or final.
        - action must be a JSON object, never a string.
        - action.tool must be one available tool.
        - Use final when no tool is needed or observations already answer the user.
        - Keep thought under 12 words and final concise.
        """
        if !availableTools.isEmpty {
            sys += "\nAvailable tools:\n"
            sys += compactStructuredToolList(availableTools, userMessage: request.userMessage)
            sys += "\n"
        } else {
            sys += "\nNo tools are available. Emit final JSON only.\n"
        }
        let appPrompt = boundedStructuredContextNote(sanitizeSystemPromptForStructuredOutput(request.systemPrompt))
        if !appPrompt.isEmpty {
            sys += "\nContext note, lower priority than JSON/tool rules:\n\(appPrompt)\n"
        }
        sys += "\nRouting hints: current web/research -> web.search; local files/notes -> rag.search; save user preference -> memory.save; recall stored memory -> memory.recall; weather -> weather; draft email -> mail.draft; scheduled agent run -> trigger.create. Do not include attachment bodies or local source snippets in this routing turn."
        return sys
    }

    static func buildAgentUserTurn(
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        stepIndex: Int,
        scratchpad: String
    ) -> String {
        var out = ""
        let context = sanitizedHistoryContext(request.history.map { ($0.role.messageRole, $0.content) })
        let userMessage = sanitizedStructuredUserMessage(request.userMessage)
        if !context.isEmpty {
            out += "Conversation context, for reference only. Do not imitate its formatting:\n\(context)\n\n"
        }
        out += "User request:\n\(userMessage)"
        if stepIndex > 0 {
            out += "\n\nPrior structured turns and observations:\n\(compactStructuredScratchpad(scratchpad))"
            out += "\n\nEmit the next JSON object now. If the observations already answer the user, choose final. If another tool is absolutely required, action must be an object like {\"tool\":\"tool.id\",\"args\":{}}; never emit action as a string."
        } else if shouldForceActionSchema(request: request, availableTools: availableTools, stepIndex: stepIndex, hasObservations: false) {
            out += "\n\nEmit the first JSON object now. This tool-backed request requires an action before any final answer. Use exactly one available tool id. Allowed tool IDs: \(allowedToolIDsList(availableTools)). Do not emit final or {}."
        } else if availableTools.isEmpty {
            out += "\n\nEmit the first JSON object now. No tools are available, so emit final only."
        } else {
            out += "\n\nEmit the first JSON object now. Choose either action or final."
        }
        return out
    }

    static func structuredAgentResponseFormat(
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        stepIndex: Int,
        hasObservations: Bool
    ) -> LLMResponseFormat {
        if shouldForceActionSchema(request: request, availableTools: availableTools, stepIndex: stepIndex, hasObservations: hasObservations) {
            return .constrainedJSON(schema: structuredAgentActionResponseSchema)
        }
        if availableTools.isEmpty {
            return .constrainedJSON(schema: structuredAgentFinalResponseSchema)
        }
        return .constrainedJSON(schema: structuredAgentResponseSchema)
    }

    static func shouldForceActionSchema(
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        stepIndex: Int,
        hasObservations: Bool
    ) -> Bool {
        guard stepIndex == 0, !hasObservations, !availableTools.isEmpty else { return false }
        let routing = IntentRouter.classify(sanitizedStructuredUserMessage(request.userMessage))
        return IntentRouter.intentRequiresTool(routing) && !routing.requiresClarification
    }

    static func structuredToolSourceIDs(
        secureIDs: Set<String>,
        optionIDs: Set<String>,
        routingIDs: Set<String>
    ) -> Set<String> {
        let secure = Set(secureIDs.map(ToolRouteGuard.canonicalToolID))
        let options = Set(optionIDs.map(ToolRouteGuard.canonicalToolID))
        let routing = Set(routingIDs.map(ToolRouteGuard.canonicalToolID))
        let scoped: Set<String>
        if !options.isEmpty, !routing.isEmpty {
            let intersection = options.intersection(routing)
            scoped = intersection.isEmpty ? options : intersection
        } else if !options.isEmpty {
            scoped = options
        } else if !routing.isEmpty {
            scoped = routing
        } else {
            return secure
        }
        let secured = scoped.intersection(secure)
        return secured.isEmpty ? scoped : secured
    }

    static func shouldStopAfterFirstWebObservation(
        request: AgentKernelRequest,
        actionTool: String,
        observations: [(tool: String, result: String)]
    ) -> Bool {
        guard request.traceCorrelation?.e2eRunID != nil || request.traceCorrelation?.scenarioID?.hasPrefix("training-") == true else { return false }
        let prompt = sanitizedStructuredUserMessage(request.userMessage)
        let routing = IntentRouter.classify(prompt)
        guard routing.intent == .webSearch,
              ToolRouteGuard.canonicalToolID(actionTool) == "web.search",
              hasUsableObservation(for: .webSearch, observations: observations) else {
            return false
        }
        let lowerPrompt = prompt.lowercased()
        return !lowerPrompt.contains("fetch")
            && !lowerPrompt.contains("open the url")
            && !lowerPrompt.contains("open this url")
            && !lowerPrompt.contains("read the full")
    }

    static func shouldStopAfterToolResult(_ status: ToolResultStatus) -> Bool {
        status != .success
    }

    static func phoneCallContinuationAfterContactObservation(
        routing: IntentRoutingDecision,
        actionTool: String,
        observation: String,
        availableToolIDs: Set<String>
    ) -> AgentPhoneCallContinuation? {
        guard routing.intent == .phoneCall,
              ToolRouteGuard.canonicalToolID(actionTool) == "contacts.search" else {
            return nil
        }
        return SlotAgentService.phoneCallContinuation(
            afterContactObservation: observation,
            availableToolIDs: availableToolIDs,
            routing: routing
        )
    }

    static func toolRequiredFinalNeedsAction(
        _ final: String,
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        observations: [(tool: String, result: String)]
    ) -> Bool {
        guard observations.isEmpty, !availableTools.isEmpty else { return false }
        let routing = IntentRouter.classify(sanitizedStructuredUserMessage(request.userMessage))
        guard IntentRouter.intentRequiresTool(routing), !routing.requiresClarification else { return false }
        return structuredFinalIsGenericFallback(final) || structuredFinalIsPlaceholder(final)
    }

    static func postprocessStructuredFinalAnswer(
        _ finalAnswer: String,
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        observations: [(tool: String, result: String)],
        steps: [AgentStep]
    ) -> String {
        let prompt = sanitizedStructuredUserMessage(request.userMessage)
        let routing = IntentRouter.classify(prompt)
        if routing.intent == .weather,
           weatherFinalOverstatesPrecipitation(finalAnswer: finalAnswer, observations: observations) {
            let weatherObservation = observations
                .last(where: { ToolRouteGuard.canonicalToolID($0.tool) == "weather" })?
                .result
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let observationText = weatherObservation?.isEmpty == false ? weatherObservation! : "the weather observation"
            return "Weather update: \(observationText). No precipitation was reported in the weather observation."
        }
        if let memoryFinal = memorySaveRecallFinalIfApplicable(routing: routing, prompt: prompt, steps: steps) {
            return memoryFinal
        }
        if routing.intent == .webSearch,
           hasUsableObservation(for: .webSearch, observations: observations),
           webFinalRequiresObservationFallback(finalAnswer) {
            return deterministicObservationFallback(observations: observations, intent: .webSearch)
                ?? "I found web results, but could not synthesize a grounded final answer from them."
        }
        if routing.intent == .rag || routing.intent == .files,
           structuredFinalIsGenericFallback(finalAnswer),
           hasUsableObservation(for: routing.intent, observations: observations),
           let deterministic = deterministicObservationFallback(observations: observations, intent: routing.intent) {
            return deterministic
        }
        if let requiredMemoryAction = nextRequiredMemoryAction(
            memoryPlan: MemoryCommandPlan.saveThenRecall(from: prompt),
            steps: steps,
            availableToolIDs: Set(availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        ) {
            return "I could not complete the memory request because the required \(requiredMemoryAction.tool) action did not run."
        }
        return finalAnswer
    }

    static func deterministicObservationFallback(observations: [(tool: String, result: String)], intent: UserIntent) -> String? {
        guard !observations.isEmpty else { return nil }
        if intent == .webSearch {
            return deterministicWebSummaryFallback(observations: observations)
                ?? deterministicWebResultFallback(observations: observations)
        }
        if intent == .rag || intent == .files {
            if let emptyState = ragOrFilesEmptyObservationFinal(observations: observations) {
                return emptyState
            }
            let sourced = observations.prefix(3).enumerated().map { index, obs in
                "[\(index + 1)] \(compactObservationResult(obs.result, limit: 700))"
            }
            return "Summary\n\(sourced.joined(separator: "\n"))\n\nKey modules\nNo explicit modules were present in the retrieved snippets unless named above."
        }
        guard let last = observations.last else { return nil }
        let compact = compactObservationResult(last.result, limit: 1_200)
        return compact.isEmpty ? nil : compact
    }

    static func deterministicWebSummaryFallback(observations: [(tool: String, result: String)]) -> String? {
        let joined = observations
            .filter {
                let tool = ToolRouteGuard.canonicalToolID($0.tool)
                return tool == "web.search" || tool == "web.fetch"
            }
            .map(\.result)
            .joined(separator: "\n")
        let candidates = webSummaryCandidates(from: joined)
        let useful = candidates.isEmpty
            ? joined.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            : candidates
        let lines = useful.filter { line in
            let lower = line.lowercased()
            return line.count >= 16
                && !lower.hasPrefix("search results for:")
                && !lower.hasPrefix("web search results:")
                && !lower.hasPrefix("http")
                && !lower.contains("<lumen_web_payload")
                && !lower.contains("\"mediakind\"")
                && !webFinalRequiresObservationFallback(line)
        }
        let bullets = prioritizeWebCandidates(lines).prefix(2).map { "- \(compactObservationResult($0, limit: 220))" }
        guard bullets.count >= 2 else { return nil }
        return "Summary:\n\(bullets.joined(separator: "\n"))"
    }

    static func deterministicWebResultFallback(observations: [(tool: String, result: String)]) -> String? {
        let joined = observations
            .filter {
                let tool = ToolRouteGuard.canonicalToolID($0.tool)
                return tool == "web.search" || tool == "web.fetch"
            }
            .map(\.result)
            .joined(separator: "\n")
        let candidates = prioritizeWebCandidates(webSummaryCandidates(from: joined))
            .filter { candidate in
                let lower = candidate.lowercased()
                return candidate.count >= 12
                    && !lower.hasPrefix("search results for:")
                    && !lower.hasPrefix("web search results:")
                    && !lower.contains("<lumen_web_payload")
                    && !webFinalRequiresObservationFallback(candidate)
            }
        let bullets = candidates.prefix(3).map { "- \(compactObservationResult($0, limit: 180))" }
        guard !bullets.isEmpty else {
            let compact = compactObservationResult(joined, limit: 500)
            return compact.isEmpty ? nil : "Web results found:\n- \(compact)"
        }
        if bullets.count == 1 {
            return "Web results found:\n\(bullets[0])\n- The remaining result payload did not include enough snippet detail for a stronger synthesis."
        }
        return "Web results found:\n\(bullets.joined(separator: "\n"))"
    }

    static func webSummaryCandidates(from text: String) -> [String] {
        var candidates = webPayloadCandidates(from: text)
        let fallbackText = strippingWebPayloadBlocks(from: text)
        let patterns = [
            #"(?is)\{[^{}]*"title"\s*:\s*"([^"]+)"[^{}]*"snippet"\s*:\s*"([^"]+)"[^{}]*\}"#,
            #"(?is)\{[^{}]*"snippet"\s*:\s*"([^"]+)"[^{}]*"title"\s*:\s*"([^"]+)"[^{}]*\}"#,
            #"(?is)"title"\s*:\s*"([^"]+)".{0,900}?"snippet"\s*:\s*"([^"]+)""#,
            #"(?is)"snippet"\s*:\s*"([^"]+)".{0,900}?"title"\s*:\s*"([^"]+)""#,
            #"(?is)"title"\s*:\s*"([^"]+)""#
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let ns = fallbackText as NSString
            for match in regex.matches(in: fallbackText, range: NSRange(location: 0, length: ns.length)) {
                if match.numberOfRanges >= 3 {
                    candidates.append("\(ns.substring(with: match.range(at: 1))): \(ns.substring(with: match.range(at: 2)))")
                } else if match.numberOfRanges >= 2 {
                    candidates.append(ns.substring(with: match.range(at: 1)))
                }
            }
        }
        if !candidates.isEmpty {
            return dedupedWebCandidates(candidates.map(decodeJSONStringEscapes).filter { !$0.isEmpty })
        }
        return text.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    static func strippingWebPayloadBlocks(from text: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: #"(?is)<lumen_web_payload[^>]*>.*?</lumen_web_payload>"#) else { return text }
        return regex.stringByReplacingMatches(in: text, range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "")
    }

    static func webPayloadCandidates(from text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: #"(?is)<lumen_web_payload[^>]*>(.*?)</lumen_web_payload>"#) else { return [] }
        let ns = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).flatMap { match -> [String] in
            guard match.numberOfRanges >= 2 else { return [] }
            let payload = ns.substring(with: match.range(at: 1))
            guard let data = payload.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
            var items: [String] = []
            for key in ["results", "media"] {
                guard let array = object[key] as? [[String: Any]] else { continue }
                for item in array {
                    guard let title = (item["title"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !title.isEmpty else { continue }
                    let snippet = (item["snippet"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
                    let source = (item["source"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let snippet, !snippet.isEmpty {
                        items.append("\(title): \(snippet)")
                    } else if let source, !source.isEmpty {
                        items.append("\(title) - \(source)")
                    } else {
                        items.append(title)
                    }
                }
            }
            return items
        }
    }

    static func webFinalRequiresObservationFallback(_ finalAnswer: String) -> Bool {
        let text = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        if text.isEmpty { return true }
        if lower.contains("no direct answer from web search") { return true }
        if lower.hasPrefix("search results for:") || lower.hasPrefix("web search results:") { return true }
        if lower.contains("search results for:") && (lower.contains("\nhttp") || lower.contains("\n- http")) { return true }
        if RoutingJSONLeakDetector.containsInternalRoutingJSON(text) { return true }
        if text.range(of: #"(?is)^\s*(https?://\S+)\s*$"#, options: .regularExpression) != nil { return true }
        if text.range(of: #"(?is)^\s*(?:check\s+out|see|read|visit|open|here(?:'s| is))\b[^\n]{0,180}https?://\S+\s*\.?\s*$"#, options: .regularExpression) != nil { return true }
        return false
    }

    static func hasUsableObservation(for intent: UserIntent, observations: [(tool: String, result: String)]) -> Bool {
        observations.contains {
            let tool = ToolRouteGuard.canonicalToolID($0.tool)
            let result = $0.result.trimmingCharacters(in: .whitespacesAndNewlines)
            guard isUsableObservationResult(result) else { return false }
            switch intent {
            case .webSearch:
                return tool == "web.search" || tool == "web.fetch"
            case .rag, .files:
                return tool == "rag.search" || tool == "files.read"
            default:
                return ToolRegistry.find(id: tool) != nil
            }
        }
    }

    static func isUsableObservationResult(_ result: String) -> Bool {
        let text = result.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return false }
        let lower = text.lowercased()
        return !lower.hasPrefix("unknown tool:")
            && !lower.contains(" is disabled")
            && !lower.contains("tool disabled")
            && !lower.contains("disabled. enable it in tools")
    }

    static func ragOrFilesEmptyObservationFinal(observations: [(tool: String, result: String)]) -> String? {
        for observation in observations.reversed() {
            let tool = ToolRouteGuard.canonicalToolID(observation.tool)
            guard tool == "rag.search" || tool == "files.read" else { continue }
            let result = observation.result.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !result.isEmpty else { continue }
            let lower = result.lowercased()
            if lower.contains("retrieval is unavailable")
                || lower.contains("rag storage unavailable")
                || lower.contains("storage unavailable")
                || lower.contains("index unavailable") {
                return "RAG retrieval is unavailable right now. \(compactObservationResult(result, limit: 220))"
            }
            if lower.contains("no matching snippets")
                || lower.contains("no matching results")
                || lower.contains("no snippets found")
                || lower.contains("no matches found")
                || lower.contains("no results found") {
                return "No matching snippets were found in the local index."
            }
            if lower.contains("index is empty")
                || lower.contains("local index appears empty")
                || lower.contains("no imported files")
                || lower.contains("nothing has been indexed") {
                return "The local retrieval index is empty. Import or reindex files before searching."
            }
        }
        return nil
    }

    static func structuredFinalIsGenericFallback(_ finalAnswer: String) -> Bool {
        let lower = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return lower.isEmpty
            || lower.contains("i'm ready. please ask again")
            || lower.contains("please ask again or tell me what you'd like to do next")
            || lower.contains("tool output could not be validated")
            || lower.contains("could not be validated")
            || lower.contains("i couldn't produce a confident answer")
            || lower.contains("i couldn't find a confident answer")
    }

    static func structuredFinalIsPlaceholder(_ final: String) -> Bool {
        let text = final.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        if SchemaPlaceholderDetector.isPlaceholderFinal(text) { return true }
        if lower.range(of: #"^\[\s*(insert|add|include|provide)\b[^\]]+\]\s*$"#, options: .regularExpression) != nil { return true }
        return lower.contains("[insert local weather information]")
            || lower.contains("[insert weather")
            || lower.contains("<tool result>")
            || lower.contains("<tool_output>")
    }

    static func weatherFinalOverstatesPrecipitation(finalAnswer: String, observations: [(tool: String, result: String)]) -> Bool {
        let answer = finalAnswer.lowercased()
        let recommendsPrecipitationAction = answer.contains("umbrella")
            || answer.contains("likely raining")
            || answer.contains("it's raining")
            || answer.contains("it is raining")
        guard recommendsPrecipitationAction else { return false }
        let weatherObservation = observations
            .filter { ToolRouteGuard.canonicalToolID($0.tool) == "weather" }
            .map(\.result)
            .joined(separator: "\n")
            .lowercased()
        let precipitationSignals = ["rain", "raining", "drizzle", "precip", "precipitation", "shower", "forecasted rain", "chance of rain", "probability of precipitation", "freezing rain", "snow", "thunderstorm"]
        return !precipitationSignals.contains { weatherObservation.contains($0) }
    }

    static func repairedMemoryActionIfNeeded(
        modelAction: AgentAction,
        memoryPlan: MemoryCommandPlan?,
        steps: [AgentStep],
        availableToolIDs: Set<String>
    ) -> (action: AgentAction, reflection: AgentStep?) {
        guard let required = nextRequiredMemoryAction(memoryPlan: memoryPlan, steps: steps, availableToolIDs: availableToolIDs) else {
            return (modelAction, nil)
        }
        let modelTool = ToolRouteGuard.canonicalToolID(modelAction.tool)
        let requiredTool = ToolRouteGuard.canonicalToolID(required.tool)
        if modelTool == requiredTool, memoryActionArgumentsMatch(modelAction, required: required) {
            return (modelAction, nil)
        }
        return (
            required,
            AgentStep(kind: .reflection, content: "Memory save-then-recall invariant repaired \(modelTool) into \(requiredTool) before tool execution.")
        )
    }

    static func nextRequiredMemoryAction(memoryPlan: MemoryCommandPlan?, steps: [AgentStep], availableToolIDs: Set<String>) -> AgentAction? {
        guard let memoryPlan else { return nil }
        let actionToolIDs = steps.filter { $0.kind == .action }.compactMap(\.toolID).map(ToolRouteGuard.canonicalToolID)
        if !actionToolIDs.contains("memory.save") {
            guard availableToolIDs.contains("memory.save") else { return nil }
            return AgentAction(tool: "memory.save", args: ["content": .string(memoryPlan.saveContent), "kind": .string("fact")])
        }
        if !actionToolIDs.contains("memory.recall") {
            guard availableToolIDs.contains("memory.recall") else { return nil }
            return AgentAction(tool: "memory.recall", args: ["query": .string(memoryPlan.recallQuery)])
        }
        return nil
    }

    static func memoryActionArgumentsMatch(_ action: AgentAction, required: AgentAction) -> Bool {
        let requiredKeys = Set(required.args.keys)
        guard Set(action.args.keys).isSuperset(of: requiredKeys) else { return false }
        return requiredKeys.allSatisfy { action.args[$0]?.stringValue == required.args[$0]?.stringValue }
    }

    static func repairedMapsSearchActionIfNeeded(
        modelAction: AgentAction?,
        routing: IntentRoutingDecision,
        prompt: String,
        steps: [AgentStep],
        availableToolIDs: Set<String>
    ) -> (action: AgentAction, reflection: AgentStep)? {
        guard let required = nextRequiredMapsSearchAction(
            routing: routing,
            prompt: prompt,
            steps: steps,
            availableToolIDs: availableToolIDs
        ) else {
            return nil
        }
        let modelTool = modelAction.map { ToolRouteGuard.canonicalToolID($0.tool) } ?? "none"
        guard modelTool != "maps.search" else { return nil }
        return (
            required,
            AgentStep(kind: .reflection, content: "Maps search continuation repaired \(modelTool) into maps.search after location observation.")
        )
    }

    static func nextRequiredMapsSearchAction(
        routing: IntentRoutingDecision,
        prompt: String,
        steps: [AgentStep],
        availableToolIDs: Set<String>
    ) -> AgentAction? {
        guard routing.intent == .maps, availableToolIDs.contains("maps.search") else { return nil }
        let actionToolIDs = steps.filter { $0.kind == .action }.compactMap(\.toolID).map(ToolRouteGuard.canonicalToolID)
        guard actionToolIDs.contains("location.current"), !actionToolIDs.contains("maps.search") else { return nil }
        let observationToolIDs = steps.filter { $0.kind == .observation }.compactMap(\.toolID).map(ToolRouteGuard.canonicalToolID)
        guard observationToolIDs.contains("location.current") else { return nil }
        return DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: prompt,
            availableToolIDs: availableToolIDs
        )
        .first { ToolRouteGuard.canonicalToolID($0.tool) == "maps.search" }
    }

    static func memorySaveRecallFinalIfApplicable(routing: IntentRoutingDecision, prompt: String, steps: [AgentStep]) -> String? {
        guard routing.intent == .memory else { return nil }
        let actionSteps = steps.filter { $0.kind == .action }
        let actionToolIDs = actionSteps.compactMap(\.toolID).map(ToolRouteGuard.canonicalToolID)
        guard actionToolIDs.contains("memory.save"), actionToolIDs.contains("memory.recall") else { return nil }
        let lowerPrompt = prompt.lowercased()
        guard lowerPrompt.contains("tell me what you remembered")
            || lowerPrompt.contains("what you remembered")
            || lowerPrompt.contains("what did you remember")
        else { return nil }
        guard let savedContent = actionSteps.first(where: { ToolRouteGuard.canonicalToolID($0.toolID ?? "") == "memory.save" })?.toolArgs?["content"] else { return nil }
        return "I remember that \(rememberedPreference(from: savedContent))."
    }

    static func rememberedPreference(from content: String) -> String {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "you prefer concise bullet points" }
        if let range = trimmed.range(of: "I prefer ", options: [.caseInsensitive]) {
            let preference = preferenceFragment(String(trimmed[range.upperBound...]))
            if !preference.isEmpty { return "you prefer \(preference)" }
        }
        if let range = trimmed.range(of: "prefer ", options: [.caseInsensitive]) {
            let preference = preferenceFragment(String(trimmed[range.upperBound...]))
            if !preference.isEmpty { return "you prefer \(preference)" }
        }
        if let range = trimmed.range(of: "Remember that ", options: [.caseInsensitive]) {
            let remembered = preferenceFragment(String(trimmed[range.upperBound...]))
            if !remembered.isEmpty { return remembered }
        }
        return preferenceFragment(trimmed)
    }

    static func preferenceFragment(_ text: String) -> String {
        var fragment = text
        if let range = fragment.range(of: ", then", options: [.caseInsensitive]) {
            fragment = String(fragment[..<range.lowerBound])
        }
        for separator in [".", "\n", ";", "?", "!"] {
            if let range = fragment.range(of: separator) {
                fragment = String(fragment[..<range.lowerBound])
            }
        }
        return fragment.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func approvalBoundaryFinal(for toolID: String, action: AgentAction) -> String {
        switch toolID {
        case "alarm.request_authorization":
            return "Approval required for alarm.request_authorization. I did not request alarm authorization yet."
        case "alarm.schedule", "alarm.countdown", "alarm.pause", "alarm.resume", "alarm.stop", "alarm.snooze", "alarm.cancel":
            return "Approval required for \(toolID). I did not change alarms yet."
        case "calendar.create":
            return "Approval required for calendar.create. I did not create an event yet."
        case "reminders.create":
            return "Approval required for reminders.create. I did not create a reminder yet."
        case "mail.draft":
            return "Approval required for mail.draft. I did not prepare or send the email yet."
        case "messages.draft":
            return "Approval required for messages.draft. I did not prepare or send the message yet."
        case "trigger.create":
            return "Approval required for trigger.create. I did not schedule an agent run yet."
        case "phone.call":
            return "Approval required for phone.call. I did not place the call yet."
        case "camera.capture":
            return "Approval required for camera.capture. I did not open the camera yet."
        default:
            return "Approval required for \(action.displayContent). I did not run it yet."
        }
    }

    static func userVisibleToolObservation(toolID: String, result: ToolResult) -> String {
        let text = result.displayText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty { return text }
        let modelText = result.modelText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelText.isEmpty { return modelText }
        if let errorCode = result.errorCode, !errorCode.isEmpty {
            return "\(toolID) finished with status \(result.status.rawValue): \(errorCode)."
        }
        return "\(toolID) finished with status \(result.status.rawValue) and no user-visible output."
    }

    static func agentJSONMissingDecisionRetryUserTurn(from userTurn: String, rawOutput: String, allowedToolIDs: [String]) -> String {
        let clipped = String(redactedForDiagnostics(rawOutput).trimmingCharacters(in: .whitespacesAndNewlines).prefix(300))
        return """
        \(compactAgentJSONUserTurnForPreflight(userTurn))

        Previous live agent-json attempt emitted a JSON object with no action or final:
        \(clipped)

        This turn requires a tool action before any final answer. Emit exactly one action JSON object now.
        Use one available tool id only. Allowed tool IDs: \(allowedToolIDsList(allowedToolIDs)). Do not emit {}, final, prose, markdown, schema, status, approvalPrompt, or tool metadata fields.
        {"action":{"tool":"<allowed tool id>","args":{}}}
        /no_think
        Start with { and finish after the matching }. Output JSON only.
        """
    }

    static func agentJSONContextCompactionRequest(from request: GenerateRequest) -> GenerateRequest {
        GenerateRequest(
            id: request.id,
            sessionID: request.sessionID,
            systemPrompt: truncateSystemPromptForAgentJSONPreflight(request.systemPrompt),
            history: [],
            userMessage: compactAgentJSONUserTurnForPreflight(request.userMessage),
            temperature: min(request.temperature, 0.05),
            topP: min(request.topP, 0.6),
            repetitionPenalty: max(request.repetitionPenalty, 1.05),
            maxTokens: min(max(request.maxTokens / 2, structuredTurnMinTokenCap), 224),
            modelName: request.modelName,
            relevantMemories: [],
            attachments: [],
            responseFormat: request.responseFormat,
            seed: request.seed,
            developerTraceModeEnabled: false,
            reasoningCaptureEnabled: false,
            reasoningTraceBudgetCharacters: request.reasoningTraceBudgetCharacters,
            allowsMemoryPressureContinuation: request.allowsMemoryPressureContinuation
        )
    }

    static func compactAgentJSONUserTurnForPreflight(_ userTurn: String) -> String {
        var requestText = userTurn
        if let marker = requestText.range(of: "User request:") {
            requestText = String(requestText[marker.upperBound...])
        }
        if let emit = requestText.range(of: "\n\nEmit ") {
            requestText = String(requestText[..<emit.lowerBound])
        }
        if let prior = requestText.range(of: "\n\nPrior structured turns and observations:") {
            requestText = String(requestText[..<prior.lowerBound])
        }
        requestText = String(requestText.trimmingCharacters(in: .whitespacesAndNewlines).prefix(600))
        return """
        User request:
        \(requestText)

        Emit one JSON object now: action with an available tool, or final if no tool is needed.
        /no_think
        """
    }

    static func compactStructuredToolList(_ tools: [ToolDefinition], userMessage: String) -> String {
        let routing = IntentRouter.classify(sanitizedStructuredUserMessage(userMessage))
        let preferredIDs = Set(routing.allowedToolIDs.map(ToolRouteGuard.canonicalToolID))
        let ordered = tools.enumerated().sorted { left, right in
            let leftPreferred = preferredIDs.contains(ToolRouteGuard.canonicalToolID(left.element.id))
            let rightPreferred = preferredIDs.contains(ToolRouteGuard.canonicalToolID(right.element.id))
            if leftPreferred != rightPreferred { return leftPreferred && !rightPreferred }
            return left.offset < right.offset
        }
        return ordered.prefix(12).map { _, tool in
            "- \(ToolRouteGuard.canonicalToolID(tool.id)): \(compactToolDescription(tool))"
        }.joined(separator: "\n")
    }

    static func compactToolDescription(_ tool: ToolDefinition) -> String {
        let text = tool.description.trimmingCharacters(in: .whitespacesAndNewlines)
        let firstSentence = text.split(separator: ".").first.map(String.init)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? text
        let args = tool.capabilityContract.arguments
            .prefix(4)
            .map { $0.required ? "\($0.name)*" : $0.name }
            .joined(separator: ",")
        let argsPart = args.isEmpty ? nil : "args: \(args)"
        let combined = [firstSentence, argsPart].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: ". ")
        return String(combined.prefix(structuredToolDescriptionCharCap))
    }

    static func sanitizedHistoryContext(_ history: [(role: MessageRole, content: String)]) -> String {
        let recent = history.suffix(3)
        var lines: [String] = []
        var used = 0
        for item in recent {
            let role: String
            switch item.role {
            case .user: role = "User"
            case .assistant: role = "Assistant"
            case .system: role = "System"
            case .tool: role = "Tool"
            }
            let content = sanitizeHistoryContent(item.content)
            guard !content.isEmpty else { continue }
            let line = "\(role): \(content)"
            let cost = line.count + 1
            if used + cost > structuredHistoryTotalCharCap { break }
            lines.append(line)
            used += cost
        }
        return lines.joined(separator: "\n")
    }

    static func sanitizeHistoryContent(_ content: String) -> String {
        var text = stripInternalGrounding(from: content)
        text = text.replacingOccurrences(of: #"```[\s\S]*?```"#, with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: #"</?[A-Za-z_][A-Za-z0-9_.:-]*(?:\s+[^<>]*?)?/?>"#, with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: #"([{}\[\]`|])\1+"#, with: "$1", options: .regularExpression)
        text = text.replacingOccurrences(of: "<json>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "</json>", with: "", options: .caseInsensitive)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        text = text.replacingOccurrences(of: "\n", with: " ")
        while text.contains("  ") { text = text.replacingOccurrences(of: "  ", with: " ") }
        return String(text.prefix(structuredHistoryTurnCharCap))
    }

    static func sanitizeSystemPromptForStructuredOutput(_ systemPrompt: String) -> String {
        var trimmed = stripInternalGrounding(from: systemPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        for clause in ["Use fenced code blocks.", "Use fenced code blocks", "Think step-by-step.", "Think step-by-step", "Think step by step.", "Think step by step"] {
            trimmed = trimmed.replacingOccurrences(of: clause, with: "")
        }
        let blockedPhrases = ["markdown", "code fence", "code fences", "fenced code block", "fenced code blocks", "headings", "step-by-step", "step by step"]
        return trimmed.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { sentence in
                !sentence.isEmpty && !blockedPhrases.contains { phrase in
                    sentence.lowercased().contains(phrase)
                }
            }
            .joined(separator: "\n")
    }

    static func stripInternalGrounding(from text: String) -> String {
        var stripped = text
        for marker in ["<!-- LUMEN_GROUNDING_V1 -->", "[AVAILABLE LOCAL TOOLS]", "[RUNTIME POLICY]", "[LOCAL MEMORY]", "[LOCAL SOURCES]"] {
            if let range = stripped.range(of: marker, options: [.caseInsensitive]) {
                stripped = String(stripped[..<range.lowerBound])
            }
        }
        return stripped
    }

    static func sanitizedStructuredUserMessage(_ userMessage: String) -> String {
        let stripped = stripInternalGrounding(from: userMessage).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else {
            return String(userMessage.trimmingCharacters(in: .whitespacesAndNewlines).prefix(structuredUserMessageCharCap))
        }
        return String(stripped.prefix(structuredUserMessageCharCap))
    }

    static func boundedStructuredContextNote(_ text: String) -> String {
        String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(structuredContextNoteCharCap))
    }

    static func compactStructuredScratchpad(_ scratchpad: String) -> String {
        let compact = scratchpad
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        guard compact.count > structuredScratchpadCharCap else { return compact }
        return String(compact.suffix(structuredScratchpadCharCap))
    }

    static func compactScratchpadObservation(_ text: String) -> String {
        var compact = text.replacingOccurrences(of: "\n", with: " ")
        while compact.contains("  ") { compact = compact.replacingOccurrences(of: "  ", with: " ") }
        return compact.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func strictToolExecutableTurn(_ turn: AgentTurn) -> AgentTurn {
        guard turn.hadNoise, turn.action != nil else { return turn }
        return AgentTurn(thought: nil, action: nil, final: nil, parseError: .noisyOutput, hadNoise: true)
    }

    static func runtimeFailureParseError(from raw: String) -> AgentTurnParseError? {
        let lower = raw.lowercased()
        if lower.contains("prompt exceeded context window before generation")
            || lower.contains("prompt exceeds shared chat context window")
            || lower.contains("failed to initialize context: prompt exceeds") {
            return .contextWindowExceeded
        }
        return nil
    }

    static func agentJSONEmptyStreamReason(streamStarted: Bool, textChunkCount: Int, finalChunkReceived: Bool, taskCancelled: Bool, maxTokensEffective: Int) -> String {
        if maxTokensEffective <= 0 { return "decodeBudgetZero" }
        if taskCancelled, textChunkCount == 0 { return "cancelledBeforeFirstToken" }
        if !streamStarted { return "runtimeUnavailable" }
        if finalChunkReceived, textChunkCount == 0 { return "completedWithoutText" }
        if textChunkCount == 0 { return "stoppedBeforeFirstToken" }
        return "unknownEmptyStream"
    }

    static func structuredTurnMaxTokens(from requestedMaxTokens: Int) -> Int {
        min(max(requestedMaxTokens, structuredTurnMinTokenCap), structuredTurnMaxTokenCap)
    }

    static func agentTemperature(from userTemperature: Double) -> Double {
        min(max(userTemperature, 0.0), 0.15)
    }

    static func agentTopP(from userTopP: Double) -> Double {
        min(max(userTopP, 0.1), 0.85)
    }

    static func allowedToolIDsList(_ tools: [ToolDefinition]) -> String {
        allowedToolIDsList(tools.map { ToolRouteGuard.canonicalToolID($0.id) })
    }

    static func allowedToolIDsList(_ toolIDs: [String]) -> String {
        let ids = Array(Set(toolIDs.map(ToolRouteGuard.canonicalToolID))).sorted()
        return ids.isEmpty ? "none" : ids.joined(separator: ", ")
    }

    static func truncateSystemPromptForAgentJSONPreflight(_ systemPrompt: String) -> String {
        String(systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).prefix(1_200))
    }

    static func compactObservationResult(_ result: String, limit: Int) -> String {
        var compact = result.replacingOccurrences(of: "\n", with: " ")
        while compact.contains("  ") { compact = compact.replacingOccurrences(of: "  ", with: " ") }
        return String(compact.trimmingCharacters(in: .whitespacesAndNewlines).prefix(limit))
    }

    static func redactedForDiagnostics(_ raw: String) -> String {
        var text = raw
        text = text.replacingOccurrences(
            of: #"(?is)<think>.*?</think>"#,
            with: "<think>…redacted…</think>",
            options: .regularExpression
        )
        text = text.replacingOccurrences(
            of: #"(?is)<\|start_thinking\|>.*?<\|end_thinking\|>"#,
            with: "<|start_thinking|>…redacted…<|end_thinking|>",
            options: .regularExpression
        )
        return ModelOutputSanitizer.boundedPrefix(text, limit: 4_000)
    }

    static func prioritizeWebCandidates(_ candidates: [String]) -> [String] {
        candidates.sorted { webCandidatePriority($0) > webCandidatePriority($1) }
    }

    static func webCandidatePriority(_ text: String) -> Int {
        let lower = text.lowercased()
        if lower.contains("developer.apple.com") || lower.contains("swift.org") || lower.contains("docs.swift.org") { return 3 }
        if lower.contains("swift") || lower.contains("concurrency") || lower.contains("actor") || lower.contains("task") { return 2 }
        return 1
    }

    static func dedupedWebCandidates(_ candidates: [String]) -> [String] {
        var seen: Set<String> = []
        var unique: [String] = []
        for candidate in candidates {
            let key = normalizedWebCandidateTitle(candidate)
            guard !key.isEmpty, !seen.contains(key) else { continue }
            seen.insert(key)
            unique.append(candidate)
        }
        return unique
    }

    static func normalizedWebCandidateTitle(_ candidate: String) -> String {
        let title = candidate.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? candidate
        let normalized = title.lowercased().map { character -> Character in
            character.isLetter || character.isNumber ? character : " "
        }
        return String(normalized).split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    static func decodeJSONStringEscapes(_ text: String) -> String {
        let quoted = "\"\(text.replacingOccurrences(of: "\"", with: "\\\""))\""
        guard let data = quoted.data(using: .utf8),
              let decoded = try? JSONSerialization.jsonObject(with: data) as? String else {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

#if DEBUG
extension StructuredAgentKernelExecutor {
    static func structuredAgentResponseSchemaForTests(
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        stepIndex: Int,
        hasObservations: Bool
    ) -> String {
        switch structuredAgentResponseFormat(
            request: request,
            availableTools: availableTools,
            stepIndex: stepIndex,
            hasObservations: hasObservations
        ) {
        case .constrainedJSON(let schema):
            return schema
        case .plainText, .json, .toolCallJSON:
            return ""
        }
    }

    static func deterministicObservationFallbackForTests(
        observations: [(tool: String, result: String)],
        intent: UserIntent
    ) -> String? {
        deterministicObservationFallback(observations: observations, intent: intent)
    }

    static func shouldStopAfterFirstWebObservationForTests(
        request: AgentKernelRequest,
        actionTool: String,
        observations: [(tool: String, result: String)]
    ) -> Bool {
        shouldStopAfterFirstWebObservation(request: request, actionTool: actionTool, observations: observations)
    }

    static func structuredToolSourceIDsForTests(
        secureIDs: Set<String>,
        optionIDs: Set<String>,
        routingIDs: Set<String>
    ) -> Set<String> {
        structuredToolSourceIDs(secureIDs: secureIDs, optionIDs: optionIDs, routingIDs: routingIDs)
    }

    static func shouldStopAfterToolResultForTests(_ status: ToolResultStatus) -> Bool {
        shouldStopAfterToolResult(status)
    }

    static func phoneCallContinuationAfterContactObservationForTests(
        routing: IntentRoutingDecision,
        actionTool: String,
        observation: String,
        availableToolIDs: Set<String>
    ) -> AgentPhoneCallContinuation? {
        phoneCallContinuationAfterContactObservation(
            routing: routing,
            actionTool: actionTool,
            observation: observation,
            availableToolIDs: availableToolIDs
        )
    }

    static func toolRequiredFinalNeedsActionForTests(
        _ final: String,
        request: AgentKernelRequest,
        availableTools: [ToolDefinition],
        observations: [(tool: String, result: String)]
    ) -> Bool {
        toolRequiredFinalNeedsAction(final, request: request, availableTools: availableTools, observations: observations)
    }

    static func approvalBoundaryFinalForTests(toolID: String, action: AgentAction) -> String {
        approvalBoundaryFinal(for: toolID, action: action)
    }

    static func ragOrFilesEmptyObservationFinalForTests(observations: [(tool: String, result: String)]) -> String? {
        ragOrFilesEmptyObservationFinal(observations: observations)
    }

    static func repairedMemoryActionIfNeededForTests(
        modelAction: AgentAction,
        memoryPlan: MemoryCommandPlan?,
        steps: [AgentStep],
        availableToolIDs: Set<String>
    ) -> (action: AgentAction, reflection: AgentStep?) {
        repairedMemoryActionIfNeeded(
            modelAction: modelAction,
            memoryPlan: memoryPlan,
            steps: steps,
            availableToolIDs: availableToolIDs
        )
    }

    static func nextRequiredMemoryActionForTests(
        memoryPlan: MemoryCommandPlan?,
        steps: [AgentStep],
        availableToolIDs: Set<String>
    ) -> AgentAction? {
        nextRequiredMemoryAction(
            memoryPlan: memoryPlan,
            steps: steps,
            availableToolIDs: availableToolIDs
        )
    }

    static func repairedMapsSearchActionIfNeededForTests(
        modelAction: AgentAction?,
        routing: IntentRoutingDecision,
        prompt: String,
        steps: [AgentStep],
        availableToolIDs: Set<String>
    ) -> (action: AgentAction, reflection: AgentStep)? {
        repairedMapsSearchActionIfNeeded(
            modelAction: modelAction,
            routing: routing,
            prompt: prompt,
            steps: steps,
            availableToolIDs: availableToolIDs
        )
    }
}
#endif

private extension AgentKernelRequest {
    var toolInvocationSourceForStructuredAgent: ToolInvocationSource {
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

private extension AgentKernelSource {
    var isForegroundForStructuredAgent: Bool {
        switch self {
        case .chat, .voice, .appIntent, .diagnostics, .benchmark:
            return true
        case .trigger:
            return false
        }
    }
}
