import Foundation

nonisolated enum AgentWorkflowSlot: String, Codable, Sendable, CaseIterable, Identifiable {
    case cortex
    case executor
    case mouth
    case mimicry
    case rem
    case fleet
    case embedding
    case runtime
    case unknown

    var id: String { rawValue }

    static func normalized(_ value: String?) -> AgentWorkflowSlot? {
        guard let raw = value?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        let lowered = raw.lowercased()
        if let slot = AgentWorkflowSlot(rawValue: lowered) { return slot }
        if lowered.contains("cortex") { return .cortex }
        if lowered.contains("executor") { return .executor }
        if lowered.contains("mouth") || lowered.contains("bouche") { return .mouth }
        if lowered.contains("mimicry") { return .mimicry }
        if lowered.contains("rem") { return .rem }
        if lowered.contains("fleet") { return .fleet }
        if lowered.contains("embedding") { return .embedding }
        return nil
    }

    static func infer(kind: PersistentRuntimeDiagnosticSignalKind, phase: String, values: [String: String]) -> AgentWorkflowSlot {
        if let explicit = normalized(Self.value("activeAdapterSlot", in: values) ?? Self.value("adapterSlot", in: values) ?? Self.value("slot", in: values)) {
            return explicit
        }

        let normalizedPhase = phase.lowercased()
        switch kind {
        case .fallbackUsed:
            return .mouth
        case .llamaPromptBudget, .llamaFirstToken, .llamaComplete, .llamaCancel, .llamaFailure:
            return .fleet
        case .groundingCost:
            return .rem
        case .chatRuntimeTrace:
            if normalizedPhase.contains("routing") || normalizedPhase.contains("planned") || normalizedPhase.contains("budget") {
                return .cortex
            }
            if normalizedPhase.contains("action") || normalizedPhase.contains("observation") || normalizedPhase.contains("approval") {
                return .executor
            }
            if normalizedPhase.contains("final") || normalizedPhase.contains("clarification") {
                return .mouth
            }
            if normalizedPhase.contains("ground") {
                return .rem
            }
            return .runtime
        default:
            return .runtime
        }
    }

    private static func value(_ key: String, in values: [String: String]) -> String? {
        values[key] ?? values[key.lowercased()]
    }
}

nonisolated enum AgentWorkflowEventStatus: String, Codable, Sendable {
    case running
    case done
    case failed
    case cancelled
    case fallback
    case info
}

nonisolated struct AgentWorkflowEvent: Codable, Sendable, Identifiable, Equatable {
    let id: UUID
    let createdAt: Date
    let kind: String
    let phase: String
    let slot: AgentWorkflowSlot
    let status: AgentWorkflowEventStatus
    let turnID: String?
    let conversationID: String?
    let intent: String?
    let selectedToolID: String?
    let allowedToolIDs: [String]
    let durationMs: Int?
    let firstTokenLatencyMs: Int?
    let tokensPerSecond: Double?
    let promptChars: Int?
    let outputChars: Int?
    let fallbackReason: String?
    let parseError: String?
    let rawValues: [String: String]

    init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        kind: String,
        phase: String,
        slot: AgentWorkflowSlot,
        status: AgentWorkflowEventStatus,
        turnID: String?,
        conversationID: String?,
        intent: String?,
        selectedToolID: String?,
        allowedToolIDs: [String],
        durationMs: Int?,
        firstTokenLatencyMs: Int?,
        tokensPerSecond: Double?,
        promptChars: Int?,
        outputChars: Int?,
        fallbackReason: String?,
        parseError: String?,
        rawValues: [String: String]
    ) {
        self.id = id
        self.createdAt = createdAt
        self.kind = kind
        self.phase = phase
        self.slot = slot
        self.status = status
        self.turnID = turnID
        self.conversationID = conversationID
        self.intent = intent
        self.selectedToolID = selectedToolID
        self.allowedToolIDs = allowedToolIDs
        self.durationMs = durationMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.tokensPerSecond = tokensPerSecond
        self.promptChars = promptChars
        self.outputChars = outputChars
        self.fallbackReason = fallbackReason
        self.parseError = parseError
        self.rawValues = rawValues
    }
}

nonisolated struct AgentWorkflowSnapshot: Codable, Sendable, Equatable {
    let generatedAt: Date
    let events: [AgentWorkflowEvent]
    let activeBySlot: [String: AgentWorkflowEvent]
    let lastEventBySlot: [String: AgentWorkflowEvent]
    let completedCountBySlot: [String: Int]
    let fallbackCount: Int
    let errorCount: Int
    let totalDurationMsBySlot: [String: Int]

    var touchedSlots: [String] {
        Array(Set(events.map { $0.slot.rawValue })).sorted()
    }
}

nonisolated final class AgentWorkflowMonitor: @unchecked Sendable {
    static let shared = AgentWorkflowMonitor()

    private let lock = NSLock()
    private var observerID: UUID?
    private var events: [AgentWorkflowEvent] = []
    private let maxEvents: Int

    init(maxEvents: Int = 500, startObserving: Bool = false) {
        self.maxEvents = max(10, maxEvents)
        if startObserving {
            _ = start()
        }
    }

    deinit {
        stop()
    }

    @discardableResult
    func start() -> Bool {
        lock.lock()
        if observerID != nil {
            lock.unlock()
            return false
        }
        lock.unlock()

        let id = PersistentRuntimeDiagnosticsObserver.shared.addObserver { [weak self] signal in
            self?.ingest(signal)
        }

        lock.lock()
        observerID = id
        lock.unlock()
        return true
    }

    func stop() {
        let id: UUID?
        lock.lock()
        id = observerID
        observerID = nil
        lock.unlock()
        if let id {
            PersistentRuntimeDiagnosticsObserver.shared.removeObserver(id)
        }
    }

    func reset() {
        lock.lock()
        events.removeAll()
        lock.unlock()
    }

    func ingestForTests(_ signal: PersistentRuntimeDiagnosticSignal) {
        ingest(signal)
    }

    func ingest(_ signal: PersistentRuntimeDiagnosticSignal) {
        guard let event = Self.event(from: signal) else { return }
        lock.lock()
        events.append(event)
        if events.count > maxEvents {
            events.removeFirst(events.count - maxEvents)
        }
        lock.unlock()
    }

    func snapshot() -> AgentWorkflowSnapshot {
        let copied: [AgentWorkflowEvent]
        lock.lock()
        copied = events
        lock.unlock()

        var active: [String: AgentWorkflowEvent] = [:]
        var last: [String: AgentWorkflowEvent] = [:]
        var completed: [String: Int] = [:]
        var durations: [String: Int] = [:]
        var fallbackCount = 0
        var errorCount = 0

        for event in copied {
            let key = event.slot.rawValue
            last[key] = event
            switch event.status {
            case .running:
                active[key] = event
            case .done:
                active[key] = nil
                completed[key, default: 0] += 1
            case .failed:
                active[key] = nil
                errorCount += 1
            case .cancelled:
                active[key] = nil
            case .fallback:
                fallbackCount += 1
            case .info:
                break
            }
            if let durationMs = event.durationMs {
                durations[key, default: 0] += durationMs
            }
        }

        return AgentWorkflowSnapshot(
            generatedAt: Date(),
            events: copied,
            activeBySlot: active,
            lastEventBySlot: last,
            completedCountBySlot: completed,
            fallbackCount: fallbackCount,
            errorCount: errorCount,
            totalDurationMsBySlot: durations
        )
    }

    func jsonReportData(pretty: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if pretty {
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        }
        return try encoder.encode(snapshot())
    }

    private static func event(from signal: PersistentRuntimeDiagnosticSignal) -> AgentWorkflowEvent? {
        let values = signal.values
        let phase = value("phase", in: values) ?? signal.kind.rawValue
        let slot = AgentWorkflowSlot.infer(kind: signal.kind, phase: phase, values: values)
        let selectedTool = value("toolID", in: values) ?? value("selectedToolID", in: values)
        let allowed = value("allowedToolIDs", in: values)?
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty } ?? []

        return AgentWorkflowEvent(
            createdAt: signal.at,
            kind: signal.kind.rawValue,
            phase: phase,
            slot: slot,
            status: status(kind: signal.kind, phase: phase, values: values),
            turnID: value("turnID", in: values),
            conversationID: value("conversationID", in: values),
            intent: value("intent", in: values),
            selectedToolID: selectedTool,
            allowedToolIDs: allowed,
            durationMs: intValue(value("elapsedMs", in: values) ?? value("generationElapsedMs", in: values)),
            firstTokenLatencyMs: intValue(value("firstTokenLatencyMs", in: values) ?? value("latencyMs", in: values)),
            tokensPerSecond: doubleValue(value("tokensPerSecond", in: values) ?? value("decodeTokensPerSecond", in: values)),
            promptChars: intValue(value("promptChars", in: values)),
            outputChars: intValue(value("outputChars", in: values) ?? value("finalChars", in: values)),
            fallbackReason: value("reason", in: values) ?? value("fallbackReason", in: values),
            parseError: value("parseError", in: values),
            rawValues: values
        )
    }

    private static func status(kind: PersistentRuntimeDiagnosticSignalKind, phase: String, values: [String: String]) -> AgentWorkflowEventStatus {
        let normalizedPhase = phase.lowercased()
        if kind == .fallbackUsed || normalizedPhase.contains("fallback") { return .fallback }
        if normalizedPhase.contains("error") || kind == .llamaFailure { return .failed }
        if normalizedPhase.contains("cancel") || kind == .llamaCancel || kind == .slotAgentCancel { return .cancelled }
        if normalizedPhase.contains("end") || normalizedPhase.contains("done") || normalizedPhase.contains("final") || kind == .llamaComplete || kind == .slotAgentEnd || kind == .slotAgentDoneYielded {
            return .done
        }
        if normalizedPhase.contains("start") || kind == .slotAgentStart {
            return .running
        }
        return .info
    }

    private static func value(_ key: String, in values: [String: String]) -> String? {
        values[key] ?? values[key.lowercased()]
    }

    private static func intValue(_ value: String?) -> Int? {
        guard let value else { return nil }
        return Int(value)
    }

    private static func doubleValue(_ value: String?) -> Double? {
        guard let value else { return nil }
        return Double(value)
    }
}
