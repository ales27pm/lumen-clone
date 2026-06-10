import Foundation
import Testing
@testable import Lumen

@Suite(.serialized)
struct AgentWorkflowMonitorTests {
    @Test func workflowMonitorClassifiesSlotAgentSignalsBySlotAndStatus() throws {
        let monitor = AgentWorkflowMonitor(maxEvents: 20)

        monitor.ingestForTests(.init(kind: .chatRuntimeTrace, values: [
            "phase": "routing",
            "intent": "weather",
            "turnID": "turn-1",
            "conversationID": "conversation-1",
            "allowedToolIDs": "location.current,weather"
        ], at: Date()))

        monitor.ingestForTests(.init(kind: .chatRuntimeTrace, values: [
            "phase": "action_selected",
            "toolID": "weather",
            "argKeys": "location",
            "turnID": "turn-1",
            "conversationID": "conversation-1"
        ], at: Date()))

        monitor.ingestForTests(.init(kind: .fallbackUsed, values: [
            "reason": "empty-after-sanitization",
            "source": "final-output-sanitizer"
        ], at: Date()))

        let snapshot = monitor.snapshot()
        #expect(snapshot.events.count == 3)
        #expect(snapshot.events[0].slot == .cortex)
        #expect(snapshot.events[1].slot == .executor)
        #expect(snapshot.events[1].selectedToolID == "weather")
        #expect(snapshot.events[2].slot == .mouth)
        #expect(snapshot.fallbackCount == 1)
        #expect(snapshot.touchedSlots.contains("cortex"))
        #expect(snapshot.touchedSlots.contains("executor"))
        #expect(snapshot.touchedSlots.contains("mouth"))
    }

    @Test func workflowMonitorTracksDurationsAndCompletionBySlot() throws {
        let monitor = AgentWorkflowMonitor(maxEvents: 20)
        monitor.ingestForTests(.init(kind: .slotAgentStart, values: ["promptChars": "42"], at: Date()))
        monitor.ingestForTests(.init(kind: .slotAgentGroundingComplete, values: [
            "path": "normal-agent",
            "elapsedMs": "37",
            "toolCount": "3"
        ], at: Date()))
        monitor.ingestForTests(.init(kind: .llamaComplete, values: [
            "elapsedMs": "1200",
            "firstTokenLatencyMs": "300",
            "tokensPerSecond": "12.5"
        ], at: Date()))

        let snapshot = monitor.snapshot()
        #expect(snapshot.totalDurationMsBySlot["runtime"] == 37)
        #expect(snapshot.totalDurationMsBySlot["fleet"] == 1200)
        #expect(snapshot.completedCountBySlot["fleet"] == 1)
        #expect(snapshot.events.last?.firstTokenLatencyMs == 300)
        #expect(snapshot.events.last?.tokensPerSecond == 12.5)
    }

    @Test func workflowMonitorReportIsJSONEncodable() throws {
        let monitor = AgentWorkflowMonitor(maxEvents: 5)
        monitor.ingestForTests(.init(kind: .chatRuntimeTrace, values: [
            "phase": "final",
            "finalChars": "18",
            "turnID": "turn-2"
        ], at: Date()))

        let data = try monitor.jsonReportData()
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("lastEventBySlot"))
        #expect(text.contains("mouth"))
    }
}
