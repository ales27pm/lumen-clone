import XCTest
@testable import Lumen

final class ImproveLoopSampleGateTests: XCTestCase {
    func testQuarantinesLegacyToolNamespace() {
        let repair = AgentBehaviorRepairSample(
            id: UUID(),
            createdAt: Date(),
            agent: "cortex",
            violationCode: "tool_not_allowed_by_runtime_router",
            promptPrefix: "What is my name",
            expected: "memory.recall",
            badOutput: "contacts.lookup",
            correctedOutput: "memory.recall",
            lesson: "Use canonical memory route.",
            curriculum: "routing"
        )
        let audit = AgentBehaviorAuditReport(
            passed: false,
            score: 0.2,
            generatedAt: Date(),
            traceCount: 1,
            violationCount: 1,
            sourceCommit: "test",
            violations: [],
            recommendations: [],
            repairSamples: [repair]
        )

        let dataset = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: audit,
            traces: [],
            scenarioResults: [],
            sourceCommit: "test"
        )

        XCTAssertTrue(dataset.acceptedTraining.isEmpty)
        XCTAssertEqual(dataset.quarantinedSamples.first?.sampleType, .rejectedLegacyToolNamespace)
        XCTAssertEqual(dataset.counters.legacyToolNamespaceRejected, 1)
    }

    func testAcceptsCorrectedCanonicalRepairSample() {
        let repair = AgentBehaviorRepairSample(
            id: UUID(),
            createdAt: Date(),
            agent: "mouth",
            violationCode: "final_answer_style",
            promptPrefix: "Observation: Saved: User's name is Alexis",
            expected: "friendly final",
            badOutput: "Saved: User's name is Alexis",
            correctedOutput: "Got it — I’ll remember your name is Alexis.",
            lesson: "Turn safe observations into concise user-visible responses.",
            curriculum: "mouth"
        )
        let audit = AgentBehaviorAuditReport(
            passed: false,
            score: 0.8,
            generatedAt: Date(),
            traceCount: 1,
            violationCount: 1,
            sourceCommit: "test",
            violations: [],
            recommendations: [],
            repairSamples: [repair]
        )

        let dataset = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: audit,
            traces: [],
            scenarioResults: [],
            sourceCommit: "test"
        )

        XCTAssertEqual(dataset.acceptedTraining.count, 1)
        XCTAssertEqual(dataset.acceptedTraining.first?.sampleType, .mouthObservationFinal)
        XCTAssertTrue(dataset.quarantinedSamples.isEmpty)
    }

    func testParseErrorTraceBecomesRegressionNotTraining() {
        let trace = AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "cortex",
            stage: "orchestrator-json",
            intent: "memory",
            promptPrefix: "What is my name?",
            rawOutputPrefix: "I hit an internal response-format issue.",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["memory.recall"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: "noJSONObject",
            emittedFinalInActionTurn: false
        )

        let dataset = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: nil,
            traces: [trace],
            scenarioResults: [],
            sourceCommit: "test"
        )

        XCTAssertTrue(dataset.acceptedTraining.isEmpty)
        XCTAssertEqual(dataset.regressionTests.first?.sampleType, .cortexJSONContract)
    }

    func testCPUWatchdogDegradedIsRuntimeResourceFallbackNotTraining() {
        let repair = AgentBehaviorRepairSample(
            id: UUID(),
            createdAt: Date(),
            agent: "cortex",
            violationCode: "rag_grounding",
            promptPrefix: "Search local docs and summarize modules.",
            expected: "grounded RAG final",
            badOutput: "I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cpu-watchdog-degraded.",
            correctedOutput: "Mention modules and sources.",
            lesson: "RAG should ground answers.",
            curriculum: "rag"
        )
        let audit = AgentBehaviorAuditReport(
            passed: false,
            score: 0.2,
            generatedAt: Date(),
            traceCount: 1,
            violationCount: 1,
            sourceCommit: "test",
            violations: [],
            recommendations: [],
            repairSamples: [repair]
        )

        let dataset = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: audit,
            traces: [],
            scenarioResults: [],
            sourceCommit: "test"
        )

        XCTAssertTrue(dataset.acceptedTraining.isEmpty)
        XCTAssertEqual(dataset.quarantinedSamples.first?.sampleType, .rejectedResourceFallback)
        XCTAssertEqual(dataset.counters.resourceFallbackRejected, 1)
    }

    func testRAGEmptyIndexIsQuarantinedNotTraining() {
        let repair = AgentBehaviorRepairSample(
            id: UUID(),
            createdAt: Date(),
            agent: "mouth",
            violationCode: "rag_grounding",
            promptPrefix: "Search my files for architecture notes and summarize key modules.",
            expected: "honest empty retrieval final",
            badOutput: "No matching documents found. The local document index appears empty. Import local files and try again.",
            correctedOutput: "Key modules: core module details were retrieved from local file snippets [1].",
            lesson: "Do not hallucinate RAG sources.",
            curriculum: "rag"
        )
        let audit = AgentBehaviorAuditReport(
            passed: false,
            score: 0.2,
            generatedAt: Date(),
            traceCount: 1,
            violationCount: 1,
            sourceCommit: "test",
            violations: [],
            recommendations: [],
            repairSamples: [repair]
        )

        let dataset = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: audit,
            traces: [],
            scenarioResults: [],
            sourceCommit: "test"
        )

        XCTAssertTrue(dataset.acceptedTraining.isEmpty)
        XCTAssertEqual(dataset.quarantinedSamples.first?.sampleType, .rejectedArchitectureFailure)
    }

    func testInternalRoutingJSONLeakageIsQuarantinedNotTraining() {
        let repair = AgentBehaviorRepairSample(
            id: UUID(),
            createdAt: Date(),
            agent: "mouth",
            violationCode: "final_leaked_internal_json",
            promptPrefix: "Search web and summarize.",
            expected: "web summary",
            badOutput: #"{"intent":"webSearch","nextModel":"rag","reasoningSummary":"bad","requiresApproval":false,"sourceFile":"ios/Lumen/Models/ToolDefinition.swift"}"#,
            correctedOutput: "Use structured concurrency and MainActor isolation.",
            lesson: "Do not leak internal routing JSON.",
            curriculum: "web"
        )
        let audit = AgentBehaviorAuditReport(
            passed: false,
            score: 0.2,
            generatedAt: Date(),
            traceCount: 1,
            violationCount: 1,
            sourceCommit: "test",
            violations: [],
            recommendations: [],
            repairSamples: [repair]
        )

        let dataset = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: audit,
            traces: [],
            scenarioResults: [],
            sourceCommit: "test"
        )

        XCTAssertTrue(dataset.acceptedTraining.isEmpty)
        XCTAssertEqual(dataset.quarantinedSamples.first?.sampleType, .rejectedArchitectureFailure)
    }

    func testCanonicalizesTraceAllowedToolComparison() {
        let trace = AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .toolAction,
            slot: "cortex",
            stage: "action",
            intent: "memory",
            promptPrefix: "What is my name?",
            rawOutputPrefix: "memory.search",
            selectedToolID: "memory.search",
            toolArguments: [:],
            allowedToolIDs: ["memory.recall"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false
        )

        let dataset = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: nil,
            traces: [trace],
            scenarioResults: [],
            sourceCommit: "test"
        )

        XCTAssertEqual(dataset.regressionTests.first?.canonicalToolID, "memory.recall")
        XCTAssertEqual(dataset.regressionTests.first?.allowedToolIDs, ["memory.recall"])
    }
}
