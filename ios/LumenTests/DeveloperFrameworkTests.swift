import XCTest
@testable import Lumen

final class DeveloperFrameworkTests: XCTestCase {
    func testOnlyLiveE2EOwnsScenarioPassFail() {
        for layer in DeveloperEvidenceLayer.allCases {
            if layer == .e2eTestReport {
                XCTAssertTrue(layer.ownsLiveE2EScenarios)
                XCTAssertEqual(layer.sourceLayer, "e2eTestReport")
            } else {
                XCTAssertFalse(layer.ownsLiveE2EScenarios)
            }
        }
    }

    func testEvidenceLayerBaselineIncludesAllLayers() {
        let baseline = DeveloperEvidenceLayerStatus.baseline()

        XCTAssertEqual(baseline.count, DeveloperEvidenceLayer.allCases.count)
        XCTAssertTrue(baseline.contains { $0.layer == .agentGroundingRuntimeAudit })
        XCTAssertTrue(baseline.contains { $0.layer == .agentBehaviorTraceRecorder })
        XCTAssertTrue(baseline.contains { $0.layer == .e2eTestReport && $0.status == "live owner" })
    }

    func testWorkflowActionsExposeExpectedExports() {
        XCTAssertEqual(DeveloperWorkflowAction.exportRuntimeAudit.title, "Export runtime audit package")
        XCTAssertEqual(DeveloperWorkflowAction.exportLiveE2E.systemImage, "arrow.up.doc")
        XCTAssertEqual(DeveloperWorkflowAction.exportRecentTraces.title, "Export recent runtime traces")
    }

    func testLiveE2EExportEnvelopeMatchesImproveLoopIngestionContract() throws {
        let report = E2ETestReport(
            id: UUID(),
            startedAt: Date(),
            finishedAt: Date(),
            passed: 0,
            failed: 0,
            results: []
        )
        let result = try EvidenceLayerExporter.writeLiveE2EReport(report)

        let data = try Data(contentsOf: result.url)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let policy = try XCTUnwrap(object["exportPolicy"] as? [String: Any])
        let payload = try XCTUnwrap(object["payload"] as? [String: Any])

        XCTAssertEqual(policy["format"] as? String, "live-e2e-test-report-json")
        XCTAssertEqual(policy["sourceLayer"] as? String, "e2eTestReport")
        XCTAssertEqual(policy["ownsLiveE2EScenarios"] as? Bool, true)
        XCTAssertEqual(policy["includesDeterministicStaticScenarios"] as? Bool, false)
        XCTAssertEqual(payload["failed"] as? Int, 0)
        XCTAssertTrue(result.url.lastPathComponent.hasPrefix("lumen-live-e2e-report-redacted-v1-"))
        XCTAssertEqual(result.url.pathComponents.suffix(2).first, "LumenEvidenceLayerExports")
    }

    func testLiveE2EExportRedactsAllFreeFormRuntimeContent() throws {
        let secretPrompt = "Find the private contact named Secret Person"
        let secretFinal = "Secret Person has a private appointment tomorrow."
        let secretEvent = "Tool returned private@example.test and message identifier 12345"
        let secretPerformanceNote = "Loaded private model path /private/model.gguf"
        let sensitiveMetadataKey = "987-65-4321"
        let callerCorrelationToken = "caller-token-\(sensitiveMetadataKey)"
        let reportID = UUID(uuidString: "C1111111-1111-4111-8111-111111111111")!
        let resultID = UUID(uuidString: "C2222222-2222-4222-8222-222222222222")!
        let eventID = UUID(uuidString: "C3333333-3333-4333-8333-333333333333")!
        let e2eRunID = UUID(uuidString: "C4444444-4444-4444-8444-444444444444")!
        let agentRunID = UUID(uuidString: "C5555555-5555-4555-8555-555555555555")!
        let result = E2ETestResult(
            id: resultID,
            scenarioID: "privacy-export",
            kind: "live",
            title: "Private scenario title",
            prompt: secretPrompt,
            expectedIntent: "contacts.search",
            actualIntent: "contacts.search",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            correlationToken: callerCorrelationToken,
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.modelBackedRequired.rawValue,
            passed: false,
            failures: ["Private failure mentions Secret Person"],
            finalText: secretFinal,
            missingHints: ["Secret Person"],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [
                E2ETestEvent(
                    id: eventID,
                    createdAt: Date(),
                    scenarioID: "privacy-export",
                    phase: "toolResult",
                    message: secretEvent
                )
            ],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: secretFinal,
            sanitizedFinalPrefix: secretFinal,
            rawFinalHadUnsafeLeakage: true,
            sanitizedFinalRemovedArtifacts: ["toolPayload"],
            outputHygieneFailures: ["Leaked Secret Person"],
            performanceMatrix: E2EPerformanceMatrix(
                aneUtilizationPercent: nil,
                eventDensityCPUProxyPercent: 1,
                gpuUtilizationPercent: nil,
                peakRAMMB: 2,
                averageRAMMB: 1,
                sampleCount: 1,
                notes: [secretPerformanceNote],
                accelerationDiagnostics: nil
            ),
            metadata: [
                "failureKind": "toolObservationLeak",
                sensitiveMetadataKey: secretEvent
            ]
        )
        let report = E2ETestReport(
            id: reportID,
            startedAt: Date(),
            finishedAt: Date(),
            passed: 0,
            failed: 1,
            results: [result]
        )

        let export = try EvidenceLayerExporter.writeLiveE2EReport(report)
        let secondExport = try EvidenceLayerExporter.writeLiveE2EReport(report)
        let data = try Data(contentsOf: export.url)
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))

        for secret in [
            secretPrompt, secretFinal, secretEvent, secretPerformanceNote, "Secret Person",
            "private@example.test", sensitiveMetadataKey, callerCorrelationToken,
            reportID.uuidString, resultID.uuidString, eventID.uuidString,
            e2eRunID.uuidString, agentRunID.uuidString,
        ] {
            XCTAssertFalse(text.contains(secret))
        }
        XCTAssertTrue(text.contains("privacyRedacted"))
        XCTAssertTrue(text.contains("[redacted sha256="))
        XCTAssertTrue(text.contains("toolObservationLeak"))
        let exportedResult = try XCTUnwrap(export.envelope.payload.results.first)
        let secondExportedResult = try XCTUnwrap(secondExport.envelope.payload.results.first)
        XCTAssertTrue(exportedResult.metadata.keys.allSatisfy {
            $0 == "failureKind" || $0 == "privacyRedacted" || $0.hasPrefix("metadata_")
        })
        XCTAssertTrue(exportedResult.correlationToken?.hasPrefix("corr_hash_v2_") == true)
        XCTAssertNotEqual(export.envelope.payload.id, secondExport.envelope.payload.id)
        XCTAssertNotEqual(exportedResult.id, secondExportedResult.id)
        XCTAssertNotEqual(exportedResult.events.first?.id, secondExportedResult.events.first?.id)
        XCTAssertNotEqual(exportedResult.correlationToken, secondExportedResult.correlationToken)
    }

    func testE2ELogStorePersistsOnlyPrivacySafeRepresentationsAndPurgesLegacyFiles() throws {
        let directory = try E2ETestLogStore.reportsDirectory()
        let legacyURL = directory.appendingPathComponent("e2e-results.jsonl")
        let rawCanary = "Private Calendar Canary 98f39b"
        let sensitiveMetadataKey = "987-65-4321"
        let callerCorrelationToken = "caller-token-\(sensitiveMetadataKey)"
        let reportID = UUID(uuidString: "D1111111-1111-4111-8111-111111111111")!
        let resultID = UUID(uuidString: "D2222222-2222-4222-8222-222222222222")!
        let eventID = UUID(uuidString: "D3333333-3333-4333-8333-333333333333")!
        let e2eRunID = UUID(uuidString: "D4444444-4444-4444-8444-444444444444")!
        try rawCanary.write(to: legacyURL, atomically: true, encoding: .utf8)

        _ = try E2ETestLogStore.reportsDirectory()
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyURL.path))

        let result = E2ETestResult(
            id: resultID,
            scenarioID: "privacy-persistence",
            title: "Private persistence title",
            prompt: rawCanary,
            expectedIntent: "calendar.read",
            actualIntent: "calendar.read",
            e2eRunID: e2eRunID,
            correlationToken: callerCorrelationToken,
            requiresAgentRun: true,
            passed: false,
            failures: [rawCanary],
            finalText: rawCanary,
            missingHints: [rawCanary],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [
                E2ETestEvent(
                    id: eventID,
                    createdAt: Date(),
                    scenarioID: "privacy-persistence",
                    phase: "toolResult",
                    message: rawCanary
                )
            ],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: rawCanary,
            sanitizedFinalPrefix: rawCanary,
            rawFinalHadUnsafeLeakage: true,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [rawCanary],
            metadata: [sensitiveMetadataKey: rawCanary]
        )
        E2ETestLogStore.append(result)
        E2ETestLogStore.writeLatest(E2ETestReport(
            id: reportID,
            startedAt: Date(),
            finishedAt: Date(),
            passed: 0,
            failed: 1,
            results: [result]
        ))

        let persistedURLs = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ).filter { $0.lastPathComponent.contains("redacted-v1") }
        XCTAssertFalse(persistedURLs.isEmpty)
        for url in persistedURLs {
            let text = try String(contentsOf: url, encoding: .utf8)
            for canary in [
                rawCanary, sensitiveMetadataKey, callerCorrelationToken,
                reportID.uuidString, resultID.uuidString, eventID.uuidString,
                e2eRunID.uuidString,
            ] {
                XCTAssertFalse(text.contains(canary))
            }
        }
    }
}
