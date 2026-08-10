import Foundation
import XCTest
@testable import Lumen

final class AgentParseDiagnosticsPersistenceTests: XCTestCase {
    func testVersionedPersistencePurgesLegacyFilesAndKeepsCanariesOffDisk() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let legacyFailureURL = directory.appendingPathComponent("agent-parse-failures.jsonl")
        let legacyNoiseURL = directory.appendingPathComponent("agent-parse-noise.jsonl")
        try Data("Legacy Failure Person Canary".utf8).write(to: legacyFailureURL)
        try Data("Legacy Noise Person Canary".utf8).write(to: legacyNoiseURL)

        let failure = makeFailureTrace()
        let noise = makeNoiseTrace()
        try AgentParseFailureRecorder.persist(failure, in: directory)
        try AgentParseNoiseRecorder.persist(noise, in: directory)
        // Exercise both the protected create and protected append paths.
        try AgentParseFailureRecorder.persist(failure, in: directory)
        try AgentParseNoiseRecorder.persist(noise, in: directory)

        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyFailureURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyNoiseURL.path))

        let failureURL = directory.appendingPathComponent(AgentParseDiagnosticsFile.failure)
        let noiseURL = directory.appendingPathComponent(AgentParseDiagnosticsFile.noise)
        let failureData = try Data(contentsOf: failureURL)
        let noiseData = try Data(contentsOf: noiseURL)
        let persistedText = String(decoding: failureData + noiseData, as: UTF8.self)
        for canary in privacyCanaries(for: failure, noise: noise) {
            XCTAssertFalse(
                persistedText.localizedCaseInsensitiveContains(canary),
                "Persistent parse diagnostics leaked canary: \(canary)"
            )
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let persistedFailure = try decoder.decode(
            AgentParseFailureTrace.self,
            from: Data(try XCTUnwrap(failureData.split(separator: 0x0A).first))
        )
        let persistedNoise = try decoder.decode(
            AgentParseNoiseTrace.self,
            from: Data(try XCTUnwrap(noiseData.split(separator: 0x0A).first))
        )
        XCTAssertEqual(failureData.split(separator: 0x0A).count, 2)
        XCTAssertEqual(noiseData.split(separator: 0x0A).count, 2)
        XCTAssertNotEqual(persistedFailure.id, failure.id)
        XCTAssertNotEqual(persistedNoise.id, noise.id)
        XCTAssertTrue(persistedFailure.isPrivacySafePersistentDiagnostic)
        XCTAssertTrue(persistedNoise.isPrivacySafePersistentDiagnostic)

        for url in [failureURL, noiseURL] {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            XCTAssertEqual(
                attributes[.protectionKey] as? FileProtectionType,
                FileProtectionType.complete,
                "Expected complete protection for \(url.lastPathComponent)"
            )
        }
    }

    func testUnexpectedLegacyArtifactTypeFailsClosedBeforeVersionedWrite() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("agent-parse-failures.jsonl", isDirectory: true),
            withIntermediateDirectories: false
        )

        XCTAssertThrowsError(try AgentParseFailureRecorder.persist(makeFailureTrace(), in: directory)) { error in
            XCTAssertEqual(
                error as? AgentParseDiagnosticsStorageError,
                .unexpectedArtifactType("agent-parse-failures.jsonl")
            )
        }
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(AgentParseDiagnosticsFile.failure).path
            )
        )
    }

    func testPersistentSummaryLoadersRejectRawRecordsInVersionedFiles() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        var failureLine = try encoder.encode(makeFailureTrace())
        failureLine.append(0x0A)
        var noiseLine = try encoder.encode(makeNoiseTrace())
        noiseLine.append(0x0A)
        try failureLine.write(
            to: directory.appendingPathComponent(AgentParseDiagnosticsFile.failure),
            options: [.atomic]
        )
        try noiseLine.write(
            to: directory.appendingPathComponent(AgentParseDiagnosticsFile.noise),
            options: [.atomic]
        )

        let failureSummary = try AgentParseFailureSummaryLoader.load(
            fromPersistentDirectory: directory
        )
        let noiseSummary = try AgentParseNoiseSummaryLoader.load(
            fromPersistentDirectory: directory
        )
        XCTAssertEqual(failureSummary.totalLines, 1)
        XCTAssertEqual(failureSummary.decodedLines, 0)
        XCTAssertEqual(failureSummary.skippedLines, 1)
        XCTAssertEqual(noiseSummary.totalLines, 1)
        XCTAssertEqual(noiseSummary.decodedLines, 0)
        XCTAssertEqual(noiseSummary.skippedLines, 1)
    }

    func testPersistentSummaryLoaderPurgesLegacyFileInsteadOfMigratingIt() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let legacyURL = directory.appendingPathComponent("agent-parse-failures.jsonl")
        try Data("Legacy Summary Person Canary".utf8).write(to: legacyURL)

        let summary = try AgentParseFailureSummaryLoader.load(fromPersistentDirectory: directory)

        XCTAssertEqual(summary.totalLines, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyURL.path))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(AgentParseDiagnosticsFile.failure).path
            )
        )
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("AgentParseDiagnosticsPersistenceTests-\(UUID().uuidString)", isDirectory: true)
    }

    private func makeFailureTrace() -> AgentParseFailureTrace {
        AgentParseFailureTrace(
            id: UUID(uuidString: "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE")!,
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            parseError: "Parse Error Person Canary",
            modelName: "Private Model Person Canary",
            temperature: 0.2,
            topP: 0.9,
            maxTokens: 128,
            stepIndex: 1,
            systemPromptPrefix: "System Prompt Person Canary",
            userTurnPrefix: "User Turn Person Canary",
            rawOutputPrefix: "Raw Output Person Canary",
            streamedThoughtPrefix: "Thought Person Canary",
            streamedFinalPrefix: "Final Person Canary",
            selectedJSONPrefix: "Selected JSON Person Canary",
            prefixNoise: "Prefix Noise Person Canary",
            suffixNoise: "Suffix Noise Person Canary"
        )
    }

    private func makeNoiseTrace() -> AgentParseNoiseTrace {
        AgentParseNoiseTrace(
            id: UUID(uuidString: "11111111-2222-4333-8444-555555555555")!,
            createdAt: Date(timeIntervalSince1970: 1_800_000_001),
            modelName: "Noise Model Person Canary",
            temperature: 0.3,
            topP: 0.8,
            maxTokens: 256,
            stepIndex: 2,
            systemPromptPrefix: "Noise System Person Canary",
            userTurnPrefix: "Noise User Person Canary",
            rawOutputPrefix: "Noise Raw Person Canary",
            selectedJSONPrefix: "Noise JSON Person Canary",
            prefixNoise: "Noise Prefix Person Canary",
            suffixNoise: "Noise Suffix Person Canary"
        )
    }

    private func privacyCanaries(
        for failure: AgentParseFailureTrace,
        noise: AgentParseNoiseTrace
    ) -> [String] {
        [
            failure.id.uuidString,
            failure.parseError,
            failure.modelName,
            failure.systemPromptPrefix,
            failure.userTurnPrefix,
            failure.rawOutputPrefix,
            failure.streamedThoughtPrefix,
            failure.streamedFinalPrefix,
            failure.selectedJSONPrefix,
            failure.prefixNoise,
            failure.suffixNoise,
            noise.id.uuidString,
            noise.modelName,
            noise.systemPromptPrefix,
            noise.userTurnPrefix,
            noise.rawOutputPrefix,
            noise.selectedJSONPrefix,
            noise.prefixNoise,
            noise.suffixNoise,
        ].compactMap { $0 }
    }
}
