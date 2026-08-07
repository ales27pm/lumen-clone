import Foundation
import XCTest
@testable import Lumen

@MainActor
final class ModelDownloaderResumeTests: XCTestCase {
    func testIdentityBindsImmutableArtifactFieldsInsteadOfMutableCatalogID() {
        let first = makeModel(id: "catalog-a")
        let renamed = makeModel(id: "catalog-b")
        let differentRepo = makeModel(id: "catalog-a", repoID: "owner/other")
        let differentPath = makeModel(id: "catalog-a", sourcePath: "releases/other.gguf")
        let differentRevision = makeModel(id: "catalog-a", revision: String(repeating: "b", count: 40))
        let differentDigest = makeModel(id: "catalog-a", digest: String(repeating: "c", count: 64))

        let identity = ModelDownloadIdentity(model: first)
        XCTAssertEqual(identity, ModelDownloadIdentity(model: renamed))
        XCTAssertNotEqual(identity, ModelDownloadIdentity(model: differentRepo))
        XCTAssertNotEqual(identity, ModelDownloadIdentity(model: differentPath))
        XCTAssertNotEqual(identity, ModelDownloadIdentity(model: differentRevision))
        XCTAssertNotEqual(identity, ModelDownloadIdentity(model: differentDigest))
        XCTAssertEqual(identity.persistenceKey.count, 64)
        XCTAssertTrue(identity.persistenceKey.allSatisfy(\.isHexDigit))
    }

    func testResumeEnvelopeRejectsAnotherArtifactAndLegacyUnboundBytes() throws {
        let identity = ModelDownloadIdentity(model: makeModel(id: "one"))
        let otherIdentity = ModelDownloadIdentity(
            model: makeModel(id: "two", revision: String(repeating: "d", count: 40))
        )
        let resumeData = Data([0x01, 0x02, 0x03])
        let encoded = try ModelDownloadResumeEnvelope.encoded(identity: identity, resumeData: resumeData)

        XCTAssertEqual(
            ModelDownloadResumeEnvelope.resumeData(from: encoded, matching: identity),
            resumeData
        )
        XCTAssertNil(ModelDownloadResumeEnvelope.resumeData(from: encoded, matching: otherIdentity))
        XCTAssertNil(
            ModelDownloadResumeEnvelope.resumeData(
                from: Data("legacy raw resume bytes".utf8),
                matching: identity
            )
        )
    }

    func testTransportFailurePolicyPreservesRetriableFailuresAndExplicitIntentWins() {
        let timeout = NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut)
        let offline = NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet)
        let cancelled = NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)

        XCTAssertEqual(
            ModelDownloadFailurePolicy.disposition(
                for: timeout,
                startedFromResumeData: false,
                explicitIntent: nil
            ),
            .retriableTransport
        )
        XCTAssertEqual(
            ModelDownloadFailurePolicy.disposition(
                for: offline,
                startedFromResumeData: true,
                explicitIntent: nil
            ),
            .retriableTransport
        )
        XCTAssertEqual(
            ModelDownloadFailurePolicy.disposition(
                for: cancelled,
                startedFromResumeData: true,
                explicitIntent: .pause
            ),
            .explicitPause
        )
        XCTAssertEqual(
            ModelDownloadFailurePolicy.disposition(
                for: timeout,
                startedFromResumeData: true,
                explicitIntent: .cancel
            ),
            .explicitCancel
        )
        XCTAssertEqual(
            ModelDownloadFailurePolicy.disposition(
                for: cancelled,
                startedFromResumeData: false,
                explicitIntent: nil
            ),
            .terminal
        )
    }

    func testRejectedResumeDataRetriesFreshOnlyForAResumedTask() {
        let cannotResume = NSError(
            domain: NSURLErrorDomain,
            code: ModelDownloadFailurePolicy.cannotResumeErrorCode
        )

        XCTAssertEqual(
            ModelDownloadFailurePolicy.disposition(
                for: cannotResume,
                startedFromResumeData: true,
                explicitIntent: nil
            ),
            .retryWithoutResumeData
        )
        XCTAssertEqual(
            ModelDownloadFailurePolicy.disposition(
                for: cannotResume,
                startedFromResumeData: false,
                explicitIntent: nil
            ),
            .terminal
        )
    }

    func testSingleFlightCompletionFanoutDrainsEverySubscriberExactlyOnce() {
        let identity = ModelDownloadIdentity(model: makeModel(id: "shared"))
        let fanout = ModelDownloadCompletionFanout()
        var completed: [String] = []
        fanout.append({ _ in completed.append("first") }, for: identity)
        fanout.append({ _ in completed.append("second") }, for: identity)

        XCTAssertEqual(fanout.count(for: identity), 2)
        let handlers = fanout.take(for: identity)
        XCTAssertEqual(handlers.count, 2)
        handlers.forEach { $0(URL(fileURLWithPath: "/verified/model.gguf")) }

        XCTAssertEqual(completed, ["first", "second"])
        XCTAssertEqual(fanout.count(for: identity), 0)
        XCTAssertTrue(fanout.take(for: identity).isEmpty)
    }

    func testAtomicInstallReplacesDestinationOnlyAfterRenameSucceeds() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-downloader-atomic-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let destination = root.appendingPathComponent("model.gguf")
        let staging = root.appendingPathComponent(".staging-valid")
        try Data("verified-old".utf8).write(to: destination)
        try Data("verified-new".utf8).write(to: staging)

        try ModelDownloader.atomicallyInstallValidatedStagingFile(staging, at: destination)
        XCTAssertEqual(try Data(contentsOf: destination), Data("verified-new".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path))

        let missingStaging = root.appendingPathComponent(".staging-missing")
        XCTAssertThrowsError(
            try ModelDownloader.atomicallyInstallValidatedStagingFile(missingStaging, at: destination)
        )
        XCTAssertEqual(try Data(contentsOf: destination), Data("verified-new".utf8))
    }

    private func makeModel(
        id: String,
        repoID: String = "owner/repository",
        sourcePath: String = "releases/model.gguf",
        revision: String = String(repeating: "a", count: 40),
        digest: String = String(repeating: "b", count: 64)
    ) -> CatalogModel {
        CatalogModel(
            id: id,
            name: "Model \(id)",
            repoId: repoID,
            fileName: "model.gguf",
            parameters: "1B",
            quantization: "Q4_K_M",
            sizeBytes: 64,
            role: .chat,
            description: "Synthetic downloader fixture.",
            tags: [],
            sourceRevision: revision,
            expectedSHA256: digest,
            sourcePath: sourcePath
        )
    }
}
