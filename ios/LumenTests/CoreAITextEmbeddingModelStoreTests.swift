import CryptoKit
import Foundation
import Testing
@testable import Lumen

struct CoreAITextEmbeddingModelStoreTests {
    @Test func pinsOnlyTextWeightsTokenizerConfigurationAndNotices() throws {
        let files = CoreAITextEmbeddingModelStore.pinnedFiles
        #expect(files.count == 9)
        #expect(files.map(\.path) == files.map(\.path).sorted())
        #expect(files.allSatisfy { !$0.path.contains("vision") && !$0.path.contains("audio") })
        #expect(files.first(where: { $0.path.hasSuffix("main.mlirb") })?.sha256 == CoreAITextEmbeddingPolicy.graphDigest)
        #expect(files.first(where: { $0.path == "model/tokenizer/tokenizer.json" })?.sha256 == CoreAITextEmbeddingPolicy.tokenizerDigest)
        #expect(files.first(where: { $0.path == "model/config.json" })?.sha256 == CoreAITextEmbeddingPolicy.configurationDigest)
        #expect(Set(["LICENSE", "NOTICE", "model/MODEL-ATTRIBUTION.json"]).isSubset(of: Set(files.map(\.path))))
    }

    @Test func importsOwnedCopyAndChecksEveryFileOnLoad() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let installed = try fixture.storage.configuredPackage()
        #expect(installed.rootURL.path.hasPrefix(fixture.owned.path + "/"))
        #expect(installed.identifier == CoreAITextEmbeddingPolicy.baseIdentifier)
        #expect(installed.configuration.embedding_dim == 768)
        try FileManager.default.removeItem(at: fixture.source)
        #expect(try fixture.storage.configuredPackage().rootURL == installed.rootURL)
        try Data("changed tokenizer".utf8).write(to: installed.tokenizerURL)
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.integrity) { try fixture.storage.configuredPackage() }
    }

    @Test func metadataOnlySummaryDoesNotVerifyPayloadOrLoadEngine() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let installed = try fixture.storage.configuredPackage()
        try FileManager.default.removeItem(at: installed.modelURL.appendingPathComponent("main.mlirb"))
        let summary = try fixture.storage.configurationSummary()
        #expect(summary.maximumTokens == 2_048)
        #expect(summary.nativeDimensions == 768)
        #expect(summary.artifactDigest == installed.artifactDigest)
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.integrity) { try fixture.storage.configuredPackage() }
    }

    @Test func failedReplacementPreservesPreviousInstallation() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let previous = try fixture.storage.configuredPackage().rootURL
        try Data("mismatched graph".utf8).write(to: fixture.source.appendingPathComponent(Self.graphPath))
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.integrity) { try fixture.storage.install(source: fixture.source) }
        #expect(try fixture.storage.configuredPackage().rootURL == previous)
        #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.owned.path).allSatisfy { !$0.hasPrefix(".import-") })
    }

    @Test func rejectsMismatchedMainHashEvenWithMatchingFileInventory() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try Data(repeating: 0, count: 32).write(to: fixture.source.appendingPathComponent(Self.hashPath))
        let changed = try fixture.identities()
        let storage = CoreAITextEmbeddingModelStore.Storage(directory: fixture.owned, expectedFiles: changed)
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.integrity) { try storage.install(source: fixture.source) }
        #expect(!storage.isConfigured)
    }

    @Test func ignoresDownloadReceiptsButCopiesOnlyPinnedPayload() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let receipt = fixture.source.appendingPathComponent(".cache/huggingface/download/model/config.json.metadata")
        try FileManager.default.createDirectory(at: receipt.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("synthetic local download receipt".utf8).write(to: receipt)
        try Data("finder metadata".utf8).write(to: fixture.source.appendingPathComponent(".DS_Store"))
        try fixture.storage.install(source: fixture.source)
        let root = try fixture.storage.configuredPackage().rootURL
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(".cache").path))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(".DS_Store").path))
    }

    @Test func rejectsVisionAudioOrUnexpectedPayload() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let extra = fixture.source.appendingPathComponent("model/embeddinggemma-2-vision.aimodel/main.mlirb")
        try FileManager.default.createDirectory(at: extra.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("vision weights are excluded".utf8).write(to: extra)
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.invalidPackage) { try fixture.storage.install(source: fixture.source) }
        #expect(!fixture.storage.isConfigured)
    }

    @Test func rejectsSymlinkEvenWhenTargetIsInsidePackage() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let tokenizer = fixture.source.appendingPathComponent("model/tokenizer/tokenizer.json")
        try FileManager.default.removeItem(at: tokenizer)
        try FileManager.default.createSymbolicLink(at: tokenizer, withDestinationURL: fixture.source.appendingPathComponent("NOTICE"))
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.localDirectory) { try fixture.storage.install(source: fixture.source) }
        #expect(!fixture.storage.isConfigured)
    }

    @Test func rejectsMissingLicenseAndUnsupportedConfiguration() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try FileManager.default.removeItem(at: fixture.source.appendingPathComponent("LICENSE"))
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.invalidPackage) { try fixture.storage.install(source: fixture.source) }
        try Data("synthetic license".utf8).write(to: fixture.source.appendingPathComponent("LICENSE"))
        try fixture.configuration(asset: "../outside.aimodel")
        let changed = try fixture.identities()
        let storage = CoreAITextEmbeddingModelStore.Storage(directory: fixture.owned, expectedFiles: changed)
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.invalidPackage) { try storage.install(source: fixture.source) }
    }

    @Test func rejectsTraversalAndChangedIdentity() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let pointer = fixture.owned.appendingPathComponent("active.json")
        var document = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: pointer)) as? [String: Any])
        document["installation"] = "../outside"
        try JSONSerialization.data(withJSONObject: document).write(to: pointer)
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.integrity) { try fixture.storage.configuredPackage() }
        try fixture.storage.remove()
        #expect(!fixture.storage.isConfigured)
    }

    @Test func rejectsChangedNoticesAndRemovesOwnedCopy() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let root = try fixture.storage.configuredPackage().rootURL
        try Data("changed notice".utf8).write(to: root.appendingPathComponent("NOTICE"))
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.integrity) { try fixture.storage.configuredPackage() }
        try fixture.storage.remove()
        #expect(!fixture.storage.isConfigured)
        #expect(!FileManager.default.fileExists(atPath: root.path))
        #expect(throws: CoreAITextEmbeddingModelStore.Failure.notConfigured) { try fixture.storage.configuredPackage() }
    }

    private static let graphPath = "model/embeddinggemma-2-text-mm.aimodel/main.mlirb"
    private static let hashPath = "model/embeddinggemma-2-text-mm.aimodel/main.hash"

    private struct Fixture {
        let root: URL
        let source: URL
        let owned: URL
        let expectedFiles: [CoreAITextEmbeddingModelStore.FileIdentity]
        var storage: CoreAITextEmbeddingModelStore.Storage { .init(directory: owned, expectedFiles: expectedFiles) }

        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
            source = root.appendingPathComponent("source", isDirectory: true)
            owned = root.appendingPathComponent("owned", isDirectory: true)
            for file in CoreAITextEmbeddingModelStore.pinnedFiles {
                let url = source.appendingPathComponent(file.path)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data("synthetic \(file.path); not a real model".utf8).write(to: url)
            }
            let graph = try Data(contentsOf: source.appendingPathComponent(Self.graphPath))
            try Data(SHA256.hash(data: graph)).write(to: source.appendingPathComponent(Self.hashPath))
            try Self.writeConfiguration(source: source)
            expectedFiles = try Self.identities(source: source)
        }

        func identities() throws -> [CoreAITextEmbeddingModelStore.FileIdentity] { try Self.identities(source: source) }
        private static func identities(source: URL) throws -> [CoreAITextEmbeddingModelStore.FileIdentity] {
            try CoreAITextEmbeddingModelStore.pinnedFiles.map { item in
                let data = try Data(contentsOf: source.appendingPathComponent(item.path))
                return .init(path: item.path, bytes: Int64(data.count), sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
            }
        }

        func configuration(asset: String) throws { try Self.writeConfiguration(source: source, asset: asset) }
        private static func writeConfiguration(source: URL, asset: String = "embeddinggemma-2-text-mm.aimodel") throws {
            let value: [String: Any] = [
                "base_model": "google/embeddinggemma-2", "format": "coreai", "asset": asset,
                "functions": CoreAITextEmbeddingPolicy.functions, "buckets": CoreAITextEmbeddingPolicy.buckets,
                "max_tokens": 2_048, "precision_mode": "ane", "hidden_size": 512,
                "embedding_dim": 768, "matryoshka_dims": [768, 512, 256, 128],
                "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
                "inputs": ["input_ids": "fixture", "attention_mask": "fixture", "mm_embeds": "fixture", "mm_mask": "fixture"],
                "output": ["embedding": "fixture"],
                "prompts": ["document": CoreAITextEmbeddingPolicy.documentPrefix,
                            "search_query": CoreAITextEmbeddingPolicy.queryPrefix,
                            "sentence_similarity": "task: sentence similarity | query: ",
                            "code_retrieval": "task: code retrieval | query: "]
            ]
            try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: source.appendingPathComponent("model/config.json"))
        }
        private static let graphPath = CoreAITextEmbeddingModelStoreTests.graphPath
        private static let hashPath = CoreAITextEmbeddingModelStoreTests.hashPath
        func remove() { try? FileManager.default.removeItem(at: root) }
    }
}
