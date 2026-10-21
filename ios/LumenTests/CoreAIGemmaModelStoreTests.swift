import CryptoKit
import Foundation
import Testing
@testable import Lumen

struct CoreAIGemmaModelStoreTests {
    @Test func importedContextMetadataDoesNotLoadOrValidateModelPayload() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.metadata(context: 32_768)
        try fixture.storage.install(source: fixture.source)
        let installed = try fixture.storage.configuredURL()
        try FileManager.default.removeItem(at: installed.appendingPathComponent("gemma.aimodel/main.mlirb"))
        #expect(try fixture.storage.configuredContextLength() == 32_768)
        #expect(throws: CoreAIGemmaModelStore.Failure.integrity) { try fixture.storage.configuredURL() }
    }

    @Test func importsOwnedCopyAndChecksEveryPayloadOnLoad() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let installed = try fixture.storage.configuredURL()
        #expect(installed.path.hasPrefix(fixture.owned.path + "/"))
        #expect(installed != fixture.source)
        try FileManager.default.removeItem(at: fixture.source)
        #expect(try fixture.storage.configuredURL() == installed)
        try Data("changed tokenizer".utf8).write(to: installed.appendingPathComponent("tokenizer/tokenizer.json"))
        #expect(throws: CoreAIGemmaModelStore.Failure.integrity) { try fixture.storage.configuredURL() }
    }

    @Test func detectsChangedPerLayerEmbeddings() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let installed = try fixture.storage.configuredURL()
        try Data("mutated PLE".utf8).write(to: installed.appendingPathComponent("gemma_ple.safetensors"))
        #expect(throws: CoreAIGemmaModelStore.Failure.integrity) { try fixture.storage.configuredURL() }
    }

    @Test func failedReplacementPreservesPreviousModel() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let original = try fixture.storage.configuredURL()
        try Data(repeating: 0, count: 32).write(to: fixture.source.appendingPathComponent("gemma.aimodel/main.hash"))
        #expect(throws: CoreAIGemmaModelStore.Failure.integrity) { try fixture.storage.install(source: fixture.source) }
        #expect(try fixture.storage.configuredURL() == original)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: fixture.owned.path)
        #expect(remaining.allSatisfy { !$0.hasPrefix(".import-") })
    }

    @Test func rejectsTraversalFromModelMetadata() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.metadata(main: "../outside.aimodel")
        #expect(throws: CoreAIGemmaModelStore.Failure.invalidPackage) { try fixture.storage.install(source: fixture.source) }
        #expect(!fixture.storage.isConfigured)
    }

    @Test func rejectsSymbolicLinksEvenWhenTargetIsInsideSourceFolder() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let linked = fixture.source.appendingPathComponent("tokenizer/linked.json")
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: fixture.source.appendingPathComponent("tokenizer/tokenizer.json"))
        #expect(throws: CoreAIGemmaModelStore.Failure.localDirectory) { try fixture.storage.install(source: fixture.source) }
        #expect(!fixture.storage.isConfigured)
    }

    @Test func rejectsMissingTokenizerAndUnsupportedContext() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.metadata(context: 131_072)
        #expect(throws: CoreAIGemmaModelStore.Failure.invalidPackage) { try fixture.storage.install(source: fixture.source) }
        try fixture.metadata()
        try FileManager.default.removeItem(at: fixture.source.appendingPathComponent("tokenizer/tokenizer_config.json"))
        #expect(throws: CoreAIGemmaModelStore.Failure.invalidPackage) { try fixture.storage.install(source: fixture.source) }
    }

    @Test func detectsUnexpectedFilesAndRemovesInstalledCopy() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let installed = try fixture.storage.configuredURL()
        try Data("unexpected".utf8).write(to: installed.appendingPathComponent("unexpected.bin"))
        #expect(throws: CoreAIGemmaModelStore.Failure.integrity) { try fixture.storage.configuredURL() }
        try fixture.storage.remove()
        #expect(!fixture.storage.isConfigured)
        #expect(!FileManager.default.fileExists(atPath: installed.path))
        #expect(throws: CoreAIGemmaModelStore.Failure.notConfigured) { try fixture.storage.configuredURL() }
    }

    @Test func rejectsEscapingInstallationPointer() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try fixture.storage.install(source: fixture.source)
        let pointer = fixture.owned.appendingPathComponent("active.json")
        let pointerBytes = try Data(contentsOf: pointer)
        let pointerObject = try JSONSerialization.jsonObject(with: pointerBytes)
        var document = try #require(pointerObject as? [String: Any])
        document["installation"] = "../outside"
        try JSONSerialization.data(withJSONObject: document).write(to: pointer)
        #expect(throws: CoreAIGemmaModelStore.Failure.integrity) { try fixture.storage.configuredURL() }
        try fixture.storage.remove()
        #expect(!fixture.storage.isConfigured)
        #expect(!FileManager.default.fileExists(atPath: fixture.owned.path))
    }

    private struct Fixture {
        let root: URL
        let source: URL
        let owned: URL
        var storage: CoreAIGemmaModelStore.Storage { .init(directory: owned) }

        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
            source = root.appendingPathComponent("source", isDirectory: true)
            owned = root.appendingPathComponent("owned", isDirectory: true)
            for directory in ["gemma.aimodel", "tokenizer"] {
                try FileManager.default.createDirectory(at: source.appendingPathComponent(directory), withIntermediateDirectories: true)
            }
            let bytecode = Data("synthetic Core AI bytecode; not a real model".utf8)
            try bytecode.write(to: source.appendingPathComponent("gemma.aimodel/main.mlirb"))
            try Data(SHA256.hash(data: bytecode)).write(to: source.appendingPathComponent("gemma.aimodel/main.hash"))
            for path in ["gemma.aimodel/metadata.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja", "gemma_ple.safetensors"] {
                try Data("synthetic fixture".utf8).write(to: source.appendingPathComponent(path))
            }
            try metadata()
        }

        func metadata(main: String = "gemma.aimodel", context: Int = 8_192) throws {
            let document: [String: Any] = [
                "kind": "llm", "assets": ["main": main],
                "auxiliary_assets": ["per_layer_embeddings": "gemma_ple.safetensors"],
                "language": ["tokenizer": "google/gemma-4-E2B-it", "max_context_length": context, "embedded_tokenizer": true] as [String: Any]
            ]
            let data = try JSONSerialization.data(withJSONObject: document)
            try data.write(to: source.appendingPathComponent("metadata.json"))
        }

        func remove() { try? FileManager.default.removeItem(at: root) }
    }
}
