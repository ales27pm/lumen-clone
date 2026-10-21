import Foundation
import XCTest
@testable import Lumen

final class CoreAITextEmbeddingPolicyTests: XCTestCase {
    func testExplicitPurposesApplyTheModelSpecificPrefixExactlyOnce() throws {
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.formattedText(.init(text: "Question", purpose: .retrievalQuery)),
                       "task: search result | query: Question")
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.formattedText(.init(text: "Passage", purpose: .retrievalDocument(title: nil))),
                       "title: none | text: Passage")
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.formattedText(.init(text: "Passage", purpose: .retrievalDocument(title: "Example"))),
                       "title: Example | text: Passage")
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.formattedText(.init(text: "Sentence", purpose: .semanticSimilarity)),
                       "task: sentence similarity | query: Sentence")
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.formattedText(.init(text: "Function", purpose: .codeQuery)),
                       "task: code retrieval | query: Function")
        XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.formattedText(.init(text: "search_query: Legacy"))) { error in
            XCTAssertEqual(error as? CoreAITextEmbeddingError, .invalidRequest)
        }
        XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.formattedText(.init(text: " \n", purpose: .retrievalQuery)))
    }

    func testTokenLimitIncludesPrefixBOSAndEOSAndNeverSilentlyTruncates() throws {
        let body = [Int32](repeating: 100, count: 2_046)
        let ids = try CoreAITextEmbeddingPolicy.tokenIDs(body)
        XCTAssertEqual(ids.count, 2_048)
        XCTAssertEqual(ids.first, 2)
        XCTAssertEqual(ids.last, 1)
        XCTAssertEqual(Array(ids.dropFirst().dropLast()), body)
        XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.tokenIDs(body + [101])) { error in
            XCTAssertEqual(error as? CoreAITextEmbeddingError, .inputTooLong)
        }
        let invalidBodies: [[Int32]] = [[], [-1], [262_144]]
        for invalid in invalidBodies {
            XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.tokenIDs(invalid))
        }
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.bucket(forTokenCount: 129), 288)
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.bucket(forTokenCount: 2_048), 2_048)
        XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.bucket(forTokenCount: 2_049))
        XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.bucket(forTokenCount: 0))
    }

    func testMRLDefaultAndAllSupportedDimensionsUseUnitVectorsAndDistinctSpaces() throws {
        XCTAssertEqual(try CoreAITextEmbeddingPolicy.requestedDimensions(.init(text: "x", purpose: .retrievalQuery)), 256)
        let source = (1...768).map(Double.init)
        var identifiers = Set<String>()
        for dimension in [128, 256, 512, 768] {
            let values = try CoreAITextEmbeddingPolicy.normalizedVector(source, dimensions: dimension)
            XCTAssertEqual(values.count, dimension)
            XCTAssertEqual(values.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-12)
            XCTAssertEqual(values[0] / values[1], 0.5, accuracy: 1e-12)
            identifiers.insert(CoreAITextEmbeddingPolicy.modelIdentifier(dimensions: dimension))
        }
        XCTAssertEqual(identifiers.count, 4)
        XCTAssertEqual(CoreAITextEmbeddingPolicy.formatVersion, 4)
        for dimension in [0, 1, 255, 384, 1_024] {
            XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.requestedDimensions(.init(text: "x", dimensions: dimension)))
        }
    }

    func testMalformedVectorsDoNotBecomeEmptyOrZeroSuccess() throws {
        for vector in [[Double](), [Double](repeating: 0, count: 768), [Double](repeating: 1, count: 767)] {
            XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.normalizedVector(vector, dimensions: 256)) { error in
                XCTAssertEqual(error as? CoreAITextEmbeddingError, .invalidOutput)
            }
        }
        for bad in [Double.nan, Double.infinity, -Double.infinity] {
            var vector = [Double](repeating: 1, count: 768)
            vector[767] = bad // Validate all native values before MRL slicing.
            XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.normalizedVector(vector, dimensions: 128))
        }
    }

    func testPinnedTextConfigurationRejectsIncompatibleGraphContracts() throws {
        let data = try fixture("CoreAITextEmbeddingConfiguration")
        let config = try CoreAITextEmbeddingPolicy.validateConfiguration(data)
        XCTAssertEqual(config.buckets, [32, 64, 128, 288, 512, 1024, 2048])
        XCTAssertEqual(config.embedding_dim, 768)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for (key, bad) in [("max_tokens", 8192), ("hidden_size", 768), ("bos_token_id", 1)] {
            var changed = object
            changed[key] = bad
            XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.validateConfiguration(JSONSerialization.data(withJSONObject: changed)))
        }
        object["functions"] = ["mm_32"]
        XCTAssertThrowsError(try CoreAITextEmbeddingPolicy.validateConfiguration(JSONSerialization.data(withJSONObject: object)))
    }

    func testPinnedPublicAuthorTokenizerCasesRetainExactUTF8AndBPEBehavior() throws {
        let data = try fixture("CoreAITextEmbeddingTokenizer")
        let tokenizer = try GemmaTokenizer(data: data)
        struct Fixture: Decodable {
            struct Row: Decodable { let kind: String; let text: String; let ids: [Int32] }
            struct Provenance: Decodable { let revision: String; let tokenizerSHA256: String }
            let provenance: Provenance
            let cases: [Row]
        }
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        XCTAssertEqual(fixture.provenance.revision, CoreAITextEmbeddingPolicy.revision)
        XCTAssertEqual(fixture.provenance.tokenizerSHA256, CoreAITextEmbeddingPolicy.tokenizerDigest)
        XCTAssertEqual(fixture.cases.count, 11)
        for row in fixture.cases {
            let prefix: String
            switch row.kind {
            case "q": prefix = CoreAITextEmbeddingPolicy.queryPrefix
            case "d": prefix = CoreAITextEmbeddingPolicy.documentPrefix
            default: prefix = ""
            }
            XCTAssertEqual(tokenizer.encodeWithSpecials(prefix + row.text), row.ids)
        }
        // The compact fixture retains every merge relevant to its selected inputs;
        // it is a test vocabulary and cannot pass production's full-tokenizer hash.
        XCTAssertEqual(tokenizer.bosID, 2)
        XCTAssertEqual(tokenizer.eosID, 1)
        XCTAssertEqual(tokenizer.padID, 0)
        XCTAssertEqual(tokenizer.encode("<bos><eos>"), [2, 1])
    }

    func testExactJSONPreservesLeadingBOMAndCanonicalUnicodeDistinctions() throws {
        let data = Data("{\"bom\":\"\\ufeff\",\"é\":1,\"e\\u0301\":2}".utf8)
        let parsed = try ExactJSON.parse(data)
        XCTAssertEqual(parsed["bom"]?.stringBytes, [0xEF, 0xBB, 0xBF])
        XCTAssertEqual(parsed.pairs?.count, 3)
        XCTAssertEqual(parsed.pairs?[1].key, [0xC3, 0xA9])
        XCTAssertEqual(parsed.pairs?[2].key, [0x65, 0xCC, 0x81])
    }

    private func fixture(_ name: String) throws -> Data {
        let bundle = Bundle(for: CoreAITextEmbeddingPolicyTests.self)
        if let resource = bundle.url(forResource: name, withExtension: "json", subdirectory: "Fixtures")
            ?? bundle.url(forResource: name, withExtension: "json") {
            return try Data(contentsOf: resource)
        }
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        return try Data(contentsOf: source.appendingPathComponent("Fixtures/\(name).json"))
    }
}
