import Foundation
import XCTest
@testable import Lumen

@MainActor
final class MemoryEmbeddingIdentityTests: XCTestCase {
    func testExistingInitializerDoesNotInferEmbeddingIdentityFromVectorLength() {
        let memory = MemoryItem(content: "Synthetic unindexed memory", embedding: [0.5, -0.25])

        XCTAssertEqual(memory.embedding, [0.5, -0.25])
        XCTAssertNil(memory.embeddingFormatVersion)
        XCTAssertNil(memory.embeddingModelIdentifier)
        XCTAssertNil(memory.embeddingDimension)
    }

    func testInitializerAcceptsExplicitEmbeddingMetadataWithoutChangingMemoryFields() {
        let memory = MemoryItem(
            content: "Synthetic project preference",
            kind: .preference,
            source: "test",
            embedding: [0.5, -0.25],
            topic: "identity",
            expiresAt: Date(timeIntervalSince1970: 2_000_000_000),
            freshnessClass: .durable,
            embeddingFormatVersion: 7,
            embeddingModelIdentifier: "test:sha256:known-model",
            embeddingDimension: 2
        )

        XCTAssertEqual(memory.content, "Synthetic project preference")
        XCTAssertEqual(memory.memoryKind, .preference)
        XCTAssertEqual(memory.source, "test")
        XCTAssertEqual(memory.topic, "identity")
        XCTAssertEqual(memory.expiresAt, Date(timeIntervalSince1970: 2_000_000_000))
        XCTAssertEqual(memory.freshnessClass, MemoryFreshnessClass.durable.rawValue)
        XCTAssertEqual(memory.embedding, [0.5, -0.25])
        XCTAssertEqual(memory.embeddingFormatVersion, 7)
        XCTAssertEqual(memory.embeddingModelIdentifier, "test:sha256:known-model")
        XCTAssertEqual(memory.embeddingDimension, 2)
    }

}
