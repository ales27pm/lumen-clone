import Foundation
import SwiftData
import XCTest
@testable import Lumen

@MainActor
final class MemoryEmbeddingCompatibilityTests: XCTestCase {
    func testLegacyUnversionedDiskStorePreservesMemoriesAndLeavesEmbeddingIdentityUnknown() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MemoryEmbeddingCompatibility-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let storeURL = directory.appendingPathComponent("memories.store")
        let fixtures = Self.legacyFixtures
        let legacySchema = Schema([MemoryItemBeforeEmbeddingMetadata.MemoryItem.self])
        let currentSchema = Schema([MemoryItem.self])

        // Both schemas must describe the same entity. A renamed test entity could
        // otherwise exercise a new empty table instead of the upgrade boundary.
        XCTAssertEqual(legacySchema.entities.map(\.name), currentSchema.entities.map(\.name))

        try autoreleasepool {
            let container = try Self.openStore(schema: legacySchema, url: storeURL)
            let context = ModelContext(container)
            context.autosaveEnabled = false
            for fixture in fixtures {
                context.insert(fixture.makeLegacyMemory())
            }
            try context.save()
            XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItemBeforeEmbeddingMetadata.MemoryItem>()), fixtures.count)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: storeURL.path))

        let replacementVector = [0.25, -0.5, 0.75, 0.125]
        let replacementFormatVersion = 7
        let replacementModelIdentifier = "test:sha256:known-reembedding-model"

        // No migration plan or fixture rewrite: this is the same unversioned
        // Schema -> ModelContainer opening path used by the production app.
        try autoreleasepool {
            let container = try Self.openStore(schema: currentSchema, url: storeURL)
            let context = ModelContext(container)
            context.autosaveEnabled = false
            let memories = try context.fetch(FetchDescriptor<MemoryItem>())
            XCTAssertEqual(memories.count, fixtures.count)
            for fixture in fixtures {
                let memory = try XCTUnwrap(memories.first { $0.id == fixture.id })
                fixture.assertPreserved(in: memory)
                XCTAssertEqual(memory.embedding, fixture.embedding)
                XCTAssertNil(memory.embeddingFormatVersion)
                XCTAssertNil(memory.embeddingModelIdentifier)
                XCTAssertNil(memory.embeddingDimension)
            }

            // Simulate persistence after a successful explicit re-embedding.
            // Metadata and its replacement vector are saved together; migration
            // itself must never stamp a legacy vector with a guessed identity.
            let refreshed = try XCTUnwrap(memories.first { $0.id == fixtures[0].id })
            refreshed.embedding = replacementVector
            refreshed.embeddingFormatVersion = replacementFormatVersion
            refreshed.embeddingModelIdentifier = replacementModelIdentifier
            refreshed.embeddingDimension = replacementVector.count
            try context.save()
        }

        try autoreleasepool {
            let container = try Self.openStore(schema: currentSchema, url: storeURL)
            let context = ModelContext(container)
            context.autosaveEnabled = false
            let memories = try context.fetch(FetchDescriptor<MemoryItem>())
            XCTAssertEqual(memories.count, fixtures.count)
            for (index, fixture) in fixtures.enumerated() {
                let memory = try XCTUnwrap(memories.first { $0.id == fixture.id })
                fixture.assertPreserved(in: memory)
                if index == 0 {
                    XCTAssertEqual(memory.embedding, replacementVector)
                    XCTAssertEqual(memory.embeddingFormatVersion, replacementFormatVersion)
                    XCTAssertEqual(memory.embeddingModelIdentifier, replacementModelIdentifier)
                    XCTAssertEqual(memory.embeddingDimension, replacementVector.count)
                } else {
                    XCTAssertEqual(memory.embedding, fixture.embedding)
                    XCTAssertNil(memory.embeddingFormatVersion)
                    XCTAssertNil(memory.embeddingModelIdentifier)
                    XCTAssertNil(memory.embeddingDimension)
                }
            }
        }
    }

    func testExistingInitializerDoesNotInferEmbeddingIdentityFromVectorLength() {
        let memory = MemoryItem(content: "Synthetic legacy-compatible memory", embedding: [0.5, -0.25])

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
            topic: "compatibility",
            expiresAt: Date(timeIntervalSince1970: 2_000_000_000),
            freshnessClass: .durable,
            embeddingFormatVersion: 7,
            embeddingModelIdentifier: "test:sha256:known-model",
            embeddingDimension: 2
        )

        XCTAssertEqual(memory.content, "Synthetic project preference")
        XCTAssertEqual(memory.memoryKind, .preference)
        XCTAssertEqual(memory.source, "test")
        XCTAssertEqual(memory.topic, "compatibility")
        XCTAssertEqual(memory.expiresAt, Date(timeIntervalSince1970: 2_000_000_000))
        XCTAssertEqual(memory.freshnessClass, MemoryFreshnessClass.durable.rawValue)
        XCTAssertEqual(memory.embedding, [0.5, -0.25])
        XCTAssertEqual(memory.embeddingFormatVersion, 7)
        XCTAssertEqual(memory.embeddingModelIdentifier, "test:sha256:known-model")
        XCTAssertEqual(memory.embeddingDimension, 2)
    }

    private static func openStore(schema: Schema, url: URL) throws -> ModelContainer {
        let configuration = ModelConfiguration(
            "MemoryEmbeddingCompatibility",
            schema: schema,
            url: url,
            cloudKitDatabase: .none
        )
        return try ModelContainer(for: schema, configurations: [configuration])
    }

    private static var legacyFixtures: [LegacyMemoryFixture] {
        [
            LegacyMemoryFixture(
                id: UUID(uuidString: "13131313-1313-4313-8313-131313131313")!,
                content: "Synthetic pinned project memory",
                source: "manual",
                createdAt: Date(timeIntervalSince1970: 1_600_000_001),
                embedding: [0.125, -0.75, 0.5],
                kind: "project",
                isPinned: true,
                topic: "migration fixture",
                expiresAt: Date(timeIntervalSince1970: 2_000_000_001),
                freshnessClass: "durable"
            ),
            LegacyMemoryFixture(
                id: UUID(uuidString: "24242424-2424-4424-8424-242424242424")!,
                content: "Synthetic sparse memory without a vector",
                source: "conversation",
                createdAt: Date(timeIntervalSince1970: 1_600_000_002),
                embedding: [],
                kind: "fact",
                isPinned: false,
                topic: nil,
                expiresAt: nil,
                freshnessClass: nil
            ),
            LegacyMemoryFixture(
                id: UUID(uuidString: "35353535-3535-4535-8535-353535353535")!,
                content: "Synthetic expired memory retained during migration",
                source: "tool",
                createdAt: Date(timeIntervalSince1970: 1_600_000_003),
                embedding: [-0.25, 0.25],
                kind: "preference",
                isPinned: false,
                topic: "historical",
                expiresAt: Date(timeIntervalSince1970: 1_600_000_004),
                freshnessClass: "volatile"
            )
        ]
    }
}

private struct LegacyMemoryFixture {
    let id: UUID
    let content: String
    let source: String
    let createdAt: Date
    let embedding: [Double]
    let kind: String
    let isPinned: Bool
    let topic: String?
    let expiresAt: Date?
    let freshnessClass: String?

    func makeLegacyMemory() -> MemoryItemBeforeEmbeddingMetadata.MemoryItem {
        let memory = MemoryItemBeforeEmbeddingMetadata.MemoryItem(content: content)
        memory.id = id
        memory.source = source
        memory.createdAt = createdAt
        memory.embedding = embedding
        memory.kind = kind
        memory.isPinned = isPinned
        memory.topic = topic
        memory.expiresAt = expiresAt
        memory.freshnessClass = freshnessClass
        return memory
    }

    @MainActor
    func assertPreserved(in memory: MemoryItem, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(memory.id, id, file: file, line: line)
        XCTAssertEqual(memory.content, content, file: file, line: line)
        XCTAssertEqual(memory.source, source, file: file, line: line)
        XCTAssertEqual(memory.createdAt, createdAt, file: file, line: line)
        XCTAssertEqual(memory.kind, kind, file: file, line: line)
        XCTAssertEqual(memory.isPinned, isPinned, file: file, line: line)
        XCTAssertEqual(memory.topic, topic, file: file, line: line)
        XCTAssertEqual(memory.expiresAt, expiresAt, file: file, line: line)
        XCTAssertEqual(memory.freshnessClass, freshnessClass, file: file, line: line)
    }
}

// Frozen persisted shape from MemoryItem before embedding metadata was added.
// Keep the leaf model name and all ten attributes/defaults unchanged. Intentionally
// use Schema([MemoryItem.self]) rather than introducing a versioned source store.
private enum MemoryItemBeforeEmbeddingMetadata {
    @Model
    final class MemoryItem {
        var id: UUID = UUID()
        var content: String = ""
        var source: String = "manual"
        var createdAt: Date = Date()
        var embedding: [Double] = []
        var kind: String = "fact"
        var isPinned: Bool = false
        var topic: String?
        var expiresAt: Date?
        var freshnessClass: String?

        init(content: String) {
            self.content = content
        }
    }
}
