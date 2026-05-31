import Foundation
import SwiftData
import Testing
@testable import Lumen

@MainActor
struct MemoryCascadeTests {
    @Test func recallReturnsTieredItemsForVectorizedAndCondensed() async throws {
        let context = try makeInMemoryContext()
        seedRecallFixture(context: context)

        let result = await MemoryCascade.recall(
            query: "travel plans and sprint goals",
            history: [
                (role: .user, content: "Let's review travel plans."),
                (role: .assistant, content: "Sure, and we should also lock sprint goals.")
            ],
            context: context
        )

        #expect(!result.condensed.isEmpty)
        #expect(result.condensed.allSatisfy { $0.scope == .remCondensed })
        #expect(result.vectorized.allSatisfy { $0.scope == .conversation || $0.scope == .project || $0.scope == .userPreference })

        let hasEmbeddingRuntime = await AppLlamaService.shared.hasSemanticEmbeddingRuntime
        if hasEmbeddingRuntime {
            #expect(!result.vectorized.isEmpty)
        }
    }

    @Test func condenseIfNeededCreatesCascadeRemCondensedItems() async throws {
        let context = try makeInMemoryContext()
        seedCondenseFixture(context: context)

        try await MemoryCascade.condenseIfNeeded(context: context, minimumCount: 6)

        let descriptor = FetchDescriptor<MemoryItem>(
            predicate: #Predicate<MemoryItem> { item in
                item.source == "rem-condensed"
            }
        )
        let condensedItems = try context.fetch(descriptor)

        #expect(!condensedItems.isEmpty)
        #expect(condensedItems.contains { item in
            (item.topic ?? "").hasPrefix("cascade:")
        })
    }

    private func makeInMemoryContext() throws -> ModelContext {
        let schema = Schema([MemoryItem.self])
        let configuration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: schema, configurations: [configuration])
        return ModelContext(container)
    }

    private func seedRecallFixture(context: ModelContext) {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let fixtures: [(content: String, kind: MemoryKind, source: String, topic: String?, secondsOffset: TimeInterval)] = [
            ("Book flights to Portland for conference", .project, "manual", "travel", -400),
            ("Sprint goal: finish memory cascade tests", .conversation, "manual", "work", -300),
            ("User prefers aisle seat for travel", .preference, "auto", "travel", -200),
            ("Condensed travel and sprint notes", .conversation, "rem-condensed", "cascade:travel", -100)
        ]

        for fixture in fixtures {
            let item = MemoryItem(
                content: fixture.content,
                kind: fixture.kind,
                source: fixture.source,
                embedding: [1.0, 0.0, 0.0],
                topic: fixture.topic
            )
            item.createdAt = now.addingTimeInterval(fixture.secondsOffset)
            context.insert(item)
        }

        try? context.save()
    }

    private func seedCondenseFixture(context: ModelContext) {
        let now = Date(timeIntervalSince1970: 1_700_100_000)

        for index in 0..<7 {
            let item = MemoryItem(
                content: "Meeting note \(index): discuss launch checklist and owners",
                kind: .conversation,
                source: "manual",
                embedding: [0.8, 0.1, 0.1],
                topic: "team-sync"
            )
            item.createdAt = now.addingTimeInterval(TimeInterval(index))
            context.insert(item)
        }

        let existing = MemoryItem(
            content: "Existing condensed summary for travel",
            kind: .conversation,
            source: "rem-condensed",
            embedding: [0.3, 0.3, 0.4],
            topic: "cascade:travel"
        )
        existing.createdAt = now.addingTimeInterval(-50)
        context.insert(existing)

        try? context.save()
    }
}
