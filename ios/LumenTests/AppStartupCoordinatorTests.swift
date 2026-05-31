import Foundation
import Testing
import SwiftData
@testable import Lumen

@MainActor
struct AppStartupCoordinatorTests {
    enum TestError: Error { case failed }

    @Test func startupFailureTransitionsToFailedState() async throws {
        var coordinator = AppStartupCoordinator()
        let appState = AppState()

        await coordinator.initialize(
            appState: appState,
            createContainer: { throw TestError.failed },
            bootstrap: { _, _ in }
        )

        guard case .failed(let context) = coordinator.state else {
            Issue.record("Expected failed state")
            return
        }
        #expect(context.stage == .container)
        #expect(!context.domain.isEmpty)
    }

    @Test func startupBootstrapFailureTracksBootstrapStage() async throws {
        var coordinator = AppStartupCoordinator()
        let appState = AppState()

        let schema = Schema([
            Conversation.self,
            ChatMessage.self,
            MemoryItem.self,
            StoredModel.self,
            RAGChunk.self,
            Trigger.self,
        ])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)

        await coordinator.initialize(
            appState: appState,
            createContainer: {
                try ModelContainer(for: schema, configurations: [config])
            },
            bootstrap: { _, _ in
                throw AppStartupCoordinator.StartupError(stage: .bootstrap, underlying: TestError.failed)
            }
        )

        guard case .failed(let context) = coordinator.state else {
            Issue.record("Expected failed state")
            return
        }
        #expect(context.stage == .bootstrap)
        #expect(!context.domain.isEmpty)
    }

    @Test func retryAfterFailureTransitionsToReady() async throws {
        var coordinator = AppStartupCoordinator()
        let appState = AppState()
        var attempts = 0

        await coordinator.initialize(
            appState: appState,
            createContainer: {
                attempts += 1
                if attempts == 1 { throw TestError.failed }
                let schema = Schema([
                    Conversation.self,
                    ChatMessage.self,
                    MemoryItem.self,
                    StoredModel.self,
                    RAGChunk.self,
                    Trigger.self,
                ])
                let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
                return try ModelContainer(for: schema, configurations: [config])
            },
            bootstrap: { _, _ in }
        )

        #expect(attempts == 1)
        #expect({ if case .failed = coordinator.state { true } else { false } }())

        await coordinator.initialize(
            appState: appState,
            createContainer: {
                attempts += 1
                let schema = Schema([
                    Conversation.self,
                    ChatMessage.self,
                    MemoryItem.self,
                    StoredModel.self,
                    RAGChunk.self,
                    Trigger.self,
                ])
                let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
                return try ModelContainer(for: schema, configurations: [config])
            },
            bootstrap: { _, _ in }
        )

        #expect(attempts == 2)
        #expect({ if case .ready = coordinator.state { true } else { false } }())
    }
    @Test func continueInLimitedModeTransitionsToReady() async throws {
        var coordinator = AppStartupCoordinator()
        let appState = AppState()

        await coordinator.initialize(
            appState: appState,
            createContainer: { throw TestError.failed },
            bootstrap: { _, _ in }
        )

        coordinator.continueInLimitedMode(appState: appState)

        #expect({ if case .ready = coordinator.state { true } else { false } }())
    }

    @Test func applicationSupportDirectoryIsCreatedBeforePersistentContainer() throws {
        let temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LumenStartupTests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let applicationSupport = temporaryRoot
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Application Support", isDirectory: true)

        try AppStartupCoordinator.ensureDirectoryExists(applicationSupport)

        var isDirectory: ObjCBool = false
        #expect(FileManager.default.fileExists(atPath: applicationSupport.path, isDirectory: &isDirectory))
        #expect(isDirectory.boolValue)

        try? FileManager.default.removeItem(at: temporaryRoot)
    }

}
