import Foundation
import SwiftUI
import SwiftData
import OSLog

@MainActor
final class AppStartupCoordinator {
    enum Stage: String {
        case container
        case bootstrap
        case groundingResources
        case modelLoader
        case triggers
        case remCycle
    }

    struct FailureContext: Error, Equatable {
        let stage: Stage
        let message: String
        let domain: String
        let code: Int

        var summary: String { "\(domain) (\(code)): \(message)" }
    }

    enum State: Equatable {
        case loading
        case ready(ModelContainer)
        case failed(FailureContext)
    }

    private let logger = Logger(subsystem: "ai.lumen.app", category: "startup")
    private(set) var state: State = .loading

    func initialize(
        appState: AppState,
        createContainer: @MainActor @Sendable () throws -> ModelContainer = AppStartupCoordinator.defaultContainerFactory,
        bootstrap: @MainActor @Sendable (AppState, ModelContext) async throws -> Void = AppStartupCoordinator.defaultBootstrap
    ) async {
        state = .loading
        appState.runtime.startBoot()
        do {
            let container = try createContainer()
            appState.runtime.updateBootStep(id: "container", detail: "SwiftData container ready", state: .complete)

            let ctx = ModelContext(container)
            try await bootstrap(appState, ctx)
            state = .ready(container)
        } catch {
            let failure = Self.failureContext(stage: currentStage(error), from: error)
            emitFailureTelemetry(failure)
            state = .failed(failure)
        }
    }

    private func currentStage(_ error: Error) -> Stage {
        (error as? StartupError)?.stage ?? .container
    }

    private func emitFailureTelemetry(_ failure: FailureContext) {
        logger.error("startup_failed stage=\(failure.stage.rawValue, privacy: .public) domain=\(failure.domain, privacy: .public) code=\(failure.code, privacy: .public) message=\(failure.message, privacy: .private)")
    }

    private static func failureContext(stage: Stage, from error: Error) -> FailureContext {
        let baseError: Error
        if let startupError = error as? StartupError {
            baseError = startupError.underlying
        } else {
            baseError = error
        }
        let nsError = baseError as NSError
        return FailureContext(stage: stage, message: nsError.localizedDescription, domain: nsError.domain, code: nsError.code)
    }

    func continueInLimitedMode(appState: AppState) {
        do {
            let container = try Self.inMemoryContainerFactory()
            SharedContainer.shared = container
            appState.runtime.completeBootCore()
            state = .ready(container)
        } catch {
            let failure = Self.failureContext(stage: .container, from: error)
            emitFailureTelemetry(failure)
            state = .failed(failure)
        }
    }

    private static func inMemoryContainerFactory() throws -> ModelContainer {
        try makeContainer(isStoredInMemoryOnly: true)
    }

    private static func defaultContainerFactory() throws -> ModelContainer {
        try ensureApplicationSupportDirectoryExists()
        return try makeContainer(isStoredInMemoryOnly: false)
    }

    static func ensureApplicationSupportDirectoryExists(fileManager: FileManager = .default) throws {
        guard let applicationSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            throw CocoaError(.fileNoSuchFile)
        }
        try ensureDirectoryExists(applicationSupportURL, fileManager: fileManager)
    }

    static func ensureDirectoryExists(_ url: URL, fileManager: FileManager = .default) throws {
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private static func makeContainer(isStoredInMemoryOnly: Bool) throws -> ModelContainer {
        let config = ModelConfiguration(schema: appSchema, isStoredInMemoryOnly: isStoredInMemoryOnly)
        return try ModelContainer(for: appSchema, configurations: [config])
    }

    private static var appSchema: Schema {
        Schema([
            Conversation.self,
            ChatMessage.self,
            MemoryItem.self,
            StoredModel.self,
            RAGChunk.self,
            Trigger.self,
        ])
    }

    private static func defaultBootstrap(appState: AppState, ctx: ModelContext) async throws {
        try await withStage(.bootstrap) {
            try LumenModelSlotContract.validateCompletenessAtStartup()
            await ModelLaunchBootstrap.ensureFleetDownloaded(appState: appState, context: ctx)
        }

        try await withStage(.groundingResources) {
            appState.runtime.updateBootStep(id: "grounding", detail: "Verifying bundled agent grounding resources", state: .running)
            try BundledAgentGroundingStore.shared.verifyRequiredResources()
            _ = try BundledAgentGroundingStore.shared.loadManifest()
            _ = try BundledAgentGroundingStore.shared.loadFleetSystemPrompts()
            appState.runtime.updateBootStep(id: "grounding", detail: "Bundled agent grounding resources ready", state: .complete)
        }

        try await withStage(.modelLoader) {
            appState.runtime.updateBootStep(id: "loader", detail: "Loading active chat and embedding models", state: .running)
            let storedModels = (try? ctx.fetch(FetchDescriptor<StoredModel>())) ?? []
            await ModelLoader.loadAtLaunch(appState: appState, stored: storedModels)
            appState.runtime.updateBootStep(id: "loader", detail: "Model runtime initialized", state: .complete)
        }

        try await withStage(.triggers) {
            appState.runtime.updateBootStep(id: "triggers", detail: "Registering background tasks", state: .running)
            TriggerScheduler.shared.registerTasks()
            TriggerScheduler.shared.scheduleBackgroundRefresh()
            await TriggerScheduler.shared.requestPermission()
            appState.runtime.updateBootStep(id: "triggers", detail: "Background tasks ready", state: .complete)
        }

        try await withStage(.remCycle) {
            await RemCycleService.runIfDue(context: ctx, appState: appState, reason: "launch")
            appState.runtime.completeBootCore()
        }
    }

    private static func withStage(_ stage: Stage, operation: () async throws -> Void) async throws {
        do {
            try await operation()
        } catch {
            throw StartupError(stage: stage, underlying: error)
        }
    }

    struct StartupError: Error {
        let stage: Stage
        let underlying: Error
    }
}

@main
struct LumenApp: App {
    @UIApplicationDelegateAdaptor(LumenAppDelegate.self) private var appDelegate
    @State private var appState = AppState()
    @State private var startup = AppStartupCoordinator()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            Group {
                switch startup.state {
                case .loading:
                    BootSplashView()
                case .ready(let container):
                    RootView()
                        .modelContainer(container)
                        .onChange(of: scenePhase) { _, phase in
                            if phase == .active {
                                Task { @MainActor in
                                    let ctx = ModelContext(container)
                                    await TriggerScheduler.shared.fireDueTriggers(context: ctx, appState: appState)
                                    await RemCycleService.runIfDue(context: ctx, appState: appState, reason: "scene-active")
                                }
                            }
                        }
                case .failed(let failure):
                    StartupFailureView(failure: failure) {
                        await startup.initialize(appState: appState)
                        if case .ready(let container) = startup.state {
                            SharedContainer.shared = container
                        }
                    } safeModeAction: {
                        startup.continueInLimitedMode(appState: appState)
                    }
                }
            }
            .environment(appState)
            .environment(VoiceService.shared)
            .preferredColorScheme(.dark)
            .tint(Theme.accent)
            .onOpenURL { url in
                MicrosoftGraphURLHandler.handle(url)
            }
            .task {
                guard case .loading = startup.state else { return }
                await startup.initialize(appState: appState)
                if case .ready(let container) = startup.state {
                    SharedContainer.shared = container
                }
            }
        }
    }
}

private struct StartupFailureView: View {
    let failure: AppStartupCoordinator.FailureContext
    let retryAction: () async -> Void
    let safeModeAction: () -> Void
    @State private var isRetrying = false

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 42))
                .foregroundStyle(.yellow)
            Text("Couldn’t Start Lumen")
                .font(.title2.weight(.semibold))
            Text("Lumen hit an initialization problem and couldn’t finish startup.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Text("Stage: \(failure.stage.rawValue)\n\(failure.summary)")
                .font(.caption)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            Button("Retry") {
                guard !isRetrying else { return }
                isRetrying = true
                Task {
                    defer { isRetrying = false }
                    await retryAction()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRetrying)

            Button("Continue in Limited Mode", action: safeModeAction)
                .buttonStyle(.bordered)
        }
        .padding(24)
    }
}
