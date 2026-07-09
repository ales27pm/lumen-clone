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
        bootstrap: @escaping @Sendable (AppState, ModelContext) async throws -> Void = AppStartupCoordinator.defaultBootstrap
    ) async {
        state = .loading
        appState.runtime.startBoot()
        do {
            let container = try createContainer()
            appState.runtime.updateBootStep(id: "container", detail: "SwiftData container ready", state: .complete)

            // Complete core boot immediately so the UI renders.
            // Heavy work (grounding resources, fleet checks, model loading)
            // continues in a background detached task.
            appState.runtime.completeBootCore()
            state = .ready(container)

            // Defer heavy bootstrap work to a truly detached background task.
            // This MUST NOT inherit @MainActor isolation.
            // Capture parameters locally so they can escape into the Task.detached closure.
            let capturedAppState = appState
            let capturedContainer = container
            let capturedBootstrap = bootstrap
            Task.detached(priority: .medium) {
                let ctx = ModelContext(capturedContainer)
                do {
                    try await capturedBootstrap(capturedAppState, ctx)
                } catch {
                    Logger(subsystem: "ai.lumen.app", category: "startup").warning("Background bootstrap completed with error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                }
            }
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

    nonisolated static func uiTestingBootstrap(appState: AppState, ctx: ModelContext) async throws {
        await MainActor.run {
            appState.runtime.updateBootStep(id: "grounding", detail: "Skipped for UI tests", state: .complete)
            appState.runtime.updateBootStep(id: "triggers", detail: "Skipped for UI tests", state: .complete)
            appState.runtime.dismissBootSplash()
        }
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

    /// Runs light bootstrap work in the background after the UI is already visible.
    /// This is nonisolated — it runs on the Task.detached executor, NOT @MainActor.
    /// All UI updates go through `await MainActor.run { ... }`.
    ///
    /// IMPORTANT: Fleet model checks and model loading are deliberately NOT done here.
    /// They are deferred to on-demand loading (ModelLoader.ensureChatLoaded / ensureEmbedLoaded)
    /// when the user first interacts with the chat. This keeps startup under 1 second
    /// and prevents watchdog (0x8BADF00D) kills from blocking the main actor.
    nonisolated private static func defaultBootstrap(appState: AppState, ctx: ModelContext) async throws {
        // Phase 1: Lightweight validation (fast, no I/O heavy work)
        try await withStage(.bootstrap) {
            try LumenModelSlotContract.validateCompletenessAtStartup()
        }

        // Phase 2: Grounding resources — parsed on background actor
        await loadGroundingResourcesOnBackground(appState: appState)

        // Phase 3: Triggers
        await MainActor.run {
            appState.runtime.updateBootStep(id: "triggers", detail: "Registering background tasks", state: .running)
        }
        await BackgroundOrchestrator.shared.register()
        await BackgroundOrchestrator.shared.schedule()
        await BackgroundOrchestrator.shared.requestPermission()
        await MainActor.run {
            appState.runtime.updateBootStep(id: "triggers", detail: "Background tasks ready", state: .complete)
        }

        // Boot is complete. Dismiss the splash immediately — models load on-demand.
        await MainActor.run {
            appState.runtime.bootHeadline = "Lumen is ready"
            appState.runtime.dismissBootSplash()
        }
    }

    /// Loads grounding resources (492KB JSON manifests) on a background actor
    /// so JSONDecoder does not block the main thread.
    private static func loadGroundingResourcesOnBackground(appState: AppState) async {
        await MainActor.run {
            appState.runtime.updateBootStep(id: "grounding", detail: "Verifying bundled agent grounding resources", state: .running)
        }
        do {
            try await GroundingResourceLoader.shared.verifyRequiredResourcesAsync()
            _ = try await GroundingResourceLoader.shared.loadManifestAsync()
            _ = try await GroundingResourceLoader.shared.loadFleetSystemPromptsAsync()
            _ = try await GroundingResourceLoader.shared.loadRuntimeGroundingBundleAsync()
            _ = try await GroundingResourceLoader.shared.loadRuntimeGroundingPromptAsync()
            await MainActor.run {
                appState.runtime.updateBootStep(id: "grounding", detail: "Bundled agent grounding resources ready", state: .complete)
            }
        } catch {
            Logger(subsystem: "ai.lumen.app", category: "startup").warning("Grounding resources unavailable error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public). Continuing in limited mode.")
            await MainActor.run {
                appState.runtime.updateBootStep(id: "grounding", detail: "Grounding resources unavailable — limited mode", state: .complete)
            }
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
                            SceneTransitionCoordinator.shared.handleScenePhaseChange(phase)
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
                if LumenLaunchArguments.isUITesting {
                    await startup.initialize(appState: appState, bootstrap: AppStartupCoordinator.uiTestingBootstrap)
                } else {
                    await startup.initialize(appState: appState)
                }
                if case .ready(let container) = startup.state {
                    SharedContainer.shared = container
                    if LumenLaunchArguments.isUITesting {
                        appState.runtime.dismissBootSplash()
                    } else {
                        await PersistentRuntimeDiagnosticsRunner.shared.resumeIfEnabled()
                    }
                }
            }
        }
    }
}

enum LumenLaunchArguments {
    static var isUITesting: Bool {
        ProcessInfo.processInfo.arguments.contains("--lumen-ui-tests")
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
