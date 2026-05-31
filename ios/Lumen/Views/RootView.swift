import SwiftUI
import SwiftData

struct RootView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query private var storedModels: [StoredModel]
    @State private var selection: MenuItem? = .chat
    @State private var columnVisibility: NavigationSplitViewVisibility = .automatic

    enum MenuItem: Hashable, Identifiable, CaseIterable {
        case chat, outlook, models, memory, sources, triggers, tools, settings
        var id: Self { self }
        var title: String {
            switch self {
            case .chat: return "Chat"
            case .outlook: return "Outlook"
            case .models: return "Models"
            case .memory: return "Memory"
            case .sources: return "Sources"
            case .triggers: return "Triggers"
            case .tools: return "Tools"
            case .settings: return "Settings"
            }
        }
        var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.text.bubble.right"
            case .outlook: return "envelope.badge.shield.half.filled"
            case .models: return "cpu"
            case .memory: return "brain"
            case .sources: return "externaldrive"
            case .triggers: return "alarm"
            case .tools: return "wrench.and.screwdriver"
            case .settings: return "gearshape"
            }
        }
    }

    var body: some View {
        ZStack {
            NavigationSplitView(columnVisibility: $columnVisibility) {
                List(MenuItem.allCases, selection: $selection) { item in
                    NavigationLink(value: item) {
                        Label(item.title, systemImage: item.systemImage)
                    }
                }
                .navigationTitle("Lumen")
                .listStyle(.sidebar)
            } detail: {
                NavigationStack {
                    detailView(for: selection ?? .chat)
                }
            }
            .tint(Theme.accent)
            .task {
                await ModelLoader.loadAtLaunch(appState: appState, stored: storedModels)
            }
            .task(id: appState.activeChatModelID) {
                await ModelLoader.syncChat(appState: appState, stored: storedModels)
            }
            .task(id: appState.activeEmbeddingModelID) {
                await ModelLoader.syncEmbed(appState: appState, stored: storedModels)
            }
            .onChange(of: storedModels.count) { _, _ in
                Task {
                    await ModelLoader.loadAtLaunch(appState: appState, stored: storedModels)
                }
            }

            if appState.runtime.bootSplashVisible {
                BootSplashView()
                    .zIndex(10)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: appState.runtime.bootSplashVisible)
    }

    @ViewBuilder
    private func detailView(for item: MenuItem) -> some View {
        switch item {
        case .chat: ChatHomeView()
        case .outlook: OutlookMailView()
        case .models: ModelsView()
        case .memory: MemoryView()
        case .sources: SourcesView()
        case .triggers: TriggersView()
        case .tools: ToolsView()
        case .settings: SettingsView()
        }
    }
}


