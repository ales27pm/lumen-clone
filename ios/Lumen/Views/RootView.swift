import SwiftUI
import SwiftData

struct RootView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query private var storedModels: [StoredModel]
    @State private var selection: MenuItem? = LumenLaunchArguments.isUITesting ? .settings : .chat
    @State private var columnVisibility: NavigationSplitViewVisibility = .automatic

    enum MenuSection: String, Identifiable, CaseIterable {
        case work
        case knowledge
        case automation
        case system

        var id: String { rawValue }

        var title: String {
            switch self {
            case .work: return "Work"
            case .knowledge: return "Knowledge"
            case .automation: return "Automation"
            case .system: return "System"
            }
        }

        var items: [MenuItem] {
            switch self {
            case .work: return [.chat, .outlook, .models]
            case .knowledge: return [.memory, .sources, .philosophies]
            case .automation: return [.triggers, .tools]
            case .system: return [.settings]
            }
        }
    }

    enum MenuItem: Hashable, Identifiable, CaseIterable {
        case chat, outlook, models, memory, sources, philosophies, triggers, tools, settings
        var id: Self { self }
        var title: String {
            switch self {
            case .chat: return "Chat"
            case .outlook: return "Outlook"
            case .models: return "Models"
            case .memory: return "Memory"
            case .sources: return "Sources"
            case .philosophies: return "Philosophies"
            case .triggers: return "Triggers"
            case .tools: return "Tools"
            case .settings: return "Settings"
            }
        }
        var subtitle: String {
            switch self {
            case .chat: return "Plan, ask, and execute"
            case .outlook: return "Mail with guardrails"
            case .models: return "Local runtime control"
            case .memory: return "Recall and capture"
            case .sources: return "Grounded files"
            case .philosophies: return "Reasoning profiles"
            case .triggers: return "Scheduled runs"
            case .tools: return "Action registry"
            case .settings: return "Preferences"
            }
        }
        var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.text.bubble.right"
            case .outlook: return "envelope.badge.shield.half.filled"
            case .models: return "cpu"
            case .memory: return "brain"
            case .sources: return "externaldrive"
            case .philosophies: return "sparkles.rectangle.stack"
            case .triggers: return "alarm"
            case .tools: return "wrench.and.screwdriver"
            case .settings: return "gearshape"
            }
        }
        var accessibilityIdentifier: String {
            switch self {
            case .chat: return "root.chat"
            case .outlook: return "root.outlook"
            case .models: return "root.models"
            case .memory: return "root.memory"
            case .sources: return "root.sources"
            case .philosophies: return "root.philosophies"
            case .triggers: return "root.triggers"
            case .tools: return "root.tools"
            case .settings: return "root.settings"
            }
        }
    }

    var body: some View {
        ZStack {
            AppBackground()

            NavigationSplitView(columnVisibility: $columnVisibility) {
                List(selection: $selection) {
                    LumenSidebarHeader()
                        .listRowSeparator(.hidden)
                        .listRowBackground(Color.clear)
                        .listRowInsets(EdgeInsets(top: 12, leading: 14, bottom: 12, trailing: 14))

                    ForEach(MenuSection.allCases) { section in
                        Section(section.title) {
                            ForEach(section.items) { item in
                                NavigationLink(value: item) {
                                    LumenSidebarRow(item: item, isSelected: selection == item)
                                }
                                .listRowInsets(EdgeInsets(top: 4, leading: 12, bottom: 4, trailing: 12))
                                .listRowBackground(Color.clear)
                            }
                        }
                    }
                }
                .navigationTitle("")
                .navigationBarTitleDisplayMode(.inline)
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
            } detail: {
                NavigationStack {
                    detailView(for: selection ?? .chat)
                }
            }
            .tint(Theme.accent)
            .navigationSplitViewStyle(.balanced)


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
        case .philosophies: AlgorithmicPhilosophiesView()
        case .triggers: TriggersView()
        case .tools: ToolsView()
        case .settings: SettingsView()
        }
    }
}

private struct LumenSidebarHeader: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            LumenBrandAsset(kind: .wordmarkLockup, accessibilityLabel: "Lumen")
                .frame(maxWidth: 220)
                .clipShape(.rect(cornerRadius: 10))

            HStack(spacing: 8) {
                StatusDot(color: Theme.accent, size: 7)
                Text("On-device agent workspace")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(1)
            }
        }
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(Theme.surfaceHigh)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }
}

private struct LumenSidebarRow: View {
    let item: RootView.MenuItem
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 11) {
            Image(systemName: item.systemImage)
                .font(.body.weight(.semibold))
                .foregroundStyle(isSelected ? LumenBrand.midnight : Theme.accent)
                .frame(width: 34, height: 34)
                .background {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(isSelected ? Theme.accent : Theme.accent.opacity(0.10))
                }

            VStack(alignment: .leading, spacing: 2) {
                Text(item.title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                Text(item.subtitle)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(item.accessibilityIdentifier)
    }
}
