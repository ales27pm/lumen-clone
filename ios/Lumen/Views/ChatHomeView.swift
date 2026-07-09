import SwiftUI
import SwiftData

struct ChatHomeView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]
    @Query private var storedModels: [StoredModel]

    @State private var selectedConversation: Conversation?
    @State private var showingConversations = false
    @State private var showingModelPicker = false
    @State private var showingOnboarding = false
    @State private var pendingDrafts: [UUID: String] = [:]

    var body: some View {
        NavigationStack {
            ZStack {
                AppBackground()
                if let convo = selectedConversation ?? conversations.first {
                    ChatView(
                        conversation: convo,
                        initialDraft: pendingDrafts[convo.id],
                        onInitialDraftConsumed: { pendingDrafts[convo.id] = nil }
                    )
                        .id(convo.id)
                } else {
                    EmptyChatPlaceholder(
                        modelName: activeModelName,
                        onNew: { createConversation() },
                        onPrompt: { createConversation(seedPrompt: $0) }
                    )
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    LumenIconControl(
                        systemImage: "sidebar.left",
                        accessibilityLabel: "Open conversations",
                        action: { showingConversations = true }
                    )
                }
                ToolbarItem(placement: .principal) {
                    Button {
                        showingModelPicker = true
                    } label: {
                        activeModelPill
                    }
                    .buttonStyle(.plain)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    LumenIconControl(
                        systemImage: "square.and.pencil",
                        accessibilityLabel: "Start new conversation",
                        action: { createConversation() }
                    )
                }
            }
        }
        .sheet(isPresented: $showingConversations) {
            ConversationsSheet(selected: $selectedConversation)
                .presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $showingModelPicker) {
            ModelPickerSheet()
                .presentationDetents([.medium])
        }
        .sheet(isPresented: $showingOnboarding) {
            OnboardingSheet()
                .presentationDetents([.large])
        }
        .onAppear {
            if conversations.isEmpty {
                createConversation()
            }
            if storedModels.isEmpty && !LumenLaunchArguments.isUITesting {
                showingOnboarding = true
            }
        }
    }

    private var activeModelPill: some View {
        let name = storedModels.first { $0.id.uuidString == appState.activeChatModelID }?.name ?? "No model"
        return HStack(spacing: 6) {
            LumenBrandAsset(kind: .assistantMark)
                .frame(width: 24, height: 24)
                .clipShape(.rect(cornerRadius: 7))
            VStack(alignment: .leading, spacing: 1) {
                Text(appState.isGenerating ? "Lumen is working" : "Lumen")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                Text(name)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
            }
            Image(systemName: "chevron.down")
                .font(.caption2)
                .foregroundStyle(Theme.textTertiary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Theme.surfaceHigh, in: Capsule())
        .overlay { Capsule().strokeBorder(Theme.border, lineWidth: 1) }
    }

    private func createConversation(seedPrompt: String? = nil) {
        let title = seedPrompt.map { String($0.prefix(42)) } ?? "New Chat"
        let convo = Conversation(title: title, systemPrompt: appState.systemPrompt, modelName: activeModelName)
        modelContext.insert(convo)
        try? modelContext.save()
        if let seedPrompt {
            pendingDrafts[convo.id] = seedPrompt
        }
        selectedConversation = convo
    }

    private var activeModelName: String? {
        storedModels.first { $0.id.uuidString == appState.activeChatModelID }?.name
    }
}

struct EmptyChatPlaceholder: View {
    var modelName: String?
    var onNew: () -> Void
    var onPrompt: (String) -> Void

    private let prompts = [
        "Turn this messy idea into a plan with next actions.",
        "Analyze an attached file and extract decisions.",
        "Find the safest way to automate this task.",
        "Draft a concise reply using my recent context."
    ]

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                Spacer(minLength: 34)

                ZStack {
                    LumenBrandAsset(kind: .mark)
                        .frame(maxWidth: 280)
                        .opacity(0.48)
                        .accessibilityHidden(true)

                    LumenBrandAsset(kind: .verticalLogo, accessibilityLabel: "Lumen")
                        .frame(maxWidth: 180)
                        .clipShape(.rect(cornerRadius: 16))
                        .shadow(color: LumenBrand.lumen.opacity(0.24), radius: 24, y: 10)
                }
                .frame(height: 240)

                VStack(spacing: 8) {
                    Text("Start with an outcome")
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                        .multilineTextAlignment(.center)
                    Text(modelName ?? "Choose a local model to unlock full chat.")
                        .font(.footnote)
                        .foregroundStyle(Theme.textSecondary)
                        .multilineTextAlignment(.center)
                        .lineLimit(2)
                }
                .frame(maxWidth: 360)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 10)], spacing: 10) {
                    ForEach(prompts, id: \.self) { prompt in
                        Button {
                            onPrompt(prompt)
                        } label: {
                            HStack(alignment: .top, spacing: 9) {
                                Image(systemName: "sparkle")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(Theme.accent)
                                    .frame(width: 22, height: 22)
                                Text(prompt)
                                    .font(.footnote.weight(.medium))
                                    .foregroundStyle(Theme.textPrimary)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: 0)
                            }
                            .padding(12)
                            .frame(maxWidth: .infinity, minHeight: 64, alignment: .topLeading)
                            .background(Theme.surfaceHigh, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .overlay {
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .strokeBorder(Theme.border, lineWidth: 1)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
                .frame(maxWidth: 520)

                Button(action: onNew) {
                    Label("New chat", systemImage: "square.and.pencil")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(LumenBrand.midnight)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 12)
                        .background(Theme.accent, in: Capsule())
                }
                .buttonStyle(.plain)
                .padding(.top, 2)

                Spacer(minLength: 34)
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
