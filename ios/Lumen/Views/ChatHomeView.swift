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

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
                if let convo = selectedConversation ?? conversations.first {
                    ChatView(conversation: convo)
                        .id(convo.id)
                } else {
                    EmptyChatPlaceholder(onNew: createConversation)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showingConversations = true
                    } label: {
                        Image(systemName: "list.bullet")
                            .font(.body)
                    }
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
                    Button {
                        createConversation()
                    } label: {
                        Image(systemName: "square.and.pencil")
                            .font(.body)
                    }
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
            if storedModels.isEmpty {
                showingOnboarding = true
            }
        }
    }

    private var activeModelPill: some View {
        let name = storedModels.first { $0.id.uuidString == appState.activeChatModelID }?.name ?? "No model"
        return HStack(spacing: 6) {
            StatusDot(color: appState.isGenerating ? Theme.accent : Theme.textTertiary, size: 6)
            Text(name)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
            Image(systemName: "chevron.down")
                .font(.caption2)
                .foregroundStyle(Theme.textTertiary)
        }
    }

    private func createConversation() {
        let convo = Conversation(title: "New Chat", systemPrompt: appState.systemPrompt, modelName: activeModelName)
        modelContext.insert(convo)
        try? modelContext.save()
        selectedConversation = convo
    }

    private var activeModelName: String? {
        storedModels.first { $0.id.uuidString == appState.activeChatModelID }?.name
    }
}

struct EmptyChatPlaceholder: View {
    var onNew: () -> Void
    var body: some View {
        VStack(spacing: 14) {
            Spacer()
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 36))
                .foregroundStyle(Theme.textTertiary)
            Text("No conversation")
                .font(.body.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
            Text("Start a new chat to begin.")
                .font(.footnote)
                .foregroundStyle(Theme.textSecondary)
            Button(action: onNew) {
                Text("New chat")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Theme.accent)
                    .clipShape(.rect(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            .padding(.top, 6)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
