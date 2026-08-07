import SwiftUI
import SwiftData
import Foundation

struct ChatHomeView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]
    @Query private var storedModels: [StoredModel]
    @AppStorage("verifiedModelProvisioningReceipt") private var provisioningReceiptData: Data?

    @State private var selectedConversation: Conversation?
    @State private var showingConversations = false
    @State private var showingModelPicker = false
    @State private var showingOnboarding = false
    @State private var pendingDrafts: [UUID: String] = [:]
    @State private var conversationPersistenceError: String?

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
        .sheet(isPresented: $showingOnboarding, onDismiss: {
            if ModelProvisioningReceipt.status() != .completed {
                _ = ModelProvisioningReceipt.markDeferred()
            }
        }) {
            OnboardingSheet()
                .presentationDetents([.large])
        }
        .onAppear {
            if conversations.isEmpty {
                createConversation()
            }
        }
        .task(id: provisioningValidationKey) {
            await refreshProvisioningPresentation()
        }
        .alert("Conversation not saved", isPresented: Binding(
            get: { conversationPersistenceError != nil },
            set: { if !$0 { conversationPersistenceError = nil } }
        )) {
            Button("OK", role: .cancel) { conversationPersistenceError = nil }
        } message: {
            Text(conversationPersistenceError ?? "The new conversation could not be saved.")
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
        let outcome = ConversationPersistenceCoordinator.attemptSave(
            estimatedBytes: title.utf8.count + appState.systemPrompt.utf8.count,
            operation: "chat-home.create",
            save: { try modelContext.save() }
        )
        guard case .saved = outcome else {
            modelContext.delete(convo)
            if case .failed(let failure) = outcome {
                conversationPersistenceError = failure.userMessage
            } else {
                conversationPersistenceError = "The new conversation could not be saved."
            }
            return
        }
        if let seedPrompt {
            pendingDrafts[convo.id] = seedPrompt
        }
        selectedConversation = convo
    }

    private var activeModelName: String? {
        storedModels.first { $0.id.uuidString == appState.activeChatModelID }?.name
    }

    private var provisioningValidationKey: String {
        let records = storedModels
            .map { "\($0.id.uuidString)|\($0.repoId)|\($0.fileName)|\($0.localPath)|\($0.downloadedAt.timeIntervalSince1970)" }
            .sorted()
            .joined(separator: "\n")
        return [
            provisioningReceiptData?.base64EncodedString() ?? "none",
            appState.activeChatModelID ?? "none",
            appState.activeEmbeddingModelID ?? "none",
            records
        ].joined(separator: "\n")
    }

    private func refreshProvisioningPresentation() async {
        guard !LumenLaunchArguments.isUITesting else {
            showingOnboarding = false
            return
        }
        switch ModelProvisioningReceipt.status() {
        case .deferred:
            showingOnboarding = false
        case .completed:
            let valid = await ModelLaunchBootstrap.isProvisionedSelectionValid(
                appState: appState,
                context: modelContext
            )
            guard !Task.isCancelled else { return }
            if !valid {
                ModelProvisioningReceipt.invalidate()
            }
            showingOnboarding = !valid
        case .consented, .none:
            showingOnboarding = true
        }
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
