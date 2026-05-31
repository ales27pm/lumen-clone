import SwiftUI
import SwiftData

struct ConversationsSheet: View {
    @Binding var selected: Conversation?
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]
    @State private var search = ""

    var filtered: [Conversation] {
        guard !search.isEmpty else { return conversations }
        return conversations.filter { $0.title.localizedCaseInsensitiveContains(search) }
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(filtered) { convo in
                    Button {
                        selected = convo
                        dismiss()
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: convo.isPinned ? "pin.fill" : "bubble.left")
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(convo.title)
                                    .font(.body)
                                    .lineLimit(1)
                                Text(convo.preview)
                                    .font(.caption)
                                    .foregroundStyle(Theme.textSecondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                            Text(convo.updatedAt, style: .relative)
                                .font(.caption2)
                                .foregroundStyle(Theme.textTertiary)
                        }
                        .padding(.vertical, 2)
                    }
                    .swipeActions(edge: .leading) {
                        Button {
                            convo.isPinned.toggle()
                            try? modelContext.save()
                        } label: {
                            Label(convo.isPinned ? "Unpin" : "Pin", systemImage: "pin")
                        }
                        .tint(.orange)
                    }
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) {
                            modelContext.delete(convo)
                            try? modelContext.save()
                            if selected?.id == convo.id { selected = nil }
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
            }
            .searchable(text: $search, prompt: "Search chats")
            .navigationTitle("Conversations")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
