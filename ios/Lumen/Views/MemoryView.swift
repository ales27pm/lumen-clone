import SwiftUI
import SwiftData

struct MemoryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \MemoryItem.createdAt, order: .reverse) private var items: [MemoryItem]
    @State private var search = ""
    @State private var showAdd = false
    @State private var showWipeAlert = false
    @State private var showExport = false
    @State private var exportText = ""

    var filtered: [MemoryItem] {
        guard !search.isEmpty else { return items }
        return items.filter { $0.content.localizedCaseInsensitiveContains(search) }
    }

    var pinned: [MemoryItem] { filtered.filter { $0.isPinned } }
    var recent: [MemoryItem] { filtered.filter { !$0.isPinned }.prefix(20).map { $0 } }
    var byTopic: [(String, [MemoryItem])] {
        let rest = filtered.filter { !$0.isPinned }
        let grouped = Dictionary(grouping: rest) { item in
            let trimmedTopic = item.topic?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let trimmedTopic, !trimmedTopic.isEmpty {
                return trimmedTopic
            }
            return item.memoryKind.label
        }
        return grouped.sorted { $0.key < $1.key }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
                ScrollView {
                    VStack(spacing: 18) {
                        summaryRow
                        if filtered.isEmpty {
                            emptyState
                        } else {
                            if !pinned.isEmpty {
                                section(title: "Pinned", items: pinned)
                            }
                            section(title: search.isEmpty ? "Recent" : "Results", items: search.isEmpty ? recent : filtered)
                            if search.isEmpty {
                                ForEach(byTopic, id: \.0) { topic, list in
                                    section(title: topic, items: list)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                    .padding(.bottom, 40)
                }
            }
            .searchable(text: $search, prompt: "Search memory")
            .navigationTitle("Memory")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button { showAdd = true } label: { Label("Add memory", systemImage: "plus") }
                        Button {
                            exportText = MemoryStore.exportJSON(context: modelContext)
                            showExport = true
                        } label: { Label("Export JSON", systemImage: "square.and.arrow.up") }
                        Button(role: .destructive) { showWipeAlert = true } label: { Label("Wipe all", systemImage: "trash") }
                    } label: {
                        Image(systemName: "ellipsis")
                    }
                }
            }
            .sheet(isPresented: $showAdd) {
                AddMemorySheet()
                    .presentationDetents([.medium])
            }
            .sheet(isPresented: $showExport) {
                NavigationStack {
                    ScrollView {
                        Text(exportText)
                            .font(.caption.monospaced())
                            .foregroundStyle(Theme.textSecondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .textSelection(.enabled)
                    }
                    .navigationTitle("Export")
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Copy") {
                                UIPasteboard.general.string = exportText
                            }
                        }
                    }
                }
            }
            .alert("Wipe all memory?", isPresented: $showWipeAlert) {
                Button("Wipe", role: .destructive) {
                    try? MemoryStore.wipeEverything(context: modelContext)
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This permanently deletes every memory, including pinned items.")
            }
        }
    }

    private func section(title: String, items: [MemoryItem]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
                .padding(.leading, 2)
            VStack(spacing: 8) {
                ForEach(items) { item in
                    MemoryRow(item: item,
                              onDelete: { delete(item) },
                              onTogglePin: { togglePin(item) })
                }
            }
        }
    }

    private var summaryRow: some View {
        HStack(spacing: 12) {
            Image(systemName: "brain")
                .font(.body)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 28, height: 28)
            VStack(alignment: .leading, spacing: 1) {
                Text("\(items.count) memories")
                    .font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                Text("Vectorized locally in SQLite")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer()
            if !items.filter(\.isPinned).isEmpty {
                Text("\(items.filter(\.isPinned).count) pinned")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(12)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "tray")
                .font(.title)
                .foregroundStyle(Theme.textTertiary)
            Text("No memories yet")
                .font(.body)
                .foregroundStyle(Theme.textPrimary)
            Text("Chat with Lumen and facts will appear here automatically.")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(40)
    }

    private func delete(_ item: MemoryItem) {
        modelContext.delete(item)
        try? modelContext.save()
    }

    private func togglePin(_ item: MemoryItem) {
        item.isPinned.toggle()
        try? modelContext.save()
        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
    }
}

struct MemoryRow: View {
    @Bindable var item: MemoryItem
    var onDelete: () -> Void
    var onTogglePin: () -> Void
    @State private var isEditing = false

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: item.memoryKind.icon)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 22, height: 22)
                .padding(.top, 2)
            VStack(alignment: .leading, spacing: 4) {
                if isEditing {
                    TextField("Memory", text: $item.content, axis: .vertical)
                        .font(.footnote)
                        .foregroundStyle(Theme.textPrimary)
                        .textFieldStyle(.plain)
                } else {
                    Text(item.content)
                        .font(.footnote)
                        .foregroundStyle(Theme.textPrimary)
                }
                HStack(spacing: 6) {
                    Text(item.memoryKind.label)
                        .font(.caption2)
                        .foregroundStyle(Theme.textSecondary)
                    Text("·")
                        .font(.caption2)
                        .foregroundStyle(Theme.textTertiary)
                    if item.isPinned {
                        Image(systemName: "pin.fill")
                            .font(.caption2)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    Text(item.createdAt, style: .relative)
                        .font(.caption2).foregroundStyle(Theme.textTertiary)
                }
            }
            Spacer(minLength: 0)
            Menu {
                Button { isEditing.toggle() } label: {
                    Label(isEditing ? "Done" : "Edit", systemImage: isEditing ? "checkmark" : "pencil")
                }
                Button { onTogglePin() } label: {
                    Label(item.isPinned ? "Unpin" : "Pin", systemImage: item.isPinned ? "pin.slash" : "pin")
                }
                Button(role: .destructive, action: onDelete) {
                    Label("Delete", systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis")
                    .foregroundStyle(Theme.textTertiary)
                    .padding(6)
            }
        }
        .padding(12)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }
}

struct AddMemorySheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @State private var content: String = ""
    @State private var kind: MemoryKind = .fact
    @State private var topic: String = ""
    @State private var pinned: Bool = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Content") {
                    TextField("What should I remember?", text: $content, axis: .vertical)
                        .lineLimit(3...6)
                }
                Section("Type") {
                    Picker("Kind", selection: $kind) {
                        ForEach(MemoryKind.allCases, id: \.self) { k in
                            Text(k.label).tag(k)
                        }
                    }
                    .pickerStyle(.menu)
                }
                Section("Topic (optional)") {
                    TextField("e.g. projects, people", text: $topic)
                }
                Section {
                    Toggle("Pin this memory", isOn: $pinned)
                }
                Section {
                    Button("Save") {
                        Task {
                            try? await MemoryStore.remember(
                                content,
                                kind: kind,
                                source: "manual",
                                topic: topic.isEmpty ? nil : topic,
                                context: modelContext
                            )
                            if pinned, let saved = try? modelContext.fetch(FetchDescriptor<MemoryItem>()).first(where: { $0.content == content.trimmingCharacters(in: .whitespaces) }) {
                                saved.isPinned = true
                                try? modelContext.save()
                            }
                            dismiss()
                        }
                    }
                    .disabled(content.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .navigationTitle("Add Memory")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
