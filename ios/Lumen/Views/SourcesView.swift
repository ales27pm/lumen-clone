import SwiftUI
import SwiftData
import UniformTypeIdentifiers

struct SourcesView: View {
    private enum ReindexTarget {
        case files
        case photos

        var confirmationTitle: String {
            switch self {
            case .files: "Reindex imported files?"
            case .photos: "Reindex photo metadata?"
            }
        }

        var confirmationMessage: String {
            switch self {
            case .files:
                "This will clear and rebuild the existing local file index from your imported files."
            case .photos:
                "This will clear and rebuild the existing local photo metadata index for the last six months."
            }
        }

        var confirmationButtonTitle: String {
            switch self {
            case .files: "Reindex Files"
            case .photos: "Reindex Photos"
            }
        }
    }

    @Environment(\.modelContext) private var modelContext
    @Query(
        filter: #Predicate<RAGChunk> {
            !$0.sourceType.starts(with: "__lumen_rag_replacement_staging__:")
        },
        sort: \RAGChunk.createdAt,
        order: .reverse
    ) private var chunks: [RAGChunk]
    @State private var showFilePicker = false
    @State private var showNoteSheet = false
    @State private var pendingReindex: ReindexTarget?
    @State private var busy = false
    @State private var status: String?

    private var counts: [RAGSourceType: Int] {
        var out: [RAGSourceType: Int] = [:]
        for c in chunks { out[c.kind, default: 0] += 1 }
        return out
    }

    var body: some View {
        NavigationStack {
            ZStack {
                AppBackground()
                ScrollView {
                    VStack(spacing: 18) {
                        headerRow
                        VStack(spacing: 8) {
                            ForEach(RAGSourceType.allCases, id: \.self) { type in
                                NavigationLink {
                                    SourceDetailView(type: type)
                                } label: {
                                    sourceRow(type: type, count: counts[type] ?? 0)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        if let status {
                            Text(status)
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                    .padding(.bottom, 40)
                }
            }
            .navigationTitle("Sources")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button { showFilePicker = true } label: { Label("Import file", systemImage: "doc.badge.plus") }
                        Button { showNoteSheet = true } label: { Label("Add note", systemImage: "note.text.badge.plus") }
                        Button { pendingReindex = .files } label: { Label("Reindex files", systemImage: "arrow.clockwise") }
                        Button { pendingReindex = .photos } label: { Label("Reindex photos (6mo)", systemImage: "photo.stack") }
                        Divider()
                        Button(role: .destructive) {
                            wipeIndex()
                        } label: {
                            Label("Wipe index", systemImage: "trash")
                        }
                    } label: {
                        Image(systemName: "ellipsis")
                    }
                }
            }
            .fileImporter(isPresented: $showFilePicker,
                          allowedContentTypes: [.plainText, .pdf, .text, .utf8PlainText],
                          allowsMultipleSelection: true) { result in
                if case .success(let urls) = result {
                    importFiles(urls: urls)
                }
            }
            .sheet(isPresented: $showNoteSheet) {
                AddNoteSheet()
                    .presentationDetents([.medium, .large])
            }
            .confirmationDialog(
                pendingReindex?.confirmationTitle ?? "Reindex local sources?",
                isPresented: Binding(
                    get: { pendingReindex != nil },
                    set: { isPresented in
                        if !isPresented { pendingReindex = nil }
                    }
                ),
                titleVisibility: .visible
            ) {
                if let target = pendingReindex {
                    Button(target.confirmationButtonTitle, role: .destructive) {
                        pendingReindex = nil
                        confirmReindex(target)
                    }
                }
                Button("Cancel", role: .cancel) {
                    pendingReindex = nil
                }
            } message: {
                if let target = pendingReindex {
                    Text(target.confirmationMessage)
                }
            }
            .overlay {
                if busy {
                    ZStack {
                        Color.black.opacity(0.4).ignoresSafeArea()
                        ProgressView("Indexing…")
                            .padding(24)
                            .background(Theme.surface)
                            .clipShape(.rect(cornerRadius: 12))
                    }
                }
            }
        }
    }

    private var headerRow: some View {
        HStack(spacing: 12) {
            Image(systemName: "externaldrive.badge.plus")
                .font(.body)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 28, height: 28)
            VStack(alignment: .leading, spacing: 1) {
                Text("\(chunks.count) chunks indexed")
                    .font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                Text("Embeddings stored locally in SQLite")
                    .font(.caption).foregroundStyle(Theme.textSecondary)
            }
            Spacer()
        }
        .padding(12)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }

    private func sourceRow(type: RAGSourceType, count: Int) -> some View {
        HStack(spacing: 10) {
            Image(systemName: type.icon)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(type.label).font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                Text("\(count) chunks").font(.caption).foregroundStyle(Theme.textSecondary)
            }
            Spacer()
            Image(systemName: "chevron.right")
                .font(.caption).foregroundStyle(Theme.textTertiary)
        }
        .padding(12)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }

    private func wipeIndex() {
        do {
            try RAGStore.wipe(nil, context: modelContext)
            status = "Wiped the full local index."
        } catch {
            status = "Could not wipe index: \(error.localizedDescription)"
        }
    }

    private func importFiles(urls: [URL]) {
        Task {
            busy = true; defer { busy = false }
            var total = 0
            var failures: [String] = []
            for u in urls {
                let imported = FileStore.importFileWithDiagnostics(from: u)
                guard let dest = imported.url else {
                    failures.append("import failed: \(imported.diagnostic ?? "import_failed")")
                    continue
                }
                let result = await RAGStore.indexFileWithDiagnostics(url: dest, context: modelContext)
                total += result.indexedCount
                if result.didIndexAllChunks == false, let diagnostic = result.diagnostic {
                    failures.append("\(dest.lastPathComponent): \(diagnostic)")
                }
            }
            if failures.isEmpty {
                status = "Indexed \(total) new chunks from \(urls.count) file(s)."
            } else {
                status = "Indexed \(total) chunks; \(failures.count) file(s) degraded or failed. \(failures.prefix(2).joined(separator: "; "))"
            }
        }
    }

    private func reindexFiles() {
        Task {
            busy = true; defer { busy = false }
            guard let container = SharedContainer.shared else {
                status = "RAG storage unavailable. Diagnostic: swiftdata_shared_container_unavailable."
                return
            }
            // Bulk replacement may roll back its transaction on failure. Keep it
            // isolated from unrelated unsaved edits owned by this view context.
            let result = await RAGStore.indexImportedFilesWithDiagnostics(context: ModelContext(container))
            status = MemoryTools.ragIndexFilesMessage(from: result)
        }
    }

    private func reindexPhotos() {
        Task {
            busy = true; defer { busy = false }
            guard let container = SharedContainer.shared else {
                status = "RAG storage unavailable. Diagnostic: swiftdata_shared_container_unavailable."
                return
            }
            let result = await RAGStore.indexPhotosWithDiagnostics(
                monthsBack: 6,
                context: ModelContext(container)
            )
            status = MemoryTools.ragIndexPhotosMessage(from: result)
        }
    }

    private func confirmReindex(_ target: ReindexTarget) {
        switch target {
        case .files:
            reindexFiles()
        case .photos:
            reindexPhotos()
        }
    }
}

struct SourceDetailView: View {
    let type: RAGSourceType
    @Environment(\.modelContext) private var modelContext
    @State private var items: [RAGChunk] = []
    @State private var status: String?

    var body: some View {
        ZStack {
            AppBackground()
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(items) { chunk in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(chunk.sourceName)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(Theme.textPrimary)
                            Text(chunk.content)
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                                .lineLimit(6)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .background(Theme.surface)
                        .clipShape(.rect(cornerRadius: 10))
                        .overlay {
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(Theme.border, lineWidth: 1)
                        }
                    }
                    if let status {
                        Text(status)
                            .font(.footnote)
                            .foregroundStyle(Theme.textSecondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.top, 8)
                    }
                    if items.isEmpty {
                        Text("No \(type.label.lowercased()) indexed yet.")
                            .font(.footnote)
                            .foregroundStyle(Theme.textSecondary)
                            .padding(.top, 40)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .padding(.bottom, 40)
            }
        }
        .navigationTitle(type.label)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(role: .destructive) {
                    wipeCurrentSourceType()
                } label: {
                    Image(systemName: "trash")
                }
            }
        }
        .onAppear(perform: reload)
    }

    private func wipeCurrentSourceType() {
        do {
            try RAGStore.wipe(type, context: modelContext)
            reload()
            status = "Wiped \(type.label.lowercased()) chunks."
        } catch {
            status = "Could not wipe \(type.label.lowercased()): \(error.localizedDescription)"
        }
    }

    private func reload() {
        let result = RAGStore.chunksWithDiagnostics(for: type, context: modelContext)
        items = result.chunks
        if result.mode == "failed", let diagnostic = result.diagnostic {
            status = "Could not load \(type.label.lowercased()) chunks. Diagnostic: \(diagnostic)."
        }
    }
}

struct AddNoteSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @State private var title = ""
    @State private var body_ = ""
    @State private var saving = false
    @State private var status: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Title") {
                    TextField("Note title", text: $title)
                }
                Section("Content") {
                    TextEditor(text: $body_)
                        .frame(minHeight: 200)
                        .font(.footnote)
                }
                Section {
                    Button("Save & index") { save() }
                        .disabled(title.isEmpty || body_.isEmpty || saving)
                    if let status {
                        Text(status)
                            .font(.footnote)
                            .foregroundStyle(Theme.textSecondary)
                    }
                }
            }
            .navigationTitle("Add Note")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private func save() {
        Task {
            saving = true
            let result = await RAGStore.indexNoteWithDiagnostics(title: title, body: body_, context: modelContext)
            saving = false
            if result.mode == .indexed || result.mode == .partial {
                dismiss()
            } else {
                status = "Could not index note. Diagnostic: \(MemoryTools.diagnosticText(result.diagnostic))."
            }
        }
    }
}
