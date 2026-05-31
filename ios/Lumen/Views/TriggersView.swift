import SwiftUI
import SwiftData

struct TriggersView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Trigger.createdAt, order: .reverse) private var triggers: [Trigger]
    @State private var showEditor = false
    @State private var editing: Trigger?

    private var nextUp: [Trigger] {
        triggers.filter { !$0.isPaused && ($0.nextFireAt ?? $0.computeNextFire()) != nil }
            .sorted { ($0.nextFireAt ?? .distantFuture) < ($1.nextFireAt ?? .distantFuture) }
    }
    private var paused: [Trigger] { triggers.filter(\.isPaused) }

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
                ScrollView {
                    VStack(spacing: 18) {
                        headerRow
                        if triggers.isEmpty {
                            emptyState
                        } else {
                            if !nextUp.isEmpty { section("Next up", items: nextUp) }
                            if !paused.isEmpty { section("Paused", items: paused) }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                    .padding(.bottom, 40)
                }
            }
            .navigationTitle("Triggers")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { editing = nil; showEditor = true } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showEditor) {
                TriggerEditorSheet(existing: editing)
                    .presentationDetents([.large])
            }
            .onAppear {
                TriggerScheduler.shared.refreshNextFireTimes(context: modelContext)
            }
        }
    }

    private func section(_ title: String, items: [Trigger]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
                .padding(.leading, 2)
            VStack(spacing: 8) {
                ForEach(items) { t in
                    TriggerRow(trigger: t,
                               onRun: { runNow(t) },
                               onTogglePause: { togglePause(t) },
                               onEdit: { editing = t; showEditor = true },
                               onDelete: { delete(t) })
                }
            }
        }
    }

    private var headerRow: some View {
        HStack(spacing: 12) {
            Image(systemName: "bolt.horizontal")
                .font(.body)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 28, height: 28)
            VStack(alignment: .leading, spacing: 1) {
                Text("\(triggers.count) triggers")
                    .font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                Text("Agent runs in the background and notifies you")
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

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "alarm")
                .font(.title)
                .foregroundStyle(Theme.textTertiary)
            Text("No triggers yet").font(.body).foregroundStyle(Theme.textPrimary)
            Text("Schedule the agent to run on a timer or at a specific time.")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(Theme.textSecondary)
            Button {
                editing = nil; showEditor = true
            } label: {
                Text("New trigger")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(Theme.accent)
                    .clipShape(.rect(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            .padding(.top, 6)
        }
        .padding(32)
    }

    private func runNow(_ t: Trigger) {
        Task {
            _ = await TriggerScheduler.shared.runTrigger(t, context: modelContext, appState: appState, notify: false)
            try? modelContext.save()
        }
    }
    private func togglePause(_ t: Trigger) {
        t.isPaused.toggle()
        t.nextFireAt = t.isPaused ? nil : t.computeNextFire()
        try? modelContext.save()
    }
    private func delete(_ t: Trigger) {
        modelContext.delete(t)
        try? modelContext.save()
    }
}

struct TriggerRow: View {
    @Bindable var trigger: Trigger
    var onRun: () -> Void
    var onTogglePause: () -> Void
    var onEdit: () -> Void
    var onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: trigger.kind.icon)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .frame(width: 22, height: 22)
                    .padding(.top, 2)
                VStack(alignment: .leading, spacing: 3) {
                    Text(trigger.title)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(Theme.textPrimary)
                    Text(trigger.prompt)
                        .font(.footnote)
                        .foregroundStyle(Theme.textSecondary)
                        .lineLimit(2)
                    HStack(spacing: 6) {
                        Text(trigger.kind.label)
                            .font(.caption2).foregroundStyle(Theme.textSecondary)
                        Text("·").font(.caption2).foregroundStyle(Theme.textTertiary)
                        Text(nextLabel)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(Theme.textSecondary)
                        if trigger.isPaused {
                            Text("·").font(.caption2).foregroundStyle(Theme.textTertiary)
                            Text("paused").font(.caption2).foregroundStyle(.orange)
                        }
                    }
                }
                Spacer(minLength: 0)
                Menu {
                    Button(action: onRun) { Label("Run now", systemImage: "play.fill") }
                    Button(action: onTogglePause) {
                        Label(trigger.isPaused ? "Resume" : "Pause", systemImage: trigger.isPaused ? "play" : "pause")
                    }
                    Button(action: onEdit) { Label("Edit", systemImage: "pencil") }
                    Button(role: .destructive, action: onDelete) { Label("Delete", systemImage: "trash") }
                } label: {
                    Image(systemName: "ellipsis")
                        .foregroundStyle(Theme.textTertiary)
                        .padding(6)
                }
            }
            if let last = trigger.lastResult, !last.isEmpty {
                Text(last)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Theme.surfaceHigh)
                    .clipShape(.rect(cornerRadius: 8))
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

    private var nextLabel: String {
        if trigger.isPaused { return "paused" }
        if let d = trigger.nextFireAt ?? trigger.computeNextFire() {
            return d.formatted(date: .abbreviated, time: .shortened)
        }
        return "—"
    }
}

struct TriggerEditorSheet: View {
    var existing: Trigger?
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext

    @State private var title: String = ""
    @State private var prompt: String = ""
    @State private var kind: TriggerScheduleType = .once
    @State private var onceDate: Date = Date().addingTimeInterval(3600)
    @State private var dailyTime: Date = {
        Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: Date()) ?? Date()
    }()
    @State private var intervalMinutes: Int = 60
    @State private var beforeMinutes: Int = 15

    var body: some View {
        NavigationStack {
            Form {
                Section("What should Lumen do?") {
                    TextField("Title", text: $title)
                    TextField("Prompt", text: $prompt, axis: .vertical)
                        .lineLimit(3...6)
                }

                Section("Schedule") {
                    Picker("Type", selection: $kind) {
                        ForEach(TriggerScheduleType.allCases, id: \.self) { k in
                            Label(k.label, systemImage: k.icon).tag(k)
                        }
                    }
                    switch kind {
                    case .once:
                        DatePicker("Fire at", selection: $onceDate)
                    case .daily:
                        DatePicker("Time", selection: $dailyTime, displayedComponents: .hourAndMinute)
                    case .interval:
                        Stepper(value: $intervalMinutes, in: 15...1440, step: 15) {
                            HStack { Text("Every"); Spacer(); Text("\(intervalMinutes) min").foregroundStyle(Theme.textSecondary) }
                        }
                    case .beforeNextEvent:
                        Stepper(value: $beforeMinutes, in: 5...120, step: 5) {
                            HStack { Text("Minutes before"); Spacer(); Text("\(beforeMinutes)").foregroundStyle(Theme.textSecondary) }
                        }
                    }
                }

                Section {
                    Button("Save") { save() }
                        .disabled(title.trimmingCharacters(in: .whitespaces).isEmpty ||
                                  prompt.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .navigationTitle(existing == nil ? "New Trigger" : "Edit Trigger")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onAppear(perform: load)
        }
    }

    private func load() {
        guard let e = existing else { return }
        title = e.title
        prompt = e.prompt
        kind = e.kind
        if let d = e.fireDate { onceDate = d }
        if let m = e.timeOfDayMinutes {
            dailyTime = Calendar.current.date(bySettingHour: m / 60, minute: m % 60, second: 0, of: Date()) ?? Date()
        }
        if let s = e.intervalSeconds { intervalMinutes = max(15, Int(s / 60)) }
        if let b = e.beforeNextEventMinutes { beforeMinutes = b }
    }

    private func save() {
        let target: Trigger
        if let e = existing {
            target = e
            target.title = title
            target.prompt = prompt
            target.scheduleType = kind.rawValue
        } else {
            target = Trigger(title: title, prompt: prompt, scheduleType: kind)
            modelContext.insert(target)
        }

        switch kind {
        case .once:
            target.fireDate = onceDate
            target.timeOfDayMinutes = nil
            target.intervalSeconds = nil
            target.beforeNextEventMinutes = nil
        case .daily:
            let comps = Calendar.current.dateComponents([.hour, .minute], from: dailyTime)
            target.timeOfDayMinutes = (comps.hour ?? 9) * 60 + (comps.minute ?? 0)
            target.fireDate = nil
            target.intervalSeconds = nil
            target.beforeNextEventMinutes = nil
        case .interval:
            target.intervalSeconds = TimeInterval(intervalMinutes * 60)
            target.fireDate = nil
            target.timeOfDayMinutes = nil
            target.beforeNextEventMinutes = nil
        case .beforeNextEvent:
            target.beforeNextEventMinutes = beforeMinutes
            target.fireDate = nil
            target.timeOfDayMinutes = nil
            target.intervalSeconds = nil
        }
        target.isPaused = false
        target.nextFireAt = target.computeNextFire()
        try? modelContext.save()
        Task {
            await TriggerScheduler.shared.requestPermission()
            TriggerScheduler.shared.scheduleBackgroundRefresh()
        }
        dismiss()
    }
}
