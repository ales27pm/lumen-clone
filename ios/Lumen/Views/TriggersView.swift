import SwiftUI
import SwiftData

struct TriggersView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Trigger.createdAt, order: .reverse) private var triggers: [Trigger]
    @State private var showEditor = false
    @State private var editing: Trigger?
    @State private var persistenceFailure: TriggerPersistenceFailure?
    @State private var pendingRetry: PendingTriggerMutation?
    @State private var autonomousExecutionSuspended = false
    @State private var successfullyDeletedTriggerIDs: Set<UUID> = []

    private enum PendingTriggerMutation {
        case setPaused(Trigger, Bool)
        case delete(Trigger)
        case persistRun(Trigger)
        case resolveSuspension
    }

    private var visibleTriggers: [Trigger] {
        triggers.filter { !successfullyDeletedTriggerIDs.contains($0.id) }
    }

    private var nextUp: [Trigger] {
        visibleTriggers.filter { !$0.isPaused && ($0.nextFireAt ?? $0.computeNextFire()) != nil }
            .sorted { ($0.nextFireAt ?? .distantFuture) < ($1.nextFireAt ?? .distantFuture) }
    }
    private var paused: [Trigger] { visibleTriggers.filter(\.isPaused) }

    var body: some View {
        NavigationStack {
            ZStack {
                AppBackground()
                ScrollView {
                    VStack(spacing: 18) {
                        headerRow
                        if autonomousExecutionSuspended {
                            persistenceSafetyBanner
                        }
                        if visibleTriggers.isEmpty {
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
                autonomousExecutionSuspended = TriggerScheduler.shared.isAutonomousExecutionSuspended
                TriggerScheduler.shared.refreshNextFireTimes(context: modelContext)
            }
            .alert(item: $persistenceFailure) { failure in
                Alert(
                    title: Text(failure.alertTitle),
                    message: Text(failure.userMessage),
                    primaryButton: .default(Text("Retry")) {
                        retryPendingMutation()
                    },
                    secondaryButton: .cancel(Text("Dismiss")) {
                        pendingRetry = nil
                    }
                )
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
                Text("\(visibleTriggers.count) triggers")
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

    private var persistenceSafetyBanner: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.shield.fill")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 3) {
                Text("Automatic runs suspended")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                Text("A trigger change could not be saved. Choose the intended action again to retry safely.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Button("Resolve safely") {
                    resolveAutonomousExecutionSuspension()
                }
                .font(.caption.weight(.semibold))
                .buttonStyle(.bordered)
                .accessibilityIdentifier("triggers.resolvePersistenceSafetyInterlock")
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.orange.opacity(0.55), lineWidth: 1)
        }
        .accessibilityIdentifier("triggers.persistenceSafetyInterlock")
    }

    private func runNow(_ t: Trigger) {
        Task {
            let outcome = await TriggerScheduler.shared.runTriggerWithPersistenceOutcome(
                t,
                context: modelContext,
                appState: appState,
                notify: false
            )
            if case .persistenceFailed(let failure) = outcome {
                autonomousExecutionSuspended = true
                pendingRetry = .persistRun(t)
                persistenceFailure = failure
            }
        }
    }

    private func togglePause(_ t: Trigger) {
        setPaused(t, paused: !t.isPaused)
    }

    private func setPaused(_ trigger: Trigger, paused: Bool) {
        let operation: TriggerPersistenceOperation = paused ? .pause : .resume
        let safetyToken = TriggerScheduler.persistenceSafetyToken(operation: operation, triggerID: trigger.id)
        let previousPaused = trigger.isPaused
        let previousNextFireAt = trigger.nextFireAt
        trigger.isPaused = paused
        trigger.nextFireAt = paused ? nil : trigger.computeNextFire()

        let outcome = TriggerPersistenceCoordinator.attempt(
            operation: operation,
            save: { try modelContext.save() },
            restore: {
                trigger.isPaused = previousPaused
                trigger.nextFireAt = previousNextFireAt
            },
            onSaved: { resumeAndScheduleAutonomousExecution(triggerID: trigger.id) },
            onFailure: { suspendAutonomousExecution(safetyToken: safetyToken) }
        )
        handle(outcome, retry: .setPaused(trigger, paused))
    }

    private func delete(_ t: Trigger) {
        let safetyToken = TriggerScheduler.persistenceSafetyToken(operation: .delete, triggerID: t.id)
        let outcome = TriggerPersistenceCoordinator.delete(
            triggerID: t.id,
            container: modelContext.container,
            onSaved: {
                successfullyDeletedTriggerIDs.insert(t.id)
                resumeAndScheduleAutonomousExecution(triggerID: t.id)
            },
            onFailure: { suspendAutonomousExecution(safetyToken: safetyToken) }
        )
        handle(outcome, retry: .delete(t))
    }

    private func handle(_ outcome: TriggerPersistenceCoordinator.Outcome, retry: PendingTriggerMutation) {
        switch outcome {
        case .saved:
            pendingRetry = nil
            persistenceFailure = nil
        case .failed(let failure):
            autonomousExecutionSuspended = true
            pendingRetry = retry
            persistenceFailure = failure
        }
    }

    private func retryPendingMutation() {
        guard let retry = pendingRetry else { return }
        pendingRetry = nil
        switch retry {
        case .setPaused(let trigger, let paused):
            setPaused(trigger, paused: paused)
        case .delete(let trigger):
            delete(trigger)
        case .persistRun(let trigger):
            persistPendingRun(trigger)
        case .resolveSuspension:
            resolveAutonomousExecutionSuspension()
        }
    }

    private func persistPendingRun(_ trigger: Trigger) {
        let safetyToken = TriggerScheduler.persistenceSafetyToken(operation: .run, triggerID: trigger.id)
        let outcome = TriggerPersistenceCoordinator.attempt(
            operation: .run,
            save: { try modelContext.save() },
            restore: {},
            onSaved: { resumeAndScheduleAutonomousExecution(triggerID: trigger.id) },
            onFailure: { suspendAutonomousExecution(safetyToken: safetyToken) }
        )
        handle(outcome, retry: .persistRun(trigger))
    }

    private func resumeAndScheduleAutonomousExecution(triggerID: UUID) {
        TriggerScheduler.shared.resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: triggerID)
        autonomousExecutionSuspended = TriggerScheduler.shared.isAutonomousExecutionSuspended
        TriggerScheduler.shared.scheduleBackgroundRefresh()
    }

    private func resolveAutonomousExecutionSuspension() {
        let outcome = TriggerScheduler.shared.resolveAutonomousExecutionSuspension(container: modelContext.container)
        autonomousExecutionSuspended = TriggerScheduler.shared.isAutonomousExecutionSuspended
        handle(outcome, retry: .resolveSuspension)
    }

    private func suspendAutonomousExecution(safetyToken: String) -> Bool {
        TriggerScheduler.shared.suspendAutonomousExecutionAfterPersistenceFailure(token: safetyToken)
        autonomousExecutionSuspended = true
        return true
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
    @State private var persistenceFailure: TriggerPersistenceFailure?

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
            .alert(item: $persistenceFailure) { failure in
                Alert(
                    title: Text(failure.alertTitle),
                    message: Text(failure.userMessage),
                    primaryButton: .default(Text("Retry")) { save() },
                    secondaryButton: .cancel(Text("Dismiss"))
                )
            }
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
        let previousSnapshot: TriggerMutationSnapshot?
        let safetyToken: String?
        if let e = existing {
            target = e
            previousSnapshot = TriggerMutationSnapshot(trigger: e)
            safetyToken = TriggerScheduler.persistenceSafetyToken(operation: .update, triggerID: e.id)
            target.title = title
            target.prompt = prompt
            target.scheduleType = kind.rawValue
        } else {
            target = Trigger(title: title, prompt: prompt, scheduleType: kind)
            previousSnapshot = nil
            safetyToken = nil
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

        let outcome = TriggerPersistenceCoordinator.attempt(
            operation: existing == nil ? .create : .update,
            save: { try modelContext.save() },
            restore: {
                if let previousSnapshot {
                    previousSnapshot.restore(target)
                } else {
                    modelContext.delete(target)
                }
            },
            onSaved: {
                TriggerScheduler.shared.resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: target.id)
            },
            onFailure: {
                guard let safetyToken else {
                    return TriggerScheduler.shared.isAutonomousExecutionSuspended
                }
                TriggerScheduler.shared.suspendAutonomousExecutionAfterPersistenceFailure(token: safetyToken)
                return true
            }
        )
        switch outcome {
        case .saved:
            persistenceFailure = nil
            Task {
                await TriggerScheduler.shared.requestPermission()
                TriggerScheduler.shared.scheduleBackgroundRefresh()
            }
            dismiss()
        case .failed(let failure):
            persistenceFailure = failure
        }
    }
}

private struct TriggerMutationSnapshot {
    let title: String
    let prompt: String
    let scheduleType: String
    let fireDate: Date?
    let timeOfDayMinutes: Int?
    let intervalSeconds: TimeInterval?
    let beforeNextEventMinutes: Int?
    let isPaused: Bool
    let nextFireAt: Date?

    init(trigger: Trigger) {
        title = trigger.title
        prompt = trigger.prompt
        scheduleType = trigger.scheduleType
        fireDate = trigger.fireDate
        timeOfDayMinutes = trigger.timeOfDayMinutes
        intervalSeconds = trigger.intervalSeconds
        beforeNextEventMinutes = trigger.beforeNextEventMinutes
        isPaused = trigger.isPaused
        nextFireAt = trigger.nextFireAt
    }

    func restore(_ trigger: Trigger) {
        trigger.title = title
        trigger.prompt = prompt
        trigger.scheduleType = scheduleType
        trigger.fireDate = fireDate
        trigger.timeOfDayMinutes = timeOfDayMinutes
        trigger.intervalSeconds = intervalSeconds
        trigger.beforeNextEventMinutes = beforeNextEventMinutes
        trigger.isPaused = isPaused
        trigger.nextFireAt = nextFireAt
    }
}
