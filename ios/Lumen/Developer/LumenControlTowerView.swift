import SwiftUI

struct LumenControlTowerView: View {
    @State private var model = LumenControlTowerModel()

    var body: some View {
        List {
            Section("Live Workflow") {
                HStack {
                    Label(model.monitoringEnabled ? "Monitoring active" : "Monitoring paused", systemImage: model.monitoringEnabled ? "dot.radiowaves.left.and.right" : "pause.circle")
                    Spacer()
                    Button(model.monitoringEnabled ? "Pause" : "Start") {
                        model.toggleMonitoring()
                    }
                }
                Button {
                    model.refresh()
                } label: {
                    Label("Refresh snapshot", systemImage: "arrow.clockwise")
                }
                Button(role: .destructive) {
                    model.reset()
                } label: {
                    Label("Reset workflow buffer", systemImage: "trash")
                }
            }

            Section("Cockpit") {
                LabeledContent("Events", value: "\(model.snapshot.events.count)")
                LabeledContent("Touched slots", value: model.snapshot.touchedSlots.joined(separator: " / "))
                LabeledContent("Fallbacks", value: "\(model.snapshot.fallbackCount)")
                LabeledContent("Errors", value: "\(model.snapshot.errorCount)")
                LabeledContent("Scenario bank", value: "\(model.coverage.scenarioCount) scenarios / \(model.coverage.toolCount) tools")
                if !model.coverage.missingToolIDs.isEmpty {
                    Text("Tools below minimum scenario coverage: \(model.coverage.missingToolIDs.prefix(8).joined(separator: ", "))")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }

            Section("Adapter Slots") {
                ForEach(AgentWorkflowSlot.allCases) { slot in
                    AgentWorkflowSlotRow(
                        slot: slot,
                        latest: model.snapshot.lastEventBySlot[slot.rawValue],
                        completedCount: model.snapshot.completedCountBySlot[slot.rawValue] ?? 0,
                        totalDurationMs: model.snapshot.totalDurationMsBySlot[slot.rawValue] ?? 0
                    )
                }
            }

            Section("Latest Timeline") {
                ForEach(model.snapshot.events.suffix(30).reversed()) { event in
                    AgentWorkflowEventRow(event: event)
                }
            }

            Section("Tool Scenario Bank") {
                ForEach(model.scenarioPreview) { entry in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(entry.toolID)
                                .font(.caption.monospaced().weight(.semibold))
                            Spacer()
                            Text(entry.kind.rawValue)
                                .font(.caption2.monospaced())
                                .foregroundStyle(Theme.textSecondary)
                        }
                        Text(entry.prompt)
                            .font(.caption)
                        Text("slots: \(entry.requiredSlots.map(\.rawValue).joined(separator: " → "))")
                            .font(.caption2.monospaced())
                            .foregroundStyle(Theme.textSecondary)
                    }
                    .padding(.vertical, 2)
                }
            }

            Section("Unified Report") {
                NavigationLink {
                    DeveloperConsoleTextView(title: "Control Tower JSON", bodyText: model.reportText())
                } label: {
                    Label("Open JSON snapshot", systemImage: "doc.text.magnifyingglass")
                }
            }
        }
        .navigationTitle("Control Tower")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            model.startIfNeeded()
            model.refresh()
        }
    }
}

@MainActor
@Observable
final class LumenControlTowerModel {
    var monitoringEnabled = false
    var snapshot = AgentWorkflowMonitor.shared.snapshot()
    var coverage = ToolScenarioBank.coverageSummary()
    var scenarioPreview: [ToolScenarioBankEntry] = Array(ToolScenarioBank.entries().prefix(24))

    func startIfNeeded() {
        _ = AgentWorkflowMonitor.shared.start()
        monitoringEnabled = AgentWorkflowMonitor.shared.isMonitoring
        refresh()
    }

    func toggleMonitoring() {
        if AgentWorkflowMonitor.shared.isMonitoring {
            AgentWorkflowMonitor.shared.stop()
        } else {
            _ = AgentWorkflowMonitor.shared.start()
        }
        monitoringEnabled = AgentWorkflowMonitor.shared.isMonitoring
        refresh()
    }

    func refresh() {
        monitoringEnabled = AgentWorkflowMonitor.shared.isMonitoring
        snapshot = AgentWorkflowMonitor.shared.snapshot()
        coverage = ToolScenarioBank.coverageSummary()
        scenarioPreview = Array(ToolScenarioBank.entries().prefix(24))
    }

    func reset() {
        AgentWorkflowMonitor.shared.reset()
        refresh()
    }

    func reportText() -> String {
        do {
            let data = try AgentWorkflowMonitor.shared.jsonReportData()
            return String(decoding: data, as: UTF8.self)
        } catch {
            return "Control Tower report encoding failed: \(error.localizedDescription)"
        }
    }
}

private struct AgentWorkflowSlotRow: View {
    let slot: AgentWorkflowSlot
    let latest: AgentWorkflowEvent?
    let completedCount: Int
    let totalDurationMs: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label(slot.rawValue, systemImage: icon)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(latest?.status.rawValue ?? "idle")
                    .font(.caption.monospaced())
                    .foregroundStyle(statusColor)
            }
            HStack {
                Text("completed \(completedCount)")
                Text("total \(totalDurationMs)ms")
                if let selectedToolID = latest?.selectedToolID {
                    Text(selectedToolID)
                }
            }
            .font(.caption.monospaced())
            .foregroundStyle(Theme.textSecondary)
            if let latest {
                Text("\(latest.kind) / \(latest.phase)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(.vertical, 2)
    }

    private var icon: String {
        switch slot {
        case .cortex: return "brain.head.profile"
        case .executor: return "hammer"
        case .mouth: return "text.bubble"
        case .mimicry: return "theatermasks"
        case .rem: return "moon.zzz"
        case .fleet: return "cpu"
        case .embedding: return "point.3.connected.trianglepath.dotted"
        case .runtime: return "gearshape.2"
        case .unknown: return "questionmark.circle"
        }
    }

    private var statusColor: Color {
        switch latest?.status {
        case .running: return .green
        case .failed: return .red
        case .fallback: return .orange
        case .cancelled: return .yellow
        case .done: return .blue
        case .info, nil: return Theme.textSecondary
        }
    }
}

private struct AgentWorkflowEventRow: View {
    let event: AgentWorkflowEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(event.slot.rawValue)
                    .font(.caption.monospaced().weight(.semibold))
                Text(event.status.rawValue)
                    .font(.caption2.monospaced())
                    .foregroundStyle(statusColor)
                Spacer()
                if let durationMs = event.durationMs {
                    Text("\(durationMs)ms")
                        .font(.caption2.monospaced())
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            Text("\(event.kind) / \(event.phase)")
                .font(.caption)
            HStack {
                if let intent = event.intent { Text(intent) }
                if let tool = event.selectedToolID { Text(tool) }
                if let fallback = event.fallbackReason { Text("fallback: \(fallback)") }
            }
            .font(.caption2.monospaced())
            .foregroundStyle(Theme.textSecondary)
        }
        .padding(.vertical, 2)
    }

    private var statusColor: Color {
        switch event.status {
        case .running: return .green
        case .done: return .blue
        case .failed: return .red
        case .cancelled: return .yellow
        case .fallback: return .orange
        case .info: return Theme.textSecondary
        }
    }
}
