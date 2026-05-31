import SwiftUI

struct FleetStatusCard: View {
    let snapshot: LumenModelFleetSnapshot
    let progresses: [String: DownloadProgress]
    let loadedPaths: Set<String>
    let onRepair: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: snapshot.isRunnableV1 ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(snapshot.isRunnableV1 ? Theme.accent : .orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Fleet v1")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text(summaryText)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                Button("Repair") { onRepair() }
                    .font(.caption.weight(.semibold))
                    .buttonStyle(.bordered)
            }

            VStack(spacing: 8) {
                ForEach(LumenModelSlot.allCases) { slot in
                    HStack(spacing: 10) {
                        Text(slot.displayName)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Theme.textPrimary)
                            .frame(width: 76, alignment: .leading)

                        VStack(alignment: .leading, spacing: 2) {
                            if let assignment = snapshot.assignment(for: slot) {
                                Text("\(assignment.displayName) · \(assignment.parameters) · \(assignment.quantization)")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(Theme.textSecondary)
                                    .lineLimit(1)
                                Text(statusText(for: assignment))
                                    .font(.caption2.weight(.medium))
                                    .foregroundStyle(statusColor(for: assignment))
                            } else {
                                Text("missing")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.orange)
                            }
                        }
                        Spacer()
                    }
                    .padding(8)
                    .background(Theme.surfaceHigh.opacity(0.55))
                    .clipShape(.rect(cornerRadius: 8))
                }
            }

            let active = activeDownloads
            if !active.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Downloads")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.textSecondary)
                    ForEach(active, id: \.id) { item in
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text(item.name)
                                    .font(.caption2.weight(.medium))
                                    .foregroundStyle(Theme.textPrimary)
                                    .lineLimit(1)
                                Spacer()
                                Text(item.label)
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(item.isFailed ? .red : Theme.textSecondary)
                            }
                            if !item.isFailed {
                                ProgressView(value: item.progress.fractionCompleted)
                                    .tint(Theme.accent)
                            }
                        }
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }

    private var summaryText: String {
        let mode = snapshot.mode.displayName
        if snapshot.missingSlots.isEmpty { return "All logical slots assigned · \(mode)." }
        return "Missing: \(snapshot.missingSlots.map(\.displayName).joined(separator: ", ")) · \(mode)"
    }

    private func statusText(for assignment: LumenModelAssignment) -> String {
        if loadedPaths.contains(assignment.localPath) { return "runtime resident · loaded" }
        if snapshot.runtimeResidentSlots.contains(assignment.slot) { return "runtime resident · not loaded" }
        if snapshot.targetResidentSlots.contains(assignment.slot) {
            return assignment.slot.shouldRunOnlyWhenIdle ? "target resident · idle-only · pending" : "target resident · pending"
        }
        return "not resident"
    }

    private func statusColor(for assignment: LumenModelAssignment) -> Color {
        if loadedPaths.contains(assignment.localPath) { return Theme.accent }
        if snapshot.runtimeResidentSlots.contains(assignment.slot) { return .orange }
        return snapshot.targetResidentSlots.contains(assignment.slot) ? Theme.textSecondary : Theme.textTertiary
    }

    private var activeDownloads: [(id: String, name: String, progress: DownloadProgress, label: String, isFailed: Bool)] {
        LumenModelFleetCatalog.allFleetModels.compactMap { model in
            guard let progress = progresses[model.id] else { return nil }
            switch progress.state {
            case .downloading:
                return (model.id, model.name, progress, "\(Int(progress.fractionCompleted * 100))%", false)
            case .paused:
                return (model.id, model.name, progress, "paused", false)
            case .failed(let message):
                return (model.id, model.name, progress, "failed: \(message)", true)
            case .queued, .completed:
                return nil
            }
        }
    }
}
