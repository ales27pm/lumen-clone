import SwiftUI

struct AgentStepsPanel: View {
    let steps: [AgentStep]
    @State var expanded: Bool

    private var visibleSteps: [AgentStep] {
        AgentVisibleContentSanitizer.sanitizedSteps(steps)
    }

    var body: some View {
        let displaySteps = visibleSteps
        if !displaySteps.isEmpty {
            VStack(alignment: .leading, spacing: 0) {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "sparkles")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                        Text("\(displaySteps.count) reasoning step\(displaySteps.count == 1 ? "" : "s")")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                        Spacer()
                        Image(systemName: expanded ? "chevron.up" : "chevron.down")
                            .font(.caption2)
                            .foregroundStyle(Theme.textTertiary)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                }
                .buttonStyle(.plain)

                if expanded {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(Array(displaySteps.enumerated()), id: \.element.id) { idx, step in
                            AgentStepRow(step: step, index: idx + 1, isLast: idx == displaySteps.count - 1)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 10)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.surface)
            .clipShape(.rect(cornerRadius: 10))
            .overlay {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            }
        }
    }
}

struct AgentStepRow: View {
    let step: AgentStep
    let index: Int
    let isLast: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(spacing: 0) {
                Image(systemName: step.icon)
                    .font(.caption)
                    .foregroundStyle(tint)
                    .frame(width: 20, height: 20)
                if !isLast {
                    Rectangle()
                        .fill(Theme.border)
                        .frame(width: 1)
                        .frame(maxHeight: .infinity)
                }
            }
            .frame(minHeight: 28)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(step.label)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    if step.kind == .action, let toolID = step.toolID,
                       let tool = ToolRegistry.find(id: toolID) {
                        Text(tool.name)
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                }
                Text(step.content)
                    .font(step.kind == .action || step.kind == .observation ? .caption.monospaced() : .caption)
                    .foregroundStyle(Theme.textSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .padding(.vertical, 1)
        }
    }

    private var tint: Color {
        switch step.kind {
        case .thought: Theme.accent
        case .action: Color(red: 0.95, green: 0.7, blue: 0.4)
        case .approvalBoundary: Color(red: 0.98, green: 0.55, blue: 0.3)
        case .observation: Color(red: 0.5, green: 0.85, blue: 0.6)
        case .reflection: Color(red: 0.75, green: 0.6, blue: 0.95)
        }
    }
}
