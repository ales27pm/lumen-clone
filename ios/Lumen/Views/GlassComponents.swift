import SwiftUI

struct Surface<Content: View>: View {
    var cornerRadius: CGFloat = 12
    var padding: CGFloat = 14
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(padding)
            .background {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(Theme.surfaceHigh)
            }
            .clipShape(.rect(cornerRadius: cornerRadius))
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            }
    }
}

extension View {
    func surface(cornerRadius: CGFloat = 12, padding: CGFloat = 14) -> some View {
        self
            .padding(padding)
            .background {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(Theme.surfaceHigh)
            }
            .clipShape(.rect(cornerRadius: cornerRadius))
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            }
    }

    func glassCard(cornerRadius: CGFloat = 12, tint: Color = .clear) -> some View {
        self.surface(cornerRadius: cornerRadius, padding: 0)
    }

    func lumenTouchTarget(_ size: CGFloat = 44) -> some View {
        self.frame(minWidth: size, minHeight: size)
    }
}

struct StatusDot: View {
    var color: Color
    var size: CGFloat = 8
    var body: some View {
        Circle().fill(color).frame(width: size, height: size)
    }
}

struct LumenBrandAsset: View {
    enum Kind {
        case mark
        case assistantMark
        case wordmarkLockup
        case verticalLogo

        var assetName: String {
            switch self {
            case .mark: return "LumenMark"
            case .assistantMark: return "LumenAssistantMark"
            case .wordmarkLockup: return "LumenWordmarkLockup"
            case .verticalLogo: return "LumenVerticalLogo"
            }
        }
    }

    var kind: Kind
    var accessibilityLabel: String?

    var body: some View {
        Image(kind.assetName)
            .resizable()
            .scaledToFit()
            .accessibilityLabel(accessibilityLabel ?? "")
            .accessibilityHidden(accessibilityLabel == nil)
    }
}

struct LumenStatusChip: View {
    var title: String
    var systemImage: String
    var tint: Color = Theme.accent

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.caption.weight(.semibold))
            Text(title)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(tint.opacity(0.12), in: Capsule())
        .overlay {
            Capsule()
                .strokeBorder(tint.opacity(0.28), lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }
}

struct LumenIconControl: View {
    var systemImage: String
    var accessibilityLabel: String
    var isProminent: Bool = false
    var tint: Color = Theme.textSecondary
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.body.weight(isProminent ? .semibold : .medium))
                .foregroundStyle(isProminent ? LumenBrand.midnight : tint)
                .frame(width: 44, height: 44)
                .background {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(isProminent ? Theme.accent : Theme.surfaceHigh)
                        .overlay(isProminent ? Color.clear : Color.white.opacity(0.03))
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(isProminent ? Theme.accent.opacity(0.35) : Theme.border, lineWidth: 1)
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }
}

nonisolated func formatBytes(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .binary
    formatter.allowedUnits = [.useGB, .useMB]
    return formatter.string(fromByteCount: bytes)
}
