import SwiftUI

struct Surface<Content: View>: View {
    var cornerRadius: CGFloat = 10
    var padding: CGFloat = 14
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(padding)
            .background(Theme.surface)
            .clipShape(.rect(cornerRadius: cornerRadius))
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            }
    }
}

extension View {
    func surface(cornerRadius: CGFloat = 10, padding: CGFloat = 14) -> some View {
        self
            .padding(padding)
            .background(Theme.surface)
            .clipShape(.rect(cornerRadius: cornerRadius))
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            }
    }

    func glassCard(cornerRadius: CGFloat = 10, tint: Color = .clear) -> some View {
        self.surface(cornerRadius: cornerRadius, padding: 0)
    }
}

struct StatusDot: View {
    var color: Color
    var size: CGFloat = 8
    var body: some View {
        Circle().fill(color).frame(width: size, height: size)
    }
}

nonisolated func formatBytes(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .binary
    formatter.allowedUnits = [.useGB, .useMB]
    return formatter.string(fromByteCount: bytes)
}
