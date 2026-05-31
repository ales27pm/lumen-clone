import SwiftUI

struct AppBackground: View {
    var body: some View {
        LumenBrandBackground(intensity: 0.72)
    }
}

struct AuroraBackground: View {
    var body: some View {
        LumenBrandBackground(intensity: 0.84)
    }
}

enum Theme {
    static let background = LumenBrand.midnight
    static let surface = LumenBrand.glass
    static let surfaceHigh = LumenBrand.glassHigh
    static let border = LumenBrand.edge
    static let borderStrong = LumenBrand.edgeStrong
    static let textPrimary = LumenBrand.text
    static let textSecondary = LumenBrand.textMuted
    static let textTertiary = LumenBrand.textFaint
    static let accent = LumenBrand.lumen
}
