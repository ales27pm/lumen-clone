import SwiftUI

nonisolated enum LumenBrand {
    static let midnight = Color(red: 0.012, green: 0.015, blue: 0.026)
    static let deepSpace = Color(red: 0.024, green: 0.030, blue: 0.052)
    static let ink = Color(red: 0.035, green: 0.043, blue: 0.070)
    static let glass = Color.white.opacity(0.070)
    static let glassHigh = Color.white.opacity(0.105)
    static let edge = Color.white.opacity(0.125)
    static let edgeStrong = Color.white.opacity(0.210)
    static let ember = Color(red: 1.000, green: 0.780, blue: 0.360)
    static let lumen = Color(red: 1.000, green: 0.920, blue: 0.640)
    static let corona = Color(red: 0.640, green: 0.800, blue: 1.000)
    static let plasma = Color(red: 0.470, green: 0.570, blue: 1.000)
    static let violet = Color(red: 0.640, green: 0.430, blue: 1.000)
    static let text = Color.white.opacity(0.960)
    static let textMuted = Color.white.opacity(0.620)
    static let textFaint = Color.white.opacity(0.400)

    static let markGradient = AngularGradient(
        colors: [ember, lumen, corona, plasma, violet, ember],
        center: .center,
        startAngle: .degrees(-130),
        endAngle: .degrees(230)
    )

    static let haloGradient = RadialGradient(
        colors: [lumen.opacity(0.60), corona.opacity(0.26), plasma.opacity(0.09), .clear],
        center: .center,
        startRadius: 0,
        endRadius: 170
    )
}

struct LumenAssistantMark: View {
    var showsWordmark: Bool = false
    var size: CGFloat = 92

    var body: some View {
        HStack(spacing: showsWordmark ? 14 : 0) {
            ZStack {
                Circle()
                    .fill(LumenBrand.haloGradient)
                    .frame(width: size * 1.85, height: size * 1.85)
                    .blur(radius: size * 0.13)

                Circle()
                    .fill(LumenBrand.markGradient)
                    .frame(width: size, height: size)
                    .shadow(color: LumenBrand.lumen.opacity(0.56), radius: size * 0.28)
                    .shadow(color: LumenBrand.plasma.opacity(0.35), radius: size * 0.46)

                Circle()
                    .strokeBorder(Color.white.opacity(0.42), lineWidth: max(1, size * 0.020))
                    .frame(width: size, height: size)

                Circle()
                    .fill(Color.white.opacity(0.96))
                    .frame(width: size * 0.19, height: size * 0.19)
                    .offset(x: -size * 0.15, y: -size * 0.12)
                    .blur(radius: size * 0.01)

                LumenInnerGlyph()
                    .stroke(Color.white.opacity(0.82), style: StrokeStyle(lineWidth: max(1.4, size * 0.035), lineCap: .round, lineJoin: .round))
                    .frame(width: size * 0.56, height: size * 0.56)
                    .shadow(color: .white.opacity(0.65), radius: size * 0.06)
            }
            .frame(width: size * 1.42, height: size * 1.42)

            if showsWordmark {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Lumen")
                        .font(.system(size: size * 0.30, weight: .semibold, design: .rounded))
                        .foregroundStyle(LumenBrand.text)
                    Text("on-device light")
                        .font(.system(size: size * 0.105, weight: .medium, design: .rounded))
                        .foregroundStyle(LumenBrand.textMuted)
                        .tracking(1.2)
                        .textCase(.uppercase)
                }
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Lumen assistant light mark")
    }
}

private struct LumenInnerGlyph: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let w = rect.width
        let h = rect.height
        path.move(to: CGPoint(x: w * 0.50, y: h * 0.09))
        path.addCurve(to: CGPoint(x: w * 0.82, y: h * 0.42), control1: CGPoint(x: w * 0.66, y: h * 0.12), control2: CGPoint(x: w * 0.80, y: h * 0.25))
        path.addCurve(to: CGPoint(x: w * 0.50, y: h * 0.91), control1: CGPoint(x: w * 0.85, y: h * 0.65), control2: CGPoint(x: w * 0.68, y: h * 0.84))
        path.addCurve(to: CGPoint(x: w * 0.18, y: h * 0.42), control1: CGPoint(x: w * 0.32, y: h * 0.84), control2: CGPoint(x: w * 0.15, y: h * 0.65))
        path.addCurve(to: CGPoint(x: w * 0.50, y: h * 0.09), control1: CGPoint(x: w * 0.20, y: h * 0.25), control2: CGPoint(x: w * 0.34, y: h * 0.12))

        path.move(to: CGPoint(x: w * 0.50, y: h * 0.23))
        path.addLine(to: CGPoint(x: w * 0.50, y: h * 0.77))
        path.move(to: CGPoint(x: w * 0.29, y: h * 0.50))
        path.addLine(to: CGPoint(x: w * 0.71, y: h * 0.50))
        return path
    }
}

struct LumenBrandBackground: View {
    var intensity: Double = 1.0

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [LumenBrand.midnight, LumenBrand.deepSpace, LumenBrand.ink],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            RadialGradient(
                colors: [LumenBrand.lumen.opacity(0.26 * intensity), LumenBrand.corona.opacity(0.14 * intensity), .clear],
                center: UnitPoint(x: 0.52, y: 0.38),
                startRadius: 0,
                endRadius: 360
            )
            .blendMode(.screen)

            RadialGradient(
                colors: [LumenBrand.plasma.opacity(0.20 * intensity), .clear],
                center: UnitPoint(x: 0.20, y: 0.86),
                startRadius: 0,
                endRadius: 280
            )
            .blendMode(.screen)

            RadialGradient(
                colors: [LumenBrand.violet.opacity(0.16 * intensity), .clear],
                center: UnitPoint(x: 0.88, y: 0.18),
                startRadius: 0,
                endRadius: 260
            )
            .blendMode(.screen)
        }
        .ignoresSafeArea()
    }
}

struct LumenLightBeam: View {
    var body: some View {
        GeometryReader { proxy in
            let w = proxy.size.width
            let h = proxy.size.height
            Path { path in
                path.move(to: CGPoint(x: w * 0.50, y: h * 0.24))
                path.addLine(to: CGPoint(x: w * 0.08, y: h * 0.95))
                path.addLine(to: CGPoint(x: w * 0.92, y: h * 0.95))
                path.closeSubpath()
            }
            .fill(
                LinearGradient(
                    colors: [LumenBrand.lumen.opacity(0.28), LumenBrand.corona.opacity(0.10), .clear],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .blur(radius: 18)
            .blendMode(.screen)
        }
        .allowsHitTesting(false)
    }
}
