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
    var processIntensity: Double = 1.0

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 24.0)) { timeline in
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
                .scaleEffect(CGFloat(1.0 + 0.018 * realtimePulse(timeline.date, speed: 0.11)))

                RadialGradient(
                    colors: [LumenBrand.plasma.opacity(0.20 * intensity), .clear],
                    center: UnitPoint(x: 0.20, y: 0.86),
                    startRadius: 0,
                    endRadius: 280
                )
                .blendMode(.screen)
                .offset(x: CGFloat(14 * realtimePulse(timeline.date, speed: 0.07)), y: CGFloat(-10 * realtimePulse(timeline.date, speed: 0.05)))

                RadialGradient(
                    colors: [LumenBrand.violet.opacity(0.16 * intensity), .clear],
                    center: UnitPoint(x: 0.88, y: 0.18),
                    startRadius: 0,
                    endRadius: 260
                )
                .blendMode(.screen)
                .offset(x: CGFloat(-12 * realtimePulse(timeline.date, speed: 0.06)), y: CGFloat(16 * realtimePulse(timeline.date, speed: 0.09)))

                LatentLiturgyProcessField(date: timeline.date, intensity: intensity * processIntensity)
                    .blendMode(.screen)
                    .opacity(0.94)
            }
        }
        .ignoresSafeArea()
    }

    private func realtimePulse(_ date: Date, speed: Double) -> Double {
        sin(date.timeIntervalSinceReferenceDate * speed)
    }
}

private struct LatentLiturgyProcessField: View {
    let date: Date
    let intensity: Double

    var body: some View {
        Canvas(opaque: false, rendersAsynchronously: true) { context, size in
            guard size.width > 0, size.height > 0 else { return }
            let clock = date.timeIntervalSinceReferenceDate
            let dayProgress = dayFraction(for: date)
            let center = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
            let radius = min(size.width, size.height)
            let clauses = hiddenClauses(in: size, dayProgress: dayProgress)

            drawClauseHalos(clauses, in: &context, clock: clock, radius: radius)
            drawParticleScript(clauses, center: center, in: &context, size: size, clock: clock, dayProgress: dayProgress)
            drawBreathingMargins(in: &context, size: size, clock: clock)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private func dayFraction(for date: Date) -> Double {
        let calendar = Calendar.autoupdatingCurrent
        let hour: Int = calendar.component(.hour, from: date)
        let minute: Int = calendar.component(.minute, from: date)
        let second: Int = calendar.component(.second, from: date)
        let nanosecond: Int = calendar.component(.nanosecond, from: date)

        let wholeSeconds: Int = (hour * 3_600) + (minute * 60) + second
        let fractionalSeconds: Double = Double(nanosecond) / 1_000_000_000.0
        return (Double(wholeSeconds) + fractionalSeconds) / 86_400.0
    }

    private func hiddenClauses(in size: CGSize, dayProgress: Double) -> [CGPoint] {
        let count = 7
        let golden = Double.pi * (3 - sqrt(5))
        let radius = min(size.width, size.height)
        let center = CGPoint(x: size.width * 0.5, y: size.height * 0.5)

        return (0..<count).map { index in
            let normalized = Double(index) / Double(max(1, count - 1))
            let angle = -Double.pi / 2 + Double(index) * golden + dayProgress * Double.pi * 2 * 0.08
            let clauseRadius = radius * CGFloat(0.12 + 0.25 * sqrt(normalized + 0.08))
            return CGPoint(
                x: center.x + cosCGFloat(angle) * clauseRadius,
                y: center.y + sinCGFloat(angle) * clauseRadius
            )
        }
    }

    private func drawClauseHalos(_ clauses: [CGPoint], in context: inout GraphicsContext, clock: TimeInterval, radius: CGFloat) {
        for (index, clause) in clauses.enumerated() {
            let breath = 0.5 + 0.5 * sin(clock * 0.45 + Double(index) * 0.72)
            let haloRadius = radius * CGFloat(0.035 + 0.018 * breath)
            let rect = CGRect(x: clause.x - haloRadius, y: clause.y - haloRadius, width: haloRadius * 2, height: haloRadius * 2)
            var path = Path(ellipseIn: rect)
            context.stroke(
                path,
                with: .color(LumenBrand.lumen.opacity((0.05 + 0.035 * breath) * intensity)),
                lineWidth: CGFloat(0.8 + breath * 0.8)
            )

            let innerRadius = haloRadius * 0.32
            path = Path(ellipseIn: CGRect(x: clause.x - innerRadius, y: clause.y - innerRadius, width: innerRadius * 2, height: innerRadius * 2))
            context.fill(path, with: .color(LumenBrand.ember.opacity(0.018 * intensity)))
        }
    }

    private func drawParticleScript(
        _ clauses: [CGPoint],
        center: CGPoint,
        in context: inout GraphicsContext,
        size: CGSize,
        clock: TimeInterval,
        dayProgress: Double
    ) {
        let streamCount = 30
        let radius = min(size.width, size.height)

        for stream in 0..<streamCount {
            let streamSeed = Double(stream) * 12.9898 + dayProgress * 37.719
            let clause = clauses[stream % max(1, clauses.count)]
            let startAngle = streamSeed * 0.43 + clock * (0.018 + Double(stream % 7) * 0.0018)
            var point = CGPoint(
                x: clause.x + cosCGFloat(startAngle) * radius * CGFloat(0.07 + pseudo(streamSeed) * 0.20),
                y: clause.y + sinCGFloat(startAngle) * radius * CGFloat(0.07 + pseudo(streamSeed + 8.3) * 0.20)
            )
            var path = Path()
            path.move(to: point)

            let steps = 22
            for step in 0..<steps {
                let nearest = nearestClause(to: point, clauses: clauses)
                let dx = nearest.x - point.x
                let dy = nearest.y - point.y
                let distance = max(18, sqrt(dx * dx + dy * dy))
                let orbital = atan2(Double(dy), Double(dx)) + Double.pi / 2
                let centerAngle = atan2(Double(center.y - point.y), Double(center.x - point.x))
                let turbulence = sin(Double(step) * 0.55 + streamSeed + clock * 0.11) * 0.72
                let angle = orbital * 0.62 + centerAngle * 0.38 + turbulence
                let stride = radius * CGFloat(0.006 + 0.010 * pseudo(streamSeed + Double(step) * 1.7)) * CGFloat(1.0 + 60.0 / Double(distance))
                let next = CGPoint(x: point.x + cosCGFloat(angle) * stride, y: point.y + sinCGFloat(angle) * stride)
                let control = CGPoint(
                    x: (point.x + next.x) * 0.5 + cosCGFloat(angle + Double.pi / 2) * stride * 0.65,
                    y: (point.y + next.y) * 0.5 + sinCGFloat(angle + Double.pi / 2) * stride * 0.65
                )
                path.addQuadCurve(to: next, control: control)
                point = next
            }

            let warmth = 0.5 + 0.5 * sin(clock * 0.23 + streamSeed)
            let color = stream % 5 == 0 ? LumenBrand.corona : (warmth > 0.55 ? LumenBrand.ember : LumenBrand.lumen)
            context.stroke(
                path,
                with: .color(color.opacity((0.020 + 0.026 * warmth) * intensity)),
                style: StrokeStyle(lineWidth: 0.42 + CGFloat(warmth) * 0.82, lineCap: .round, lineJoin: .round)
            )
        }
    }

    private func drawBreathingMargins(in context: inout GraphicsContext, size: CGSize, clock: TimeInterval) {
        let insetCount = 8
        for index in 0..<insetCount {
            let inset = CGFloat(index) * 18
            let rect = CGRect(x: inset, y: inset, width: max(0, size.width - inset * 2), height: max(0, size.height - inset * 2))
            let opacity = (0.010 + 0.006 * sin(clock * 0.31 + Double(index))) * intensity
            context.stroke(
                Path(roundedRect: rect, cornerRadius: 24 + CGFloat(index) * 3),
                with: .color(LumenBrand.corona.opacity(max(0, opacity))),
                lineWidth: 0.55
            )
        }
    }

    private func nearestClause(to point: CGPoint, clauses: [CGPoint]) -> CGPoint {
        clauses.min { lhs, rhs in
            let ldx = lhs.x - point.x
            let ldy = lhs.y - point.y
            let rdx = rhs.x - point.x
            let rdy = rhs.y - point.y
            return ldx * ldx + ldy * ldy < rdx * rdx + rdy * rdy
        } ?? point
    }

    private func pseudo(_ value: Double) -> Double {
        let raw = sin(value * 12.9898) * 43758.5453
        return raw - floor(raw)
    }

    private func cosCGFloat(_ value: Double) -> CGFloat {
        CGFloat(cos(value))
    }

    private func sinCGFloat(_ value: Double) -> CGFloat {
        CGFloat(sin(value))
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
