#if canImport(CarPlay)
import CarPlay
import SwiftData
import UIKit

@MainActor
final class CarPlayVoiceSceneDelegate: UIResponder, CPTemplateApplicationSceneDelegate {
    private enum VoiceStateID {
        static let ready = "ready"
        static let listening = "listening"
        static let thinking = "thinking"
        static let speaking = "speaking"
        static let unavailable = "unavailable"
    }

    private var interfaceController: CPInterfaceController?
    private var voiceTemplate: CPVoiceControlTemplate?

    func templateApplicationScene(
        _ templateApplicationScene: CPTemplateApplicationScene,
        didConnect interfaceController: CPInterfaceController
    ) {
        self.interfaceController = interfaceController
        interfaceController.prefersDarkUserInterfaceStyle = true

        let template = makeRootTemplate()
        interfaceController.setRootTemplate(template, animated: false) { [weak self] success, error in
            guard success else {
                self?.presentConnectionError(error)
                return
            }
        }
    }

    func templateApplicationScene(
        _ templateApplicationScene: CPTemplateApplicationScene,
        didDisconnectInterfaceController interfaceController: CPInterfaceController
    ) {
        if self.interfaceController === interfaceController {
            VoiceService.shared.stopListening()
            VoiceService.shared.stopSpeaking()
            self.interfaceController = nil
            voiceTemplate = nil
        }
    }
}

private extension CarPlayVoiceSceneDelegate {

    func makeRootTemplate() -> CPListTemplate {
        let askItem = CPListItem(text: "Ask Lumen", detailText: "Start voice assistant")
        askItem.handler = { [weak self] _, completion in
            Task { @MainActor in
                await self?.startAskLumenSession()
                completion()
            }
        }

        let stopItem = CPListItem(text: "Stop / Cancel", detailText: "Stop listening or speaking")
        stopItem.handler = { _, completion in
            Task { @MainActor in
                VoiceService.shared.stopListening()
                VoiceService.shared.stopSpeaking()
                completion()
            }
        }

        let phoneItem = CPListItem(text: "Open on iPhone", detailText: "Use Lumen on your phone")
        phoneItem.handler = { [weak self] _, completion in
            Task { @MainActor in
                self?.presentAlert(title: "Open Lumen on iPhone", message: "Use your iPhone for full chat, settings, and permissions.")
                completion()
            }
        }

        let diagnosticsItem = CPListItem(text: "Diagnostics", detailText: "Check voice readiness")
        diagnosticsItem.handler = { [weak self] _, completion in
            Task { @MainActor in
                self?.presentDiagnostics()
                completion()
            }
        }

        return CPListTemplate(title: "Lumen", sections: [
            CPListSection(items: [askItem, stopItem, phoneItem, diagnosticsItem])
        ])
    }

    func startAskLumenSession() async {
        guard let interfaceController else { return }
        guard ProcessInfo.processInfo.thermalState != .critical else {
            presentUnavailable("Device is too warm; try again shortly.")
            return
        }
        guard await VoiceService.shared.requestPermissions() else {
            presentUnavailable("Open Lumen on iPhone to allow microphone and speech recognition.")
            return
        }

        let voice = makeVoiceTemplate(scale: interfaceController.carTraitCollection.displayScale)
        voiceTemplate = voice
        interfaceController.pushTemplate(voice, animated: true) { [weak self] success, error in
            Task { @MainActor in
                guard success else {
                    self?.presentConnectionError(error)
                    return
                }
                self?.voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.listening)
                await VoiceService.shared.startListening { [weak self] transcript in
                    Task { @MainActor in
                        await self?.runCarPlayPrompt(transcript)
                    }
                }
                if let lastError = VoiceService.shared.lastError, !VoiceService.shared.isListening {
                    self?.presentUnavailable(lastError)
                }
            }
        }
    }

    func runCarPlayPrompt(_ prompt: String) async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.ready)
            return
        }
        guard ProcessInfo.processInfo.thermalState != .critical else {
            presentUnavailable("Device is too warm; try again shortly.")
            return
        }
        guard let container = SharedContainer.shared else {
            presentUnavailable("Lumen context is unavailable. Check your iPhone.")
            return
        }

        voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.thinking)
        let ctx = ModelContext(container)
        let settings = SettingsSnapshot.loadFromDisk()
        let result = await HeadlessAgentKernelRunner.run(
            prompt: trimmed,
            settings: settings,
            context: ctx,
            maxSteps: min(2, settings.maxAgentSteps),
            source: .appIntent
        )
        let answer = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !answer.isEmpty else {
            speakUnavailable("Lumen could not produce a response. Try again shortly.")
            return
        }

        voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.speaking)
        VoiceService.shared.speak(String(answer.prefix(500)), voiceID: settings.voiceID, rate: settings.speakingRate) { [weak self] in
            self?.voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.ready)
        }
    }

    func makeVoiceTemplate(scale: CGFloat) -> CPVoiceControlTemplate {
        let states = [
            CPVoiceControlState(
                identifier: VoiceStateID.ready,
                titleVariants: ["Ask Lumen", "Lumen"],
                image: CarPlayVoiceArtwork.staticImage(
                    symbolName: "sparkles",
                    palette: .ready,
                    scale: scale
                ),
                repeats: false
            ),
            CPVoiceControlState(
                identifier: VoiceStateID.listening,
                titleVariants: ["Listening", "Go ahead"],
                image: CarPlayVoiceArtwork.pulsingImage(
                    symbolName: "waveform",
                    palette: .listening,
                    scale: scale
                ),
                repeats: true
            ),
            CPVoiceControlState(
                identifier: VoiceStateID.thinking,
                titleVariants: ["Working on it", "Thinking"],
                image: CarPlayVoiceArtwork.pulsingImage(
                    symbolName: "brain.head.profile",
                    palette: .thinking,
                    scale: scale
                ),
                repeats: true
            ),
            CPVoiceControlState(
                identifier: VoiceStateID.speaking,
                titleVariants: ["Answering", "Speaking"],
                image: CarPlayVoiceArtwork.pulsingImage(
                    symbolName: "speaker.wave.2.fill",
                    palette: .speaking,
                    scale: scale
                ),
                repeats: true
            ),
            CPVoiceControlState(
                identifier: VoiceStateID.unavailable,
                titleVariants: ["Check iPhone", "Unavailable"],
                image: CarPlayVoiceArtwork.staticImage(
                    symbolName: "exclamationmark.triangle.fill",
                    palette: .unavailable,
                    scale: scale
                ),
                repeats: false
            )
        ]
        return CPVoiceControlTemplate(voiceControlStates: states)
    }

    func presentUnavailable(_ message: String) {
        voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.unavailable)
        presentAlert(title: "Lumen unavailable", message: message)
        VoiceService.shared.speak(message, voiceID: nil, rate: 0.46)
    }

    func speakUnavailable(_ message: String) {
        voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.unavailable)
        VoiceService.shared.speak(message, voiceID: nil, rate: 0.46) { [weak self] in
            self?.voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.ready)
        }
    }

    func presentDiagnostics() {
        let voice = VoiceService.shared
        let status = voice.lastError ?? (voice.isListening ? "Listening" : voice.isSpeaking ? "Speaking" : "Ready")
        presentAlert(title: "Lumen Diagnostics", message: status)
    }

    func presentAlert(title: String, message: String) {
        let action = CPAlertAction(title: "OK", style: .default) { [weak self] _ in
            self?.interfaceController?.dismissTemplate(animated: true, completion: nil)
        }
        let alert = CPAlertTemplate(titleVariants: [title, message], actions: [action])
        interfaceController?.presentTemplate(alert, animated: true, completion: nil)
    }

    func presentConnectionError(_ error: Error?) {
        let action = CPAlertAction(title: "OK", style: .default) { [weak self] _ in
            self?.interfaceController?.dismissTemplate(animated: true, completion: nil)
        }
        let alert = CPAlertTemplate(
            titleVariants: [error.map { "Lumen CarPlay unavailable: \(RuntimeMetricErrorSanitizer.code(for: $0))" } ?? "Lumen CarPlay unavailable"],
            actions: [action]
        )
        interfaceController?.presentTemplate(alert, animated: true, completion: nil)
        voiceTemplate?.activateVoiceControlState(withIdentifier: VoiceStateID.unavailable)
    }
}

private enum CarPlayVoiceArtwork {
    struct Palette {
        let backgroundTop: UIColor
        let backgroundBottom: UIColor
        let accent: UIColor
        let accentMuted: UIColor

        static let ready = Palette(
            backgroundTop: UIColor(red: 0.06, green: 0.08, blue: 0.11, alpha: 1),
            backgroundBottom: UIColor(red: 0.01, green: 0.11, blue: 0.15, alpha: 1),
            accent: UIColor(red: 0.31, green: 0.91, blue: 0.78, alpha: 1),
            accentMuted: UIColor(red: 0.15, green: 0.38, blue: 0.39, alpha: 1)
        )

        static let listening = Palette(
            backgroundTop: UIColor(red: 0.04, green: 0.09, blue: 0.15, alpha: 1),
            backgroundBottom: UIColor(red: 0.01, green: 0.17, blue: 0.26, alpha: 1),
            accent: UIColor(red: 0.20, green: 0.74, blue: 1.00, alpha: 1),
            accentMuted: UIColor(red: 0.08, green: 0.30, blue: 0.47, alpha: 1)
        )

        static let thinking = Palette(
            backgroundTop: UIColor(red: 0.11, green: 0.07, blue: 0.16, alpha: 1),
            backgroundBottom: UIColor(red: 0.21, green: 0.08, blue: 0.22, alpha: 1),
            accent: UIColor(red: 0.86, green: 0.49, blue: 1.00, alpha: 1),
            accentMuted: UIColor(red: 0.42, green: 0.18, blue: 0.48, alpha: 1)
        )

        static let speaking = Palette(
            backgroundTop: UIColor(red: 0.13, green: 0.09, blue: 0.03, alpha: 1),
            backgroundBottom: UIColor(red: 0.25, green: 0.17, blue: 0.02, alpha: 1),
            accent: UIColor(red: 1.00, green: 0.75, blue: 0.25, alpha: 1),
            accentMuted: UIColor(red: 0.48, green: 0.31, blue: 0.09, alpha: 1)
        )

        static let unavailable = Palette(
            backgroundTop: UIColor(red: 0.18, green: 0.06, blue: 0.06, alpha: 1),
            backgroundBottom: UIColor(red: 0.28, green: 0.07, blue: 0.07, alpha: 1),
            accent: UIColor(red: 1.00, green: 0.36, blue: 0.30, alpha: 1),
            accentMuted: UIColor(red: 0.52, green: 0.16, blue: 0.14, alpha: 1)
        )
    }

    static func staticImage(symbolName: String, palette: Palette, scale: CGFloat) -> UIImage {
        render(symbolName: symbolName, palette: palette, scale: scale, pulse: 0.18, glow: 0.55)
    }

    static func pulsingImage(symbolName: String, palette: Palette, scale: CGFloat) -> UIImage {
        let frames = stride(from: 0, to: 12, by: 1).map { step -> UIImage in
            let phase = CGFloat(step) / 11
            let pulse = 0.12 + sin(phase * .pi) * 0.22
            let glow = 0.38 + sin(phase * .pi) * 0.42
            return render(symbolName: symbolName, palette: palette, scale: scale, pulse: pulse, glow: glow)
        }
        return UIImage.animatedImage(with: frames, duration: 1.15) ?? frames[0]
    }

    private static func render(
        symbolName: String,
        palette: Palette,
        scale: CGFloat,
        pulse: CGFloat,
        glow: CGFloat
    ) -> UIImage {
        let size = CGSize(width: 148, height: 148)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = max(scale, 2)
        format.opaque = false

        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { context in
            let rect = CGRect(origin: .zero, size: size)
            let bounds = rect.insetBy(dx: 7, dy: 7)
            let cgContext = context.cgContext

            drawBackground(in: cgContext, bounds: bounds, palette: palette)
            drawRings(in: cgContext, bounds: bounds, palette: palette, pulse: pulse, glow: glow)
            drawSymbol(symbolName, in: bounds, palette: palette)
        }
    }

    private static func drawBackground(in context: CGContext, bounds: CGRect, palette: Palette) {
        let path = UIBezierPath(roundedRect: bounds, cornerRadius: 34)
        context.saveGState()
        path.addClip()

        let colors = [palette.backgroundTop.cgColor, palette.backgroundBottom.cgColor] as CFArray
        let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0, 1])
        context.drawLinearGradient(
            gradient!,
            start: CGPoint(x: bounds.midX, y: bounds.minY),
            end: CGPoint(x: bounds.midX, y: bounds.maxY),
            options: []
        )
        context.restoreGState()

        palette.accent.withAlphaComponent(0.32).setStroke()
        path.lineWidth = 2
        path.stroke()
    }

    private static func drawRings(
        in context: CGContext,
        bounds: CGRect,
        palette: Palette,
        pulse: CGFloat,
        glow: CGFloat
    ) {
        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        let outerRadius = min(bounds.width, bounds.height) * (0.39 + pulse)
        let innerRadius = min(bounds.width, bounds.height) * 0.30

        context.saveGState()
        context.setShadow(offset: .zero, blur: 24, color: palette.accent.withAlphaComponent(glow).cgColor)
        palette.accentMuted.withAlphaComponent(0.88).setFill()
        UIBezierPath(ovalIn: CGRect(
            x: center.x - outerRadius,
            y: center.y - outerRadius,
            width: outerRadius * 2,
            height: outerRadius * 2
        )).fill()
        context.restoreGState()

        palette.accent.withAlphaComponent(0.34 + glow * 0.28).setStroke()
        let ringPath = UIBezierPath(ovalIn: CGRect(
            x: center.x - innerRadius,
            y: center.y - innerRadius,
            width: innerRadius * 2,
            height: innerRadius * 2
        ))
        ringPath.lineWidth = 3
        ringPath.stroke()
    }

    private static func drawSymbol(_ symbolName: String, in bounds: CGRect, palette: Palette) {
        let symbolConfiguration = UIImage.SymbolConfiguration(pointSize: 52, weight: .semibold)
        let symbol = UIImage(systemName: symbolName, withConfiguration: symbolConfiguration)
            ?? UIImage(systemName: "circle.fill", withConfiguration: symbolConfiguration)
        let image = symbol?.withTintColor(.white, renderingMode: .alwaysOriginal)
        let side: CGFloat = 66
        let symbolRect = CGRect(
            x: bounds.midX - side / 2,
            y: bounds.midY - side / 2,
            width: side,
            height: side
        )

        palette.accent.withAlphaComponent(0.16).setFill()
        UIBezierPath(ovalIn: symbolRect.insetBy(dx: -10, dy: -10)).fill()
        image?.draw(in: symbolRect)
    }
}
#endif
