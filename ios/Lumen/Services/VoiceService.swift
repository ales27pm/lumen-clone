import Foundation
import AVFoundation
import Speech
import Observation

nonisolated struct VoiceInputReadinessSnapshot: Equatable, Sendable {
    var isInputAvailable: Bool
    var availableInputCount: Int?
    var currentRouteInputCount: Int
    var currentRouteOutputCount: Int
    var availableInputsSummary: String
    var routeInputsSummary: String
    var routeOutputsSummary: String
    var sampleRate: Double?
    var channelCount: UInt32?
    var exceptionError: String?
}

nonisolated enum VoiceInputReadiness {
    static let unavailableMessage = "Voice input is unavailable in the current CarPlay audio route. Try again after the microphone route is active."

    static func failureReason(for snapshot: VoiceInputReadinessSnapshot) -> String? {
        if let exceptionError = snapshot.exceptionError, !exceptionError.isEmpty {
            return "\(unavailableMessage) AVAudio reported: \(exceptionError)"
        }
        guard snapshot.isInputAvailable else { return unavailableMessage }
        let hasAvailableInput = (snapshot.availableInputCount ?? 0) > 0
        let hasCurrentInput = snapshot.currentRouteInputCount > 0
        guard hasAvailableInput || hasCurrentInput else { return unavailableMessage }
        guard let sampleRate = snapshot.sampleRate, sampleRate > 0 else { return unavailableMessage }
        guard let channelCount = snapshot.channelCount, channelCount > 0 else { return unavailableMessage }
        return nil
    }
}

struct VoiceAudioStartupResult: Equatable, Sendable {
    var succeeded: Bool
    var error: String?

    static let success = VoiceAudioStartupResult(succeeded: true, error: nil)
    static func failure(_ error: String) -> VoiceAudioStartupResult {
        VoiceAudioStartupResult(succeeded: false, error: error)
    }
}

struct VoiceAudioStartup {
    typealias TapHandler = (AVAudioPCMBuffer, AVAudioTime) -> Void

    var activateAudioSession: () -> VoiceAudioStartupResult
    var inputReadinessSnapshot: (AVAudioEngine) -> VoiceInputReadinessSnapshot
    var installInputTap: (AVAudioEngine, @escaping TapHandler) -> VoiceAudioStartupResult
    var prepareAndStartEngine: (AVAudioEngine) -> VoiceAudioStartupResult
    var stopEngine: (AVAudioEngine) -> Void
    var removeInputTap: (AVAudioEngine) -> Void

    static let live = VoiceAudioStartup(
        activateAudioSession: {
            do {
                let session = AVAudioSession.sharedInstance()
                try session.setCategory(.playAndRecord, mode: .spokenAudio, options: [.duckOthers, .defaultToSpeaker, .allowBluetoothHFP])
                try session.setActive(true, options: .notifyOthersOnDeactivation)
                return .success
            } catch {
                return .failure("Audio session error: \(error.localizedDescription)")
            }
        },
        inputReadinessSnapshot: { engine in
            let session = AVAudioSession.sharedInstance()
            let availableInputs = session.availableInputs
            let route = session.currentRoute
            var inputNode: AVAudioInputNode?
            var format: AVAudioFormat?
            var exceptionError: NSString?

            let inputOK = AudioExceptionCatcher.`try`({
                inputNode = engine.inputNode
            }, error: &exceptionError)
            if inputOK, let inputNode {
                var formatError: NSString?
                let formatOK = AudioExceptionCatcher.`try`({
                    format = inputNode.outputFormat(forBus: 0)
                }, error: &formatError)
                if !formatOK {
                    exceptionError = formatError
                }
            }

            return VoiceInputReadinessSnapshot(
                isInputAvailable: session.isInputAvailable,
                availableInputCount: availableInputs?.count,
                currentRouteInputCount: route.inputs.count,
                currentRouteOutputCount: route.outputs.count,
                availableInputsSummary: Self.portSummary(availableInputs ?? []),
                routeInputsSummary: Self.portSummary(route.inputs),
                routeOutputsSummary: Self.portSummary(route.outputs),
                sampleRate: format?.sampleRate,
                channelCount: format?.channelCount,
                exceptionError: exceptionError as String?
            )
        },
        installInputTap: { engine, tap in
            var exceptionError: NSString?
            var format: AVAudioFormat?
            let ok = AudioExceptionCatcher.`try`({
                let input = engine.inputNode
                format = input.outputFormat(forBus: 0)
                input.installTap(onBus: 0, bufferSize: 1024, format: format, block: tap)
            }, error: &exceptionError)
            guard ok else {
                return .failure(exceptionError as String? ?? "AVAudio input tap setup failed.")
            }
            return .success
        },
        prepareAndStartEngine: { engine in
            var exceptionError: NSString?
            var thrownError: Error?
            let ok = AudioExceptionCatcher.`try`({
                engine.prepare()
                do {
                    try engine.start()
                } catch {
                    thrownError = error
                }
            }, error: &exceptionError)
            guard ok else {
                return .failure(exceptionError as String? ?? "AVAudio engine start failed.")
            }
            if let thrownError {
                return .failure("Audio engine failed: \(thrownError.localizedDescription)")
            }
            return .success
        },
        stopEngine: { engine in
            var ignoredError: NSString?
            _ = AudioExceptionCatcher.`try`({
                if engine.isRunning {
                    engine.stop()
                }
            }, error: &ignoredError)
        },
        removeInputTap: { engine in
            var ignoredError: NSString?
            _ = AudioExceptionCatcher.`try`({
                engine.inputNode.removeTap(onBus: 0)
            }, error: &ignoredError)
        }
    )

    private static func portSummary(_ ports: [AVAudioSessionPortDescription]) -> String {
        guard !ports.isEmpty else { return "none" }
        return ports
            .map { "\($0.portType.rawValue):\($0.portName)" }
            .joined(separator: "|")
    }
}

@MainActor
@Observable
final class VoiceService: NSObject {
    static let shared = VoiceService()

    var isListening: Bool = false
    var isSpeaking: Bool = false
    var liveTranscript: String = ""
    var inputLevel: Double = 0
    var lastError: String?

    @ObservationIgnored private let audioEngine = AVAudioEngine()
    @ObservationIgnored private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    @ObservationIgnored private var recognitionTask: SFSpeechRecognitionTask?
    @ObservationIgnored private let synthesizer = AVSpeechSynthesizer()
    @ObservationIgnored private var recognizer: SFSpeechRecognizer?
    @ObservationIgnored private var onFinal: ((String) -> Void)?
    @ObservationIgnored private var onSpeechEnd: (() -> Void)?
    @ObservationIgnored private var cancellationID: UUID?
    @ObservationIgnored private var ttsCancellationID: UUID?
    @ObservationIgnored private var audioStartup = VoiceAudioStartup.live
    @ObservationIgnored private var inputTapInstalled = false
    @ObservationIgnored private var audioSessionObserverTokens: [NSObjectProtocol] = []

    override init() {
        super.init()
        recognizer = SFSpeechRecognizer(locale: Locale.current) ?? SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        synthesizer.delegate = self
        registerAudioSessionObservers()
    }

    deinit {
        for token in audioSessionObserverTokens {
            NotificationCenter.default.removeObserver(token)
        }
    }

    // MARK: - Permissions

    func requestPermissions() async -> Bool {
        let speechStatus: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in cont.resume(returning: status) }
        }
        guard speechStatus == .authorized else {
            lastError = "Speech recognition not authorized."
            return false
        }
        let micOK: Bool = await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { cont.resume(returning: $0) }
        }
        if !micOK { lastError = "Microphone not authorized." }
        return micOK
    }

    /// Starts listening for speech input with continuous transcript updates.
    /// - Parameters:
    ///   - onFinal: Called with the final transcript when speech ends.

    @discardableResult
    func startListening(
        permissionsAlreadyGranted: Bool = false,
        diagnosticSource: String = "voice-start",
        onFinal: @escaping (String) -> Void
    ) async -> Bool {
        if !permissionsAlreadyGranted {
            guard await requestPermissions() else { return false }
        }
        resetListeningState()
        stopSpeaking()
        lastError = nil

        let activation = audioStartup.activateAudioSession()
        guard activation.succeeded else {
            lastError = activation.error ?? "Audio session could not be activated."
            resetListeningState()
            return false
        }

        let readiness = audioStartup.inputReadinessSnapshot(audioEngine)
        if let failureReason = VoiceInputReadiness.failureReason(for: readiness) {
            lastError = failureReason
            recordVoiceStartupFailure(source: diagnosticSource, snapshot: readiness, reason: failureReason)
            resetListeningState()
            return false
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if #available(iOS 16, *) { request.addsPunctuation = true }
        recognitionRequest = request

        self.onFinal = onFinal
        liveTranscript = ""

        guard let recognizer, recognizer.isAvailable else {
            lastError = "Speech recognizer unavailable."
            resetListeningState()
            return false
        }

        cancellationID = AppCancellationBus.shared.registerCancellation({ [weak self] in
            Task { @MainActor [weak self] in self?.stopListening() }
        }, category: .voiceRecognition)

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                if let result {
                    self.liveTranscript = result.bestTranscription.formattedString
                    if result.isFinal {
                        let text = self.liveTranscript
                        let onFinal = self.onFinal
                        self.resetListeningState()
                        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            onFinal?(text)
                        }
                    }
                }
                if error != nil {
                    self.resetListeningState()
                }
            }
        }

        let tapResult = audioStartup.installInputTap(audioEngine) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
            self?.updateLevel(from: buffer)
        }
        guard tapResult.succeeded else {
            let reason = tapResult.error ?? VoiceInputReadiness.unavailableMessage
            lastError = reason
            recordVoiceStartupFailure(source: diagnosticSource, snapshot: readiness, reason: reason)
            resetListeningState()
            return false
        }
        inputTapInstalled = true

        let startResult = audioStartup.prepareAndStartEngine(audioEngine)
        guard startResult.succeeded else {
            let reason = startResult.error ?? VoiceInputReadiness.unavailableMessage
            lastError = reason
            recordVoiceStartupFailure(source: diagnosticSource, snapshot: readiness, reason: reason)
            resetListeningState()
            return false
        }
        isListening = true
        return true
    }

    func stopListening() {
        resetListeningState()
    }

    private func resetListeningState() {
        audioStartup.stopEngine(audioEngine)
        if inputTapInstalled {
            audioStartup.removeInputTap(audioEngine)
            inputTapInstalled = false
        }
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        if let cancellationID {
            AppCancellationBus.shared.unregister(cancellationID, category: .voiceRecognition)
            self.cancellationID = nil
        }
        recognitionRequest = nil
        recognitionTask = nil
        onFinal = nil
        isListening = false
        inputLevel = 0
    }

    func finishListening() {
        recognitionRequest?.endAudio()
        audioStartup.stopEngine(audioEngine)
        if inputTapInstalled {
            audioStartup.removeInputTap(audioEngine)
            inputTapInstalled = false
        }
    }

    private func recordVoiceStartupFailure(source: String, snapshot: VoiceInputReadinessSnapshot, reason: String) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .voiceStartupFailure, values: [
            "source": source,
            "phase": "audio-input-readiness",
            "routeInputs": snapshot.routeInputsSummary,
            "routeOutputs": snapshot.routeOutputsSummary,
            "availableInputs": snapshot.availableInputsSummary,
            "isInputAvailable": String(snapshot.isInputAvailable),
            "sampleRate": snapshot.sampleRate.map { String($0) } ?? "nil",
            "channelCount": snapshot.channelCount.map { String($0) } ?? "nil",
            "reason": reason
        ]))
    }

    private func registerAudioSessionObservers() {
        let center = NotificationCenter.default
        let interruption = center.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            let rawType = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            Task { @MainActor [weak self] in
                self?.handleAudioSessionInterruption(rawType: rawType)
            }
        }
        let routeChange = center.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            let rawReason = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
            Task { @MainActor [weak self] in
                self?.handleAudioRouteChange(rawReason: rawReason)
            }
        }
        audioSessionObserverTokens = [interruption, routeChange]
    }

    private func handleAudioSessionInterruption(rawType: UInt?) {
        guard let rawType, let type = AVAudioSession.InterruptionType(rawValue: rawType) else { return }
        let reason = type == .began ? "interruption-began" : "interruption-ended"
        if type == .began, isListening {
            lastError = "Voice input was interrupted by the audio system. Tap the microphone to resume."
            resetListeningState()
        }
        recordVoiceAudioSessionEvent(source: "voice-audio-session", phase: "interruption", reason: reason)
    }

    private func handleAudioRouteChange(rawReason: UInt?) {
        let reason = rawReason
            .flatMap { AVAudioSession.RouteChangeReason(rawValue: $0) }
            .map(Self.routeChangeReasonName(_:)) ?? "unknown"
        if isListening {
            lastError = "Voice input route changed. Tap the microphone to resume."
            resetListeningState()
        }
        recordVoiceAudioSessionEvent(source: "voice-audio-session", phase: "route-change", reason: reason)
    }

    private func recordVoiceAudioSessionEvent(source: String, phase: String, reason: String) {
        let session = AVAudioSession.sharedInstance()
        let route = session.currentRoute
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .voiceAudioSessionEvent, values: [
            "source": source,
            "phase": phase,
            "routeInputs": Self.portSummary(route.inputs),
            "routeOutputs": Self.portSummary(route.outputs),
            "availableInputs": Self.portSummary(session.availableInputs ?? []),
            "isInputAvailable": String(session.isInputAvailable),
            "reason": reason
        ]))
    }

    private static func routeChangeReasonName(_ reason: AVAudioSession.RouteChangeReason) -> String {
        switch reason {
        case .unknown: return "unknown"
        case .newDeviceAvailable: return "new-device-available"
        case .oldDeviceUnavailable: return "old-device-unavailable"
        case .categoryChange: return "category-change"
        case .override: return "override"
        case .wakeFromSleep: return "wake-from-sleep"
        case .noSuitableRouteForCategory: return "no-suitable-route-for-category"
        case .routeConfigurationChange: return "route-configuration-change"
        @unknown default: return "unknown-new-route-change-reason"
        }
    }

    private static func portSummary(_ ports: [AVAudioSessionPortDescription]) -> String {
        guard !ports.isEmpty else { return "none" }
        return ports
            .map { "\($0.portType.rawValue):\($0.portName)" }
            .joined(separator: "|")
    }

    #if DEBUG
    func configureAudioStartupForTests(_ startup: VoiceAudioStartup) {
        resetListeningState()
        audioStartup = startup
    }

    func resetAudioStartupForTests() {
        resetListeningState()
        audioStartup = .live
    }

    func markInputTapInstalledForTests(_ installed: Bool) {
        inputTapInstalled = installed
    }

    var inputTapInstalledForTests: Bool {
        inputTapInstalled
    }
    #endif

    private func updateLevel(from buffer: AVAudioPCMBuffer) {
        guard let data = buffer.floatChannelData?[0] else { return }
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return }
        var sum: Float = 0
        for i in 0..<frames { sum += data[i] * data[i] }
        let rms = sqrt(sum / Float(frames))
        let level = max(0, min(1, Double(rms) * 8))
        Task { @MainActor in
            self.inputLevel = level * 0.6 + self.inputLevel * 0.4
        }
    }

    // MARK: - Speaking

    func speak(_ text: String, voiceID: String?, rate: Double, onComplete: (() -> Void)? = nil) {
        let trimmed = FinalOutputSanitizer.sanitizeUserVisibleText(text).text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { onComplete?(); return }
        onSpeechEnd = onComplete
        let utterance = AVSpeechUtterance(string: trimmed)
        if let voiceID, let v = AVSpeechSynthesisVoice(identifier: voiceID) {
            utterance.voice = v
        } else {
            utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.identifier)
                ?? AVSpeechSynthesisVoice(language: "en-US")
        }
        utterance.rate = Float(max(0.1, min(0.8, rate)))
        utterance.pitchMultiplier = 1.0
        utterance.preUtteranceDelay = 0.02
        isSpeaking = true
        registerTTSCancellation()
        synthesizer.speak(utterance)
    }

    func speakChunk(_ text: String, voiceID: String?, rate: Double) {
        let trimmed = FinalOutputSanitizer.sanitizeUserVisibleText(text).text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let utterance = AVSpeechUtterance(string: trimmed)
        if let voiceID, let v = AVSpeechSynthesisVoice(identifier: voiceID) {
            utterance.voice = v
        } else {
            utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.identifier)
                ?? AVSpeechSynthesisVoice(language: "en-US")
        }
        utterance.rate = Float(max(0.1, min(0.8, rate)))
        isSpeaking = true
        registerTTSCancellation()
        synthesizer.speak(utterance)
    }

    /// Registers a handler to stop speaking on TTS cancellation.
    private func registerTTSCancellation() {
        unregisterTTSCancellation()
        ttsCancellationID = AppCancellationBus.shared.registerCancellation({ [weak self] in
            Task { @MainActor [weak self] in self?.stopSpeaking() }
        }, category: .tts)
    }

    private func unregisterTTSCancellation() {
        if let ttsCancellationID {
            AppCancellationBus.shared.unregister(ttsCancellationID, category: .tts)
            self.ttsCancellationID = nil
        }
    }

    func stopSpeaking() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        unregisterTTSCancellation()
        isSpeaking = false
    }
}

extension VoiceService: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            if !self.synthesizer.isSpeaking {
                self.unregisterTTSCancellation()
                self.isSpeaking = false
                let cb = self.onSpeechEnd
                self.onSpeechEnd = nil
                cb?()
            }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.unregisterTTSCancellation()
            self.isSpeaking = false
            self.onSpeechEnd = nil
        }
    }
}

nonisolated enum VoiceCatalog {
    struct Entry: Identifiable, Hashable, Sendable {
        var id: String
        var name: String
        var language: String
        var quality: String
    }

    @MainActor
    static func available() -> [Entry] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter {
                let currentCode = Locale.current.language.languageCode?.identifier ?? ""
                return $0.language.hasPrefix("en") || (!currentCode.isEmpty && $0.language.hasPrefix(currentCode))
            }
            .sorted { $0.name < $1.name }
            .map { v in
                let q: String
                switch v.quality {
                case .premium: q = "Premium"
                case .enhanced: q = "Enhanced"
                default: q = "Default"
                }
                return Entry(id: v.identifier, name: v.name, language: v.language, quality: q)
            }
    }
}
