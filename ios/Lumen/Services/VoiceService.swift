import Foundation
import AVFoundation
import Speech
import Observation

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

    override init() {
        super.init()
        recognizer = SFSpeechRecognizer(locale: Locale.current) ?? SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        synthesizer.delegate = self
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

    // MARK: - Listening

    func startListening(onFinal: @escaping (String) -> Void) async {
        guard await requestPermissions() else { return }
        stopListening()
        stopSpeaking()

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .spokenAudio, options: [.duckOthers, .defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            lastError = "Audio session error: \(error.localizedDescription)"
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if #available(iOS 16, *) { request.addsPunctuation = true }
        recognitionRequest = request

        self.onFinal = onFinal
        liveTranscript = ""

        guard let recognizer, recognizer.isAvailable else {
            lastError = "Speech recognizer unavailable."
            return
        }

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                if let result {
                    self.liveTranscript = result.bestTranscription.formattedString
                    if result.isFinal {
                        let text = self.liveTranscript
                        self.stopListening()
                        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            self.onFinal?(text)
                        }
                    }
                }
                if error != nil {
                    self.stopListening()
                }
            }
        }

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
            self?.updateLevel(from: buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            isListening = true
        } catch {
            lastError = "Audio engine failed: \(error.localizedDescription)"
        }
    }

    func stopListening() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionRequest = nil
        recognitionTask = nil
        isListening = false
        inputLevel = 0
    }

    func finishListening() {
        recognitionRequest?.endAudio()
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
    }

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
        synthesizer.speak(utterance)
    }

    func stopSpeaking() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
    }
}

extension VoiceService: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            if !self.synthesizer.isSpeaking {
                self.isSpeaking = false
                let cb = self.onSpeechEnd
                self.onSpeechEnd = nil
                cb?()
            }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
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
