import Foundation
import Observation

@MainActor
@Observable
final class VoiceSessionController {
    var state: VoiceSessionState = .idle
    var partialTranscript: String = ""
    var finalTranscript: String = ""
    var lastErrorReason: String?

    private let recognition = SpeechRecognitionService()
    private let synthesis = SpeechSynthesisService()
    private let legacyVoice = VoiceService.shared
    private var transcriptPollTask: Task<Void, Never>?

    func startPushToTalk(onFinal: @escaping (String) -> Void) async {
        state = .requestingPermissions
        let ok = await recognition.requestPermissions()
        guard ok else {
            state = .denied("microphone_or_speech_denied")
            return
        }
        legacyVoice.stopSpeaking()
        await legacyVoice.startListening { [weak self] text in
            Task { @MainActor in
                self?.stopTranscriptPolling()
                self?.finalTranscript = text
                self?.state = .processing
                onFinal(text)
            }
        }
        state = .listening
        partialTranscript = legacyVoice.liveTranscript
        startTranscriptPolling()
    }

    func finishListening() {
        stopTranscriptPolling()
        legacyVoice.finishListening()
        state = .processing
    }

    func pollTranscript() {
        partialTranscript = legacyVoice.liveTranscript
    }

    func startSpeaking() { state = .speaking }

    func speakChunk(_ text: String, voiceID: String?, rate: Double) {
        synthesis.stop()
        legacyVoice.speakChunk(text, voiceID: voiceID, rate: rate)
        state = .speaking
    }

    func stopSpeaking() {
        legacyVoice.stopSpeaking()
        synthesis.stop()
        if state == .speaking { state = .idle }
    }

    func handleAppDidEnterBackground() {
        if VoiceInterruptionHandler.shouldInterruptOnBackground() && (state == .listening || state == .speaking || state == .processing) {
            stopTranscriptPolling()
            legacyVoice.stopListening()
            legacyVoice.stopSpeaking()
            synthesis.stop()
            state = .interrupted
        }
    }

    func cancel() {
        stopTranscriptPolling()
        legacyVoice.stopListening()
        legacyVoice.stopSpeaking()
        synthesis.stop()
        partialTranscript = ""
        state = .idle
    }

    private func startTranscriptPolling() {
        stopTranscriptPolling()
        transcriptPollTask = Task { @MainActor in
            while !Task.isCancelled && state == .listening {
                partialTranscript = legacyVoice.liveTranscript
                try? await Task.sleep(for: .milliseconds(120))
            }
        }
    }

    private func stopTranscriptPolling() {
        transcriptPollTask?.cancel()
        transcriptPollTask = nil
    }
}
