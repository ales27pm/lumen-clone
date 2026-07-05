import Foundation
import Observation

@MainActor
@Observable
final class VoiceSessionController {
    var state: VoiceSessionState = .idle
    var partialTranscript: String = ""
    var finalTranscript: String = ""
    var lastErrorReason: String?

    private let recognition: SpeechRecognitionService
    private let synthesis: SpeechSynthesisService
    private let legacyVoice: VoiceService
    private var transcriptPollTask: Task<Void, Never>?
    private var transcriptPollCancellationID: UUID?

    init(
        recognition: SpeechRecognitionService? = nil,
        synthesis: SpeechSynthesisService? = nil,
        legacyVoice: VoiceService? = nil
    ) {
        self.recognition = recognition ?? SpeechRecognitionService()
        self.synthesis = synthesis ?? SpeechSynthesisService()
        self.legacyVoice = legacyVoice ?? .shared
    }

    func startPushToTalk(onFinal: @escaping (String) -> Void) async {
        state = .requestingPermissions
        let ok = await recognition.requestPermissions()
        guard ok else {
            lastErrorReason = "microphone_or_speech_denied"
            state = .denied("microphone_or_speech_denied")
            return
        }
        legacyVoice.stopSpeaking()
        let started = await legacyVoice.startListening(permissionsAlreadyGranted: true) { [weak self] text in
            Task { @MainActor in
                self?.stopTranscriptPolling()
                self?.finalTranscript = text
                self?.state = .processing
                onFinal(text)
            }
        }
        guard started else {
            let reason = legacyVoice.lastError ?? VoiceInputReadiness.unavailableMessage
            lastErrorReason = reason
            partialTranscript = ""
            state = .failed(reason)
            return
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
                try? await Task.sleep(for: .milliseconds(VoiceConversationPolicy.transcriptPollMilliseconds))
            }
        }
        if let transcriptPollTask {
            transcriptPollCancellationID = AppCancellationBus.shared.register(transcriptPollTask, category: .voiceRecognition)
        }
    }

    private func stopTranscriptPolling() {
        transcriptPollTask?.cancel()
        transcriptPollTask = nil
        if let transcriptPollCancellationID {
            AppCancellationBus.shared.unregister(transcriptPollCancellationID, category: .voiceRecognition)
            self.transcriptPollCancellationID = nil
        }
    }
}

nonisolated enum VoiceConversationPolicy {
    static let transcriptPollMilliseconds = 200
    static let streamingMinimumCharactersBeforeSoftBoundary = 48
}

nonisolated enum VoiceTurnCompletionPolicy {
    static func acceptsSpeechCompletion(turnID: UUID, activeSpeechTurnID: UUID?) -> Bool {
        activeSpeechTurnID == turnID
    }

    static func shouldResumeHandsFree(handsFree: Bool, turnID: UUID, activeSpeechTurnID: UUID?) -> Bool {
        handsFree && acceptsSpeechCompletion(turnID: turnID, activeSpeechTurnID: activeSpeechTurnID)
    }
}

nonisolated struct VoiceSpeechChunk: Equatable, Sendable {
    var text: String
    var nextOffset: Int
}

nonisolated enum VoiceStreamingChunker {
    private static let strongBoundaries: Set<Character> = [".", "!", "?", "\n"]
    private static let softBoundaries: Set<Character> = [",", ";", ":"]

    static func nextChunk(
        in text: String,
        startingAt offset: Int,
        finishedStreaming: Bool,
        minimumStreamingCharacters: Int = VoiceConversationPolicy.streamingMinimumCharactersBeforeSoftBoundary
    ) -> VoiceSpeechChunk? {
        let characters = Array(text)
        guard offset < characters.count else { return nil }
        let remaining = Array(characters[offset...])
        var end = remaining.count

        if !finishedStreaming {
            if let strongIndex = remaining.lastIndex(where: { strongBoundaries.contains($0) }) {
                end = strongIndex + 1
            } else if remaining.count >= minimumStreamingCharacters,
                      let softIndex = remaining.lastIndex(where: { softBoundaries.contains($0) }) {
                end = softIndex + 1
            } else {
                return nil
            }
        }

        let raw = String(remaining[..<end])
        return VoiceSpeechChunk(
            text: raw.trimmingCharacters(in: .whitespacesAndNewlines),
            nextOffset: offset + end
        )
    }
}
