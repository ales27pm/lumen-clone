import Foundation
import AVFoundation

@MainActor
final class SpeechSynthesisService: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    override init() { super.init(); synthesizer.delegate = self }
    func speak(_ text: String) { synthesizer.speak(AVSpeechUtterance(string: text)) }
    func stop() { synthesizer.stopSpeaking(at: .immediate) }
}
