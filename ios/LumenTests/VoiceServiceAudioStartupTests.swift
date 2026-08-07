import AVFoundation
import XCTest
@testable import Lumen

@MainActor
final class VoiceServiceAudioStartupTests: XCTestCase {
    override func tearDown() async throws {
        VoiceService.shared.resetAudioStartupForTests()
        try await super.tearDown()
    }

    func testStartListeningFailsGracefullyWhenInputUnavailable() async {
        let counters = AudioStartupCounters()
        let snapshot = Self.snapshot(isInputAvailable: false, sampleRate: nil, channelCount: nil)
        VoiceService.shared.configureAudioStartupForTests(Self.fakeStartup(snapshot: snapshot, counters: counters))

        let observed = DiagnosticSignalRecorder(kind: .voiceStartupFailure)
        defer { observed.stop() }

        let started = await VoiceService.shared.startListening(
            permissionsAlreadyGranted: true,
            diagnosticSource: "carplay-voice-start"
        ) { _ in }

        XCTAssertFalse(started)
        XCTAssertEqual(counters.installTapCount, 0)
        XCTAssertEqual(counters.startEngineCount, 0)
        XCTAssertTrue(VoiceService.shared.lastError?.contains("Voice input is unavailable") == true)
        XCTAssertEqual(observed.signals.last?.values["source"], "carplay-voice-start")
        XCTAssertEqual(observed.signals.last?.values["phase"], "audio-input-readiness")
        XCTAssertEqual(observed.signals.last?.values["isinputavailable"], "false")
    }

    func testInvalidInputFormatFailsBeforeTapInstallOrEngineStart() async {
        let counters = AudioStartupCounters()
        let snapshot = Self.snapshot(isInputAvailable: true, sampleRate: 0, channelCount: 0)
        VoiceService.shared.configureAudioStartupForTests(Self.fakeStartup(snapshot: snapshot, counters: counters))

        let started = await VoiceService.shared.startListening(permissionsAlreadyGranted: true) { _ in }

        XCTAssertFalse(started)
        XCTAssertEqual(counters.installTapCount, 0)
        XCTAssertEqual(counters.startEngineCount, 0)
        XCTAssertTrue(VoiceService.shared.lastError?.contains("Voice input is unavailable") == true)
    }

    func testUnsupportedOnDeviceRecognitionStopsBeforeAudioActivation() async {
        let counters = AudioStartupCounters()
        VoiceService.shared.configureAudioStartupForTests(Self.fakeStartup(snapshot: Self.validSnapshot(), counters: counters))
        VoiceService.shared.configureOnDeviceSpeechReadinessForTests(.unsupportedLocale("zz-ZZ"))

        let started = await VoiceService.shared.startListening(permissionsAlreadyGranted: true) { _ in }

        XCTAssertFalse(started)
        XCTAssertEqual(counters.activateAudioSessionCount, 0)
        XCTAssertEqual(counters.installTapCount, 0)
        XCTAssertEqual(counters.startEngineCount, 0)
        XCTAssertEqual(
            VoiceService.shared.lastError,
            "On-device speech recognition is unavailable for the selected locale (zz-ZZ)."
        )
    }

    func testOnDeviceRecognitionReadinessRejectsUnavailableRecognizer() {
        XCTAssertEqual(
            OnDeviceSpeechRecognitionReadiness.evaluate(
                recognizerExists: true,
                recognizerIsAvailable: false,
                supportsOnDeviceRecognition: true,
                localeIdentifier: "en-US"
            ),
            .recognizerUnavailable
        )
    }

    func testResetListeningStateDoesNotRemoveTapUnlessTapWasInstalled() {
        let counters = AudioStartupCounters()
        VoiceService.shared.configureAudioStartupForTests(Self.fakeStartup(snapshot: Self.validSnapshot(), counters: counters))

        VoiceService.shared.markInputTapInstalledForTests(false)
        VoiceService.shared.stopListening()
        XCTAssertEqual(counters.removeTapCount, 0)

        VoiceService.shared.markInputTapInstalledForTests(true)
        VoiceService.shared.stopListening()
        XCTAssertEqual(counters.removeTapCount, 1)
        XCTAssertFalse(VoiceService.shared.inputTapInstalledForTests)
    }

    func testCarPlayVoiceStartDoesNotCrashWhenAudioInputRouteIsUnavailable() async {
        await carPlayVoiceStartDoesNotCrashWhenAudioInputRouteIsUnavailable()
    }

    func carPlayVoiceStartDoesNotCrashWhenAudioInputRouteIsUnavailable() async {
        let counters = AudioStartupCounters()
        VoiceService.shared.configureAudioStartupForTests(Self.fakeStartup(
            snapshot: Self.snapshot(isInputAvailable: false, sampleRate: 0, channelCount: 0),
            counters: counters
        ))

        let started = await VoiceService.shared.startListening(
            permissionsAlreadyGranted: true,
            diagnosticSource: "carplay-voice-start"
        ) { _ in }

        XCTAssertFalse(started)
        XCTAssertEqual(counters.installTapCount, 0)
        XCTAssertEqual(counters.startEngineCount, 0)
    }

    private static func fakeStartup(snapshot: VoiceInputReadinessSnapshot, counters: AudioStartupCounters) -> VoiceAudioStartup {
        VoiceAudioStartup(
            activateAudioSession: {
                counters.activateAudioSessionCount += 1
                return .success
            },
            inputReadinessSnapshot: { _ in snapshot },
            installInputTap: { _, _ in
                counters.installTapCount += 1
                return .success
            },
            prepareAndStartEngine: { _ in
                counters.startEngineCount += 1
                return .success
            },
            stopEngine: { _ in
                counters.stopEngineCount += 1
            },
            removeInputTap: { _ in
                counters.removeTapCount += 1
            }
        )
    }

    private static func validSnapshot() -> VoiceInputReadinessSnapshot {
        snapshot(isInputAvailable: true, sampleRate: 44_100, channelCount: 1)
    }

    private static func snapshot(isInputAvailable: Bool, sampleRate: Double?, channelCount: UInt32?) -> VoiceInputReadinessSnapshot {
        VoiceInputReadinessSnapshot(
            isInputAvailable: isInputAvailable,
            availableInputCount: isInputAvailable ? 1 : 0,
            currentRouteInputCount: isInputAvailable ? 1 : 0,
            currentRouteOutputCount: 1,
            availableInputsSummary: isInputAvailable ? "builtInMic:iPhone Microphone" : "none",
            routeInputsSummary: isInputAvailable ? "builtInMic:iPhone Microphone" : "none",
            routeOutputsSummary: "carAudio:CarPlay",
            sampleRate: sampleRate,
            channelCount: channelCount,
            exceptionError: nil
        )
    }
}

@MainActor
private final class AudioStartupCounters {
    var activateAudioSessionCount = 0
    var installTapCount = 0
    var startEngineCount = 0
    var stopEngineCount = 0
    var removeTapCount = 0
}

private final class DiagnosticSignalRecorder {
    private var observerID: UUID?
    private let lock = NSLock()
    private var storedSignals: [PersistentRuntimeDiagnosticSignal] = []

    var signals: [PersistentRuntimeDiagnosticSignal] {
        lock.lock()
        defer { lock.unlock() }
        return storedSignals
    }

    init(kind: PersistentRuntimeDiagnosticSignalKind) {
        observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { [weak self] signal in
            guard signal.kind == kind else { return }
            self?.lock.lock()
            self?.storedSignals.append(signal)
            self?.lock.unlock()
        }
    }

    func stop() {
        guard let observerID else { return }
        PersistentRuntimeDiagnosticsObserver.shared.removeObserver(observerID)
        self.observerID = nil
    }
}

#if canImport(CarPlay)
@MainActor
final class CarPlayVoiceStartupFailureTests: XCTestCase {
    func testCarPlayStartAskResetsSessionStateWhenStartListeningReturnsFalse() {
        let delegate = CarPlayVoiceSceneDelegate()
        delegate.configureVoiceStartupFailureStateForTests()

        XCTAssertEqual(delegate.sessionStateForTests, .listening)
        XCTAssertTrue(delegate.hasListeningTimeoutTaskForTests)

        delegate.handleVoiceStartupFailureForTests("Voice input is unavailable right now.")

        XCTAssertEqual(delegate.sessionStateForTests, .unavailable)
        XCTAssertFalse(delegate.hasListeningTimeoutTaskForTests)
    }
}
#endif
