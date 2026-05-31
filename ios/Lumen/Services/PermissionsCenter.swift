import Foundation
import Observation
import UIKit
import EventKit
import Contacts
import CoreLocation
import AVFoundation
import Speech
import Photos
import CoreMotion
import HealthKit
import UserNotifications
#if canImport(AlarmKit)
import AlarmKit
#endif

enum PermissionState: Sendable, Equatable {
    case notDetermined
    case granted
    case limited
    case denied
    case restricted
    case unavailable

    var label: String {
        switch self {
        case .notDetermined: return "Not set"
        case .granted: return "Allowed"
        case .limited: return "Limited"
        case .denied: return "Denied"
        case .restricted: return "Restricted"
        case .unavailable: return "Unavailable"
        }
    }

    var systemImage: String {
        switch self {
        case .granted: return "checkmark.circle.fill"
        case .limited: return "checkmark.circle"
        case .denied, .restricted: return "xmark.circle.fill"
        case .unavailable: return "minus.circle"
        case .notDetermined: return "circle.dashed"
        }
    }

    var tint: UIColor {
        switch self {
        case .granted, .limited: return .systemGreen
        case .denied, .restricted: return .systemRed
        case .unavailable: return .systemGray
        case .notDetermined: return .systemOrange
        }
    }

    var needsSystemSettings: Bool {
        self == .denied || self == .restricted || self == .limited
    }
}

nonisolated enum PermissionKind: String, CaseIterable, Identifiable, Sendable {
    case calendar, reminders, contacts, location, microphone, speech, camera, photos, motion, health, notifications, alarms

    var id: String { rawValue }

    var title: String {
        switch self {
        case .calendar: return "Calendar"
        case .reminders: return "Reminders"
        case .contacts: return "Contacts"
        case .location: return "Location"
        case .microphone: return "Microphone"
        case .speech: return "Speech Recognition"
        case .camera: return "Camera"
        case .photos: return "Photo Library"
        case .motion: return "Motion & Fitness"
        case .health: return "Health"
        case .notifications: return "Notifications"
        case .alarms: return "Alarms"
        }
    }

    var systemImage: String {
        switch self {
        case .calendar: return "calendar"
        case .reminders: return "checklist"
        case .contacts: return "person.crop.circle"
        case .location: return "location"
        case .microphone: return "mic"
        case .speech: return "waveform"
        case .camera: return "camera"
        case .photos: return "photo.on.rectangle"
        case .motion: return "figure.walk.motion"
        case .health: return "heart.text.square"
        case .notifications: return "bell.badge"
        case .alarms: return "alarm.waves.left.and.right"
        }
    }

    var rationale: String {
        switch self {
        case .calendar: return "Create and read events you ask about."
        case .reminders: return "Add and list tasks you ask about."
        case .contacts: return "Look up people when you ask about them."
        case .location: return "Answer location-aware questions."
        case .microphone: return "Capture voice input for chat."
        case .speech: return "Transcribe voice input on-device."
        case .camera: return "Capture images when you ask."
        case .photos: return "Search and summarize your library."
        case .motion: return "Summarize steps and activity."
        case .health: return "Read activity and sleep metrics."
        case .notifications: return "Deliver trigger results in the background."
        case .alarms: return "Allow scheduling and managing AlarmKit alarms."
        }
    }

    init?(usageDescriptionKey: String) {
        switch usageDescriptionKey {
        case "NSCalendarsFullAccessUsageDescription": self = .calendar
        case "NSRemindersFullAccessUsageDescription": self = .reminders
        case "NSContactsUsageDescription": self = .contacts
        case "NSLocationWhenInUseUsageDescription": self = .location
        case "NSLocationAlwaysAndWhenInUseUsageDescription", "NSLocationAlwaysUsageDescription": self = .location
        case "NSMicrophoneUsageDescription": self = .microphone
        case "NSSpeechRecognitionUsageDescription": self = .speech
        case "NSCameraUsageDescription": self = .camera
        case "NSPhotoLibraryUsageDescription": self = .photos
        case "NSMotionUsageDescription": self = .motion
        case "NSHealthShareUsageDescription": self = .health
        case "NSAlarmKitUsageDescription": self = .alarms
        default: return nil
        }
    }
}

@MainActor
@Observable
final class PermissionsCenter {
    static let shared = PermissionsCenter()

    private(set) var states: [PermissionKind: PermissionState] = [:]
    private(set) var lastRequestMessages: [PermissionKind: String] = [:]

    @ObservationIgnored private let healthStore = HKHealthStore()
    @ObservationIgnored private let motionActivity = CMMotionActivityManager()
    @ObservationIgnored private var foregroundObserver: NSObjectProtocol?
    @ObservationIgnored private var activeObserver: NSObjectProtocol?

    private init() {
        let center = NotificationCenter.default
        foregroundObserver = center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.refreshAlarmState() }
        }
        activeObserver = center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.refreshAlarmState() }
        }
        refreshAll()
    }

    deinit {
        if let foregroundObserver { NotificationCenter.default.removeObserver(foregroundObserver) }
        if let activeObserver { NotificationCenter.default.removeObserver(activeObserver) }
    }

    func state(_ kind: PermissionKind) -> PermissionState { states[kind] ?? .notDetermined }
    func lastRequestMessage(_ kind: PermissionKind) -> String? { lastRequestMessages[kind] }

    func refreshAll() {
        for kind in PermissionKind.allCases { states[kind] = readCurrentState(kind) }
        Task { @MainActor in
            let n = await currentNotificationsState()
            states[.notifications] = n
        }
    }

    private func readCurrentState(_ kind: PermissionKind) -> PermissionState {
        switch kind {
        case .calendar: return mapEKStatus(EKEventStore.authorizationStatus(for: .event), isFullAccess: true)
        case .reminders: return mapEKStatus(EKEventStore.authorizationStatus(for: .reminder), isFullAccess: true)
        case .contacts: return mapCNStatus(CNContactStore.authorizationStatus(for: .contacts))
        case .location:
            let mgr = CLLocationManager()
            return mapCLStatus(mgr.authorizationStatus)
        case .microphone: return mapMicStatus(AVAudioApplication.shared.recordPermission)
        case .speech: return mapSpeechStatus(SFSpeechRecognizer.authorizationStatus())
        case .camera: return mapCameraStatus(AVCaptureDevice.authorizationStatus(for: .video))
        case .photos: return mapPhotosStatus(PHPhotoLibrary.authorizationStatus(for: .readWrite))
        case .motion:
            guard CMMotionActivityManager.isActivityAvailable() else { return .unavailable }
            return mapMotionStatus(CMMotionActivityManager.authorizationStatus())
        case .health:
            guard HKHealthStore.isHealthDataAvailable() else { return .unavailable }
            let asked = UserDefaults.standard.bool(forKey: "perm.health.asked")
            return asked ? .granted : .notDetermined
        case .notifications: return states[.notifications] ?? .notDetermined
        case .alarms: return currentAlarmState()
        }
    }

    private func currentNotificationsState() async -> PermissionState {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        switch settings.authorizationStatus {
        case .authorized, .ephemeral: return .granted
        case .provisional: return .limited
        case .denied: return .denied
        case .notDetermined: return .notDetermined
        @unknown default: return .notDetermined
        }
    }

    func request(_ kind: PermissionKind) async {
        lastRequestMessages[kind] = nil
        let current = state(kind)
        if current.needsSystemSettings {
            lastRequestMessages[kind] = "This permission must be changed in iOS Settings."
            openSystemSettings()
            return
        }

        switch kind {
        case .calendar:
            let ok = (try? await EKEventStore().requestFullAccessToEvents()) ?? false
            lastRequestMessages[kind] = ok ? "Calendar access granted." : "Calendar access was not granted."
        case .reminders:
            let ok = (try? await EKEventStore().requestFullAccessToReminders()) ?? false
            lastRequestMessages[kind] = ok ? "Reminder access granted." : "Reminder access was not granted."
        case .contacts:
            let ok = (try? await CNContactStore().requestAccess(for: .contacts)) ?? false
            lastRequestMessages[kind] = ok ? "Contacts access granted." : "Contacts access was not granted."
        case .location:
            await requestLocation()
        case .microphone:
            let ok = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in AVAudioApplication.requestRecordPermission { cont.resume(returning: $0) } }
            lastRequestMessages[kind] = ok ? "Microphone access granted." : "Microphone access was not granted."
        case .speech:
            let status = await withCheckedContinuation { (cont: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) } }
            lastRequestMessages[kind] = "Speech recognition result: \(status)."
        case .camera:
            let ok = await AVCaptureDevice.requestAccess(for: .video)
            lastRequestMessages[kind] = ok ? "Camera access granted." : "Camera access was not granted."
        case .photos:
            let status = await withCheckedContinuation { (cont: CheckedContinuation<PHAuthorizationStatus, Never>) in PHPhotoLibrary.requestAuthorization(for: .readWrite) { cont.resume(returning: $0) } }
            lastRequestMessages[kind] = "Photo library result: \(status)."
        case .motion:
            await requestMotion()
        case .health:
            await requestHealth()
        case .notifications:
            let ok = (try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])) ?? false
            lastRequestMessages[kind] = ok ? "Notifications granted." : "Notifications were not granted."
            TriggerScheduler.shared.lastPermissionGranted = ok
        case .alarms:
            let message = await AlarmTools.requestAuthorization()
            lastRequestMessages[kind] = message
            await refreshAlarmStateAfterAuthorization()
        }

        refreshAll()
    }

    private func refreshAlarmState() { states[.alarms] = currentAlarmState() }

    private func refreshAlarmStateAfterAuthorization() async {
        refreshAlarmState()
        try? await Task.sleep(for: .milliseconds(350))
        refreshAlarmState()
        lastRequestMessages[.alarms] = [lastRequestMessages[.alarms], "Current AlarmKit state: \(state(.alarms).label)."].compactMap { $0 }.joined(separator: "\n")
    }

    private func requestLocation() async {
        let mgr = CLLocationManager()
        if mgr.authorizationStatus == .notDetermined {
            let holder = LocationAuthWaiter()
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                holder.onChange = { cont.resume() }
                mgr.delegate = holder
                mgr.requestWhenInUseAuthorization()
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(10))
                    holder.finishOnce()
                }
                _ = holder
            }
            _ = mgr
            lastRequestMessages[.location] = "Location request completed."
        } else {
            lastRequestMessages[.location] = "Location permission must be changed in iOS Settings."
            openSystemSettings()
        }
    }

    private func requestMotion() async {
        guard CMMotionActivityManager.isActivityAvailable() else {
            lastRequestMessages[.motion] = "Motion activity is unavailable on this device."
            return
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            let end = Date()
            let start = end.addingTimeInterval(-60)
            motionActivity.queryActivityStarting(from: start, to: end, to: .main) { _, _ in cont.resume() }
        }
        lastRequestMessages[.motion] = "Motion permission request completed."
    }

    private func requestHealth() async {
        guard HKHealthStore.isHealthDataAvailable() else {
            lastRequestMessages[.health] = "Health data is unavailable on this device."
            return
        }
        let readTypes: Set<HKObjectType> = [HKQuantityType(.stepCount), HKQuantityType(.heartRate), HKQuantityType(.activeEnergyBurned), HKQuantityType(.distanceWalkingRunning), HKCategoryType(.sleepAnalysis)]
        do {
            try await healthStore.requestAuthorization(toShare: [], read: readTypes)
            UserDefaults.standard.set(true, forKey: "perm.health.asked")
            lastRequestMessages[.health] = "Health authorization request completed."
        } catch {
            lastRequestMessages[.health] = "Health authorization failed: \(error.localizedDescription)"
        }
    }

    func openSystemSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        Task { @MainActor in _ = await UIApplication.shared.open(url) }
    }

    private func mapEKStatus(_ status: EKAuthorizationStatus, isFullAccess: Bool) -> PermissionState {
        switch status {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        case .restricted: return .restricted
        case .authorized: return .granted
        case .writeOnly: return .limited
        case .fullAccess: return .granted
        @unknown default: return .notDetermined
        }
    }

    private func mapCNStatus(_ status: CNAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        case .restricted: return .restricted
        case .authorized: return .granted
        case .limited: return .limited
        @unknown default: return .notDetermined
        }
    }

    private func mapCLStatus(_ status: CLAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        case .restricted: return .restricted
        case .authorizedWhenInUse, .authorizedAlways: return .granted
        @unknown default: return .notDetermined
        }
    }

    private func mapMicStatus(_ status: AVAudioApplication.recordPermission) -> PermissionState {
        switch status {
        case .undetermined: return .notDetermined
        case .denied: return .denied
        case .granted: return .granted
        @unknown default: return .notDetermined
        }
    }

    private func mapSpeechStatus(_ status: SFSpeechRecognizerAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        case .restricted: return .restricted
        case .authorized: return .granted
        @unknown default: return .notDetermined
        }
    }

    private func mapCameraStatus(_ status: AVAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        case .restricted: return .restricted
        case .authorized: return .granted
        @unknown default: return .notDetermined
        }
    }

    private func mapPhotosStatus(_ status: PHAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        case .restricted: return .restricted
        case .authorized: return .granted
        case .limited: return .limited
        @unknown default: return .notDetermined
        }
    }

    private func mapMotionStatus(_ status: CMAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        case .restricted: return .restricted
        case .authorized: return .granted
        @unknown default: return .notDetermined
        }
    }

    private func currentAlarmState() -> PermissionState {
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            switch AlarmManager.shared.authorizationState {
            case .notDetermined: return .notDetermined
            case .authorized: return .granted
            case .denied: return .denied
            @unknown default: return .unavailable
            }
        }
#endif
        return .unavailable
    }
}

@MainActor
final class LocationAuthWaiter: NSObject, CLLocationManagerDelegate {
    /// Concurrency contract: all state mutation and callback dispatch are confined to MainActor.
    var onChange: (() -> Void)?
    private var done = false

    func finishOnce() {
        MainActor.preconditionIsolated()
        if done { return }
        done = true
        let cb = onChange
        onChange = nil
        cb?()
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            if manager.authorizationStatus != .notDetermined {
                self.finishOnce()
            }
        }
    }
}
