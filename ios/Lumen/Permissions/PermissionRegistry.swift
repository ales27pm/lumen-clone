import Foundation
import AVFoundation
import Speech
import Photos
import CoreLocation
import EventKit
import Contacts
import UserNotifications
import CoreMotion

@MainActor
final class PermissionRegistry: NSObject, CLLocationManagerDelegate {
    static let shared = PermissionRegistry()
    private let locationManager = CLLocationManager()
    private var locationContinuation: CheckedContinuation<PermissionRequestResult, Never>?
    private var networkAccessEnabled = false

    override init() {
        super.init()
        locationManager.delegate = self
    }

    func setNetworkAccessEnabled(_ enabled: Bool) {
        networkAccessEnabled = enabled
    }

    func currentStatus(for domain: PermissionDomain) async -> AssistantPermissionState {
        switch domain {
        case .microphone:
            switch AVAudioApplication.shared.recordPermission {
            case .granted: return .granted
            case .denied: return .denied
            case .undetermined: return .notDetermined
            @unknown default: return .unknown
            }
        case .speechRecognition:
            return mapSpeech(SFSpeechRecognizer.authorizationStatus())
        case .camera:
            return mapAV(AVCaptureDevice.authorizationStatus(for: .video))
        case .photoLibrary:
            return mapPhoto(PHPhotoLibrary.authorizationStatus(for: .readWrite))
        case .locationWhenInUse:
            switch locationManager.authorizationStatus {
            case .authorizedWhenInUse, .authorizedAlways: return .granted
            case .denied: return .denied
            case .restricted: return .restricted
            case .notDetermined: return .notDetermined
            @unknown default: return .unknown
            }
        case .calendars:
            return mapEventKit(EKEventStore.authorizationStatus(for: .event))
        case .reminders:
            return mapEventKit(EKEventStore.authorizationStatus(for: .reminder))
        case .contacts:
            switch CNContactStore.authorizationStatus(for: .contacts) {
            case .authorized: return .granted
            case .denied: return .denied
            case .restricted: return .restricted
            case .notDetermined: return .notDetermined
            @unknown default: return .unknown
            }
        case .notifications:
            let settings = await UNUserNotificationCenter.current().notificationSettings()
            switch settings.authorizationStatus {
            case .authorized, .provisional, .ephemeral: return .granted
            case .denied: return .denied
            case .notDetermined: return .notDetermined
            @unknown default: return .unknown
            }
        case .localNetwork:
            return Bundle.main.object(forInfoDictionaryKey: "NSLocalNetworkUsageDescription") == nil ? .unavailable : .unknown
        case .motion:
            return CMMotionActivityManager.isActivityAvailable() ? .unknown : .unavailable
        case .appIntents, .filesUserSelected:
            return .granted
        case .networkAccess:
            return networkAccessEnabled ? .granted : .denied
        }
    }

    func request(_ domain: PermissionDomain) async -> PermissionRequestResult {
        switch domain {
        case .microphone:
            let granted = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
                AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
            }
            return .init(domain: domain, state: granted ? .granted : .denied, message: granted ? "Granted" : "Denied")
        case .speechRecognition:
            let status: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
            }
            return .init(domain: domain, state: mapSpeech(status), message: "Speech authorization updated")
        case .camera:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            return .init(domain: domain, state: granted ? .granted : .denied, message: granted ? "Granted" : "Denied")
        case .photoLibrary:
            let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            return .init(domain: domain, state: mapPhoto(status), message: "Photo authorization updated")
        case .locationWhenInUse:
            guard locationContinuation == nil else {
                return .init(domain: domain, state: await currentStatus(for: domain), message: "Location request already in progress")
            }
            return await withCheckedContinuation { continuation in
                self.locationContinuation = continuation
                locationManager.requestWhenInUseAuthorization()
            }
        case .calendars:
            let store = EKEventStore()
            let granted = (try? await store.requestFullAccessToEvents()) ?? false
            return .init(domain: domain, state: granted ? .granted : .denied, message: granted ? "Granted" : "Denied")
        case .reminders:
            let store = EKEventStore()
            let granted = (try? await store.requestFullAccessToReminders()) ?? false
            return .init(domain: domain, state: granted ? .granted : .denied, message: granted ? "Granted" : "Denied")
        case .contacts:
            let store = CNContactStore()
            let granted = (try? await store.requestAccess(for: .contacts)) ?? false
            return .init(domain: domain, state: granted ? .granted : .denied, message: granted ? "Granted" : "Denied")
        case .notifications:
            let granted = (try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])) ?? false
            return .init(domain: domain, state: granted ? .granted : .denied, message: granted ? "Granted" : "Denied")
        case .networkAccess:
            return .init(domain: domain, state: await currentStatus(for: domain), message: "Controlled by in-app setting")
        default:
            return .init(domain: domain, state: await currentStatus(for: domain), message: "No runtime prompt")
        }
    }

    func userFacingReason(for domain: PermissionDomain) -> String {
        "Lumen uses \(domain.rawValue) only for explicit tool requests."
    }

    func requiredInfoPlistKeys(for domain: PermissionDomain) -> [String] {
        switch domain {
        case .microphone: return ["NSMicrophoneUsageDescription"]
        case .speechRecognition: return ["NSSpeechRecognitionUsageDescription"]
        case .camera: return ["NSCameraUsageDescription"]
        case .photoLibrary: return ["NSPhotoLibraryUsageDescription"]
        case .locationWhenInUse: return ["NSLocationWhenInUseUsageDescription"]
        case .calendars: return ["NSCalendarsUsageDescription", "NSCalendarsFullAccessUsageDescription"]
        case .reminders: return ["NSRemindersUsageDescription", "NSRemindersFullAccessUsageDescription"]
        case .contacts: return ["NSContactsUsageDescription"]
        case .localNetwork: return ["NSLocalNetworkUsageDescription"]
        default: return [String]()
        }
    }

    func diagnostics() async -> [PermissionDomain: AssistantPermissionState] {
        var out: [PermissionDomain: AssistantPermissionState] = [:]
        for domain in PermissionDomain.allCases {
            out[domain] = await currentStatus(for: domain)
        }
        return out
    }

    private func mapEventKit(_ status: EKAuthorizationStatus) -> AssistantPermissionState {
        switch status {
        case .fullAccess: return .granted
        case .writeOnly: return .limited
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        @unknown default: return .unknown
        }
    }

    private func mapSpeech(_ status: SFSpeechRecognizerAuthorizationStatus) -> AssistantPermissionState {
        switch status {
        case .authorized: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        @unknown default: return .unknown
        }
    }

    private func mapPhoto(_ status: PHAuthorizationStatus) -> AssistantPermissionState {
        switch status {
        case .authorized, .limited: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        @unknown default: return .unknown
        }
    }

    private func mapAV(_ status: AVAuthorizationStatus) -> AssistantPermissionState {
        switch status {
        case .authorized: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        @unknown default: return .unknown
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        guard let continuation = locationContinuation else { return }
        locationContinuation = nil
        Task { @MainActor in
            let state = await currentStatus(for: .locationWhenInUse)
            continuation.resume(returning: .init(domain: .locationWhenInUse, state: state, message: "Location authorization updated"))
        }
    }
}
