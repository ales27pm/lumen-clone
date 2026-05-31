import Foundation
import CoreLocation

/// One-shot location fetch with a per-call delegate — no shared singleton state.
@MainActor
enum LocationProbe {
    @MainActor private static var activeDelegates: [UUID: AnyObject] = [:]

    static func currentCoordinate(timeout: TimeInterval = 8) async -> CLLocationCoordinate2D? {
        let manager = CLLocationManager()
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        let status = manager.authorizationStatus
        if status == .denied || status == .restricted {
            return nil
        }

        return await withCheckedContinuation { (cont: CheckedContinuation<CLLocationCoordinate2D?, Never>) in
            let token = UUID()
            let delegate = SingleShotLocationDelegate(manager: manager) { coord in
                activeDelegates[token] = nil
                cont.resume(returning: coord)
            }
            activeDelegates[token] = delegate
            manager.delegate = delegate

            Task { @MainActor in
                try? await Task.sleep(for: .seconds(timeout))
                delegate.finish(with: nil)
            }

            delegate.begin()
        }
    }

    static func currentDescription(timeout: TimeInterval = 8) async -> String {
        let manager = CLLocationManager()
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        let status = manager.authorizationStatus
        if status == .denied || status == .restricted {
            return "Location access was denied."
        }

        return await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            let token = UUID()
            let delegate = SingleShotDescriptionDelegate(manager: manager) { text in
                activeDelegates[token] = nil
                cont.resume(returning: text)
            }
            activeDelegates[token] = delegate
            manager.delegate = delegate

            Task { @MainActor in
                try? await Task.sleep(for: .seconds(timeout))
                delegate.finish(with: "Couldn't get location (timed out).")
            }

            delegate.begin()
        }
    }
}

private enum LocationAuthorizationAction {
    case requestLocation
    case requestWhenInUseAuthorization
    case deniedOrRestricted
    case waitForChoice
    case unknown
}

private func locationAuthorizationAction(for status: CLAuthorizationStatus) -> LocationAuthorizationAction {
    switch status {
    case .authorizedAlways, .authorizedWhenInUse:
        return .requestLocation
    case .notDetermined:
        return .requestWhenInUseAuthorization
    case .denied, .restricted:
        return .deniedOrRestricted
    @unknown default:
        return .unknown
    }
}

private func locationAuthorizationUpdateAction(for status: CLAuthorizationStatus) -> LocationAuthorizationAction {
    switch status {
    case .authorizedAlways, .authorizedWhenInUse:
        return .requestLocation
    case .notDetermined:
        return .waitForChoice
    case .denied, .restricted:
        return .deniedOrRestricted
    @unknown default:
        return .unknown
    }
}

private func logUnknownAuthorizationStatus(_ status: CLAuthorizationStatus, source: String) {
    NSLog("[LocationProbe] Unknown authorization status at %@: rawValue=%d", source, status.rawValue)
}

@MainActor
final class SingleShotLocationDelegate: NSObject, CLLocationManagerDelegate {
    /// Concurrency contract: callbacks are normalized onto MainActor before reading or mutating state.
    private let manager: CLLocationManager
    private let handler: (CLLocationCoordinate2D?) -> Void
    private var done = false

    init(manager: CLLocationManager, handler: @escaping (CLLocationCoordinate2D?) -> Void) {
        self.manager = manager
        self.handler = handler
    }

    func begin() {
        MainActor.preconditionIsolated()
        switch locationAuthorizationAction(for: manager.authorizationStatus) {
        case .requestLocation:
            manager.requestLocation()
        case .requestWhenInUseAuthorization:
            manager.requestWhenInUseAuthorization()
        case .deniedOrRestricted:
            finish(with: nil)
        case .waitForChoice:
            break
        case .unknown:
            logUnknownAuthorizationStatus(manager.authorizationStatus, source: "SingleShotLocationDelegate.begin")
            finish(with: nil)
        }
    }

    func finish(with coord: CLLocationCoordinate2D?) {
        MainActor.preconditionIsolated()
        if done { return }
        done = true
        handler(coord)
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        let coord = locations.last?.coordinate
        Task { @MainActor in
            self.finish(with: coord)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        Task { @MainActor in
            self.finish(with: nil)
        }
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            guard !self.done else { return }
            switch locationAuthorizationUpdateAction(for: manager.authorizationStatus) {
            case .requestLocation:
                manager.requestLocation()
            case .deniedOrRestricted:
                self.finish(with: nil)
            case .waitForChoice, .requestWhenInUseAuthorization:
                break
            case .unknown:
                logUnknownAuthorizationStatus(manager.authorizationStatus, source: "SingleShotLocationDelegate.locationManagerDidChangeAuthorization")
                self.finish(with: nil)
            }
        }
    }
}

@MainActor
final class SingleShotDescriptionDelegate: NSObject, CLLocationManagerDelegate {
    /// Concurrency contract: callbacks are normalized onto MainActor before reading or mutating state.
    private let manager: CLLocationManager
    private let handler: (String) -> Void
    private var done = false

    init(manager: CLLocationManager, handler: @escaping (String) -> Void) {
        self.manager = manager
        self.handler = handler
    }

    func begin() {
        MainActor.preconditionIsolated()
        switch locationAuthorizationAction(for: manager.authorizationStatus) {
        case .requestLocation:
            manager.requestLocation()
        case .requestWhenInUseAuthorization:
            manager.requestWhenInUseAuthorization()
        case .deniedOrRestricted:
            finish(with: "Location access was denied.")
        case .waitForChoice:
            break
        case .unknown:
            logUnknownAuthorizationStatus(manager.authorizationStatus, source: "SingleShotDescriptionDelegate.begin")
            finish(with: "Couldn't determine location authorization state.")
        }
    }

    func finish(with text: String) {
        MainActor.preconditionIsolated()
        if done { return }
        done = true
        handler(text)
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        let message: String
        if let loc = locations.last {
            let c = loc.coordinate
            message = String(format: "Current location: %.4f, %.4f (±%.0fm)", c.latitude, c.longitude, loc.horizontalAccuracy)
        } else {
            message = "Couldn't get location."
        }

        Task { @MainActor in
            self.finish(with: message)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        Task { @MainActor in
            self.finish(with: "Couldn't get location: \(error.localizedDescription)")
        }
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            guard !self.done else { return }
            switch locationAuthorizationUpdateAction(for: manager.authorizationStatus) {
            case .requestLocation:
                manager.requestLocation()
            case .deniedOrRestricted:
                self.finish(with: "Location access was denied.")
            case .waitForChoice, .requestWhenInUseAuthorization:
                break
            case .unknown:
                logUnknownAuthorizationStatus(manager.authorizationStatus, source: "SingleShotDescriptionDelegate.locationManagerDidChangeAuthorization")
                self.finish(with: "Couldn't determine location authorization state.")
            }
        }
    }
}
