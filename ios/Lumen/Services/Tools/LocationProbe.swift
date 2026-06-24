import Foundation
import CoreLocation

/// One-shot location fetch with a per-call delegate — no shared singleton state.
@MainActor
enum LocationProbe {
    @MainActor private static var activeDelegates: [UUID: AnyObject] = [:]

    static func currentCoordinate(timeout: TimeInterval = 8) async -> CLLocationCoordinate2D? {
        guard case .success(let coordinate) = await currentCoordinateResult(timeout: timeout) else {
            return nil
        }
        return coordinate
    }

    static func currentCoordinateResult(timeout: TimeInterval = 8) async -> LocationCoordinateProbeResult {
        let manager = CLLocationManager()
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        let status = manager.authorizationStatus
        switch locationAuthorizationAction(for: status) {
        case .deniedOrRestricted:
            return .failure(LocationCoordinateFailure.authorizationFailure(for: status))
        case .unknown:
            logUnknownAuthorizationStatus(status, source: "LocationProbe.currentCoordinateResult")
            return .failure(.unknownAuthorizationStatus(rawValue: status.rawValue))
        case .requestLocation, .requestWhenInUseAuthorization, .waitForChoice:
            break
        }

        return await withCheckedContinuation { (cont: CheckedContinuation<LocationCoordinateProbeResult, Never>) in
            let token = UUID()
            let delegate = SingleShotLocationDelegate(manager: manager) { result in
                activeDelegates[token] = nil
                cont.resume(returning: result)
            }
            activeDelegates[token] = delegate
            manager.delegate = delegate

            Task { @MainActor in
                try? await Task.sleep(for: .seconds(timeout))
                delegate.finishTimedOut()
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

enum LocationCoordinateFailure: Equatable {
    case permissionDenied
    case permissionRestricted
    case permissionNotDetermined
    case timedOut
    case unavailable(String)
    case unknownAuthorizationStatus(rawValue: Int32)

    static func authorizationFailure(for status: CLAuthorizationStatus) -> LocationCoordinateFailure {
        switch status {
        case .authorizedAlways, .authorizedWhenInUse:
            return .unavailable("Location permission is authorized, but no location was available.")
        case .denied: return .permissionDenied
        case .restricted: return .permissionRestricted
        case .notDetermined: return .permissionNotDetermined
        @unknown default: return .unknownAuthorizationStatus(rawValue: status.rawValue)
        }
    }
}

enum LocationCoordinateProbeResult {
    case success(CLLocationCoordinate2D)
    case failure(LocationCoordinateFailure)
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
    private let handler: (LocationCoordinateProbeResult) -> Void
    private var done = false

    init(manager: CLLocationManager, handler: @escaping (LocationCoordinateProbeResult) -> Void) {
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
            finish(with: .failure(LocationCoordinateFailure.authorizationFailure(for: manager.authorizationStatus)))
        case .waitForChoice:
            break
        case .unknown:
            logUnknownAuthorizationStatus(manager.authorizationStatus, source: "SingleShotLocationDelegate.begin")
            finish(with: .failure(.unknownAuthorizationStatus(rawValue: manager.authorizationStatus.rawValue)))
        }
    }

    func finish(with result: LocationCoordinateProbeResult) {
        MainActor.preconditionIsolated()
        if done { return }
        done = true
        handler(result)
    }

    func finishTimedOut() {
        MainActor.preconditionIsolated()
        guard !done else { return }
        if manager.authorizationStatus == .notDetermined {
            finish(with: .failure(.permissionNotDetermined))
        } else {
            finish(with: .failure(.timedOut))
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        let result: LocationCoordinateProbeResult
        if let coord = locations.last?.coordinate {
            result = .success(coord)
        } else {
            result = .failure(.unavailable("Location services returned no coordinates."))
        }
        Task { @MainActor in
            self.finish(with: result)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        let nsError = error as NSError
        let failure: LocationCoordinateFailure
        if nsError.domain == kCLErrorDomain,
           let code = CLError.Code(rawValue: nsError.code) {
            switch code {
            case .denied:
                failure = .permissionDenied
            case .locationUnknown, .network, .deferredFailed, .deferredNotUpdatingLocation, .promptDeclined:
                failure = .unavailable(error.localizedDescription)
            default:
                failure = .unavailable(error.localizedDescription)
            }
        } else {
            failure = .unavailable(error.localizedDescription)
        }
        Task { @MainActor in
            self.finish(with: .failure(failure))
        }
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            guard !self.done else { return }
            switch locationAuthorizationUpdateAction(for: manager.authorizationStatus) {
            case .requestLocation:
                manager.requestLocation()
            case .deniedOrRestricted:
                self.finish(with: .failure(LocationCoordinateFailure.authorizationFailure(for: manager.authorizationStatus)))
            case .waitForChoice, .requestWhenInUseAuthorization:
                break
            case .unknown:
                logUnknownAuthorizationStatus(manager.authorizationStatus, source: "SingleShotLocationDelegate.locationManagerDidChangeAuthorization")
                self.finish(with: .failure(.unknownAuthorizationStatus(rawValue: manager.authorizationStatus.rawValue)))
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
