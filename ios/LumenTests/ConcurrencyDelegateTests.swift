import CoreLocation
import Foundation
import Testing
@testable import Lumen

@MainActor
struct ConcurrencyDelegateTests {
    @Test func locationDelegateOnlyDispatchesHandlerOnceUnderRacingCallbacks() async {
        var captures: [CLLocationCoordinate2D?] = []
        let manager = CLLocationManager()
        let delegate = SingleShotLocationDelegate(manager: manager) { captures.append($0) }

        let locationA = CLLocation(latitude: 37.7749, longitude: -122.4194)
        let locationB = CLLocation(latitude: 40.7128, longitude: -74.0060)

        await withTaskGroup(of: Void.self) { group in
            group.addTask { delegate.locationManager(manager, didUpdateLocations: [locationA]) }
            group.addTask { delegate.locationManager(manager, didUpdateLocations: [locationB]) }
            group.addTask { delegate.locationManager(manager, didFailWithError: CLError(.locationUnknown)) }
        }

        try? await Task.sleep(for: .milliseconds(100))
        #expect(captures.count == 1)
    }

    @Test func descriptionDelegateKeepsFirstCompletionOrdering() async {
        var messages: [String] = []
        let manager = CLLocationManager()
        let delegate = SingleShotDescriptionDelegate(manager: manager) { messages.append($0) }
        let location = CLLocation(latitude: 51.5074, longitude: -0.1278)

        delegate.locationManager(manager, didUpdateLocations: [location])
        delegate.locationManager(manager, didFailWithError: CLError(.network))

        try? await Task.sleep(for: .milliseconds(100))

        #expect(messages.count == 1)
        #expect(messages.first?.contains("Current location:") == true)
    }

    @Test func locationAuthWaiterInvokesChangeCallbackOnlyOnce() async {
        var invocations = 0
        let waiter = LocationAuthWaiter()
        waiter.onChange = { invocations += 1 }

        waiter.finishOnce()
        waiter.finishOnce()

        #expect(invocations == 1)
    }
}
