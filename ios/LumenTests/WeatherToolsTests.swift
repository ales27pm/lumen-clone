import Testing
@testable import Lumen

struct WeatherToolsTests {
    @Test func agentCurrentLocationArgumentUsesDeviceLocation() {
        #expect(WeatherTools.isCurrentLocationRequest("current location"))
        #expect(WeatherTools.isCurrentLocationRequest("Current Location"))
        #expect(WeatherTools.isCurrentLocationRequest("current-location"))
        #expect(WeatherTools.isCurrentLocationRequest("current_location"))
        #expect(WeatherTools.isCurrentLocationRequest("this location"))
        #expect(WeatherTools.isCurrentLocationRequest("This Location"))
        #expect(WeatherTools.isCurrentLocationRequest("this-location"))
    }

    @Test func nearbyAndUserLocationAliasesUseDeviceLocation() {
        #expect(WeatherTools.isCurrentLocationRequest(""))
        #expect(WeatherTools.isCurrentLocationRequest("here"))
        #expect(WeatherTools.isCurrentLocationRequest("current"))
        #expect(WeatherTools.isCurrentLocationRequest("my location"))
        #expect(WeatherTools.isCurrentLocationRequest("device location"))
        #expect(WeatherTools.isCurrentLocationRequest("near me"))
    }

    @Test func explicitCityDoesNotUseDeviceLocation() {
        #expect(!WeatherTools.isCurrentLocationRequest("Montreal"))
        #expect(!WeatherTools.isCurrentLocationRequest("weather in Montreal"))
        #expect(!WeatherTools.isCurrentLocationRequest("Paris, France"))
    }

    @Test func deniedLocationFailureKeepsPermissionCause() {
        let message = WeatherTools.weatherLocationFailureMessage(for: .permissionDenied)
        #expect(message.contains("Location access is denied"))
        #expect(message.contains("weather in Montreal"))
    }

    @Test func timeoutLocationFailureKeepsGPSCause() {
        let message = WeatherTools.weatherLocationFailureMessage(for: .timedOut)
        #expect(message.contains("GPS timeout"))
        #expect(!message.contains("Enable Location permission"))
    }

    @Test func explicitLocationGeocodeFailureNamesLocation() {
        let message = WeatherTools.weatherGeocodingFailureMessage(for: "  Atlantis  ")
        #expect(message.contains("\"Atlantis\""))
        #expect(message.contains("more specific"))
        #expect(!message.contains("Location permission"))
    }
}
