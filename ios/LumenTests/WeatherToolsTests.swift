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
}
