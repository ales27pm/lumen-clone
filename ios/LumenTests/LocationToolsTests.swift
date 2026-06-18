import Foundation
import Testing
@testable import Lumen

@MainActor
struct LocationToolsTests {
    @Test func directionsURLUsesHTTPSAndMapsHost() {
        let url = LocationTools.directionsURL(destination: "1 Infinite Loop, Cupertino")
        #expect(url != nil)
        #expect(url?.scheme == "https")
        #expect(url?.host == "maps.apple.com")
    }

    @Test func directionsURLEncodesQueryValueCorrectly() {
        let destination = "Coffee & Tea, San Francisco"
        let url = LocationTools.directionsURL(destination: destination)
        #expect(url != nil)

        let components = URLComponents(url: url!, resolvingAgainstBaseURL: false)
        let daddr = components?.queryItems?.first(where: { $0.name == "daddr" })?.value
        #expect(daddr == destination)

        let absolute = url!.absoluteString
        #expect(absolute.contains("daddr=Coffee%20%26%20Tea,%20San%20Francisco"))
        #expect(!absolute.contains(" "))
    }

    @Test func directionsURLRejectsToolDescriptionLeakAsDestination() {
        let destination = """
        a real destination. Args: destination. Use only for navigation/route requests.
        - maps.search: Find nearby/local places in Apple Maps.
        [RUNTIME POLICY]
        legacy-interactive
        """
        #expect(LocationTools.directionsURL(destination: destination) == nil)
    }

    @Test func scheduledRecoveryMeetingsPreferWebSearchOverNearbySearch() {
        #expect(ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: "Alcoholics Anonymous meeting tomorrow"))
        #expect(!ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: "nearest pharmacy"))
    }

    @Test func dynamicPublicLookupsPreferWebSearchOverNearbySearch() {
        #expect(ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: "nearest free tax clinic tomorrow"))
        #expect(ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: "yoga class near me tonight"))
        #expect(ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: "movie showtimes closest to me today"))
        #expect(!ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: "coffee near me"))
        #expect(!ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: "nearest gas station"))
    }
}
