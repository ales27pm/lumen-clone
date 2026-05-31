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
}
