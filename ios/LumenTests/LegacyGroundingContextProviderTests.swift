import XCTest
@testable import Lumen

@MainActor
final class LegacyGroundingContextProviderTests: XCTestCase {
    func testUnavailableWithoutDirectOrShared() {
        let saved = SharedContainer.shared
        SharedContainer.shared = nil
        let provider = LegacyGroundingContextProvider(directContext: nil)
        XCTAssertNil(provider.resolveContext())
        XCTAssertEqual(provider.degradedReason, "model_context_unavailable")
        SharedContainer.shared = saved
    }
}
