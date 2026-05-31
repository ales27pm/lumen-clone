import XCTest
@testable import Lumen

@MainActor
final class PhotosToolsTests: XCTestCase {
    func testPreviousDayRangeUsesCalendarMathWhenAvailable() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let now = Date(timeIntervalSince1970: 1_700_000_000)

        let range = PhotosTools.previousDayRange(now: now, calendar: calendar)

        XCTAssertEqual(range.0, Date(timeIntervalSince1970: 1_699_920_000))
        XCTAssertEqual(range.1, Date(timeIntervalSince1970: 1_699_920_000 + 86_400))
    }

    func testPreviousDayRangeFallsBackWhenDateMathFails() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let now = Date(timeIntervalSince1970: 1_700_000_000)

        let range = PhotosTools.previousDayRange(now: now, calendar: calendar) { _, _ in nil }

        XCTAssertEqual(range.0, now.addingTimeInterval(-86_400))
        XCTAssertEqual(range.1, now)
    }
}
