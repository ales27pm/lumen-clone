import XCTest
@testable import Lumen

@MainActor
final class CalendarReadToolPolicyTests: XCTestCase {
    func testBackgroundDenied() async {
        let tool = CalendarReadTool(provider: FakeCalendarEventProvider(state: .granted))
        let inv = ToolInvocation(id: UUID(), toolID: "calendar.read", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .denied)
    }

    func testNotDeterminedPermissionIsDistinct() async {
        let result = await CalendarTools.listEventsResult(provider: FakeCalendarEventProvider(state: .notDetermined))
        XCTAssertEqual(result.status, .denied)
        XCTAssertEqual(result.errorCode, "calendar_permission_not_determined")
        XCTAssertEqual(result.structuredPayload?["availability"], "notDetermined")
        XCTAssertTrue(result.displayText.localizedCaseInsensitiveContains("not been granted yet"))
    }

    func testDeniedPermissionIsDistinct() async {
        let result = await CalendarTools.listEventsResult(provider: FakeCalendarEventProvider(state: .denied))
        XCTAssertEqual(result.status, .denied)
        XCTAssertEqual(result.errorCode, "calendar_permission_denied")
        XCTAssertEqual(result.structuredPayload?["availability"], "denied")
        XCTAssertTrue(result.displayText.localizedCaseInsensitiveContains("denied"))
    }

    func testRestrictedPermissionIsDistinct() async {
        let result = await CalendarTools.listEventsResult(provider: FakeCalendarEventProvider(state: .restricted))
        XCTAssertEqual(result.status, .denied)
        XCTAssertEqual(result.errorCode, "calendar_permission_restricted")
        XCTAssertEqual(result.structuredPayload?["availability"], "restricted")
        XCTAssertTrue(result.displayText.localizedCaseInsensitiveContains("restricted"))
    }

    func testEventKitUnavailableIsDistinct() async {
        let result = await CalendarTools.listEventsResult(provider: FakeCalendarEventProvider(state: .unavailable))
        XCTAssertEqual(result.status, .unavailable)
        XCTAssertEqual(result.errorCode, "calendar_provider_unavailable")
        XCTAssertEqual(result.structuredPayload?["availability"], "unavailable")
        XCTAssertTrue(result.displayText.localizedCaseInsensitiveContains("unavailable"))
    }

    func testGrantedEmptyResultIsSuccess() async {
        let result = await CalendarTools.listEventsResult(provider: FakeCalendarEventProvider(state: .granted, records: []))
        XCTAssertEqual(result.status, .success)
        XCTAssertNil(result.errorCode)
        XCTAssertEqual(result.structuredPayload?["availability"], "granted")
        XCTAssertEqual(result.structuredPayload?["count"], "0")
        XCTAssertTrue(result.displayText.localizedCaseInsensitiveContains("no calendar events"))
    }

    func testWriteOnlyPermissionCanCreateButCannotRead() async {
        let read = await CalendarTools.listEventsResult(provider: FakeCalendarEventProvider(state: .limited))
        XCTAssertEqual(read.status, .denied)
        XCTAssertEqual(read.errorCode, "calendar_permission_limited")
        XCTAssertEqual(read.structuredPayload?["availability"], "limited")

        let create = await CalendarTools.createEventResult(
            title: "Planning",
            startsInMinutes: 30,
            provider: FakeCalendarEventProvider(state: .limited),
            now: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(create.status, .success)
        XCTAssertNil(create.errorCode)
        XCTAssertEqual(create.structuredPayload?["availability"], "limited")
        XCTAssertEqual(create.structuredPayload?["created"], "true")
    }

    func testProviderFailureIsSanitized() async {
        let provider = FakeCalendarEventProvider(state: .granted, records: [], throwsOnEvents: true)
        let result = await CalendarTools.listEventsResult(provider: provider)
        XCTAssertEqual(result.status, .failed)
        XCTAssertEqual(result.errorCode, "calendar_provider_failure")
        XCTAssertFalse(result.displayText.localizedCaseInsensitiveContains("raw provider boom"))
        XCTAssertFalse(result.modelText.localizedCaseInsensitiveContains("raw provider boom"))
    }

    func testMalformedArgumentsFailBeforeProviderRead() async {
        let provider = FakeCalendarEventProvider(state: .granted, records: [])
        let result = await CalendarTools.listEventsResult(arguments: ["limit": "50"], provider: provider)
        XCTAssertEqual(result.status, .failed)
        XCTAssertEqual(result.errorCode, "calendar_invalid_arguments")
        XCTAssertEqual(result.structuredPayload?["failure"], "invalidArguments")
        XCTAssertEqual(provider.eventReadCount, 0)
    }

    func testProductivityCalendarListPreservesAvailabilityPayload() async {
        let tool = ProductivityLocalTool(ToolRegistry.find(id: "calendar.list")!)
        XCTAssertEqual(tool.definition.id, "calendar.list")
    }

    func testProductivityPayloadDoesNotAllowCanonicalMetadataSpoofing() {
        let payload = ProductivityLocalTool.resultPayload(
            toolID: "calendar.list",
            structuredPayload: [
                "availability": "granted",
                "toolID": "spoofed.tool",
                "implementation": "SpoofedImplementation"
            ]
        )
        XCTAssertEqual(payload["toolID"], "calendar.list")
        XCTAssertEqual(payload["implementation"], "ProductivityLocalTool")
        XCTAssertEqual(payload["availability"], "granted")
    }

    func testProductivityCalendarPermissionPromptIsForegroundOnly() {
        XCTAssertTrue(ProductivityLocalTool.shouldRequestCalendarPermission(toolID: "calendar.list", isForeground: true))
        XCTAssertTrue(ProductivityLocalTool.shouldRequestCalendarPermission(toolID: "calendar.create", isForeground: true))
        XCTAssertFalse(ProductivityLocalTool.shouldRequestCalendarPermission(toolID: "calendar.list", isForeground: false))
        XCTAssertFalse(ProductivityLocalTool.shouldRequestCalendarPermission(toolID: "reminders.list", isForeground: true))
    }

    func testWriteOnlyCalendarUsageDescriptionIsAccepted() {
        XCTAssertEqual(PermissionKind(usageDescriptionKey: "NSCalendarsWriteOnlyAccessUsageDescription"), .calendar)
        XCTAssertTrue(CalendarTools.EventKitProvider.hasCalendarUsageDescription(infoDictionary: [
            "NSCalendarsWriteOnlyAccessUsageDescription": "Create events."
        ]))
    }
}

private final class FakeCalendarEventProvider: CalendarTools.EventProvider {
    var eventReadCount = 0
    private let state: AssistantPermissionState
    private let records: [CalendarTools.CalendarEventRecord]
    private let throwsOnEvents: Bool

    init(
        state: AssistantPermissionState,
        records: [CalendarTools.CalendarEventRecord] = [],
        throwsOnEvents: Bool = false
    ) {
        self.state = state
        self.records = records
        self.throwsOnEvents = throwsOnEvents
    }

    func authorizationState() -> AssistantPermissionState {
        state
    }

    func events(start: Date, end: Date, limit: Int, titleFilter: String?) async throws -> [CalendarTools.CalendarEventRecord] {
        eventReadCount += 1
        if throwsOnEvents {
            throw FakeCalendarError.rawProviderBoom
        }
        return Array(records.prefix(limit))
    }

    func createEvent(title: String, startsInMinutes: Int, now: Date) async throws -> CalendarTools.CalendarEventRecord {
        if throwsOnEvents {
            throw FakeCalendarError.rawProviderBoom
        }
        return CalendarTools.CalendarEventRecord(
            title: title,
            start: now.addingTimeInterval(TimeInterval(startsInMinutes * 60)),
            end: now.addingTimeInterval(TimeInterval(startsInMinutes * 60 + 3600)),
            calendarTitle: "Test",
            location: nil
        )
    }
}

private enum FakeCalendarError: LocalizedError {
    case rawProviderBoom

    var errorDescription: String? {
        "raw provider boom"
    }
}
