import SwiftData
import XCTest
@testable import Lumen

@MainActor
final class NumericToolArgumentSafetyTests: XCTestCase {
    func testTriggerMinuteBoundsAndCheckedConversion() throws {
        let maximum = TriggerTools.maximumScheduleDelayMinutes
        let base = [
            "title": "Review",
            "prompt": "Review reminders",
            "schedule": "relative"
        ]

        for rawValue in ["-1", "0", String(maximum + 1), String(Int.max), String(repeating: "9", count: 128)] {
            var arguments = base
            arguments["inMinutes"] = rawValue
            XCTAssertEqual(
                failure(TriggerTools.createArguments(from: arguments)),
                .invalidArgument("inMinutes")
            )
        }

        var maximumArguments = base
        maximumArguments["inMinutes"] = String(maximum)
        let parsed = try success(TriggerTools.createArguments(from: maximumArguments))
        XCTAssertEqual(parsed.inMinutes, maximum)

        let checkedSeconds = maximum.multipliedReportingOverflow(by: 60)
        XCTAssertFalse(checkedSeconds.overflow)
        XCTAssertEqual(TriggerTools.seconds(fromMinutes: maximum), checkedSeconds.partialValue)
        XCTAssertNil(TriggerTools.seconds(fromMinutes: maximum + 1))
    }

    func testTriggerIntervalAndBeforeEventBoundsAreProductLimited() throws {
        let base = [
            "title": "Review",
            "prompt": "Review reminders",
            "schedule": "relative"
        ]

        for rawValue in [
            "60",
            String(TriggerScheduleContract.minimumIntervalSeconds - 1),
            String(TriggerTools.maximumIntervalSeconds + 1),
            String(repeating: "9", count: 128)
        ] {
            var arguments = base
            arguments["intervalSeconds"] = rawValue
            XCTAssertEqual(
                failure(TriggerTools.createArguments(from: arguments)),
                .invalidArgument("intervalSeconds")
            )
        }

        var maximumIntervalArguments = base
        maximumIntervalArguments["intervalSeconds"] = String(TriggerTools.maximumIntervalSeconds)
        XCTAssertEqual(
            try success(TriggerTools.createArguments(from: maximumIntervalArguments)).intervalSeconds,
            TriggerTools.maximumIntervalSeconds
        )

        XCTAssertEqual(TriggerScheduleContract.minimumIntervalSeconds, 900)
        XCTAssertEqual(TriggerTools.minimumIntervalSeconds, TriggerScheduleContract.minimumIntervalSeconds)
        XCTAssertEqual(
            ToolArgumentValueDomains.triggerIntervalSeconds,
            .integer(
                minimum: TriggerScheduleContract.minimumIntervalSeconds,
                maximum: TriggerTools.maximumIntervalSeconds
            )
        )
        var minimumIntervalArguments = base
        minimumIntervalArguments["schedule"] = "interval"
        minimumIntervalArguments["intervalSeconds"] = String(TriggerScheduleContract.minimumIntervalSeconds)
        XCTAssertEqual(
            try success(TriggerTools.createArguments(from: minimumIntervalArguments)).intervalSeconds,
            TriggerScheduleContract.minimumIntervalSeconds
        )

        let belowMinimum = Trigger(
            title: "Too frequent",
            prompt: "Synthetic prompt",
            scheduleType: .interval,
            intervalSeconds: TimeInterval(TriggerScheduleContract.minimumIntervalSeconds - 1)
        )
        XCTAssertNil(belowMinimum.computeNextFire())

        var invalidBeforeArguments = base
        invalidBeforeArguments["beforeMinutes"] = String(TriggerTools.maximumBeforeEventMinutes + 1)
        XCTAssertEqual(
            failure(TriggerTools.createArguments(from: invalidBeforeArguments)),
            .invalidArgument("beforeMinutes")
        )

        var maximumBeforeArguments = base
        maximumBeforeArguments["beforeMinutes"] = String(TriggerTools.maximumBeforeEventMinutes)
        XCTAssertEqual(
            try success(TriggerTools.createArguments(from: maximumBeforeArguments)).beforeMinutes,
            TriggerTools.maximumBeforeEventMinutes
        )
    }

    func testTriggerClockBoundsRejectInvalidHourAndMinuteValues() throws {
        let base = [
            "title": "Daily review",
            "prompt": "Review reminders",
            "schedule": "absolute"
        ]

        var maximumArguments = base
        maximumArguments["atTime"] = "23:59"
        let parsed = try success(TriggerTools.createArguments(from: maximumArguments))
        XCTAssertEqual(parsed.timeOfDayMinutes, 23 * 60 + 59)

        for rawValue in ["-1:00", "24:00", "23:60", "999999999999999999999999:00"] {
            var arguments = base
            arguments["atTime"] = rawValue
            XCTAssertEqual(
                failure(TriggerTools.createArguments(from: arguments)),
                .invalidArgument("atTime")
            )
        }
    }

    func testInvalidTriggerMinutesCreateNoPersistentSideEffect() async throws {
        let container = try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let savedContainer = SharedContainer.shared
        SharedContainer.shared = container
        defer { SharedContainer.shared = savedContainer }

        let response = await TriggerTools.create(args: [
            "title": "Must not persist",
            "prompt": "Synthetic prompt",
            "schedule": "relative",
            "inMinutes": "0"
        ])

        XCTAssertEqual(
            response,
            TriggerTools.invalidCreateArgumentsMessage(.invalidArgument("inMinutes"))
        )
        let context = ModelContext(container)
        XCTAssertTrue(try context.fetch(FetchDescriptor<Trigger>()).isEmpty)
    }

    func testAlarmMinuteAndSnoozeBoundsRejectBeforeRuntimeAccess() throws {
        let maximumScheduleDelay = AlarmTools.maximumScheduleDelayMinutes
        let maximumSnooze = AlarmTools.maximumSnoozeMinutes
        let now = Date(timeIntervalSince1970: 1_000)
        let base = ["title": "Wake up", "inMinutes": "10"]

        for rawValue in ["-1", "0", String(maximumScheduleDelay + 1), String(Int.max), String(repeating: "9", count: 128)] {
            var arguments = base
            arguments["inMinutes"] = rawValue
            XCTAssertEqual(
                failure(AlarmTools.scheduleArguments(from: arguments, now: now)),
                .invalidArgument("inMinutes")
            )
        }

        for rawValue in ["-1", "0", String(maximumSnooze + 1), String(Int.max), String(repeating: "9", count: 128)] {
            var arguments = base
            arguments["snoozeMinutes"] = rawValue
            XCTAssertEqual(
                failure(AlarmTools.scheduleArguments(from: arguments, now: now)),
                .invalidArgument("snoozeMinutes")
            )
        }

        var maximumArguments = base
        maximumArguments["inMinutes"] = String(maximumScheduleDelay)
        maximumArguments["snoozeMinutes"] = String(maximumSnooze)
        let parsed = try success(AlarmTools.scheduleArguments(from: maximumArguments, now: now))
        XCTAssertEqual(parsed.snoozeMinutes, maximumSnooze)

        let scheduleSeconds = maximumScheduleDelay.multipliedReportingOverflow(by: 60)
        XCTAssertFalse(scheduleSeconds.overflow)
        XCTAssertEqual(
            AlarmTools.seconds(fromScheduleDelayMinutes: maximumScheduleDelay),
            scheduleSeconds.partialValue
        )
        XCTAssertNil(AlarmTools.seconds(fromScheduleDelayMinutes: maximumScheduleDelay + 1))

        let snoozeSeconds = maximumSnooze.multipliedReportingOverflow(by: 60)
        XCTAssertFalse(snoozeSeconds.overflow)
        XCTAssertEqual(AlarmTools.seconds(fromSnoozeMinutes: maximumSnooze), snoozeSeconds.partialValue)
        XCTAssertNil(AlarmTools.seconds(fromSnoozeMinutes: maximumSnooze + 1))
    }

    func testAlarmCountdownBoundsRejectBeforeRuntimeAccess() async throws {
        let maximum = AlarmTools.maximumDurationSeconds
        for rawValue in ["-1", "0", String(maximum + 1), String(Int.max), String(repeating: "9", count: 128)] {
            XCTAssertEqual(
                countdownFailure(AlarmTools.countdownDurationSeconds(from: ["durationSeconds": rawValue])),
                .invalidArgument("durationSeconds")
            )
        }
        XCTAssertEqual(
            try countdownSuccess(AlarmTools.countdownDurationSeconds(from: ["durationSeconds": String(maximum)])),
            maximum
        )

        let response = await AlarmTools.countdown(args: [
            "title": "Must not schedule",
            "durationSeconds": String(maximum + 1)
        ])
        XCTAssertEqual(
            response,
            AlarmTools.invalidCountdownArgumentsMessage(.invalidArgument("durationSeconds"))
        )
    }

    func testInvalidAlarmScheduleIsTypedFailureWithoutAlarmKitAttempt() async {
        let response = await AlarmTools.schedule(args: [
            "title": "Must not schedule",
            "inMinutes": "0"
        ])

        XCTAssertEqual(
            response,
            AlarmTools.invalidScheduleArgumentsMessage(.invalidArgument("inMinutes"))
        )
        XCTAssertEqual(ProductivityLocalTool.alarmStatus(from: response), .failed)
        XCTAssertEqual(
            ProductivityLocalTool.alarmErrorCode(text: response, status: .failed),
            "alarmkit_invalid_arguments"
        )
    }

    private func success(
        _ result: Result<TriggerCreateArguments, TriggerCreateArgumentError>
    ) throws -> TriggerCreateArguments {
        switch result {
        case .success(let value):
            return value
        case .failure(let error):
            throw error
        }
    }

    private func failure(
        _ result: Result<TriggerCreateArguments, TriggerCreateArgumentError>
    ) -> TriggerCreateArgumentError? {
        guard case .failure(let error) = result else { return nil }
        return error
    }

    private func success(
        _ result: Result<AlarmScheduleArguments, AlarmScheduleArgumentError>
    ) throws -> AlarmScheduleArguments {
        switch result {
        case .success(let value):
            return value
        case .failure(let error):
            throw error
        }
    }

    private func failure(
        _ result: Result<AlarmScheduleArguments, AlarmScheduleArgumentError>
    ) -> AlarmScheduleArgumentError? {
        guard case .failure(let error) = result else { return nil }
        return error
    }

    private func countdownSuccess(
        _ result: Result<Int, AlarmCountdownArgumentError>
    ) throws -> Int {
        switch result {
        case .success(let value):
            return value
        case .failure(let error):
            throw error
        }
    }

    private func countdownFailure(
        _ result: Result<Int, AlarmCountdownArgumentError>
    ) -> AlarmCountdownArgumentError? {
        guard case .failure(let error) = result else { return nil }
        return error
    }
}
