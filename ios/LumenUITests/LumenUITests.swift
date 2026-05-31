//
//  LumenUITests.swift
//  LumenUITests
//
//  Created by Rork on April 20, 2026.
//

import XCTest

final class LumenUITests: XCTestCase {
    private struct DashboardStep: Codable {
        let name: String
        let status: String
        let startedAt: String
        let endedAt: String
        let durationMs: Int
        let indicators: [String: Double]
        let errorMessage: String?
    }

    private struct DashboardSummary: Codable {
        let scenario: String
        let runStartedAt: String
        let runEndedAt: String
        let totalDurationMs: Int
        let throughputStepsPerSecond: Double
        let performanceIndicators: [String: Double]
        let stepCount: Int
        let passCount: Int
        let failCount: Int
        let steps: [DashboardStep]
    }

    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
        dismissOnboardingIfNeeded()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    @MainActor
    func testDeveloperSectionIsVisibleInSettings() throws {
        openSettings()
        assertDeveloperRowsExist()
    }

    @MainActor
    func testDeveloperRowsDisplayInRequestedOrder() throws {
        openSettings()
        let order = [
            "settings.developer.runTests",
            "settings.developer.logs",
            "settings.developer.debug",
            "settings.developer.diagnostic"
        ]
        var previousMinY: CGFloat = 0
        for (index, identifier) in order.enumerated() {
            let row = developerRow(identifier)
            XCTAssertTrue(row.waitForExistence(timeout: 3), "Missing row: \(identifier)")
            let currentMinY = row.frame.minY
            if index > 0 {
                XCTAssertGreaterThan(currentMinY, previousMinY, "Row order is incorrect for \(identifier)")
            }
            previousMinY = currentMinY
        }
    }

    @MainActor
    func testDeveloperRowsRemainAccessibleAfterNavigationAwayAndBack() throws {
        openSettings()
        assertDeveloperRowsExist()

        if app.buttons["Chat"].exists {
            app.buttons["Chat"].tap()
        } else if app.staticTexts["Chat"].exists {
            app.staticTexts["Chat"].tap()
        }

        openSettings()
        assertDeveloperRowsExist()
    }

    @MainActor
    func testDeveloperRowsAreVisibleAfterAppRelaunch() throws {
        openSettings()
        assertDeveloperRowsExist()

        app.terminate()
        app.launch()
        dismissOnboardingIfNeeded()

        openSettings()
        assertDeveloperRowsExist()
    }

    @MainActor
    func testDeveloperRowsRemainHittableAfterScroll() throws {
        openSettings()
        assertDeveloperRowsExist()

        app.swipeUp()
        app.swipeDown()

        let identifiers = [
            "settings.developer.runTests",
            "settings.developer.logs",
            "settings.developer.debug",
            "settings.developer.diagnostic"
        ]
        for identifier in identifiers {
            let row = developerRow(identifier)
            XCTAssertTrue(row.waitForExistence(timeout: 3), "Missing row: \(identifier)")
            XCTAssertTrue(row.isHittable, "Row is not hittable: \(identifier)")
        }
    }

    @MainActor
    func testRunTestsButtonPresentsResultsAlert() throws {
        openSettings()

        let runTests = developerRow("settings.developer.runTests")
        XCTAssertTrue(runTests.waitForExistence(timeout: 3))
        runTests.tap()

        let alert = app.alerts["Run tests"]
        XCTAssertTrue(alert.waitForExistence(timeout: 4))
        XCTAssertTrue(alert.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "checks passed")).firstMatch.exists)
        alert.buttons["OK"].tap()
        XCTAssertFalse(alert.exists)
    }

    @MainActor
    func testLogsNavigationOpensLogsScreen() throws {
        openSettings()

        let logs = developerRow("settings.developer.logs")
        XCTAssertTrue(logs.waitForExistence(timeout: 3))
        logs.tap()

        XCTAssertTrue(app.navigationBars["Logs"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testDebugNavigationOpensDebugScreen() throws {
        openSettings()

        let debug = developerRow("settings.developer.debug")
        XCTAssertTrue(debug.waitForExistence(timeout: 3))
        debug.tap()

        XCTAssertTrue(app.navigationBars["Debug"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testDiagnosticNavigationOpensDiagnosticScreen() throws {
        openSettings()

        let diagnostic = developerRow("settings.developer.diagnostic")
        XCTAssertTrue(diagnostic.waitForExistence(timeout: 3))
        diagnostic.tap()

        XCTAssertTrue(app.navigationBars["Diagnostic"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testDeveloperFeaturesRealTimeDashboard() throws {
        let formatter = ISO8601DateFormatter()
        let runStart = Date()
        var steps: [DashboardStep] = []
        let baselineIssueCount = testRun?.totalFailureCount ?? 0

        continueAfterFailure = true
        defer { continueAfterFailure = false }

        func recordStep(_ name: String, _ body: () throws -> Void) {
            let stepStart = Date()
            let issuesBefore = self.testRun?.totalFailureCount ?? 0
            var status = "pass"
            var errorMessage: String?

            XCTContext.runActivity(named: "Dashboard Step: \(name)") { _ in
                do {
                    try body()
                } catch {
                    status = "fail"
                    errorMessage = String(describing: error)
                    XCTFail("Dashboard step '\(name)' failed: \(errorMessage!)")
                }
            }

            let issuesAfter = self.testRun?.totalFailureCount ?? 0
            if issuesAfter > issuesBefore {
                status = "fail"
                if errorMessage == nil {
                    errorMessage = "One or more XCT assertions failed during this step."
                }
            }

            let stepEnd = Date()
            let durationMs = Int(stepEnd.timeIntervalSince(stepStart) * 1_000)
            let durationSeconds = max(stepEnd.timeIntervalSince(stepStart), 0.000_1)
            steps.append(
                DashboardStep(
                    name: name,
                    status: status,
                    startedAt: formatter.string(from: stepStart),
                    endedAt: formatter.string(from: stepEnd),
                    durationMs: durationMs,
                    indicators: [
                        "durationSeconds": durationSeconds,
                        "eventsPerSecond": 1.0 / durationSeconds
                    ],
                    errorMessage: errorMessage
                )
            )

            attachStepSnapshot(stepName: name, status: status, durationMs: durationMs, errorMessage: errorMessage)
        }

        func assertElement(_ condition: @autoclosure () -> Bool, _ message: String) throws {
            if !condition() {
                throw NSError(domain: "LumenUITests", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
            }
        }

        recordStep("open_settings") { openSettings() }

        recordStep("run_tests_alert") {
            let runTests = developerRow("settings.developer.runTests")
            try assertElement(runTests.waitForExistence(timeout: 3), "Run Tests row was not visible.")
            runTests.tap()

            let runTestsAlert = app.alerts["Run tests"]
            try assertElement(runTestsAlert.waitForExistence(timeout: 4), "Run tests alert did not appear.")
            try assertElement(
                runTestsAlert.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "checks passed")).firstMatch.exists,
                "Run tests alert did not include a passing checks message."
            )
            runTestsAlert.buttons["OK"].tap()
        }

        recordStep("open_logs") {
            let logs = developerRow("settings.developer.logs")
            try assertElement(logs.waitForExistence(timeout: 3), "Logs row was not visible.")
            logs.tap()
            try assertElement(app.navigationBars["Logs"].waitForExistence(timeout: 4), "Logs screen did not open.")
            goBackIfNeeded()
        }

        recordStep("open_debug") {
            let debug = developerRow("settings.developer.debug")
            try assertElement(debug.waitForExistence(timeout: 3), "Debug row was not visible.")
            debug.tap()
            try assertElement(app.navigationBars["Debug"].waitForExistence(timeout: 4), "Debug screen did not open.")
            goBackIfNeeded()
        }

        recordStep("open_diagnostic") {
            let diagnostic = developerRow("settings.developer.diagnostic")
            try assertElement(diagnostic.waitForExistence(timeout: 3), "Diagnostic row was not visible.")
            diagnostic.tap()
            try assertElement(app.navigationBars["Diagnostic"].waitForExistence(timeout: 4), "Diagnostic screen did not open.")
            goBackIfNeeded()
        }

        recordStep("rows_still_visible") { assertDeveloperRowsExist() }

        let runEnd = Date()
        let totalMs = Int(runEnd.timeIntervalSince(runStart) * 1_000)
        let passCount = steps.filter { $0.status == "pass" }.count
        let failCount = steps.count - passCount
        let passRate = steps.isEmpty ? 0.0 : Double(passCount) / Double(steps.count)

        let summary = DashboardSummary(
            scenario: "developer_features_realtime_dashboard",
            runStartedAt: formatter.string(from: runStart),
            runEndedAt: formatter.string(from: runEnd),
            totalDurationMs: totalMs,
            throughputStepsPerSecond: Double(steps.count) / max(runEnd.timeIntervalSince(runStart), 0.001),
            performanceIndicators: [
                "p50StepDurationMs": percentileDuration(from: steps, percentile: 0.50),
                "p95StepDurationMs": percentileDuration(from: steps, percentile: 0.95),
                "maxStepDurationMs": maxDuration(from: steps),
                "passRate": passRate
            ],
            stepCount: steps.count,
            passCount: passCount,
            failCount: failCount,
            steps: steps
        )

        attachDashboardReport(summary)
        XCTAssertEqual(testRun?.totalFailureCount ?? 0, baselineIssueCount, "One or more dashboard steps failed. Inspect dashboard attachments.")
    }

    @MainActor
    func testDeveloperFeaturesEndToEndAfterRelaunch() throws {
        openSettings()
        assertDeveloperRowsExist()

        app.terminate()
        app.launch()
        dismissOnboardingIfNeeded()
        openSettings()

        let runTests = developerRow("settings.developer.runTests")
        XCTAssertTrue(runTests.waitForExistence(timeout: 3))
        runTests.tap()
        let runTestsAlert = app.alerts["Run tests"]
        XCTAssertTrue(runTestsAlert.waitForExistence(timeout: 4))
        runTestsAlert.buttons["OK"].tap()

        let logs = developerRow("settings.developer.logs")
        XCTAssertTrue(logs.waitForExistence(timeout: 3))
        logs.tap()
        XCTAssertTrue(app.navigationBars["Logs"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testLaunchPerformance() throws {
        measure(metrics: [XCTApplicationLaunchMetric()]) {
            XCUIApplication().launch()
        }
    }

    @MainActor
    private func openSettings() {
        if app.buttons["Settings"].waitForExistence(timeout: 4) {
            app.buttons["Settings"].tap()
            return
        }
        if app.staticTexts["Settings"].waitForExistence(timeout: 4) {
            app.staticTexts["Settings"].tap()
            return
        }
        if app.navigationBars.buttons.firstMatch.waitForExistence(timeout: 4) {
            app.navigationBars.buttons.firstMatch.tap()
            if app.staticTexts["Settings"].waitForExistence(timeout: 4) {
                app.staticTexts["Settings"].tap()
                return
            }
        }
        XCTFail("Unable to navigate to Settings")
    }

    @MainActor
    private func assertDeveloperRowsExist() {
        XCTAssertTrue(developerRow("settings.developer.runTests").waitForExistence(timeout: 3))
        XCTAssertTrue(developerRow("settings.developer.logs").waitForExistence(timeout: 3))
        XCTAssertTrue(developerRow("settings.developer.debug").waitForExistence(timeout: 3))
        XCTAssertTrue(developerRow("settings.developer.diagnostic").waitForExistence(timeout: 3))
    }

    @MainActor
    private func developerRow(_ identifier: String) -> XCUIElement {
        let element = app.descendants(matching: .any)[identifier]
        if element.exists {
            return element
        }

        let button = app.buttons[identifier]
        if button.exists {
            return button
        }

        return app.staticTexts[identifier]
    }

    @MainActor
    private func goBackIfNeeded() {
        let back = app.navigationBars.buttons.firstMatch
        if back.waitForExistence(timeout: 2) {
            back.tap()
        }
    }

    private func dismissOnboardingIfNeeded() {
        let skip = app.buttons["Skip for now"]
        if skip.waitForExistence(timeout: 2) {
            skip.tap()
        }
    }

    private func attachDashboardReport(_ report: DashboardSummary) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(report),
              let reportText = String(data: data, encoding: .utf8) else {
            XCTFail("Failed to serialize dashboard report")
            return
        }

        let attachment = XCTAttachment(string: reportText)
        attachment.name = "Live E2E Dashboard Metrics"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func attachStepSnapshot(stepName: String, status: String, durationMs: Int, errorMessage: String?) {
        let payload = [
            "step": stepName,
            "status": status,
            "durationMs": String(durationMs),
            "error": errorMessage ?? ""
        ]
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "\n")

        let attachment = XCTAttachment(string: payload)
        attachment.name = "Live Step \(stepName)"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func percentileDuration(from steps: [DashboardStep], percentile: Double) -> Double {
        let sorted = steps.map(\.durationMs).sorted()
        guard !sorted.isEmpty else { return 0 }
        let index = min(max(Int(Double(sorted.count - 1) * percentile), 0), sorted.count - 1)
        return Double(sorted[index])
    }

    private func maxDuration(from steps: [DashboardStep]) -> Double {
        Double(steps.map(\.durationMs).max() ?? 0)
    }
}
