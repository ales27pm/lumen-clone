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
        app = makeApp()
        app.launch()
        dismissOnboardingIfNeeded()
    }

    override func tearDownWithError() throws {
        if app != nil, app.state != .notRunning {
            app.terminate()
        }
        app = nil
    }

    @MainActor
    func testDeveloperSectionIsVisibleInSettings() throws {
        openSettings()
        assertDeveloperConsoleEntryExists()
    }

    @MainActor
    func testDeveloperConsoleEntryFollowsDebugToggles() throws {
        openSettings()
        let traceMode = app.switches["settings.developer.traceMode"]
        let reasoningCapture = app.switches["settings.developer.reasoningCapture"]
        let console = developerConsoleEntry()

        XCTAssertTrue(traceMode.waitForExistence(timeout: 3))
        XCTAssertTrue(reasoningCapture.waitForExistence(timeout: 3))
        XCTAssertTrue(console.waitForExistence(timeout: 3))
        XCTAssertGreaterThan(reasoningCapture.frame.minY, traceMode.frame.minY)
        XCTAssertGreaterThan(console.frame.minY, reasoningCapture.frame.minY)
    }

    @MainActor
    func testDeveloperConsoleEntryRemainsAccessibleAfterNavigationAwayAndBack() throws {
        openSettings()
        assertDeveloperConsoleEntryExists()

        if app.buttons["Chat"].exists {
            app.buttons["Chat"].tap()
        } else if app.staticTexts["Chat"].exists {
            app.staticTexts["Chat"].tap()
        }

        openSettings()
        assertDeveloperConsoleEntryExists()
    }

    @MainActor
    func testDeveloperConsoleEntryIsVisibleAfterAppRelaunch() throws {
        openSettings()
        assertDeveloperConsoleEntryExists()

        relaunchApp()

        openSettings()
        assertDeveloperConsoleEntryExists()
    }

    @MainActor
    func testDeveloperConsoleEntryRemainsHittableAfterScroll() throws {
        openSettings()
        assertDeveloperConsoleEntryExists()

        app.swipeUp()
        app.swipeDown()

        let console = developerConsoleEntry()
        XCTAssertTrue(console.waitForExistence(timeout: 3))
        XCTAssertTrue(console.isHittable)
    }

    @MainActor
    func testDeveloperConsoleOpensRunDashboard() throws {
        openDeveloperConsole()

        XCTAssertTrue(app.navigationBars["Developer Console"].waitForExistence(timeout: 4))
        XCTAssertTrue(app.descendants(matching: .any)["developerConsole.segmentedTabs"].waitForExistence(timeout: 4))
        XCTAssertTrue(app.staticTexts["Complete E2E Tests"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testDeveloperConsoleTelemetryTabOpens() throws {
        openDeveloperConsole()

        app.buttons["Telemetry"].tap()
        XCTAssertTrue(app.staticTexts["Live Telemetry"].waitForExistence(timeout: 4))
        XCTAssertTrue(app.staticTexts["Live Surfaces"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testDeveloperConsoleReportsTabOpens() throws {
        openDeveloperConsole()

        app.buttons["Reports"].tap()
        XCTAssertTrue(app.staticTexts["Reports"].waitForExistence(timeout: 4))
        XCTAssertTrue(app.staticTexts["Runtime debug text"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testDeveloperConsoleSurfaceNavigationOpensDiagnostics() throws {
        openDeveloperConsole()

        app.buttons["Telemetry"].tap()
        let diagnostics = app.staticTexts["Diagnostics"]
        XCTAssertTrue(diagnostics.waitForExistence(timeout: 4))
        diagnostics.tap()

        XCTAssertTrue(app.navigationBars["Diagnostics"].waitForExistence(timeout: 4))
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

        recordStep("open_developer_console") {
            let console = developerConsoleEntry()
            try assertElement(console.waitForExistence(timeout: 3), "Developer Console row was not visible.")
            console.tap()
            try assertElement(app.navigationBars["Developer Console"].waitForExistence(timeout: 4), "Developer Console did not open.")
        }

        recordStep("run_dashboard_visible") {
            try assertElement(app.staticTexts["Complete E2E Tests"].waitForExistence(timeout: 4), "Run dashboard was not visible.")
            try assertElement(app.staticTexts["Standard"].exists, "Standard E2E card was not visible.")
            try assertElement(app.staticTexts["Training"].exists, "Training E2E card was not visible.")
        }

        recordStep("telemetry_tab_visible") {
            app.buttons["Telemetry"].tap()
            try assertElement(app.staticTexts["Live Telemetry"].waitForExistence(timeout: 4), "Telemetry tab did not open.")
            try assertElement(app.staticTexts["Live Surfaces"].exists, "Live surfaces section was not visible.")
        }

        recordStep("reports_tab_visible") {
            app.buttons["Reports"].tap()
            try assertElement(app.staticTexts["Runtime debug text"].waitForExistence(timeout: 4), "Reports tab did not show runtime debug report.")
            try assertElement(app.staticTexts["Recent logs"].exists, "Reports tab did not show recent logs report.")
        }

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
        openDeveloperConsole()
        XCTAssertTrue(app.navigationBars["Developer Console"].waitForExistence(timeout: 4))

        relaunchApp()
        openSettings()

        openDeveloperConsole()
        XCTAssertTrue(app.staticTexts["Complete E2E Tests"].waitForExistence(timeout: 4))
    }

    @MainActor
    func testLaunchPerformance() throws {
        measure(metrics: [XCTApplicationLaunchMetric()]) {
            makeApp().launch()
        }
    }

    @MainActor
    private func openSettings() {
        dismissOnboardingIfNeeded()
        if app.navigationBars["Settings"].waitForExistence(timeout: 1) {
            return
        }
        let settingsRow = app.descendants(matching: .any)["root.settings"]
        if settingsRow.waitForExistence(timeout: 1) {
            settingsRow.tap()
            return
        }
        if app.buttons["Continue"].waitForExistence(timeout: 1) {
            app.buttons["Continue"].tap()
            if app.navigationBars["Settings"].waitForExistence(timeout: 1) {
                return
            }
        }
        if app.buttons["Settings"].waitForExistence(timeout: 1) {
            app.buttons["Settings"].tap()
            return
        }
        if app.buttons["Preferences"].waitForExistence(timeout: 1) {
            app.buttons["Preferences"].tap()
            return
        }
        if app.staticTexts["Settings"].waitForExistence(timeout: 1) {
            app.staticTexts["Settings"].tap()
            return
        }
        if app.staticTexts["Preferences"].waitForExistence(timeout: 1) {
            app.staticTexts["Preferences"].tap()
            return
        }
        if app.navigationBars.buttons.firstMatch.waitForExistence(timeout: 1) {
            app.navigationBars.buttons.firstMatch.tap()
            if settingsRow.waitForExistence(timeout: 1) {
                settingsRow.tap()
                return
            }
            if app.navigationBars["Settings"].waitForExistence(timeout: 1) {
                return
            }
            if app.staticTexts["Settings"].waitForExistence(timeout: 1) {
                app.staticTexts["Settings"].tap()
                return
            }
            if app.staticTexts["Preferences"].waitForExistence(timeout: 1) {
                app.staticTexts["Preferences"].tap()
                return
            }
        }
        XCTFail("Unable to navigate to Settings")
    }

    @MainActor
    private func openDeveloperConsole() {
        openSettings()
        let console = developerConsoleEntry()
        XCTAssertTrue(console.waitForExistence(timeout: 3), "Developer Console row was not visible.")
        console.tap()
    }

    @MainActor
    private func assertDeveloperConsoleEntryExists() {
        let console = developerConsoleEntry()
        XCTAssertTrue(console.waitForExistence(timeout: 3), "Developer Console row was not visible.")
        XCTAssertTrue(console.isHittable, "Developer Console row was not hittable.")
    }

    @MainActor
    private func developerConsoleEntry() -> XCUIElement {
        let identifier = "settings.developer.console"
        let element = app.descendants(matching: .any)[identifier]
        if element.exists {
            return element
        }

        let button = app.buttons[identifier]
        if button.exists {
            return button
        }

        return app.staticTexts["Developer Console"]
    }

    @MainActor
    private func goBackIfNeeded() {
        let back = app.navigationBars.buttons.firstMatch
        if back.waitForExistence(timeout: 2) {
            back.tap()
        }
    }

    private func dismissOnboardingIfNeeded() {
        let skipButton = app.buttons["Skip for now"]
        if skipButton.waitForExistence(timeout: 2) {
            skipButton.tap()
            return
        }

        let skipElement = app.descendants(matching: .any)["Skip for now"]
        if skipElement.waitForExistence(timeout: 1) {
            skipElement.tap()
        }
    }

    private func relaunchApp() {
        if app.state != .notRunning {
            app.terminate()
            XCTAssertTrue(app.wait(for: .notRunning, timeout: 5), "App did not terminate before relaunch.")
        }
        app.launch()
        dismissOnboardingIfNeeded()
    }

    private func makeApp() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments.append("--lumen-ui-tests")
        return app
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
