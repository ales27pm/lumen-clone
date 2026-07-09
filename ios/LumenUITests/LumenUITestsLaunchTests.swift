//
//  LumenUITestsLaunchTests.swift
//  LumenUITests
//
//  Created by Rork on April 20, 2026.
//

import XCTest

final class LumenUITestsLaunchTests: XCTestCase {

    override class var runsForEachTargetApplicationUIConfiguration: Bool {
        true
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testLaunch() throws {
        let app = XCUIApplication()
        switch app.state {
        case .runningForeground:
            break
        case .runningBackground, .runningBackgroundSuspended, .notRunning, .unknown:
            app.activate()
        @unknown default:
            app.activate()
        }

        // Insert steps here to perform after app launch but before taking a screenshot,
        // such as logging into a test account or navigating somewhere in the app

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Launch Screen"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
