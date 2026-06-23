import Foundation
import XCTest
@testable import Lumen

final class AgentKernelBoundaryGuardTests: XCTestCase {
    private let bridgePath = "ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift"
    private let legacyMarkers = [
        "AgentService.shared.run",
        "SlotAgentService.shared.run",
        "RolePipelineAgentService.shared.run",
        "AgentRunner.runHeadless"
    ]

    func testLegacyRuntimeCallsStayConfinedToCompatibilityBridge() throws {
        let repo = repoRoot()
        let appRoot = repo.appendingPathComponent("ios/Lumen")
        let bridge = bridgePath
        var violations: [String] = []

        guard let enumerator = FileManager.default.enumerator(
            at: appRoot,
            includingPropertiesForKeys: nil
        ) else {
            XCTFail("Could not enumerate \(appRoot.path)")
            return
        }

        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let relative = relativePath(url, root: repo)
            let text = try String(contentsOf: url, encoding: .utf8)
            for marker in legacyMarkers where text.contains(marker) && relative != bridge {
                violations.append("\(relative): \(marker)")
            }
        }

        XCTAssertTrue(
            violations.isEmpty,
            "Direct legacy runtime calls must stay behind \(bridge): \(violations.joined(separator: ", "))"
        )
    }

    func testCompatibilityBridgeIsExplicitlyDocumentedAndNarrowlyAllowlisted() throws {
        let repo = repoRoot()
        let bridgeURL = repo.appendingPathComponent(bridgePath)
        let guardURL = repo.appendingPathComponent("tools/check_agent_kernel_boundary.py")
        let bridge = try String(contentsOf: bridgeURL, encoding: .utf8)
        let guardPolicy = try String(contentsOf: guardURL, encoding: .utf8)

        XCTAssertTrue(bridge.contains("Removal condition: delete this bridge"))
        XCTAssertEqual(bridge.occurrenceCount(of: "for await event in AgentService.shared.run"), 1)
        XCTAssertEqual(bridge.occurrenceCount(of: "SlotAgentService.shared.run(request, options: options)"), 1)
        XCTAssertTrue(guardPolicy.contains("\"ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift\""))
        XCTAssertTrue(guardPolicy.contains("\"AgentService.shared.run\""))
        XCTAssertTrue(guardPolicy.contains("\"SlotAgentService.shared.run\""))
        XCTAssertFalse(guardPolicy.contains("ALLOWED_MIGRATION_FILES"))
    }

    private func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func relativePath(_ url: URL, root: URL) -> String {
        let rootPath = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath + "/") else { return path }
        return String(path.dropFirst(rootPath.count + 1))
    }
}

private extension String {
    func occurrenceCount(of needle: String) -> Int {
        components(separatedBy: needle).count - 1
    }
}
