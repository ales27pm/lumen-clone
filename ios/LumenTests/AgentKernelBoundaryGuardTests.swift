import Foundation
import XCTest
@testable import Lumen

final class AgentKernelBoundaryGuardTests: XCTestCase {
    private let bridgePath = "ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift"
    private let allowedRuntimeBoundaryPaths: Set<String> = [
        "ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift"
    ]
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
            for marker in legacyMarkers where text.contains(marker) && !allowedRuntimeBoundaryPaths.contains(relative) {
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
        let bridgeLines = bridge.split(separator: "\n", omittingEmptySubsequences: false)
        let firstNonEmptyLine = bridgeLines.first { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let lastNonEmptyLine = bridgeLines.last { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

        XCTAssertEqual(firstNonEmptyLine, "#if DEBUG")
        XCTAssertEqual(lastNonEmptyLine, "#endif")
        XCTAssertTrue(bridge.contains("Removal condition: delete this bridge"))
        XCTAssertEqual(bridge.occurrenceCount(of: "for await event in AgentService.shared.run"), 1)
        XCTAssertEqual(bridge.occurrenceCount(of: "SlotAgentService.shared.run(request, options: options)"), 2)
        XCTAssertTrue(guardPolicy.contains("\"ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift\""))
        XCTAssertTrue(guardPolicy.contains("StructuredAgentKernelExecutor.swift"))
        XCTAssertTrue(guardPolicy.contains("\"AgentService.shared.run\""))
        XCTAssertTrue(guardPolicy.contains("\"SlotAgentService.shared.run\""))
        XCTAssertTrue(guardPolicy.contains("if args.strict and not is_debug_only"))
        XCTAssertFalse(guardPolicy.contains("ALLOWED_MIGRATION_FILES"))
    }

    func testLiveE2ERunnerUsesAssistantKernelBoundary() throws {
        let repo = repoRoot()
        let runnerURL = repo.appendingPathComponent("ios/Lumen/Services/E2ETestRunner.swift")
        let runner = try String(contentsOf: runnerURL, encoding: .utf8)

        XCTAssertTrue(runner.contains("AssistantKernel.shared.run(kernelRequest, modelContext: nil)"))
        XCTAssertTrue(runner.contains("strictLiveAgentKernelRequest("))
        XCTAssertTrue(runner.contains("forceModelBackedToolPlanning: requiresStructuredAgentJSON"))
        XCTAssertTrue(runner.contains("structuredMode: requiresStructuredAgentJSON ? .requiredAgentJSON : .automatic"))
        XCTAssertTrue(runner.contains("structuredAllowedToolIDs: requiresStructuredAgentJSON ? availableTools.map(\\.id) : []"))
        XCTAssertFalse(runner.contains("Kernel migration E2E probe is DEBUG-only."))
        XCTAssertFalse(runner.contains("runLegacyAgentBridge(req, options: runOptions)"))
        XCTAssertFalse(runner.contains("Structured live E2E agent-json diagnostics require DEBUG build."))
        XCTAssertFalse(runner.contains("AgentService.shared.run(agentRequest"))
    }

    func testAssistantKernelRoutesRequiredAgentJSONToStructuredExecutor() throws {
        let repo = repoRoot()
        let kernelURL = repo.appendingPathComponent("ios/Lumen/Assistant/AssistantKernel+Streaming.swift")
        let executorURL = repo.appendingPathComponent("ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift")
        let contractsURL = repo.appendingPathComponent("ios/Lumen/Assistant/AgentKernelContracts.swift")
        let kernel = try String(contentsOf: kernelURL, encoding: .utf8)
        let executor = try String(contentsOf: executorURL, encoding: .utf8)
        let contracts = try String(contentsOf: contractsURL, encoding: .utf8)

        XCTAssertTrue(contracts.contains("case requiredAgentJSON"))
        XCTAssertTrue(contracts.contains("let structuredMode: AgentStructuredMode"))
        XCTAssertTrue(kernel.contains("request.options.structuredMode == .requiredAgentJSON"))
        XCTAssertTrue(kernel.contains("StructuredAgentKernelExecutor(kernel: self, modelContext: modelContext)"))
        XCTAssertTrue(executor.contains("AgentBehaviorTraceEmitter.recordModelTurn"))
        XCTAssertTrue(executor.contains("runtimePath: \"agent-model\""))
        XCTAssertTrue(executor.contains("StructuredToolCallValidator.validate"))
        XCTAssertTrue(executor.contains("toolRegistry.execute"))
    }

    func testReleaseToolTurnsAreNotExcludedAtKernelOrVoiceBoundary() throws {
        let repo = repoRoot()
        let checkedPaths = [
            "ios/Lumen/Assistant/AssistantKernel+Streaming.swift",
            "ios/Lumen/Voice/VoiceCommandRouter.swift"
        ]
        let forbidden = [
            "Tool-capable agent turns are excluded from this Release build",
            "Tool-capable voice turns are excluded from this Release build",
            "Legacy agent bridge is excluded from Release builds"
        ]

        for path in checkedPaths {
            let text = try String(contentsOf: repo.appendingPathComponent(path), encoding: .utf8)
            for marker in forbidden {
                XCTAssertFalse(text.contains(marker), "\(path) must not restore Release tool-turn exclusion: \(marker)")
            }
        }
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
