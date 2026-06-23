import XCTest
@testable import Lumen

final class KnowledgeGraphServiceTests: XCTestCase {
    func testBuildFromManifestAndAuditsCreatesRealGraphNodes() async {
        let graph = KnowledgeGraphService()
        await graph.buildFromManifestAndAudits(manifest: Self.makeManifest(), auditFailures: [
            RuntimeManifestFailure(
                type: "missing_live_tool",
                agent: "executor",
                expected: ["calendar.list"],
                actual: nil,
                scenario: "calendar audit",
                problem: "Calendar runtime tool missing"
            )
        ])

        let calendarResults = await graph.queryWithGNN(query: "calendar audit missing tool")
        let ids = Set(calendarResults.map(\.id))

        XCTAssertTrue(ids.contains("tool:calendar.list"))
        XCTAssertTrue(ids.contains { $0.hasPrefix("audit:missing_live_tool|executor") })
    }

    func testMultiHopTraverseReturnsBFSPathsFromManifestEdges() async {
        let graph = KnowledgeGraphService()
        await graph.buildFromManifestAndAudits(manifest: Self.makeManifest(), auditFailures: [])

        let paths = await graph.multiHopTraverse(startId: "cortex", maxHops: 2)

        XCTAssertTrue(paths.contains { $0.nodes == ["slot:cortex", "intent:calendar"] })
        XCTAssertTrue(paths.contains { $0.nodes == ["slot:cortex", "intent:calendar", "tool:calendar.list"] })
        XCTAssertTrue(paths.allSatisfy { !$0.nodes.isEmpty && $0.score > 0 })
    }

    func testQueryWithGNNUsesQueryTermsAndRanksRelevantNodes() async {
        let graph = KnowledgeGraphService()
        await graph.buildFromManifestAndAudits(manifest: Self.makeManifest(), auditFailures: [])

        let calendarResults = await graph.queryWithGNN(query: "show calendar events")
        let weatherResults = await graph.queryWithGNN(query: "weather forecast location")

        XCTAssertTrue(calendarResults.contains { $0.id == "tool:calendar.list" || $0.id == "intent:calendar" })
        XCTAssertTrue(weatherResults.contains { $0.id == "tool:weather" || $0.id == "intent:weather" })
        XCTAssertNotEqual(calendarResults.first?.id, weatherResults.first?.id)
        XCTAssertTrue(calendarResults.allSatisfy { $0.gnnScore > 0 })
    }

    private static func makeManifest() -> AgentBehaviorManifest {
        AgentBehaviorManifest(
            schemaVersion: "test",
            app: ManifestAppInfo(
                name: "Lumen",
                bundleIdentifier: "com.27pm.lumenclone",
                buildVersion: "1",
                generatedAt: nil
            ),
            sourceIntegrity: nil,
            fleet: ManifestFleet(
                contractVersion: "test",
                slots: [
                    ManifestModelSlot(
                        id: "cortex",
                        role: "orchestrator",
                        modelFamily: nil,
                        responsibilities: ["intent routing", "tool selection"]
                    ),
                    ManifestModelSlot(
                        id: "executor",
                        role: "tool_executor",
                        modelFamily: nil,
                        responsibilities: ["tool execution"]
                    ),
                    ManifestModelSlot(
                        id: "rem",
                        role: "idle_reflection",
                        modelFamily: nil,
                        responsibilities: ["manifest audit"]
                    )
                ]
            ),
            tools: [
                RuntimeToolDefinition(
                    id: "calendar.list",
                    displayName: "List Events",
                    description: "Read upcoming calendar events.",
                    requiresApproval: false,
                    permissionKey: "NSCalendarsFullAccessUsageDescription"
                ),
                RuntimeToolDefinition(
                    id: "weather",
                    displayName: "Weather",
                    description: "Get weather forecast for the current location.",
                    requiresApproval: false,
                    permissionKey: "NSLocationWhenInUseUsageDescription"
                )
            ],
            intents: [
                ManifestIntent(id: "calendar", allowedToolIDs: ["calendar.list"]),
                ManifestIntent(id: "weather", allowedToolIDs: ["weather"])
            ],
            routingMatrix: [
                ManifestRoutingEntry(intent: "calendar", allowedTools: ["calendar.list"], forbiddenTools: ["weather"]),
                ManifestRoutingEntry(intent: "weather", allowedTools: ["weather"], forbiddenTools: ["calendar.list"])
            ],
            memory: ManifestMemory(
                scopes: ["conversation"],
                freshnessClasses: [
                    ManifestFreshnessClass(id: "durable", ttlSeconds: nil, durable: true)
                ]
            ),
            sentinels: ManifestSentinels(forbiddenInUserOutput: ["<private_reasoning>"])
        )
    }
}
