import XCTest
@testable import Lumen

final class ToolApprovalPolicyTests: XCTestCase {
    private let open = SecureToolDefinition(id: "open.url", displayName: "Open", description: "", category: .sensitiveAction, requiredPermissions: [], supportsBackgroundExecution: false, requiresUserApproval: true, argumentSchemaDescription: "", resultPrivacyLevel: .moderate, maxOutputCharacters: 100)
    func testModelProposedOpenURLRequiresApproval() {
        let inv = ToolInvocation(id: UUID(), toolID: "open.url", arguments: ["url":"https://a.com"], source: .modelProposed, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: open, invocation: inv, isForeground: true, permissionStates: [:], settings: .init(networkAccessEnabled: false, userAllowlist: []))
        if case .requiresApproval = d {} else { XCTFail() }
    }

    func testNonEmptyAllowlistDeniesAbsentTool() {
        let inv = ToolInvocation(id: UUID(), toolID: "open.url", arguments: [:], source: .userInitiated, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: open, invocation: inv, isForeground: true, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: ["memory.search"]))
        if case .deny(let reason) = d { XCTAssertEqual(reason, "Tool not in user allowlist") } else { XCTFail() }
    }
}
