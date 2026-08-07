import XCTest
@testable import Lumen

final class ToolApprovalPolicyTests: XCTestCase {
    private let open = SecureToolDefinition(id: "open.url", displayName: "Open", description: "", category: .sensitiveAction, requiredPermissions: [], supportsBackgroundExecution: false, requiresUserApproval: true, argumentSchemaDescription: "", resultPrivacyLevel: .moderate, maxOutputCharacters: 100)
    private let destructive = SecureToolDefinition(id: "test.delete", displayName: "Delete", description: "", category: .destructiveAction, requiredPermissions: [], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .sensitive, maxOutputCharacters: 100)
    private let foregroundOnlyRead = SecureToolDefinition(id: "position.snapshot", displayName: "Position", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .low, maxOutputCharacters: 100)
    private let backgroundApprovalRead = SecureToolDefinition(id: "memory.export", displayName: "Export", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: true, argumentSchemaDescription: "", resultPrivacyLevel: .sensitive, maxOutputCharacters: 100)
    private let headlessLocalRead = SecureToolDefinition(id: "memory.search", displayName: "Recall", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .moderate, maxOutputCharacters: 100)
    private let headlessMutationMarkedReadOnly = SecureToolDefinition(id: "memory.save", displayName: "Save", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .moderate, maxOutputCharacters: 100)
    private let headlessProtectedRead = SecureToolDefinition(id: "contacts.lookup", displayName: "Contacts", description: "", category: .permissionRead, requiredPermissions: [.contacts], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .sensitive, maxOutputCharacters: 100)
    private let headlessPrivateReadMarkedLocal = SecureToolDefinition(id: "outlook.message.read", displayName: "Read Outlook Message", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .sensitive, maxOutputCharacters: 100)

    func testModelProposedOpenURLRequiresApproval() {
        let inv = ToolInvocation(id: UUID(), toolID: "open.url", arguments: ["url":"https://a.com"], source: .modelProposed, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: open, invocation: inv, isForeground: true, permissionStates: [:], settings: .init(networkAccessEnabled: false, userAllowlist: []))
        if case .requiresApproval = d {} else { XCTFail() }
    }

    func testUserInitiatedSensitiveActionStillRequiresApproval() {
        let inv = ToolInvocation(id: UUID(), toolID: "open.url", arguments: ["url":"https://a.com"], source: .userInitiated, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: open, invocation: inv, isForeground: true, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))
        if case .requiresApproval(let reason) = d {
            XCTAssertEqual(reason, "User approval required")
        } else {
            XCTFail()
        }
    }

    func testUserApprovedSensitiveActionCanRun() {
        let inv = ToolInvocation(id: UUID(), toolID: "open.url", arguments: ["url":"https://a.com"], source: .userApproved, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: open, invocation: inv, isForeground: true, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))
        if case .allow = d {} else { XCTFail() }
    }

    func testDestructiveActionRequiresExplicitApprovalEvenWithoutLegacyFlag() {
        let inv = ToolInvocation(id: UUID(), toolID: "test.delete", arguments: [:], source: .userInitiated, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: destructive, invocation: inv, isForeground: true, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))
        if case .requiresApproval(let reason) = d {
            XCTAssertEqual(reason, "Destructive action requires approval")
        } else {
            XCTFail()
        }
    }

    func testBackgroundTriggerDeniedWhenReadOnlyToolDoesNotSupportBackgroundExecution() {
        let inv = ToolInvocation(id: UUID(), toolID: "position.snapshot", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: foregroundOnlyRead, invocation: inv, isForeground: false, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))
        if case .deny(let reason) = d {
            XCTAssertEqual(reason, "Tool unavailable in background")
        } else {
            XCTFail()
        }
    }

    func testBackgroundTriggerDeniedWhenToolRequiresApproval() {
        let inv = ToolInvocation(id: UUID(), toolID: "memory.export", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: backgroundApprovalRead, invocation: inv, isForeground: false, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))
        if case .deny(let reason) = d {
            XCTAssertEqual(reason, "Tool requires foreground approval")
        } else {
            XCTFail()
        }
    }

    func testAppIntentAllowsOnlyBackgroundSafeLocalRead() {
        let invocation = ToolInvocation(id: UUID(), toolID: headlessLocalRead.id, arguments: [:], source: .appIntent, conversationID: nil, turnID: nil, createdAt: Date())
        let decision = ToolApprovalPolicy.decide(definition: headlessLocalRead, invocation: invocation, isForeground: false, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))

        XCTAssertEqual(decision, .allow)
    }

    func testAppIntentDeniesProtectedReadEvenWhenPermissionWasPreviouslyGranted() {
        let invocation = ToolInvocation(id: UUID(), toolID: headlessProtectedRead.id, arguments: [:], source: .appIntent, conversationID: nil, turnID: nil, createdAt: Date())
        let decision = ToolApprovalPolicy.decide(definition: headlessProtectedRead, invocation: invocation, isForeground: false, permissionStates: [.contacts: .granted], settings: .init(networkAccessEnabled: true, userAllowlist: []))

        XCTAssertEqual(decision, .deny("Tool unavailable from headless surface"))
    }

    func testAppIntentDeniesKnownMutationEvenWhenLegacyCategorySaysReadOnly() {
        let invocation = ToolInvocation(id: UUID(), toolID: headlessMutationMarkedReadOnly.id, arguments: [:], source: .appIntent, conversationID: nil, turnID: nil, createdAt: Date())
        let decision = ToolApprovalPolicy.decide(definition: headlessMutationMarkedReadOnly, invocation: invocation, isForeground: false, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))

        XCTAssertEqual(decision, .deny("Tool unavailable from headless surface"))
    }

    func testAppIntentDeniesPrivateReadEvenWhenLegacyMetadataLooksHeadlessSafe() {
        let invocation = ToolInvocation(id: UUID(), toolID: headlessPrivateReadMarkedLocal.id, arguments: [:], source: .appIntent, conversationID: nil, turnID: nil, createdAt: Date())
        let decision = ToolApprovalPolicy.decide(definition: headlessPrivateReadMarkedLocal, invocation: invocation, isForeground: false, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: []))

        XCTAssertEqual(decision, .deny("Tool unavailable from headless surface"))
    }

    func testNonEmptyAllowlistDeniesAbsentTool() {
        let inv = ToolInvocation(id: UUID(), toolID: "open.url", arguments: [:], source: .userInitiated, conversationID: nil, turnID: nil, createdAt: Date())
        let d = ToolApprovalPolicy.decide(definition: open, invocation: inv, isForeground: true, permissionStates: [:], settings: .init(networkAccessEnabled: true, userAllowlist: ["memory.search"]))
        if case .deny(let reason) = d { XCTAssertEqual(reason, "Tool not in user allowlist") } else { XCTFail() }
    }
}
