import XCTest
@testable import Lumen

final class ContactsLookupToolPolicyTests: XCTestCase {
    private struct ThrowingProvider: ContactsLookupTool.Provider {
        func search(query: String, limit: Int, includeEmails: Bool, includePhones: Bool) throws -> [[String : String]] {
            throw NSError(domain: "ContactsTest", code: 1)
        }
    }

    func testBackgroundDenied() async {
        let tool = ContactsLookupTool()
        let inv = ToolInvocation(id: UUID(), toolID: "contacts.lookup", arguments: ["query":"john"], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .denied)
    }

    func testInvalidInputIsInvalidArgs() {
        let tool = ContactsLookupTool()
        let inv = ToolInvocation(id: UUID(), toolID: "contacts.lookup", arguments: ["query":""], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = tool.executeAfterPermissionGranted(invocation: inv)
        XCTAssertEqual(res.metricsSummary, "invalid_args")
        XCTAssertEqual(res.errorCode, "invalid")
    }

    func testProviderFailureIsProviderError() {
        let tool = ContactsLookupTool(provider: ThrowingProvider())
        let inv = ToolInvocation(id: UUID(), toolID: "contacts.lookup", arguments: ["query":"john"], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = tool.executeAfterPermissionGranted(invocation: inv)
        XCTAssertEqual(res.metricsSummary, "provider_error")
        XCTAssertEqual(res.errorCode, "provider_error")
        XCTAssertFalse(res.modelText.contains("ContactsTest"))
    }
}
