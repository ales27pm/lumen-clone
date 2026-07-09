import XCTest
@testable import Lumen

@MainActor
final class ToolApprovalQueueTests: XCTestCase {
    func testApprovalBoundaryStepCreatesPendingToolMessageForChatView() {
        let step = AgentStep(
            kind: .approvalBoundary,
            content: "Approval required before running mail.draft.",
            toolID: "mail.draft",
            toolArgs: [
                "to": "sam@example.com",
                "subject": "Timeline",
                "body": "Status green"
            ]
        )

        guard let message = ChatApprovalBoundaryMapper.pendingToolMessage(for: step) else {
            XCTFail("Expected approval boundary step to create a pending tool message")
            return
        }

        XCTAssertEqual(message.messageRole, .tool)
        XCTAssertEqual(message.toolName, "mail.draft")
        XCTAssertEqual(message.status, .pendingApproval)
        XCTAssertNil(message.toolResult)

        let parsed = ToolApprovalPayloadCodec.parseLooseArguments(message.content)
        XCTAssertEqual(parsed["to"], "sam@example.com")
        XCTAssertEqual(parsed["subject"], "Timeline")
        XCTAssertEqual(parsed["body"], "Status green")

        guard let pendingActionID = ToolApprovalPayloadCodec.pendingActionID(from: parsed) else {
            XCTFail("Expected pending action id in serialized approval payload")
            return
        }
        guard let pending = ToolApprovalQueue.shared.resolve(pendingActionID) else {
            XCTFail("Expected queued approval action to be resolvable")
            return
        }
        XCTAssertEqual(pending.toolID, "mail.draft")
        XCTAssertEqual(pending.arguments.stringCoerced["to"], "sam@example.com")
        XCTAssertEqual(pending.arguments.stringCoerced["subject"], "Timeline")
        XCTAssertEqual(pending.arguments.stringCoerced["body"], "Status green")

        XCTAssertNotNil(ToolApprovalQueue.shared.consume(pendingActionID, matchingToolID: "mail.draft"))
    }

    func testPendingApprovalPayloadCarriesQueueIDAndConsumesExactArgumentsOnce() {
        let pending = ToolApprovalQueue.shared.enqueue(
            toolID: "mail.draft",
            toolName: "Draft Email",
            arguments: [
                "to": "sam@example.com",
                "subject": "Timeline",
                "body": "Status: green, ship Friday."
            ]
        )

        let payload = ToolApprovalPayloadCodec.displayArguments(
            for: pending,
            visibleArguments: pending.arguments.stringCoerced
        )
        let serialized = ToolApprovalPayloadCodec.serialize(payload)
        let parsed = ToolApprovalPayloadCodec.parseLooseArguments(serialized)

        XCTAssertEqual(ToolApprovalPayloadCodec.pendingActionID(from: parsed), pending.pendingActionID)
        let redacted = ToolArgumentRedactor.redactDisplayContent(serialized)
        XCTAssertFalse(redacted.contains(pending.pendingActionID.uuidString))

        let consumed = ToolApprovalPayloadCodec.consumePendingApproval(from: parsed, matchingToolID: pending.toolID)
        if case .success(let approved) = consumed {
            XCTAssertEqual(approved.arguments.stringCoerced["body"], "Status: green, ship Friday.")
        } else {
            XCTFail("Expected queued approval to resolve")
        }
        let replay = ToolApprovalPayloadCodec.consumePendingApproval(from: parsed, matchingToolID: pending.toolID)
        if case .failure(let error) = replay {
            XCTAssertEqual(error, .expiredOrMismatchedPendingAction)
        } else {
            XCTFail("Expected queued approval to be one-time")
        }
    }

    func testPendingApprovalPayloadIsRequiredBeforeExecutionApproval() {
        let result = ToolApprovalPayloadCodec.consumePendingApproval(
            from: ["to": "sam@example.com", "body": "hello"],
            matchingToolID: "mail.draft"
        )

        if case .failure(let error) = result {
            XCTAssertEqual(error, .missingPendingActionID)
            XCTAssertEqual(error.userMessage, "Approval request cannot be verified. Ask Lumen to prepare the action again.")
        } else {
            XCTFail("Expected missing pending action id to be rejected")
        }
    }

    func testMalformedPendingApprovalIDDoesNotResolve() {
        let parsed = ToolApprovalPayloadCodec.parseLooseArguments("pendingActionID: not-a-uuid, title: call")

        XCTAssertTrue(ToolApprovalPayloadCodec.containsPendingActionIDField(parsed))
        XCTAssertNil(ToolApprovalPayloadCodec.pendingActionID(from: parsed))
        let result = ToolApprovalPayloadCodec.consumePendingApproval(from: parsed, matchingToolID: "calendar.create")
        if case .failure(let error) = result {
            XCTAssertEqual(error, .malformedPendingActionID)
        } else {
            XCTFail("Expected malformed pending action id to be rejected")
        }
    }

    func testPendingApprovalCannotBeConsumedForDifferentTool() {
        let pending = ToolApprovalQueue.shared.enqueue(
            toolID: "mail.draft",
            toolName: "Draft Email",
            arguments: ["to": "sam@example.com", "body": "hello"]
        )

        let mismatched = ToolApprovalQueue.shared.consume(
            pending.pendingActionID,
            matchingToolID: "outlook.mail.send"
        )

        XCTAssertNil(mismatched)
        XCTAssertNil(ToolApprovalQueue.shared.resolve(pending.pendingActionID))
    }

    func testExpiredPendingApprovalCannotBeConsumed() {
        let pending = ToolApprovalQueue.shared.enqueue(
            toolID: "mail.draft",
            toolName: "Draft Email",
            arguments: ["to": "sam@example.com", "body": "hello"],
            createdAt: Date(timeIntervalSince1970: 100),
            expiresAt: Date(timeIntervalSince1970: 101)
        )

        let consumed = ToolApprovalQueue.shared.consume(
            pending.pendingActionID,
            matchingToolID: pending.toolID
        )

        XCTAssertNil(consumed)
        XCTAssertNil(ToolApprovalQueue.shared.resolve(pending.pendingActionID))
    }
}
