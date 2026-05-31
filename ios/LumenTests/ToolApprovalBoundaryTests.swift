import Testing
@testable import Lumen

struct ToolApprovalBoundaryTests {
    private let expectedApprovalTools: Set<String> = [
        "calendar.create","reminders.create","messages.draft","mail.draft","outlook.draft.create","outlook.mail.send",
        "outlook.message.mark_read","outlook.message.mark_unread","outlook.message.move","outlook.message.archive",
        "outlook.message.delete","outlook.message.reply","outlook.message.reply_all","outlook.message.forward","phone.call",
        "camera.capture","trigger.create","trigger.cancel","alarm.request_authorization","alarm.schedule","alarm.countdown",
        "alarm.pause","alarm.resume","alarm.stop","alarm.snooze","alarm.cancel"
    ]

    @Test func requiresApprovalMatrixMatchesRegistry() {
        let actual = Set(ToolRegistry.all.filter(\.requiresApproval).map(\.id))
        #expect(actual == expectedApprovalTools)
    }


}
