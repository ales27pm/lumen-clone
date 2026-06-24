import Foundation

struct FleetRuntimeCleanupResult: Sendable, Equatable {
    let unloadedSlots: [LumenModelSlot]

    var unloadedSlotSummary: String {
        guard !unloadedSlots.isEmpty else { return "none" }
        return unloadedSlots.map(\.rawValue).sorted().joined(separator: ",")
    }
}

@MainActor
enum FleetRuntimeCleanup {
    private static let optionalChatSlots: [LumenModelSlot] = [.rem, .mimicry]
    private static let nonCoreChatSlots: [LumenModelSlot] = [.rem, .mimicry, .executor, .mouth]

    @discardableResult
    static func unloadOptionalChatSlotsNow() async -> FleetRuntimeCleanupResult {
        await unloadChatSlots(optionalChatSlots)
    }

    static func unloadOptionalChatSlots() {
        Task {
            _ = await unloadOptionalChatSlotsNow()
        }
    }

    @discardableResult
    static func unloadNonCoreChatSlotsNow() async -> FleetRuntimeCleanupResult {
        await unloadChatSlots(nonCoreChatSlots)
    }

    static func unloadNonCoreChatSlots() {
        Task {
            _ = await unloadNonCoreChatSlotsNow()
        }
    }

    private static func unloadChatSlots(_ slots: [LumenModelSlot]) async -> FleetRuntimeCleanupResult {
        let loaded = await AppLlamaService.shared.loadedChatPathsBySlot
        var unloadedSlots: [LumenModelSlot] = []
        for slot in slots where loaded[slot] != nil {
            await AppLlamaService.shared.unloadChat(for: slot)
            unloadedSlots.append(slot)
        }
        return FleetRuntimeCleanupResult(unloadedSlots: unloadedSlots)
    }
}
