import Foundation

@MainActor
enum FleetRuntimeCleanup {
    static func unloadOptionalChatSlots() {
        Task {
            let loaded = await AppLlamaService.shared.loadedChatPathsBySlot
            for slot in [LumenModelSlot.rem, .mimicry] where loaded[slot] != nil {
                await AppLlamaService.shared.unloadChat(for: slot)
            }
        }
    }

    static func unloadNonCoreChatSlots() {
        Task {
            let loaded = await AppLlamaService.shared.loadedChatPathsBySlot
            for slot in [LumenModelSlot.rem, .mimicry, .executor, .mouth] where loaded[slot] != nil {
                await AppLlamaService.shared.unloadChat(for: slot)
            }
        }
    }
}
