#if COREAI_PACKAGE_IO_MEMORY_TESTS
import Foundation
import Testing
@testable import Lumen

/// The standalone runner extracts the exact production entry-point body. Runtime
/// actors and legacy pending metadata are doubles; the cancellation bus is real.
@MainActor
@Suite(.serialized)
struct NativeModelLoadCancellationTests {
    @Test func cancelsEveryRegisteredModelLoadBeforeReturning() async {
        let firstGate = Gate()
        let secondGate = Gate()
        let first = Task { await firstGate.wait() }
        let second = Task { await secondGate.wait() }
        let bus = AppCancellationBus.shared
        let firstID = bus.registerCancellation({ first.cancel() }, category: .modelLoad)
        let secondID = bus.registerCancellation({ second.cancel() }, category: .modelLoad)

        ModelLoader.cancelActiveLoads()
        // No suspension here: queuing cancellation on another actor cannot
        // satisfy this contract while that actor is synchronously hashing.
        #expect(first.isCancelled)
        #expect(second.isCancelled)
        #expect(bus.activeRegistrationCount(category: .modelLoad) == 0)

        // Cleanup also works against the deliberately failing old method.
        first.cancel()
        second.cancel()
        bus.unregister(firstID, category: .modelLoad)
        bus.unregister(secondID, category: .modelLoad)
        await firstGate.release()
        await secondGate.release()
        await first.value
        await second.value
    }

    private actor Gate {
        private var released = false
        private var waiter: CheckedContinuation<Void, Never>?
        func wait() async {
            guard !released else { return }
            await withCheckedContinuation { waiter = $0 }
        }
        func release() {
            released = true
            waiter?.resume()
            waiter = nil
        }
    }
}
#endif
