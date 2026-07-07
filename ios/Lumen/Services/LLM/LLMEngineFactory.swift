import Foundation

nonisolated enum LLMEngineFactory {
    static func makeDefaultRouter(includeUnavailableGGUF: Bool = false) async -> LLMEngineRouter {
        let router = LLMEngineRouter()
        await router.register(TinyIntentEngine(), for: .tinyIntent)

        #if DEBUG
        if includeUnavailableGGUF {
            await router.register(
                GGUFEngine(nativeBridge: UnavailableGGUFNativeBridge()),
                for: .gguf
            )
        }
        #endif

        return router
    }
}
