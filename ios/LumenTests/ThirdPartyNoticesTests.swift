import Foundation
import Testing
@testable import Lumen

struct ThirdPartyNoticesTests {
    @Test func bundledNoticesCoverEveryShippedRuntimeDependencyAndSelectableModelFamily() {
        let notices = ThirdPartyNoticesDocument.text

        #expect(notices.contains("SwiftLlama 1.2.0"))
        #expect(notices.contains("llama.cpp b6102"))
        #expect(notices.contains("Microsoft Authentication Library for Objective-C (MSAL) 1.9.0"))
        #expect(notices.contains("Microsoft IdentityCore"))
        #expect(notices.contains("Qwen3 1.7B shared base and Lumen role adapters"))
        #expect(notices.contains("Qwen3 Embedding 0.6B GGUF"))
        #expect(notices.contains("Qwen2.5 1.5B Instruct GGUF"))
        #expect(notices.contains("Nomic Embed Text v1.5 GGUF"))
        #expect(notices.contains("Apache License"))
        #expect(notices.contains("MIT License"))
    }

    @Test func signedAppResourcesDoNotRetainTheRemovedLGPLJavascriptRuntime() throws {
        #expect(Bundle.main.url(forResource: "p5.min", withExtension: "js") == nil)

        let viewer = try #require(
            Bundle.main.url(forResource: "latent_liturgy", withExtension: "html")
        )

        let viewerText = try String(contentsOf: viewer, encoding: .utf8)
        #expect(viewerText.contains(#"<script src="latent_liturgy.js"></script>"#))
        #expect(!viewerText.localizedCaseInsensitiveContains("p5"))
    }
}
