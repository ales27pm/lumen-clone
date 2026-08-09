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
        #expect(notices.contains("Qwen3 1.7B shared GGUF"))
        #expect(notices.contains("Lumen Qwen3 role adapters"))
        #expect(notices.contains("a7f6720f68f4a4567ebf7e3257041dd0b72077b518efe56890aec3516b59b9de"))
        #expect(notices.contains("883151da3764fbbfc929e8d58eb11129e66c4d54aa9f13dafb01e1505ad19c12"))
        #expect(notices.contains("17330b63f6584362ae22ad0e708c390bc6af7114c847246622cb782d4a8f026d"))
        #expect(notices.contains("The later public-corpus pipeline was not used"))
        #expect(!notices.contains("Training-source attribution for the Lumen role adapters"))
        #expect(!notices.contains("5b7b09754f85505f0406757b3d34725f50b331d3"))
        #expect(!notices.contains("7a411137cde14a39db03019a34e89dc51ff40cd4"))
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
