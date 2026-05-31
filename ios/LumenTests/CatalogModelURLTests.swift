import Foundation
import Testing
@testable import Lumen

struct CatalogModelURLTests {
    @Test func buildsDownloadURLForValidMetadata() {
        let model = CatalogModel(
            id: "valid",
            name: "Valid",
            repoId: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            fileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: []
        )

        guard case .success(let url) = model.downloadURLResult else {
            Issue.record("Expected valid metadata to build URL")
            return
        }

        #expect(url.host == "huggingface.co")
        #expect(url.path == "/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        #expect(url.query?.contains("download=true") == true)
    }

    @Test func failsForMissingRepoPath() {
        let model = CatalogModel(
            id: "missing-repo",
            name: "Invalid",
            repoId: "   ",
            fileName: "model.gguf",
            parameters: "1",
            quantization: "Q4",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: []
        )

        #expect(model.downloadURLResult == .failure(.missingRepoPath))
    }

    @Test func failsForMissingFileName() {
        let model = CatalogModel(
            id: "missing-file",
            name: "Invalid",
            repoId: "owner/repo",
            fileName: "",
            parameters: "1",
            quantization: "Q4",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: []
        )

        #expect(model.downloadURLResult == .failure(.missingFileName))
    }

    @Test func failsForInvalidCharactersInMetadata() {
        let badRepo = CatalogModel(
            id: "bad-repo",
            name: "Invalid",
            repoId: "owner/repo<>",
            fileName: "ok.gguf",
            parameters: "1",
            quantization: "Q4",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: []
        )
        let badFile = CatalogModel(
            id: "bad-file",
            name: "Invalid",
            repoId: "owner/repo",
            fileName: "model?.gguf",
            parameters: "1",
            quantization: "Q4",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: []
        )

        #expect(badRepo.downloadURLResult == .failure(.invalidRepoPathCharacters))
        #expect(badFile.downloadURLResult == .failure(.invalidFileNameCharacters))
    }
}
