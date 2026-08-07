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
            tags: [],
            sourceRevision: "91cad51170dc346986eccefdc2dd33a9da36ead9",
            expectedSHA256: "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
        )

        guard case .success(let url) = model.downloadURLResult else {
            Issue.record("Expected valid metadata to build URL")
            return
        }

        #expect(url.host == "huggingface.co")
        #expect(url.path == "/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        #expect(url.query?.contains("download=true") == true)
    }

    @Test func buildsQwen3AdapterDownloadURLAtPublishedRunPath() {
        let model = CatalogModel(
            id: "adapter",
            name: "Adapter",
            repoId: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf",
            fileName: "lumen-executor-lora.gguf",
            parameters: "LoRA",
            quantization: "GGUF",
            sizeBytes: 1,
            role: .roleAdapter,
            description: "",
            tags: [],
            sourceRevision: "f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2",
            expectedSHA256: "d3ba05ff22018a7468efa82154bd599899de7107b706b0b853758f869e6c969b",
            sourcePath: "runs/20260706T011546Z/lora_gguf/lumen-executor-lora.gguf"
        )

        guard case .success(let url) = model.downloadURLResult else {
            Issue.record("Expected valid adapter metadata to build URL")
            return
        }

        #expect(model.fileName == "lumen-executor-lora.gguf")
        #expect(model.sourcePath == "runs/20260706T011546Z/lora_gguf/lumen-executor-lora.gguf")
        #expect(url.path == "/ales27pm/lumen-qwen3-bootstrap-adapters-gguf/resolve/f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2/runs/20260706T011546Z/lora_gguf/lumen-executor-lora.gguf")
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

    @Test func failsClosedWithoutImmutableRevisionOrExpectedDigest() {
        let missingRevision = CatalogModel(
            id: "missing-revision",
            name: "Invalid",
            repoId: "owner/repo",
            fileName: "model.gguf",
            parameters: "1",
            quantization: "Q4",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: [],
            expectedSHA256: String(repeating: "a", count: 64)
        )
        let missingDigest = CatalogModel(
            id: "missing-digest",
            name: "Invalid",
            repoId: "owner/repo",
            fileName: "model.gguf",
            parameters: "1",
            quantization: "Q4",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: [],
            sourceRevision: String(repeating: "b", count: 40)
        )

        #expect(missingRevision.downloadURLResult == .failure(.missingSourceRevision))
        #expect(missingDigest.downloadURLResult == .failure(.missingExpectedSHA256))
    }

    @Test func rejectsUnsafeDestinationBasenamesEvenWhenSourcePathIsPresent() {
        let unsafeFileNames = [
            ".",
            "..",
            "../model.gguf",
            "models/model.gguf",
            "/model.gguf",
            "model.gguf/",
            "model\\escape.gguf",
            " model.gguf",
            "model.gguf ",
            "model%2Fescape.gguf",
        ]

        for fileName in unsafeFileNames {
            let model = pinnedModel(
                id: "unsafe-destination",
                fileName: fileName,
                sourcePath: "published/safe-model.gguf"
            )

            #expect(
                model.downloadURLResult == .failure(.invalidFileNameCharacters),
                "Expected unsafe destination basename to fail: \(fileName)"
            )
        }
    }

    @Test func rejectsEmptyDotTraversalAndBackslashSourcePathComponents() {
        let unsafeSourcePaths = [
            "",
            ".",
            "..",
            "../model.gguf",
            "runs/../model.gguf",
            "runs/./model.gguf",
            "runs//model.gguf",
            "/runs/model.gguf",
            "runs/model.gguf/",
            "runs\\model.gguf",
            "runs/%2e%2e/model.gguf",
            " runs/model.gguf",
            "runs/model.gguf ",
        ]

        for sourcePath in unsafeSourcePaths {
            let model = pinnedModel(
                id: "unsafe-source",
                fileName: "safe-model.gguf",
                sourcePath: sourcePath
            )

            #expect(
                model.downloadURLResult == .failure(.invalidSourcePathCharacters),
                "Expected unsafe source path to fail: \(sourcePath)"
            )
        }
    }

    @Test func rejectsMutableOrMalformedArtifactPins() {
        let invalidRevisions = [
            "main",
            "resolve/main",
            String(repeating: "a", count: 39),
            String(repeating: "g", count: 40),
        ]
        for sourceRevision in invalidRevisions {
            let model = pinnedModel(
                id: "invalid-revision",
                sourceRevision: sourceRevision
            )
            #expect(model.downloadURLResult == .failure(.invalidSourceRevision))
        }

        let invalidDigests = [
            String(repeating: "a", count: 63),
            String(repeating: "g", count: 64),
        ]
        for expectedSHA256 in invalidDigests {
            let model = pinnedModel(
                id: "invalid-digest",
                expectedSHA256: expectedSHA256
            )
            #expect(model.downloadURLResult == .failure(.invalidExpectedSHA256))
        }
    }

    @Test func everySelectableCatalogArtifactHasUniqueSafeDestinationAndImmutablePins() {
        let models = LumenModelFleetCatalog.selectableBootstrapModels
        let expectedByID: [String: (
            fileName: String,
            sourcePath: String?,
            revision: String,
            sha256: String,
            sizeBytes: Int64
        )] = [
            "fleet-bootstrap-qwen2.5-chat-base-q4": (
                "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                nil,
                "91cad51170dc346986eccefdc2dd33a9da36ead9",
                "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
                1_117_320_736
            ),
            "fleet-bootstrap-qwen2.5-embedding-nomic-q4": (
                "nomic-embed-text-v1.5.Q4_K_M.gguf",
                nil,
                "0188c9bf409793f810680a5a431e7b899c46104c",
                "d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac",
                84_106_624
            ),
            "fleet-bootstrap-qwen3-fast-shared-q4": (
                "lumen-qwen3-fast-shared-q4_k_m.gguf",
                nil,
                "8abae6d695408dbc75a134212dd616cd14549ae1",
                "a7f6720f68f4a4567ebf7e3257041dd0b72077b518efe56890aec3516b59b9de",
                1_282_439_264
            ),
            "fleet-bootstrap-qwen3-embedding-0.6b-q8": (
                "Qwen3-Embedding-0.6B-Q8_0.gguf",
                nil,
                "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
                "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
                639_150_592
            ),
            "fleet-bootstrap-qwen3-cortex-lora": (
                "lumen-cortex-lora.gguf",
                "runs/20260706T011546Z/lora_gguf/lumen-cortex-lora.gguf",
                "f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2",
                "fda964d662a4b0aee3bb73c0398fc780cef0e7dbd7294406b04bc4e3e25843ff",
                104_622_848
            ),
            "fleet-bootstrap-qwen3-executor-lora": (
                "lumen-executor-lora.gguf",
                "runs/20260706T011546Z/lora_gguf/lumen-executor-lora.gguf",
                "f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2",
                "d3ba05ff22018a7468efa82154bd599899de7107b706b0b853758f869e6c969b",
                104_622_848
            ),
            "fleet-bootstrap-qwen3-mouth-lora": (
                "lumen-mouth-lora.gguf",
                "runs/20260706T011546Z/lora_gguf/lumen-mouth-lora.gguf",
                "f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2",
                "552de3e894629f20ab26fafca2f883bb67c2934c1dc030b50658d3fbde209dd5",
                69_757_696
            ),
            "fleet-bootstrap-qwen3-mimicry-lora": (
                "lumen-mimicry-lora.gguf",
                "runs/20260706T011546Z/lora_gguf/lumen-mimicry-lora.gguf",
                "f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2",
                "1ec0799ec6767aa858fe7745623aea29854dcd31b97d1d2f9d5f76ab459061f5",
                69_757_696
            ),
            "fleet-bootstrap-qwen3-rem-lora": (
                "lumen-rem-lora.gguf",
                "runs/20260706T011546Z/lora_gguf/lumen-rem-lora.gguf",
                "f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2",
                "37431475814f072648b17bb668dec436279bb202c9ae40ab5946a3b4e648dc5d",
                104_622_848
            ),
            "fleet-bootstrap-qwen3-fleet-lora": (
                "lumen-fleet-lora.gguf",
                "runs/20260706T011546Z/lora_gguf/lumen-fleet-lora.gguf",
                "f8781f415a0ff87ea3f3a2119ab2ad96fae8fcf2",
                "0d759f87d33d1041b5487cdb7e754887d4eb5151acef1dcf08817502e67d7cb8",
                69_757_696
            ),
        ]
        let fileNames = models.map { $0.fileName.lowercased() }
        let modelIDs = models.map(\.id)
        let lowercaseHex = CharacterSet(charactersIn: "0123456789abcdef")

        #expect(!models.isEmpty)
        #expect(models.count == expectedByID.count)
        #expect(Set(fileNames).count == fileNames.count)
        #expect(Set(modelIDs).count == modelIDs.count)
        #expect(ModelCatalog.featured.count == models.count)

        for model in models {
            guard let expected = expectedByID[model.id] else {
                Issue.record("Unexpected selectable catalog model: \(model.id)")
                continue
            }
            #expect(model.fileName == expected.fileName)
            #expect(model.sourcePath == expected.sourcePath)
            #expect(model.sourceRevision == expected.revision)
            #expect(model.expectedSHA256 == expected.sha256)
            #expect(model.sizeBytes == expected.sizeBytes)
            #expect(!model.fileName.isEmpty)
            #expect(model.fileName != "." && model.fileName != "..")
            #expect(!model.fileName.contains("/") && !model.fileName.contains("\\"))
            #expect(model.sourceRevision.count == 40)
            #expect(model.sourceRevision.unicodeScalars.allSatisfy(lowercaseHex.contains))
            #expect(model.expectedSHA256.count == 64)
            #expect(model.expectedSHA256.unicodeScalars.allSatisfy(lowercaseHex.contains))

            if let sourcePath = model.sourcePath {
                let components = sourcePath.split(separator: "/", omittingEmptySubsequences: false)
                #expect(!components.isEmpty)
                #expect(
                    components.allSatisfy { component in
                        !component.isEmpty && component != "." && component != ".."
                    }
                )
                #expect(!sourcePath.contains("\\"))
            }

            guard case .success(let url) = model.downloadURLResult else {
                Issue.record("Selectable catalog model has invalid metadata: \(model.id)")
                continue
            }
            #expect(!url.path.lowercased().contains("/resolve/main/"))
            #expect(url.path.contains("/resolve/\(model.sourceRevision)/"))
        }
    }

    private func pinnedModel(
        id: String,
        fileName: String = "model.gguf",
        sourceRevision: String = String(repeating: "a", count: 40),
        expectedSHA256: String = String(repeating: "b", count: 64),
        sourcePath: String? = nil
    ) -> CatalogModel {
        CatalogModel(
            id: id,
            name: "Pinned model",
            repoId: "owner/repo",
            fileName: fileName,
            parameters: "1B",
            quantization: "Q4",
            sizeBytes: 1,
            role: .chat,
            description: "",
            tags: [],
            sourceRevision: sourceRevision,
            expectedSHA256: expectedSHA256,
            sourcePath: sourcePath
        )
    }
}
