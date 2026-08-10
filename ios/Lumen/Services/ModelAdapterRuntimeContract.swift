import Foundation

nonisolated struct LumenAdapterRoleContract: Sendable, Hashable, Identifiable {
    let roleID: String
    let slot: LumenModelSlot?
    let adapterID: String
    let adapterRepoID: String
    let adapterFileName: String
    let adapterSourcePath: String?
    let adapterSourceRevision: String
    let adapterExpectedSHA256: String
    let adapterSizeBytes: Int64
    let adapterArtifactPath: String
    let baseModelID: String
    let systemPrompt: String
    let trainRecordCount: Int
    let validationRecordCount: Int
    let expectsStructuredOutput: Bool

    var id: String { roleID }
}

nonisolated struct LumenTrainedModelRuntimeContract: Sendable, Hashable {
    let schemaVersion: String
    let family: LumenModelFamily
    let mode: String
    let sharedBaseModelID: String
    let sharedBaseRepoID: String
    let sharedBaseFileName: String
    let sharedBaseSourceRevision: String
    let sharedBaseExpectedSHA256: String
    let sharedBaseSizeBytes: Int64
    let embeddingRepoID: String?
    let embeddingFileName: String?
    let embeddingSourceRevision: String?
    let embeddingExpectedSHA256: String?
    let embeddingSizeBytes: Int64?
    let adapterRepoID: String?
    let adapterRoles: [LumenAdapterRoleContract]
    let loadBaseModelOnce: Bool
    let selectAdapterByAgentSlot: Bool
    let mergeAdaptersByDefault: Bool
    let releaseBakeAllowedWhenRuntimeCannotLoadAdapters: Bool
    let releaseBakeEnabledByDefault: Bool
    let releaseBakeManualOnly: Bool
    let releaseBakeRequiresPassingEvalGates: Bool

    var runtimeSlots: [LumenModelSlot] {
        adapterRoles.compactMap(\.slot)
    }

    var adapterRoleIDs: [String] {
        adapterRoles.map(\.roleID)
    }

    var adapterFileNames: [String] {
        adapterRoles.map(\.adapterFileName)
    }

    var diagnosticSummary: String {
        if adapterRoles.isEmpty {
            return "\(family.rawValue) baseline"
        }
        return "\(mode) base=\(sharedBaseModelID) roles=\(adapterRoleIDs.joined(separator: ","))"
    }

    var traceValues: [String: String] {
        [
            "modelFamily": family.rawValue,
            "adapterRuntime": mode,
            "trainedBaseModelID": sharedBaseModelID,
            "trainedBaseArtifact": "\(sharedBaseRepoID)/\(sharedBaseFileName)",
            "agentRoles": adapterRoleIDs.joined(separator: ","),
            "adapterSlots": runtimeSlots.map(\.rawValue).joined(separator: ","),
            "adapterCount": String(adapterRoles.count),
            "loadBaseModelOnce": String(loadBaseModelOnce),
            "selectAdapterByAgentSlot": String(selectAdapterByAgentSlot),
            "mergeAdaptersByDefault": String(mergeAdaptersByDefault),
            "releaseBakeManualOnly": String(releaseBakeManualOnly),
            "releaseBakeRequiresPassingEvalGates": String(releaseBakeRequiresPassingEvalGates)
        ]
    }

    func adapterRole(roleID: String) -> LumenAdapterRoleContract? {
        adapterRoles.first { $0.roleID == roleID }
    }

    func adapterRole(for slot: LumenModelSlot) -> LumenAdapterRoleContract? {
        adapterRoles.first { $0.slot == slot }
    }

    func matchesSharedBase(
        repoID: String,
        fileName: String,
        sizeBytes: Int64,
        expectedSHA256: String?
    ) -> Bool {
        guard let expectedSHA256,
              CatalogModel.isValidSHA256(sharedBaseExpectedSHA256),
              CatalogModel.isValidSHA256(expectedSHA256)
        else { return false }
        return repoID == sharedBaseRepoID
            && fileName == sharedBaseFileName
            && sizeBytes == sharedBaseSizeBytes
            && expectedSHA256.caseInsensitiveCompare(sharedBaseExpectedSHA256) == .orderedSame
    }

    func matchesEmbedding(repoID: String, fileName: String) -> Bool {
        guard let embeddingRepoID, let embeddingFileName else { return false }
        return repoID == embeddingRepoID && fileName == embeddingFileName
    }
}

nonisolated enum LumenTrainedModelRuntimeRegistry {
    private static let qwen3ZeroGPURunID = "20260706T011546Z"
    private static func qwen3AdapterSourcePath(_ fileName: String) -> String {
        "runs/\(qwen3ZeroGPURunID)/lora_gguf/\(fileName)"
    }

    static var selected: LumenTrainedModelRuntimeContract {
        contract(for: LumenModelFamily.persistedSelected)
    }

    static func contract(for family: LumenModelFamily) -> LumenTrainedModelRuntimeContract {
        switch family {
        case .qwen25:
            return qwen25BaselineContract
        case .qwen3:
            return qwen3AdapterBootstrapContract
        }
    }

    private static let qwen25BaselineContract = LumenTrainedModelRuntimeContract(
        schemaVersion: "lumen.trained_model_runtime_contract/1.0.0",
        family: .qwen25,
        mode: "baseline-family",
        sharedBaseModelID: "Qwen/Qwen2.5-1.5B-Instruct",
        sharedBaseRepoID: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        sharedBaseFileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        sharedBaseSourceRevision: "91cad51170dc346986eccefdc2dd33a9da36ead9",
        sharedBaseExpectedSHA256: "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        sharedBaseSizeBytes: 1_117_320_736,
        embeddingRepoID: "nomic-ai/nomic-embed-text-v1.5-GGUF",
        embeddingFileName: "nomic-embed-text-v1.5.Q4_K_M.gguf",
        embeddingSourceRevision: "0188c9bf409793f810680a5a431e7b899c46104c",
        embeddingExpectedSHA256: "d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac",
        embeddingSizeBytes: 84_106_624,
        adapterRepoID: nil,
        adapterRoles: [],
        loadBaseModelOnce: false,
        selectAdapterByAgentSlot: false,
        mergeAdaptersByDefault: false,
        releaseBakeAllowedWhenRuntimeCannotLoadAdapters: false,
        releaseBakeEnabledByDefault: false,
        releaseBakeManualOnly: true,
        releaseBakeRequiresPassingEvalGates: true
    )

    static let qwen3AdapterBootstrapContract = LumenTrainedModelRuntimeContract(
        schemaVersion: "lumen.trained_model_runtime_contract/1.0.0",
        family: .qwen3,
        mode: "adapter-first",
        sharedBaseModelID: "Qwen/Qwen3-1.7B",
        sharedBaseRepoID: "ales27pm/lumen-qwen3-bootstrap-gguf",
        sharedBaseFileName: "lumen-qwen3-fast-shared-q4_k_m.gguf",
        sharedBaseSourceRevision: "903587d86a2e8b3f05bb0fab9d42338c0add0167",
        sharedBaseExpectedSHA256: "a7f6720f68f4a4567ebf7e3257041dd0b72077b518efe56890aec3516b59b9de",
        sharedBaseSizeBytes: 1_282_439_264,
        embeddingRepoID: "Qwen/Qwen3-Embedding-0.6B-GGUF",
        embeddingFileName: "Qwen3-Embedding-0.6B-Q8_0.gguf",
        embeddingSourceRevision: "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
        embeddingExpectedSHA256: "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
        embeddingSizeBytes: 639_150_592,
        adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf",
        adapterRoles: [
            LumenAdapterRoleContract(roleID: "cortex", slot: .cortex, adapterID: "lumen-cortex-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-cortex-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-cortex-lora.gguf"), adapterSourceRevision: "dda08641ab02d0a23f0cceff342846bb94d5fa02", adapterExpectedSHA256: "fda964d662a4b0aee3bb73c0398fc780cef0e7dbd7294406b04bc4e3e25843ff", adapterSizeBytes: 104_622_848, adapterArtifactPath: "runs/20260706T011546Z/adapters/cortex", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Cortex, Lumen's routing and planning agent. Select manifest-approved tools, persist required action steps, and delegate execution to Executor.", trainRecordCount: 9573, validationRecordCount: 1689, expectsStructuredOutput: true),
            LumenAdapterRoleContract(roleID: "executor", slot: .executor, adapterID: "lumen-executor-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-executor-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-executor-lora.gguf"), adapterSourceRevision: "dda08641ab02d0a23f0cceff342846bb94d5fa02", adapterExpectedSHA256: "d3ba05ff22018a7468efa82154bd599899de7107b706b0b853758f869e6c969b", adapterSizeBytes: 104_622_848, adapterArtifactPath: "runs/20260706T011546Z/adapters/executor", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Executor, Lumen's tool-call agent. Produce strict manifest-valid tool JSON only. Never invent tools or arguments.", trainRecordCount: 591, validationRecordCount: 104, expectsStructuredOutput: true),
            LumenAdapterRoleContract(roleID: "mouth", slot: .mouth, adapterID: "lumen-mouth-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-mouth-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-mouth-lora.gguf"), adapterSourceRevision: "dda08641ab02d0a23f0cceff342846bb94d5fa02", adapterExpectedSHA256: "552de3e894629f20ab26fafca2f883bb67c2934c1dc030b50658d3fbde209dd5", adapterSizeBytes: 69_757_696, adapterArtifactPath: "runs/20260706T011546Z/adapters/mouth", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Mouth, Lumen's user-facing response agent. Explain tool results clearly without leaking internal JSON or sentinels.", trainRecordCount: 258, validationRecordCount: 45, expectsStructuredOutput: false),
            LumenAdapterRoleContract(roleID: "mimicry", slot: .mimicry, adapterID: "lumen-mimicry-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-mimicry-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-mimicry-lora.gguf"), adapterSourceRevision: "dda08641ab02d0a23f0cceff342846bb94d5fa02", adapterExpectedSHA256: "1ec0799ec6767aa858fe7745623aea29854dcd31b97d1d2f9d5f76ab459061f5", adapterSizeBytes: 69_757_696, adapterArtifactPath: "runs/20260706T011546Z/adapters/mimicry", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Mimicry, Lumen's style adaptation agent. Adapt tone within safety and privacy boundaries.", trainRecordCount: 88, validationRecordCount: 16, expectsStructuredOutput: false),
            LumenAdapterRoleContract(roleID: "rem", slot: .rem, adapterID: "lumen-rem-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-rem-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-rem-lora.gguf"), adapterSourceRevision: "dda08641ab02d0a23f0cceff342846bb94d5fa02", adapterExpectedSHA256: "37431475814f072648b17bb668dec436279bb202c9ae40ab5946a3b4e648dc5d", adapterSizeBytes: 104_622_848, adapterArtifactPath: "runs/20260706T011546Z/adapters/rem", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are REM, Lumen's reflection and repair agent. Diagnose failures, repair datasets, enforce memory policy, and produce regression samples.", trainRecordCount: 4138, validationRecordCount: 730, expectsStructuredOutput: false),
            LumenAdapterRoleContract(roleID: "fleet", slot: nil, adapterID: "lumen-fleet-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-fleet-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-fleet-lora.gguf"), adapterSourceRevision: "dda08641ab02d0a23f0cceff342846bb94d5fa02", adapterExpectedSHA256: "0d759f87d33d1041b5487cdb7e754887d4eb5151acef1dcf08817502e67d7cb8", adapterSizeBytes: 69_757_696, adapterArtifactPath: "runs/20260706T011546Z/adapters/fleet", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are part of the Lumen model fleet. Know every slot, delegation rule, memory scope, and boundary.", trainRecordCount: 4770, validationRecordCount: 842, expectsStructuredOutput: false),
        ],
        loadBaseModelOnce: true,
        selectAdapterByAgentSlot: true,
        mergeAdaptersByDefault: false,
        releaseBakeAllowedWhenRuntimeCannotLoadAdapters: true,
        releaseBakeEnabledByDefault: false,
        releaseBakeManualOnly: true,
        releaseBakeRequiresPassingEvalGates: true
    )
}
