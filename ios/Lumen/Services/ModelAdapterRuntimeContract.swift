import Foundation

nonisolated struct LumenAdapterRoleContract: Sendable, Hashable, Identifiable {
    let roleID: String
    let slot: LumenModelSlot?
    let adapterID: String
    let adapterRepoID: String
    let adapterFileName: String
    let adapterSourcePath: String?
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
    let embeddingRepoID: String?
    let embeddingFileName: String?
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

    func matchesSharedBase(repoID: String, fileName: String) -> Bool {
        repoID == sharedBaseRepoID && fileName == sharedBaseFileName
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
        embeddingRepoID: "nomic-ai/nomic-embed-text-v1.5-GGUF",
        embeddingFileName: "nomic-embed-text-v1.5.Q4_K_M.gguf",
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
        embeddingRepoID: "Qwen/Qwen3-Embedding-0.6B-GGUF",
        embeddingFileName: "Qwen3-Embedding-0.6B-Q8_0.gguf",
        adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf",
        adapterRoles: [
            LumenAdapterRoleContract(roleID: "cortex", slot: .cortex, adapterID: "lumen-cortex-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-cortex-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-cortex-lora.gguf"), adapterArtifactPath: "runs/20260706T011546Z/adapters/cortex", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Cortex, Lumen's routing and planning agent. Select manifest-approved tools, persist required action steps, and delegate execution to Executor.", trainRecordCount: 9573, validationRecordCount: 1689, expectsStructuredOutput: true),
            LumenAdapterRoleContract(roleID: "executor", slot: .executor, adapterID: "lumen-executor-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-executor-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-executor-lora.gguf"), adapterArtifactPath: "runs/20260706T011546Z/adapters/executor", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Executor, Lumen's tool-call agent. Produce strict manifest-valid tool JSON only. Never invent tools or arguments.", trainRecordCount: 591, validationRecordCount: 104, expectsStructuredOutput: true),
            LumenAdapterRoleContract(roleID: "mouth", slot: .mouth, adapterID: "lumen-mouth-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-mouth-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-mouth-lora.gguf"), adapterArtifactPath: "runs/20260706T011546Z/adapters/mouth", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Mouth, Lumen's user-facing response agent. Explain tool results clearly without leaking internal JSON or sentinels.", trainRecordCount: 258, validationRecordCount: 45, expectsStructuredOutput: false),
            LumenAdapterRoleContract(roleID: "mimicry", slot: .mimicry, adapterID: "lumen-mimicry-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-mimicry-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-mimicry-lora.gguf"), adapterArtifactPath: "runs/20260706T011546Z/adapters/mimicry", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are Mimicry, Lumen's style adaptation agent. Adapt tone within safety and privacy boundaries.", trainRecordCount: 88, validationRecordCount: 16, expectsStructuredOutput: false),
            LumenAdapterRoleContract(roleID: "rem", slot: .rem, adapterID: "lumen-rem-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-rem-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-rem-lora.gguf"), adapterArtifactPath: "runs/20260706T011546Z/adapters/rem", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are REM, Lumen's reflection and repair agent. Diagnose failures, repair datasets, enforce memory policy, and produce regression samples.", trainRecordCount: 4138, validationRecordCount: 730, expectsStructuredOutput: false),
            LumenAdapterRoleContract(roleID: "fleet", slot: nil, adapterID: "lumen-fleet-adapter", adapterRepoID: "ales27pm/lumen-qwen3-bootstrap-adapters-gguf", adapterFileName: "lumen-fleet-lora.gguf", adapterSourcePath: qwen3AdapterSourcePath("lumen-fleet-lora.gguf"), adapterArtifactPath: "runs/20260706T011546Z/adapters/fleet", baseModelID: "Qwen/Qwen3-1.7B", systemPrompt: "You are part of the Lumen model fleet. Know every slot, delegation rule, memory scope, and boundary.", trainRecordCount: 4770, validationRecordCount: 842, expectsStructuredOutput: false),
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
