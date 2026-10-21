import Foundation

struct RuntimeMetric: Codable, Sendable, Equatable {
    let timestamp: Date
    let runtimeName: String
    let taskKind: String
    let modelIDHash: String?
    let policySummary: String
    let latencyMs: Int?
    let success: Bool
    let errorCode: String?
    let thermalState: DeviceThermalState
    let lowPowerMode: Bool
    let memoryWarningCount: Int
}


enum RuntimeMetricErrorSanitizer {
    static func code(for error: Error) -> String {
        switch error {
        case let error as CoreAITextEmbeddingError:
            switch error {
            case .unsupported: return "coreai_embedding_unsupported"
            case .busy: return "coreai_embedding_busy"
            case .notPrepared: return "coreai_embedding_not_prepared"
            case .notSelected: return "coreai_embedding_not_selected"
            case .invalidRequest: return "coreai_embedding_invalid_request"
            case .inputTooLong: return "coreai_embedding_input_too_long"
            case .invalidDimensions: return "coreai_embedding_invalid_dimensions"
            case .resourceDenied: return "coreai_embedding_resource_denied"
            case .modelNotConfigured: return "coreai_embedding_model_not_configured"
            case .integrityFailure: return "coreai_embedding_integrity_failure"
            case .invalidModel: return "coreai_embedding_invalid_model"
            case .loadFailed: return "coreai_embedding_load_failed"
            case .inferenceFailed: return "coreai_embedding_inference_failed"
            case .invalidOutput: return "coreai_embedding_invalid_output"
            }
        case let error as CoreAIGemmaError:
            switch error {
            case .unsupported: return "coreai_unsupported"
            case .busy: return "coreai_busy"
            case .invalidModel: return "coreai_invalid_model"
            case .modelNotConfigured: return "coreai_model_not_configured"
            case .integrityFailure: return "coreai_integrity_failure"
            case .invalidMessages: return "coreai_invalid_messages"
            case .invalidTemperature: return "coreai_invalid_temperature"
            case .invalidTokenBudget: return "coreai_invalid_token_budget"
            case .contextFull: return "coreai_context_full"
            case .resourceDenied: return "coreai_resource_denied"
            case .loadFailed: return "coreai_load_failed"
            case .inferenceFailed: return "coreai_inference_failed"
            case .emptyOutput: return "coreai_empty_output"
            }
        case let error as AssistantKernel.KernelError:
            switch error {
            case .unsupportedTaskForTextTurn: return "unsupported_task_for_text_turn"
            case .unsupportedRuntimeForTextTurn: return "unsupported_runtime_for_text_turn"
            case .unsupportedTaskForEmbedding: return "unsupported_task_for_embedding"
            case .unsupportedRuntimeForEmbedding: return "unsupported_runtime_for_embedding"
            }
        case let error as LocalRuntimeError:
            switch error {
            case .unavailable: return "runtime_unavailable"
            }
        case let error as CoreMLRuntimeError:
            switch error {
            case .unsupportedOnPlatform: return "coreml_unsupported"
            case .modelNotConfigured: return "coreml_model_not_configured"
            case .modelNotFound: return "coreml_model_not_found"
            case .incompatibleModel: return "coreml_incompatible_model"
            case .shapeMismatch: return "coreml_shape_mismatch"
            case .experimentalRuntimeDisabled: return "coreml_experimental_runtime_disabled"
            case .computeFailure: return "coreml_compute_failure"
            }
        case is CancellationError:
            return "cancelled"
        default:
            return String(describing: type(of: error))
        }
    }
}
