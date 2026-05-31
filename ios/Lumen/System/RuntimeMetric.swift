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
        case let error as AssistantKernel.KernelError:
            switch error {
            case .unsupportedTaskForTextTurn: return "unsupported_task_for_text_turn"
            case .unsupportedRuntimeForTextTurn: return "unsupported_runtime_for_text_turn"
            }
        case let error as LocalRuntimeError:
            switch error {
            case .unavailable: return "runtime_unavailable"
            case .generationNotImplemented(let runtime): return "generation_not_implemented_\(runtime.rawValue)"
            }
        case let error as CoreMLRuntimeError:
            switch error {
            case .unsupportedOnPlatform: return "coreml_unsupported"
            case .modelNotConfigured: return "coreml_model_not_configured"
            case .modelNotFound: return "coreml_model_not_found"
            case .incompatibleModel: return "coreml_incompatible_model"
            case .shapeMismatch: return "coreml_shape_mismatch"
            case .embeddingExtractionNotImplemented: return "coreml_embedding_extraction_not_implemented"
            case .computeFailure: return "coreml_compute_failure"
            }
        case is CancellationError:
            return "cancelled"
        default:
            return String(describing: type(of: error))
        }
    }
}
