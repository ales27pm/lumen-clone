import Foundation

struct InstalledModelRecord: Sendable, Codable, Equatable, Identifiable {
    let id: String
    let catalogID: String?
    let model: LocalLLMModel
    let fileURL: URL?
    let relativePath: String?
    let sha256: String?
    let sizeBytes: Int64?
    let installedAt: Date
    let lastVerifiedAt: Date?
    let verificationStatus: ModelVerificationStatus

    var isUsable: Bool {
        switch model.backend {
        case .tinyIntent, .mock, .remote:
            return verificationStatus != .missingFile
                && verificationStatus != .hashMismatch
                && verificationStatus != .unreadable
                && verificationStatus != .unsupported
        case .gguf, .coreML:
            guard fileURL != nil else { return false }
            switch verificationStatus {
            case .verified:
                return true
            case .unverified:
                return model.expectedSHA256 == nil
            case .missingFile, .hashMismatch, .unreadable, .unsupported:
                return false
            }
        }
    }
}

enum ModelVerificationStatus: String, Sendable, Codable, Equatable, CaseIterable {
    case unverified
    case verified
    case missingFile
    case hashMismatch
    case unreadable
    case unsupported
}
