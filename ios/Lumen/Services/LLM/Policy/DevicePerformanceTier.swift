import Foundation

enum DevicePerformanceTier: String, Sendable, Codable, Equatable, CaseIterable {
    case constrained
    case balanced
    case high
    case extreme
    case simulator
    case unknown

    var defaultMaximumModelParametersBillion: Double {
        switch self {
        case .constrained:
            return 1.5
        case .balanced:
            return 3.0
        case .high:
            return 4.0
        case .extreme:
            return 8.0
        case .simulator, .unknown:
            return 1.5
        }
    }
}
