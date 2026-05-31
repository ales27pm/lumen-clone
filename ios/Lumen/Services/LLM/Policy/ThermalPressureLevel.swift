import Foundation

enum ThermalPressureLevel: String, Sendable, Codable, Equatable, CaseIterable {
    case nominal
    case fair
    case serious
    case critical
    case unknown

    var allowsDeepThink: Bool {
        self == .nominal || self == .fair
    }

    var allowsMaximumForeground: Bool {
        self == .nominal || self == .fair
    }
}
