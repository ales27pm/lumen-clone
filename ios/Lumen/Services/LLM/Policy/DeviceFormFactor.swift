import Foundation

enum DeviceFormFactor: String, Sendable, Codable, Equatable, CaseIterable {
    case iPhone
    case iPad
    case mac
    case simulator
    case unknown

    var isMobileAppleDevice: Bool {
        self == .iPhone || self == .iPad
    }

    var isSimulator: Bool {
        self == .simulator
    }
}
