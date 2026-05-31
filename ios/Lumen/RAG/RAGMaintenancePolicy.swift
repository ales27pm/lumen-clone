import Foundation

enum RAGMaintenancePolicy {
    static func allowEmbeddings(isBackground: Bool, lowPower: Bool, thermal: DeviceThermalState) -> Bool {
        if thermal == .critical { return false }
        if isBackground && lowPower { return false }
        return true
    }
}
