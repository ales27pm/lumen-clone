import Foundation

enum MemoryExtractionTrigger { case assistantTurn, userCorrection, explicitRemember, backgroundConsolidation }

enum MemoryExtractionPolicy {
    static func shouldExtract(trigger: MemoryExtractionTrigger, lowPower: Bool, isBackground: Bool, containsSensitiveToolOutput: Bool) -> Bool {
        if containsSensitiveToolOutput { return false }
        if trigger == .explicitRemember || trigger == .userCorrection { return true }
        if isBackground && lowPower { return false }
        return true
    }
}
