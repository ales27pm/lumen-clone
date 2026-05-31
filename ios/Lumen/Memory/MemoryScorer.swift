import Foundation

enum MemorySaveDecision: String, Sendable { case save, ignore, askUser, rejectSensitive }
struct MemoryScoreResult: Sendable { let score: Double; let decision: MemorySaveDecision; let reasons: [String] }

enum MemoryScorer {
    static func score(candidate: MemoryCandidate, now: Date = Date()) -> MemoryScoreResult {
        if candidate.sensitivity == .credentialLike { return .init(score: -1, decision: .rejectSensitive, reasons: ["credentialLike"]) }
        var s = candidate.confidence
        var reasons: [String] = []
        switch candidate.userExplicitness {
        case .explicitPreference: s += 2; reasons.append("explicitPreference")
        case .correction: s += 1.8; reasons.append("correction")
        case .repeatedFact: s += 1.2; reasons.append("repeatedFact")
        case .projectFact: s += 1.0; reasons.append("projectFact")
        case .inferred: s += 0.2
        case .transient: s -= 1.2; reasons.append("transientPenalty")
        }
        if candidate.sensitivity == .healthOrLegal || candidate.sensitivity == .financial { return .init(score: s, decision: .askUser, reasons: reasons + ["sensitiveDomain"]) }
        let age = now.timeIntervalSince(candidate.createdAt)
        s += max(-0.6, 0.4 - age / (60*60*24*30*6))
        let d: MemorySaveDecision = s >= 1.1 ? .save : .ignore
        return .init(score: s, decision: d, reasons: reasons)
    }
}
