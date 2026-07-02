import Foundation

nonisolated enum RoutingJSONLeakDetector {
    static func containsInternalRoutingJSON(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("\"intent\"")
            && lower.contains("\"nextmodel\"")
            && lower.contains("\"reasoningsummary\"")
    }
}
