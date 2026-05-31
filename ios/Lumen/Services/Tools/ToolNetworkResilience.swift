import Foundation
import OSLog

nonisolated enum ToolNetworkErrorClass: String, Sendable {
    case timeout
    case dns
    case client4xx
    case server5xx
    case rateLimit
    case transport
    case cancelled
    case parsing
    case circuitOpen
    case unknown
}

struct ToolRetryPolicy: Sendable {
    let maxAttempts: Int
    let baseDelay: TimeInterval
    let maxDelay: TimeInterval
    let jitterRatio: Double
}

struct ToolRequestMetrics: Sendable {
    let endpoint: String
    let latencyMs: Double
    let success: Bool
    let errorClass: ToolNetworkErrorClass?
    let retryCount: Int
    let statusCode: Int?
}

actor ToolCircuitBreaker {
    private var failures: [String: [Date]] = [:]
    private var blockedUntil: [String: Date] = [:]

    func allowRequest(endpoint: String, now: Date = .init()) -> Bool {
        guard let until = blockedUntil[endpoint] else { return true }
        if until <= now {
            blockedUntil[endpoint] = nil
            return true
        }
        return false
    }

    func record(endpoint: String, success: Bool, now: Date = .init(), threshold: Int = 3, window: TimeInterval = 120, cooldown: TimeInterval = 60) {
        if success {
            failures[endpoint] = []
            blockedUntil[endpoint] = nil
            return
        }
        var endpointFailures = failures[endpoint] ?? []
        endpointFailures.append(now)
        endpointFailures = endpointFailures.filter { now.timeIntervalSince($0) <= window }
        failures[endpoint] = endpointFailures
        if endpointFailures.count >= threshold {
            blockedUntil[endpoint] = now.addingTimeInterval(cooldown)
        }
    }
}

nonisolated enum ToolNetworkTelemetry {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "tool-network")

    static func emit(_ metrics: ToolRequestMetrics) {
        logger.log("endpoint=\(metrics.endpoint, privacy: .public) success=\(metrics.success) latency_ms=\(metrics.latencyMs, format: .fixed(precision: 1)) retries=\(metrics.retryCount) status=\(metrics.statusCode ?? -1) err=\(metrics.errorClass?.rawValue ?? "none", privacy: .public)")
        NotificationCenter.default.post(name: .toolNetworkMetrics, object: metrics)
    }
}

nonisolated extension Notification.Name {
    static let toolNetworkMetrics = Notification.Name("toolNetworkMetrics")
}

nonisolated enum ToolNetworkResilience {
    static let circuitBreaker = ToolCircuitBreaker()

    static func classify(error: Error?, response: HTTPURLResponse?) -> ToolNetworkErrorClass {
        if let status = response?.statusCode {
            if status == 429 { return .rateLimit }
            if (500..<600).contains(status) { return .server5xx }
            if (400..<500).contains(status) { return .client4xx }
        }
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut: return .timeout
            case .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed: return .dns
            case .cancelled: return .cancelled
            default: return .transport
            }
        }
        return error == nil ? .unknown : .transport
    }

    static func fallbackMessage(for errorClass: ToolNetworkErrorClass, context: String) -> String {
        switch errorClass {
        case .timeout: return "\(context) timed out. Please try again in a moment."
        case .dns: return "\(context) is currently unreachable due to a network lookup issue."
        case .rateLimit: return "\(context) is rate-limited right now. Please retry shortly."
        case .client4xx: return "\(context) request was rejected. Check inputs and try again."
        case .server5xx: return "\(context) is having server issues. Please try again later."
        case .circuitOpen: return "\(context) is temporarily unavailable after repeated failures. Try again in about a minute."
        case .parsing: return "\(context) returned an unexpected response format."
        case .cancelled: return "\(context) request was cancelled."
        case .transport, .unknown: return "\(context) is temporarily unavailable. Please retry."
        }
    }

    static func shouldRetry(errorClass: ToolNetworkErrorClass) -> Bool {
        errorClass == .timeout || errorClass == .dns || errorClass == .rateLimit || errorClass == .server5xx || errorClass == .transport
    }

    static func backoffDelay(attempt: Int, policy: ToolRetryPolicy) -> UInt64 {
        let exponential = min(policy.baseDelay * pow(2.0, Double(attempt - 1)), policy.maxDelay)
        let jitter = exponential * policy.jitterRatio * Double.random(in: 0...1)
        return UInt64((exponential + jitter) * 1_000_000_000)
    }
}
