import Foundation
import Testing
@testable import Lumen

struct ToolNetworkResilienceTests {
    @Test func classifiesRetryableAndNonRetryableErrors() {
        let timeout = URLError(.timedOut)
        #expect(ToolNetworkResilience.classify(error: timeout, response: nil) == .timeout)

        let notFound = HTTPURLResponse(url: URL(string: "https://example.com")!, statusCode: 404, httpVersion: nil, headerFields: nil)
        #expect(ToolNetworkResilience.classify(error: nil, response: notFound) == .client4xx)
        #expect(ToolNetworkResilience.shouldRetry(errorClass: .client4xx) == false)

        let serverError = HTTPURLResponse(url: URL(string: "https://example.com")!, statusCode: 503, httpVersion: nil, headerFields: nil)
        #expect(ToolNetworkResilience.classify(error: nil, response: serverError) == .server5xx)
        #expect(ToolNetworkResilience.shouldRetry(errorClass: .server5xx))

        let cancelled = URLError(.cancelled)
        #expect(ToolNetworkResilience.classify(error: cancelled, response: nil) == .cancelled)
        #expect(ToolNetworkResilience.shouldRetry(errorClass: .cancelled) == false)
    }

    @Test func circuitBreakerSuppressesAfterRepeatedFailures() async {
        let breaker = ToolCircuitBreaker()
        let endpoint = "test.endpoint"
        #expect(await breaker.allowRequest(endpoint: endpoint))

        await breaker.record(endpoint: endpoint, success: false)
        await breaker.record(endpoint: endpoint, success: false)
        await breaker.record(endpoint: endpoint, success: false)

        #expect(await breaker.allowRequest(endpoint: endpoint) == false)
    }
}
