import Foundation
import Testing
@testable import Lumen

struct RuntimeAccelerationDiagnosticsTests {
    @Test func codableRoundTrip() throws {
        let input = RuntimeAccelerationDiagnostics.forCurrentRuntime(requestedBackend: "metal", requestedGpuLayers: 999, requestedKQVOffload: true)
        let encoder = JSONEncoder()
        let data = try encoder.encode(input)
        let output = try JSONDecoder().decode(RuntimeAccelerationDiagnostics.self, from: data)
        #expect(output.backendRequested == input.backendRequested)
        #expect(output.aneUsedByCurrentRuntime == false)
    }

    @Test func oldTraceDecodesWithoutAccelerationDiagnostics() throws {
        let json = """
        {"id":"00000000-0000-0000-0000-000000000001","createdAt":"2026-01-01T00:00:00Z","event":"modelTurn","slot":"mouth","stage":"s","intent":null,"promptPrefix":"p","rawOutputPrefix":"o","selectedToolID":null,"toolArguments":{},"allowedToolIDs":[],"requiresApproval":null,"approvalMode":null,"parseError":null,"emittedFinalInActionTurn":false}
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let trace = try decoder.decode(AgentBehaviorTrace.self, from: Data(json.utf8))
        #expect(trace.accelerationDiagnostics == nil)
    }

    @Test func ggufPathReportsAneNotUsed() {
        let diag = RuntimeAccelerationDiagnostics.forCurrentRuntime(requestedBackend: "metal", requestedGpuLayers: 999, requestedKQVOffload: true)
        #expect(diag.aneUsedByCurrentRuntime == false)
        #expect(diag.aneUtilizationPercent == nil)
    }

    @Test func requestedUnverifiedWhenActualUnavailable() {
        let diag = RuntimeAccelerationDiagnostics.forCurrentRuntime(requestedBackend: "metal", requestedGpuLayers: 999, requestedKQVOffload: true, actualBackend: nil)
        #expect(diag.verificationLevel == "requested_unverified")
    }

    @Test func e2eMatrixEncodesDiagnostics() throws {
        let matrix = E2EPerformanceMatrix(
            aneUtilizationPercent: nil,
            eventDensityCPUProxyPercent: nil,
            gpuUtilizationPercent: nil,
            peakRAMMB: 0,
            averageRAMMB: 0,
            sampleCount: 0,
            notes: [],
            accelerationDiagnostics: RuntimeAccelerationDiagnostics.forCurrentRuntime(requestedBackend: "cpu", requestedGpuLayers: 0, requestedKQVOffload: false)
        )
        let data = try JSONEncoder().encode(matrix)
        let decoded = try JSONDecoder().decode(E2EPerformanceMatrix.self, from: data)
        #expect(decoded.accelerationDiagnostics != nil)
    }
}
