import SwiftUI

extension E2ETestResult {
    var statusIcon: String {
        if passed { return "checkmark.circle.fill" }
        if isRuntimePreflightNonActionable { return "thermometer.medium" }
        return "xmark.octagon.fill"
    }

    var statusColor: Color {
        if passed { return .green }
        if isRuntimePreflightNonActionable { return .orange }
        return .red
    }
}
