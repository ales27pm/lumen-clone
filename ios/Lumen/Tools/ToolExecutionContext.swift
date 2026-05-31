import Foundation
import SwiftData

struct ToolExecutionContext {
    let isForeground: Bool
    let appState: AppState?
    let modelContext: ModelContext?
    let permissionRegistry: PermissionRegistry
    let metricsStore: RuntimeMetricsStore
}
