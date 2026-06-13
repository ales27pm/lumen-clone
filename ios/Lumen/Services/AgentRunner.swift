import Foundation
import SwiftData

@MainActor
enum AgentRunner {
    /// Compatibility wrapper kept for older callers during the Agent Kernel
    /// migration. New entrypoints should call `AssistantKernel.run(...)` or a
    /// narrow kernel bridge directly.
    static func runHeadless(prompt: String, appState: AppState, context: ModelContext, maxSteps: Int? = nil) async -> (text: String, steps: [AgentStep]) {
        await HeadlessAgentKernelRunner.run(
            prompt: prompt,
            appState: appState,
            context: context,
            maxSteps: maxSteps,
            source: .appIntent
        )
    }

    /// Background-safe compatibility wrapper. It no longer calls the legacy
    /// role-pipeline service; the turn flows through the Agent Kernel.
    static func runHeadless(prompt: String, settings: SettingsSnapshot, context: ModelContext, maxSteps: Int? = nil) async -> (text: String, steps: [AgentStep]) {
        await HeadlessAgentKernelRunner.run(
            prompt: prompt,
            settings: settings,
            context: context,
            maxSteps: maxSteps,
            source: .trigger
        )
    }
}
