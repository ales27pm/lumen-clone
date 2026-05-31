import SwiftUI
import SwiftData

public struct AgentGroundingAuditView: View {
    private let auditor: RuntimeManifestAuditor
    private let scenarioRunner = RuntimeScenarioRunner()
    private let behaviorAuditor = AgentModelBehaviorAuditor()

    @Environment(\.modelContext) private var modelContext
    @Query(sort: \ChatMessage.createdAt, order: .forward) private var messages: [ChatMessage]

    @State private var report: RuntimeAgentManifestAuditReport?
    @State private var behaviorReport: AgentBehaviorAuditReport?
    @State private var scenarioResults: [RuntimeScenarioResult] = []
    @State private var errorMessage: String?
    @State private var manifestSource: String?
    @State private var usedRuntimeFallback = false
    @State private var lastExportURL: URL?
    @State private var lastExportPackage: LumenInAppDatasetPackage?
    @State private var lastLayerExportURL: URL?
    @State private var lastLayerExportLabel: String?
    @State private var isRunningLiveTraceSmokeTest = false
    @State private var lastLiveTraceSmokeSummary: String?

    public init(registryProvider: RuntimeToolRegistryProviding) {
        self.auditor = RuntimeManifestAuditor(registryProvider: registryProvider)
    }

    public var body: some View {
        List {
            Section {
                Button("Run Agent Grounding Audit") {
                    runAudit()
                }
                Button {
                    runLiveTraceSmokeTest()
                } label: {
                    if isRunningLiveTraceSmokeTest {
                        Label("Running Live Trace Smoke Test…", systemImage: "timer")
                    } else {
                        Label("Run Live Trace Smoke Test", systemImage: "waveform.path.ecg")
                    }
                }
                .disabled(isRunningLiveTraceSmokeTest)

                Button("Export Runtime Audit Package") {
                    exportRuntimeAuditPackage()
                }
                .disabled(report == nil && behaviorReport == nil && scenarioResults.isEmpty)

                if let lastLiveTraceSmokeSummary {
                    Text(lastLiveTraceSmokeSummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } footer: {
                Text("Compares the static crawler manifest against live runtime tools and recent model behaviour. Export writes an Agent Grounding runtime audit package for the offline loop: audit failures, behavior violations, and bounded diagnostic traces. Static scenario checks stay visible here but are not exported as live E2E model results. The live trace smoke test runs one real model pipeline turn and records it through AgentBehaviorTraceRecorder before export.")
            }

            Section {
                Button("Export Runtime Registry Audit") {
                    exportRuntimeRegistryAuditLayer()
                }
                .disabled(report == nil)

                Button("Export Model Behaviour Audit") {
                    exportBehaviorAuditLayer()
                }
                .disabled(behaviorReport == nil)

                Button("Export Static Scenario Checks") {
                    exportStaticScenarioLayer()
                }
                .disabled(scenarioResults.isEmpty)

                Button("Export Recent Runtime Traces") {
                    exportRecentTraceLayer()
                }

                if let lastLayerExportURL {
                    LabeledContent(lastLayerExportLabel ?? "Last layer export", value: lastLayerExportURL.lastPathComponent)
                        .font(.caption)
                    ShareLink(item: lastLayerExportURL) {
                        Label("Share Layer JSON", systemImage: "square.and.arrow.up")
                    }
                }
            } header: {
                Text("Export Individual Evidence Layers")
            } footer: {
                Text("Each export is a separate JSON envelope with an explicit sourceLayer and ownership policy. Use these for debugging one layer without polluting the live E2E dataset path.")
            }

            if let errorMessage {
                Section("Error") {
                    Text(errorMessage).foregroundStyle(.red)
                }
            }

            if let lastExportURL {
                Section("Runtime Audit Export") {
                    LabeledContent("File", value: lastExportURL.lastPathComponent)
                        .font(.caption)
                    if let lastExportPackage {
                        LabeledContent("Source layer", value: lastExportPackage.exportPolicy.sourceLayer)
                        LabeledContent("Owns live E2E", value: lastExportPackage.exportPolicy.ownsLiveE2EScenarios ? "yes" : "no")
                        LabeledContent("Static scenarios exported", value: lastExportPackage.exportPolicy.includesDeterministicStaticScenarios ? "yes" : "no")
                        LabeledContent("Traces", value: "\(lastExportPackage.recentTraces.count)")
                        LabeledContent("Allowed selections", value: "\(lastExportPackage.traceSelectedToolAllowedCount)")
                        LabeledContent("Trace parse errors", value: "\(lastExportPackage.traceParseErrorCount)")
                        LabeledContent("Runtime failures", value: "\(lastExportPackage.runtimeManifestAudit?.failures.count ?? 0)")
                        LabeledContent("Behavior violations", value: "\(lastExportPackage.behaviorAudit?.violations.count ?? 0)")
                        if lastExportPackage.recentTraces.isEmpty {
                            Label("No recent model/tool traces were exported. Run real model interactions before exporting or run the Live Trace Smoke Test above.", systemImage: "exclamationmark.triangle.fill")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }
                    ShareLink(item: lastExportURL) {
                        Label("Share Runtime Audit JSON", systemImage: "square.and.arrow.up")
                    }
                    Text("Feed this JSON into `python -m lumen_manifest_crawler improve-loop --runtime-audit <file>`. Live E2E scenario reports should come from End-to-end tests, not from Agent Grounding static scenario checks.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let report {
                Section("Runtime Registry") {
                    HStack {
                        Text(report.passed ? "Passed" : "Failed")
                        Spacer()
                        Text(scoreText(report.score))
                            .font(.headline)
                    }
                    Text(report.generatedAt.formatted())
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if let manifestSource {
                        LabeledContent("Manifest", value: manifestSource)
                            .font(.caption)
                    }
                    if usedRuntimeFallback {
                        Label("Runtime fallback was used. Add AgentBehaviorManifest.json to the app bundle or seed the Application Support manifest before trusting this as a source-of-truth audit.", systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }

                if !report.failures.isEmpty {
                    Section("Runtime Failures") {
                        ForEach(report.failures) { failure in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(failure.type).font(.headline)
                                Text(failure.problem).font(.subheadline)
                                if let actual = failure.actual {
                                    Text("Actual: \(actual)").font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                if !report.recommendedDatasetRepairs.isEmpty {
                    Section("Runtime Repair Hints") {
                        ForEach(report.recommendedDatasetRepairs, id: \.self) { repair in
                            Text(repair)
                        }
                    }
                }
            }

            if let behaviorReport {
                Section("Model Behaviour") {
                    HStack {
                        Text(behaviorReport.passed ? "Grounded" : "Drift detected")
                        Spacer()
                        Text(scoreText(behaviorReport.score))
                            .font(.headline)
                    }
                    LabeledContent("Audited traces", value: "\(behaviorReport.traceCount)")
                    LabeledContent("Violations", value: "\(behaviorReport.violationCount)")
                    if let sourceCommit = behaviorReport.sourceCommit, !sourceCommit.isEmpty {
                        LabeledContent("Static source", value: String(sourceCommit.prefix(10)))
                            .font(.caption)
                    }
                    Text(behaviorReport.generatedAt.formatted())
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if !behaviorReport.violations.isEmpty {
                    Section("Model Drift Violations") {
                        ForEach(behaviorReport.violations.prefix(30)) { violation in
                            VStack(alignment: .leading, spacing: 5) {
                                HStack {
                                    Label(violation.code, systemImage: severityIcon(violation.severity))
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(severityColor(violation.severity))
                                    Spacer()
                                    Text(violation.agent)
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.secondary)
                                }
                                Text(violation.problem)
                                    .font(.caption)
                                if !violation.promptPrefix.isEmpty {
                                    Text("Prompt: \(violation.promptPrefix)")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                                DisclosureGroup("Expected / Actual") {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Expected: \(violation.expected)")
                                        Text("Actual: \(violation.actual)")
                                    }
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                }
                                .font(.caption)
                            }
                            .padding(.vertical, 2)
                        }
                    }
                }

                if !behaviorReport.recommendations.isEmpty {
                    Section("Model Repair Recommendations") {
                        ForEach(behaviorReport.recommendations, id: \.self) { recommendation in
                            Text(recommendation)
                        }
                    }
                }
            }

            if !scenarioResults.isEmpty {
                Section {
                    ForEach(scenarioResults) { result in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(result.scenario.intent)
                                Text(result.scenario.expectedToolID)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Image(systemName: result.passed ? "checkmark.circle.fill" : "xmark.octagon.fill")
                                .foregroundStyle(result.passed ? .green : .red)
                        }
                    }
                } header: {
                    Text("Static Scenario Checks")
                } footer: {
                    Text("These are deterministic manifest sanity checks. They do not run the model and are not exported as live E2E scenario evidence by default.")
                }
            }
        }
        .navigationTitle("Agent Grounding")
    }

    private func runAudit() {
        let currentMessages = Array(messages.suffix(200))
        DispatchQueue.global(qos: .userInitiated).async {
            let loadResult = auditor.loadManifestFromStoreBundleOrRuntimeFallback()
            let auditReport = auditor.audit(manifest: loadResult.manifest)
            let scenarios = scenarioRunner.validateStaticScenarios(manifest: loadResult.manifest)
            DispatchQueue.main.async {
                let modelReport = behaviorAuditor.audit(manifest: loadResult.manifest, messages: currentMessages)
                report = auditReport
                behaviorReport = modelReport
                scenarioResults = scenarios
                manifestSource = loadResult.source
                usedRuntimeFallback = loadResult.usedRuntimeFallback
                errorMessage = nil
            }
        }
    }

    private func runLiveTraceSmokeTest() {
        isRunningLiveTraceSmokeTest = true
        lastLiveTraceSmokeSummary = "Starting live model trace smoke test…"
        errorMessage = nil
        Task { @MainActor in
            let before = AgentBehaviorTraceRecorder.recent(limit: 5_000).count
            let req = AgentRequest(
                systemPrompt: "You are Lumen. Answer concisely and do not expose hidden reasoning.",
                history: [],
                userMessage: "Live trace smoke test: explain in one sentence why a sharp chisel is safer than a dull one.",
                temperature: 0.1,
                topP: 0.7,
                repetitionPenalty: 1.05,
                maxTokens: 160,
                maxSteps: 1,
                availableTools: ToolRegistry.all,
                relevantMemories: []
            )

            var finalText = ""
            for await event in RolePipelineAgentService.shared.run(req) {
                switch event {
                case .finalDelta(let chunk):
                    finalText += chunk
                case .done(let text, _):
                    if !text.isEmpty { finalText = text }
                case .error(let message):
                    errorMessage = message
                case .step, .stepDelta:
                    break
                }
            }

            let after = AgentBehaviorTraceRecorder.recent(limit: 5_000).count
            let added = max(0, after - before)
            isRunningLiveTraceSmokeTest = false
            if added > 0 {
                lastLiveTraceSmokeSummary = "Live trace smoke test recorded \(added) trace(s). Export the runtime audit package again."
            } else if !finalText.isEmpty {
                lastLiveTraceSmokeSummary = "Smoke test generated output but no traces were recorded. Check that the latest build includes AppLlamaService trace recording."
            } else {
                lastLiveTraceSmokeSummary = "Smoke test completed without output and no traces were recorded. Confirm that a chat model is downloaded and assigned."
            }
        }
    }

    private func exportRuntimeAuditPackage() {
        do {
            let result = try InAppDatasetPackageExporter.writePackage(
                manifestSource: manifestSource ?? "unknown",
                usedRuntimeFallback: usedRuntimeFallback,
                runtimeManifestAudit: report,
                behaviorAudit: behaviorReport,
                scenarioResults: scenarioResults,
                traceLimit: 200
            )
            lastExportURL = result.url
            lastExportPackage = result.package
            errorMessage = nil
        } catch {
            errorMessage = "Runtime audit export failed: \(error.localizedDescription)"
        }
    }

    private func exportRuntimeRegistryAuditLayer() {
        guard let report else { return }
        exportLayer(
            payload: report,
            label: "Runtime registry audit",
            filePrefix: "lumen-runtime-registry-audit",
            format: "runtime-registry-audit-json",
            sourceLayer: "runtimeManifestAudit",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: false,
            notes: [
                "Compares AgentBehaviorManifest.json against the live runtime tool registry.",
                "Does not run model scenarios."
            ]
        )
    }

    private func exportBehaviorAuditLayer() {
        guard let behaviorReport else { return }
        exportLayer(
            payload: behaviorReport,
            label: "Model behaviour audit",
            filePrefix: "lumen-model-behaviour-audit",
            format: "agent-model-behaviour-audit-json",
            sourceLayer: "agentModelBehaviorAuditor",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: false,
            notes: [
                "Audits recent persisted app messages and model behaviour violations.",
                "Use this for drift and repair samples, not as an E2E scenario result."
            ]
        )
    }

    private func exportStaticScenarioLayer() {
        exportLayer(
            payload: scenarioResults,
            label: "Static scenario checks",
            filePrefix: "lumen-static-scenario-checks",
            format: "deterministic-static-scenario-checks-json",
            sourceLayer: "runtimeScenarioRunner.staticChecks",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: true,
            notes: [
                "Deterministic manifest sanity checks only.",
                "Does not run the model and must not be treated as live E2E evidence."
            ]
        )
    }

    private func exportRecentTraceLayer() {
        let traces = AgentBehaviorTraceRecorder.recent(limit: 500)
        exportLayer(
            payload: traces,
            label: "Recent runtime traces",
            filePrefix: "lumen-agent-runtime-traces",
            format: "agent-runtime-traces-json",
            sourceLayer: "agentBehaviorTraceRecorder",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: false,
            notes: [
                "Bounded recent traces captured by AgentBehaviorTraceRecorder.",
                "Empty exports indicate the recorder is not wired or no real model interactions were exercised."
            ]
        )
    }

    private func exportLayer<Payload: Encodable>(
        payload: Payload,
        label: String,
        filePrefix: String,
        format: String,
        sourceLayer: String,
        ownsLiveE2EScenarios: Bool,
        includesDeterministicStaticScenarios: Bool,
        notes: [String]
    ) {
        do {
            let result = try EvidenceLayerExporter.writeLayer(
                payload: payload,
                filePrefix: filePrefix,
                format: format,
                sourceLayer: sourceLayer,
                ownsLiveE2EScenarios: ownsLiveE2EScenarios,
                includesDeterministicStaticScenarios: includesDeterministicStaticScenarios,
                privacy: "Bounded diagnostic export generated from the local device. Review before sharing outside the improve-loop.",
                notes: notes
            )
            lastLayerExportURL = result.url
            lastLayerExportLabel = label
            errorMessage = nil
        } catch {
            errorMessage = "\(label) export failed: \(error.localizedDescription)"
        }
    }

    private func scoreText(_ score: Double) -> String {
        "\(Int((score * 100).rounded()))%"
    }

    private func severityIcon(_ severity: AgentBehaviorViolation.Severity) -> String {
        switch severity {
        case .warning: "exclamationmark.triangle"
        case .error: "xmark.octagon"
        case .critical: "flame"
        }
    }

    private func severityColor(_ severity: AgentBehaviorViolation.Severity) -> Color {
        switch severity {
        case .warning: .orange
        case .error: .red
        case .critical: .purple
        }
    }
}
