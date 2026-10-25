import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// An explicit, local export; opening Settings never loads a model or reads the ledger.
@MainActor
struct SelfPredictionSettingsSection: View {
    @State private var document: SelfPredictionExportDocument?
    @State private var isPreparing = false
    @State private var isExporting = false
    @State private var status: String?
    @State private var recordingSummary: String?

    var body: some View {
        Section {
            Button {
                prepareExport()
            } label: {
                HStack {
                    Label("Export action predictions", systemImage: "square.and.arrow.up")
                    Spacer()
                    if isPreparing { ProgressView() }
                }
            }
            .disabled(isPreparing || isExporting)
            .accessibilityIdentifier("settings.selfPrediction.export")

            if let recordingSummary {
                Text(recordingSummary)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

            if let status {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .accessibilityIdentifier("settings.selfPrediction.status")
            }
        } header: {
            Text("Action predictions")
        } footer: {
            Text("Compare the model’s predictions with the results of tool calls. Records stay on this device and exclude conversation text, tool arguments and result text. Missing predictions remain marked as missing. Storage is limited: a full or unreadable log stops recording, while actions can still run.")
        }
        .fileExporter(
            isPresented: $isExporting,
            document: document,
            contentType: .json,
            defaultFilename: "lumen-action-predictions"
        ) { result in
            switch result {
            case .success:
                status = "Prediction records exported."
            case .failure:
                status = "The export could not be saved. Your stored records are unchanged."
            }
            document = nil
        }
    }

    private func prepareExport() {
        isPreparing = true
        status = nil
        recordingSummary = nil
        Task {
            defer { isPreparing = false }
            do {
                let snapshot = try await SelfPredictionLedger.shared.exportSnapshot()
                let summary = snapshot.summary
                let warning = summary.isAtCapacity
                    ? " Storage is full; new actions may not be recorded."
                    : ""
                let gaps = SecureToolRegistry.shared.lastPredictionRecordingFailure != nil
                    ? " Some actions could not be recorded in this session."
                    : ""
                recordingSummary = "\(summary.predictionCount) predictions, \(summary.pendingCount) without a recorded result.\(warning)\(gaps)"
                document = SelfPredictionExportDocument(data: snapshot.data)
                isExporting = true
            } catch {
                // Never display file-system paths or private underlying error descriptions.
                status = "Prediction records could not be read. No records were deleted or repaired."
            }
        }
    }
}

private struct SelfPredictionExportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    let data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        guard let contents = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        data = contents
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
