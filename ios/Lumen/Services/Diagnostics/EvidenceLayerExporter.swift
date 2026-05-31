import Foundation

nonisolated struct EvidenceLayerExportPolicy: Codable, Sendable, Hashable {
    let format: String
    let sourceLayer: String
    let ownsLiveE2EScenarios: Bool
    let includesDeterministicStaticScenarios: Bool
    let privacy: String
    let notes: [String]
}

nonisolated struct EvidenceLayerEnvelope<Payload: Encodable>: Encodable {
    let schemaVersion: String
    let generatedAt: Date
    let app: InAppDatasetAppInfo
    let exportPolicy: EvidenceLayerExportPolicy
    let payload: Payload
}

nonisolated struct EvidenceLayerExportResult<Payload: Encodable> {
    let url: URL
    let envelope: EvidenceLayerEnvelope<Payload>
}

nonisolated enum EvidenceLayerExporter {
    static let schemaVersion = "1.0.0"
    private static let directoryName = "LumenEvidenceLayerExports"

    static func writeLayer<Payload: Encodable>(
        payload: Payload,
        filePrefix: String,
        format: String,
        sourceLayer: String,
        ownsLiveE2EScenarios: Bool,
        includesDeterministicStaticScenarios: Bool,
        privacy: String,
        notes: [String]
    ) throws -> EvidenceLayerExportResult<Payload> {
        let envelope = EvidenceLayerEnvelope(
            schemaVersion: schemaVersion,
            generatedAt: Date(),
            app: appInfo(),
            exportPolicy: EvidenceLayerExportPolicy(
                format: format,
                sourceLayer: sourceLayer,
                ownsLiveE2EScenarios: ownsLiveE2EScenarios,
                includesDeterministicStaticScenarios: includesDeterministicStaticScenarios,
                privacy: privacy,
                notes: notes
            ),
            payload: payload
        )
        let directory = try exportDirectory()
        let safePrefix = sanitizeFilePrefix(filePrefix)
        let url = directory.appendingPathComponent("\(safePrefix)-\(safeTimestamp(envelope.generatedAt))-\(UUID().uuidString.lowercased()).json", isDirectory: false)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(envelope).write(to: url, options: [.atomic])
        return EvidenceLayerExportResult(url: url, envelope: envelope)
    }

    static func exportDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory
        let directory = base
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func appInfo() -> InAppDatasetAppInfo {
        InAppDatasetAppInfo(
            name: Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String ?? "Lumen",
            bundleIdentifier: Bundle.main.bundleIdentifier,
            shortVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
            buildNumber: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
        )
    }

    private static func sanitizeFilePrefix(_ value: String) -> String {
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        let scalars = value.unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" }
        let collapsed = String(scalars)
            .replacingOccurrences(of: "--+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
        return collapsed.isEmpty ? "lumen-evidence-layer" : collapsed.lowercased()
    }

    private static func safeTimestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "-")
    }
}
