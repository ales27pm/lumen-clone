import Foundation
import CryptoKit
import Darwin
import Observation
import OSLog

nonisolated struct ModelDownloadIdentity: Codable, Hashable, Sendable {
    let repoID: String
    let sourcePath: String
    let sourceRevision: String
    let expectedSHA256: String

    init(model: CatalogModel) {
        repoID = model.repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        sourcePath = (model.sourcePath ?? model.fileName).trimmingCharacters(in: .whitespacesAndNewlines)
        sourceRevision = model.sourceRevision.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        expectedSHA256 = model.expectedSHA256.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    var persistenceKey: String {
        let canonical = [repoID, sourcePath, sourceRevision, expectedSHA256].joined(separator: "\u{0}")
        return SHA256.hash(data: Data(canonical.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

nonisolated struct ModelDownloadResumeEnvelope: Codable, Equatable, Sendable {
    static let currentVersion = 1

    let version: Int
    let identity: ModelDownloadIdentity
    let resumeData: Data

    init(identity: ModelDownloadIdentity, resumeData: Data) {
        version = Self.currentVersion
        self.identity = identity
        self.resumeData = resumeData
    }

    static func encoded(identity: ModelDownloadIdentity, resumeData: Data) throws -> Data {
        try PropertyListEncoder().encode(Self(identity: identity, resumeData: resumeData))
    }

    static func resumeData(from encoded: Data, matching expectedIdentity: ModelDownloadIdentity) -> Data? {
        guard let envelope = try? PropertyListDecoder().decode(Self.self, from: encoded),
              envelope.version == currentVersion,
              envelope.identity == expectedIdentity,
              !envelope.resumeData.isEmpty
        else {
            return nil
        }
        return envelope.resumeData
    }
}

nonisolated enum ModelDownloadTerminationIntent: Equatable, Sendable {
    case pause
    case cancel
}

nonisolated enum ModelDownloadErrorDisposition: Equatable, Sendable {
    case explicitPause
    case explicitCancel
    case retryWithoutResumeData
    case retriableTransport
    case terminal
}

nonisolated enum ModelDownloadFailurePolicy {
    // Foundation does not expose a Swift constant for NSURLSession's rejected-resume-data code.
    static let cannotResumeErrorCode = -1019

    static func disposition(
        for error: NSError,
        startedFromResumeData: Bool,
        explicitIntent: ModelDownloadTerminationIntent?
    ) -> ModelDownloadErrorDisposition {
        switch explicitIntent {
        case .pause:
            return .explicitPause
        case .cancel:
            return .explicitCancel
        case nil:
            break
        }

        guard error.domain == NSURLErrorDomain else { return .terminal }
        if error.code == cannotResumeErrorCode {
            return startedFromResumeData ? .retryWithoutResumeData : .terminal
        }

        let retriableCodes: Set<Int> = [
            NSURLErrorTimedOut,
            NSURLErrorCannotFindHost,
            NSURLErrorCannotConnectToHost,
            NSURLErrorNetworkConnectionLost,
            NSURLErrorDNSLookupFailed,
            NSURLErrorNotConnectedToInternet,
            NSURLErrorInternationalRoamingOff,
            NSURLErrorCallIsActive,
            NSURLErrorDataNotAllowed,
            NSURLErrorRequestBodyStreamExhausted,
            NSURLErrorBackgroundSessionWasDisconnected,
        ]
        return retriableCodes.contains(error.code) ? .retriableTransport : .terminal
    }
}

@MainActor
final class ModelDownloadCompletionFanout {
    typealias Handler = (URL) -> Void

    private var handlers: [ModelDownloadIdentity: [Handler]] = [:]

    func append(_ handler: @escaping Handler, for identity: ModelDownloadIdentity) {
        handlers[identity, default: []].append(handler)
    }

    func take(for identity: ModelDownloadIdentity) -> [Handler] {
        let pending = handlers.removeValue(forKey: identity) ?? []
        return pending
    }

    func discard(for identity: ModelDownloadIdentity) {
        handlers[identity] = nil
    }

    func hasHandlers(for identity: ModelDownloadIdentity) -> Bool {
        !(handlers[identity]?.isEmpty ?? true)
    }

    func count(for identity: ModelDownloadIdentity) -> Int {
        handlers[identity]?.count ?? 0
    }
}

@Observable
@MainActor
final class ModelDownloader: NSObject {
    static let shared = ModelDownloader()
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "model-downloader")

    var progresses: [String: DownloadProgress] = [:]

    @ObservationIgnored private var sessions: [ModelDownloadIdentity: URLSessionDownloadTask] = [:]
    @ObservationIgnored private var targets: [Int: DownloadTarget] = [:]
    @ObservationIgnored private var resumeData: [ModelDownloadIdentity: Data] = [:]
    @ObservationIgnored private var activeIdentityByModelID: [String: ModelDownloadIdentity] = [:]
    @ObservationIgnored private var subscriberModelIDs: [ModelDownloadIdentity: Set<String>] = [:]
    @ObservationIgnored private var startingOperations: [ModelDownloadIdentity: UUID] = [:]
    @ObservationIgnored private var startingDestinations: [ModelDownloadIdentity: URL] = [:]
    @ObservationIgnored private var terminationIntents: [Int: ModelDownloadTerminationIntent] = [:]
    @ObservationIgnored private var pendingPauseTaskID: [ModelDownloadIdentity: Int] = [:]
    @ObservationIgnored private let completionFanout = ModelDownloadCompletionFanout()

    private struct DownloadTarget {
        let model: CatalogModel
        let identity: ModelDownloadIdentity
        let destination: URL
        let downloadURL: URL
        let startedFromResumeData: Bool
    }

    @ObservationIgnored
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600
        config.waitsForConnectivity = true
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    static var modelsDirectory: URL { ModelStorage.modelsDirectoryURL() }
    private static func modelsDirectoryOrThrow() throws -> URL { try ModelStorage.modelsDirectoryURLOrThrow() }
    private static func resumeDirectoryOrThrow() throws -> URL { try ModelStorage.resumeDirectoryURLOrThrow() }

    func localURL(for model: CatalogModel) -> URL { Self.modelsDirectory.appendingPathComponent(model.fileName) }

    func isDownloaded(_ model: CatalogModel) async -> Bool {
        if case .success = await ModelFileIntegrity.validateDownloadedCatalogFileAsync(model, at: localURL(for: model)) {
            return true
        }
        return false
    }

    func isDownloading(_ model: CatalogModel) -> Bool {
        let identity = ModelDownloadIdentity(model: model)
        return sessions[identity] != nil || startingOperations[identity] != nil
    }

    /// Initiates a download for a catalog model, or handles it if already downloaded or downloading.
    ///
    /// If the model is already being downloaded, this returns without starting a new download. If the model is already installed locally, the completion handler is invoked immediately with the file URL. Otherwise, a download task is created and started, which will invoke the completion handler with the file URL when finished.
    ///
    /// - Parameters:
    ///   - model: The catalog model to download.
    ///   - onComplete: A closure invoked with the local file URL.
    /// - Returns: `success` if the download was initiated, is already in progress, or the file is already available; `failure` if the download URL could not be constructed.
    @discardableResult
    func start(_ model: CatalogModel, onComplete: @escaping (URL) -> Void) async -> Result<Void, CatalogModel.DownloadURLError> {
        let downloadURL: URL
        switch model.downloadURLResult {
        case .success(let url):
            downloadURL = url
        case .failure(let error):
            Self.logger.error("download_start_failed model_id=\(model.id, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "Could not start download for \(model.name): \(error.localizedDescription)"])
            return .failure(error)
        }

        let destination: URL
        do {
            destination = try Self.modelsDirectoryOrThrow().appendingPathComponent(model.fileName)
        } catch {
            Self.logger.error("download_start_failed model_id=\(model.id, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "Could not start download for \(model.name): persistent model directory unavailable."])
            return .failure(.persistentDirectoryUnavailable)
        }

        let identity = ModelDownloadIdentity(model: model)
        if let existingIdentity = activeIdentityByModelID[model.id],
           existingIdentity != identity,
           sessions[existingIdentity] != nil
                || startingOperations[existingIdentity] != nil
                || completionFanout.hasHandlers(for: existingIdentity)
                || resumeData[existingIdentity] != nil {
            Self.logger.error("download_start_failed model_id=\(model.id, privacy: .public) error_code=artifact_identity_conflict")
            NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "Could not start download for \(model.name): another immutable artifact is already using this catalog identifier."])
            return .failure(.invalidURLComponents)
        }

        if let activeTask = sessions[identity] {
            guard targets[activeTask.taskIdentifier]?.destination.standardizedFileURL == destination.standardizedFileURL else {
                Self.logger.error("download_start_failed model_id=\(model.id, privacy: .public) error_code=artifact_destination_conflict")
                return .failure(.invalidURLComponents)
            }
            registerSubscriber(model, identity: identity, onComplete: onComplete)
            NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "\(model.name) is already downloading; this request will be notified when it completes."])
            return .success(())
        }

        if startingOperations[identity] != nil {
            guard startingDestinations[identity]?.standardizedFileURL == destination.standardizedFileURL else {
                Self.logger.error("download_start_failed model_id=\(model.id, privacy: .public) error_code=artifact_destination_conflict")
                return .failure(.invalidURLComponents)
            }
            registerSubscriber(model, identity: identity, onComplete: onComplete)
            return .success(())
        }

        let startToken = UUID()
        startingOperations[identity] = startToken
        startingDestinations[identity] = destination
        registerSubscriber(model, identity: identity, onComplete: onComplete)
        defer {
            if startingOperations[identity] == startToken {
                startingOperations[identity] = nil
                startingDestinations[identity] = nil
            }
        }

        let existingValidation = await ModelFileIntegrity.validateDownloadedCatalogFileAsync(model, at: destination)
        guard startingOperations[identity] == startToken else { return .success(()) }
        if case .success(let actualSize) = existingValidation {
            completeDownload(identity: identity, model: model, destination: destination, actualSize: actualSize)
            return .success(())
        }

        let data = resumeData[identity] ?? loadPersistedResume(for: identity)
        beginDownload(model: model, identity: identity, destination: destination, downloadURL: downloadURL, resumeData: data)
        return .success(())
    }

    func pause(_ model: CatalogModel) {
        let identity = activeIdentityByModelID[model.id] ?? ModelDownloadIdentity(model: model)
        guard let task = sessions[identity] else {
            guard startingOperations[identity] != nil else { return }
            startingOperations[identity] = nil
            startingDestinations[identity] = nil
            updateProgress(
                identity: identity,
                fallbackModel: model,
                fractionCompleted: nil,
                bytesReceived: nil,
                totalBytes: model.sizeBytes,
                state: .paused
            )
            return
        }
        let taskID = task.taskIdentifier
        terminationIntents[taskID] = .pause
        pendingPauseTaskID[identity] = taskID
        task.cancel { data in
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard self.pendingPauseTaskID[identity] == taskID,
                      self.terminationIntents[taskID] == .pause
                else { return }
                if let data, !data.isEmpty {
                    self.resumeData[identity] = data
                    self.persistResume(data, for: identity)
                }
                self.pendingPauseTaskID[identity] = nil
                self.finishPaused(taskID: taskID, model: model, identity: identity, clearIntent: true)
            }
        }
    }

    func resume(_ model: CatalogModel) {
        let identity = activeIdentityByModelID[model.id] ?? ModelDownloadIdentity(model: model)
        guard sessions[identity] == nil,
              startingOperations[identity] == nil,
              completionFanout.hasHandlers(for: identity)
        else { return }
        Task { @MainActor [weak self] in
            await self?.resumePendingDownload(model, identity: identity)
        }
    }

    func cancel(_ model: CatalogModel) {
        let identity = activeIdentityByModelID[model.id] ?? ModelDownloadIdentity(model: model)
        startingOperations[identity] = nil
        startingDestinations[identity] = nil
        if let pausedTaskID = pendingPauseTaskID.removeValue(forKey: identity) {
            terminationIntents[pausedTaskID] = nil
        }
        if let task = sessions[identity] {
            let taskID = task.taskIdentifier
            terminationIntents[taskID] = .cancel
            task.cancel()
            detachTask(taskID: taskID, identity: identity, clearIntent: true)
        }
        resumeData[identity] = nil
        clearPersistedResume(for: identity)
        completionFanout.discard(for: identity)
        clearSubscriberState(identity: identity, clearProgress: true)
    }

    func deleteLocal(_ model: CatalogModel) {
        let identity = ModelDownloadIdentity(model: model)
        try? FileManager.default.removeItem(at: localURL(for: model))
        resumeData[identity] = nil
        clearPersistedResume(for: identity)
        progresses[model.id] = nil
    }

    private func registerSubscriber(
        _ model: CatalogModel,
        identity: ModelDownloadIdentity,
        onComplete: @escaping (URL) -> Void
    ) {
        completionFanout.append(onComplete, for: identity)
        subscriberModelIDs[identity, default: []].insert(model.id)
        activeIdentityByModelID[model.id] = identity
        if progresses[model.id] == nil {
            progresses[model.id] = DownloadProgress(
                fractionCompleted: 0,
                bytesReceived: 0,
                totalBytes: model.sizeBytes,
                state: .queued
            )
        }
    }

    private func beginDownload(
        model: CatalogModel,
        identity: ModelDownloadIdentity,
        destination: URL,
        downloadURL: URL,
        resumeData persistedResumeData: Data?
    ) {
        let usableResumeData = persistedResumeData.flatMap { $0.isEmpty ? nil : $0 }
        let task: URLSessionDownloadTask
        if let usableResumeData {
            resumeData[identity] = usableResumeData
            task = session.downloadTask(withResumeData: usableResumeData)
        } else {
            task = session.downloadTask(with: downloadURL)
        }

        sessions[identity] = task
        targets[task.taskIdentifier] = DownloadTarget(
            model: model,
            identity: identity,
            destination: destination,
            downloadURL: downloadURL,
            startedFromResumeData: usableResumeData != nil
        )
        updateProgress(
            identity: identity,
            fallbackModel: model,
            fractionCompleted: usableResumeData == nil ? 0 : nil,
            bytesReceived: usableResumeData == nil ? 0 : nil,
            totalBytes: model.sizeBytes,
            state: .downloading
        )
        task.resume()
    }

    private func resumePendingDownload(_ model: CatalogModel, identity: ModelDownloadIdentity) async {
        guard sessions[identity] == nil, startingOperations[identity] == nil else { return }
        guard case .success(let downloadURL) = model.downloadURLResult else {
            failPendingDownload(model: model, identity: identity, message: "Download metadata is no longer valid.")
            return
        }
        let destination: URL
        do {
            destination = try Self.modelsDirectoryOrThrow().appendingPathComponent(model.fileName)
        } catch {
            failPendingDownload(model: model, identity: identity, message: "Persistent model directory unavailable.")
            return
        }

        let startToken = UUID()
        startingOperations[identity] = startToken
        startingDestinations[identity] = destination
        defer {
            if startingOperations[identity] == startToken {
                startingOperations[identity] = nil
                startingDestinations[identity] = nil
            }
        }
        let existingValidation = await ModelFileIntegrity.validateDownloadedCatalogFileAsync(model, at: destination)
        guard startingOperations[identity] == startToken else { return }
        if case .success(let size) = existingValidation {
            completeDownload(identity: identity, model: model, destination: destination, actualSize: size)
            return
        }
        let data = resumeData[identity] ?? loadPersistedResume(for: identity)
        beginDownload(model: model, identity: identity, destination: destination, downloadURL: downloadURL, resumeData: data)
    }

    private func persistResume(_ data: Data, for identity: ModelDownloadIdentity) {
        guard !data.isEmpty else { return }
        do {
            let resumeDirectory = try Self.resumeDirectoryOrThrow()
            let url = resumeDirectory.appendingPathComponent("\(identity.persistenceKey).resume.plist")
            let encoded = try ModelDownloadResumeEnvelope.encoded(identity: identity, resumeData: data)
            try encoded.write(to: url, options: .atomic)
        } catch {
            Self.logger.error("resume_persist_failed artifact_key=\(String(identity.persistenceKey.prefix(16)), privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
        }
    }

    private func loadPersistedResume(for identity: ModelDownloadIdentity) -> Data? {
        do {
            let resumeDirectory = try Self.resumeDirectoryOrThrow()
            let url = resumeDirectory.appendingPathComponent("\(identity.persistenceKey).resume.plist")
            let encoded = try Data(contentsOf: url)
            guard let data = ModelDownloadResumeEnvelope.resumeData(from: encoded, matching: identity) else {
                try? FileManager.default.removeItem(at: url)
                Self.logger.error("resume_rejected artifact_key=\(String(identity.persistenceKey.prefix(16)), privacy: .public) reason=identity_or_format_mismatch")
                return nil
            }
            resumeData[identity] = data
            return data
        } catch {
            return nil
        }
    }

    private func clearPersistedResume(for identity: ModelDownloadIdentity) {
        guard let resumeDirectory = try? Self.resumeDirectoryOrThrow() else { return }
        let url = resumeDirectory.appendingPathComponent("\(identity.persistenceKey).resume.plist")
        try? FileManager.default.removeItem(at: url)
    }

    private func updateProgress(
        identity: ModelDownloadIdentity,
        fallbackModel: CatalogModel,
        fractionCompleted: Double?,
        bytesReceived: Int64?,
        totalBytes: Int64,
        state: DownloadProgress.State
    ) {
        let modelIDs = subscriberModelIDs[identity] ?? [fallbackModel.id]
        for modelID in modelIDs {
            let existing = progresses[modelID]
            progresses[modelID] = DownloadProgress(
                fractionCompleted: fractionCompleted ?? existing?.fractionCompleted ?? 0,
                bytesReceived: bytesReceived ?? existing?.bytesReceived ?? 0,
                totalBytes: totalBytes,
                state: state
            )
        }
    }

    private func detachTask(taskID: Int, identity: ModelDownloadIdentity, clearIntent: Bool = true) {
        if sessions[identity]?.taskIdentifier == taskID {
            sessions[identity] = nil
        }
        targets[taskID] = nil
        if clearIntent {
            terminationIntents[taskID] = nil
        }
    }

    private func finishPaused(
        taskID: Int,
        model: CatalogModel,
        identity: ModelDownloadIdentity,
        clearIntent: Bool
    ) {
        detachTask(taskID: taskID, identity: identity, clearIntent: clearIntent)
        updateProgress(
            identity: identity,
            fallbackModel: model,
            fractionCompleted: nil,
            bytesReceived: nil,
            totalBytes: model.sizeBytes,
            state: .paused
        )
    }

    private func clearSubscriberState(identity: ModelDownloadIdentity, clearProgress: Bool) {
        let modelIDs = subscriberModelIDs.removeValue(forKey: identity) ?? []
        for modelID in modelIDs {
            if activeIdentityByModelID[modelID] == identity {
                activeIdentityByModelID[modelID] = nil
            }
            if clearProgress {
                progresses[modelID] = nil
            }
        }
    }

    private func completeDownload(
        taskID: Int? = nil,
        identity: ModelDownloadIdentity,
        model: CatalogModel,
        destination: URL,
        actualSize: Int64
    ) {
        if let taskID {
            detachTask(taskID: taskID, identity: identity)
        }
        resumeData[identity] = nil
        clearPersistedResume(for: identity)
        updateProgress(
            identity: identity,
            fallbackModel: model,
            fractionCompleted: 1,
            bytesReceived: actualSize,
            totalBytes: max(actualSize, model.sizeBytes),
            state: .completed
        )
        let handlers = completionFanout.take(for: identity)
        clearSubscriberState(identity: identity, clearProgress: false)
        for handler in handlers {
            handler(destination)
        }
    }

    private func failPendingDownload(model: CatalogModel, identity: ModelDownloadIdentity, message: String) {
        resumeData[identity] = nil
        clearPersistedResume(for: identity)
        updateProgress(
            identity: identity,
            fallbackModel: model,
            fractionCompleted: nil,
            bytesReceived: nil,
            totalBytes: model.sizeBytes,
            state: .failed(message)
        )
        completionFanout.discard(for: identity)
        clearSubscriberState(identity: identity, clearProgress: false)
        NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "Download failed for \(model.name): \(message)"])
    }

    /// Cleans up staging state without removing an already verified destination.
    private func failDownload(taskID: Int, target: DownloadTarget, message: String, cleanupURLs: [URL] = []) {
        for url in cleanupURLs { try? FileManager.default.removeItem(at: url) }
        resumeData[target.identity] = nil
        clearPersistedResume(for: target.identity)
        detachTask(taskID: taskID, identity: target.identity)
        updateProgress(
            identity: target.identity,
            fallbackModel: target.model,
            fractionCompleted: nil,
            bytesReceived: nil,
            totalBytes: target.model.sizeBytes,
            state: .failed(message)
        )
        completionFanout.discard(for: target.identity)
        clearSubscriberState(identity: target.identity, clearProgress: false)
        NotificationCenter.default.post(name: .modelDownloaderInfo, object: nil, userInfo: ["message": "Download failed for \(target.model.name): \(message)"])
    }

    nonisolated static func atomicallyInstallValidatedStagingFile(_ staging: URL, at destination: URL) throws {
        let result = staging.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                Darwin.rename(sourcePath, destinationPath)
            }
        }
        guard result == 0 else {
            let code = POSIXErrorCode(rawValue: errno) ?? .EIO
            throw POSIXError(code)
        }
    }
}

extension Notification.Name {
    static let modelDownloaderInfo = Notification.Name("modelDownloaderInfo")
}

extension ModelDownloader: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let taskId = downloadTask.taskIdentifier
        Task { @MainActor in
            guard let entry = targets[taskId] else { return }
            let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : entry.model.sizeBytes
            let fraction = total > 0 ? Double(totalBytesWritten) / Double(total) : 0
            updateProgress(
                identity: entry.identity,
                fallbackModel: entry.model,
                fractionCompleted: fraction,
                bytesReceived: totalBytesWritten,
                totalBytes: total,
                state: .downloading
            )
        }
    }

    /// Installs a downloaded file after validating the HTTP response and file integrity.
    ///
    /// Verifies that the HTTP status code is in the 2xx range and that the response is not HTML or JSON. Moves the temporary file to a same-volume staging location, validates its integrity, then atomically renames it over the destination. On any validation or file operation failure, failure handling removes only staging state and leaves an existing verified destination untouched.
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let taskId = downloadTask.taskIdentifier
        let statusCode = (downloadTask.response as? HTTPURLResponse)?.statusCode
        let mimeType = downloadTask.response?.mimeType
        let fm = FileManager.default
        let modelsDirResult = Result { try ModelStorage.modelsDirectoryURLOrThrow(fileManager: fm) }
        let movedURL: URL?
        switch modelsDirResult {
        case .success(let modelsDir):
            let staging = modelsDir.appendingPathComponent(".staging-\(UUID().uuidString)")
            try? fm.removeItem(at: staging)
            do {
                try fm.moveItem(at: location, to: staging)
                movedURL = staging
            } catch {
                movedURL = nil
            }
        case .failure:
            movedURL = nil
        }

        Task { @MainActor in
            let fileManager = FileManager.default
            guard let entry = targets[taskId] else {
                if let moved = movedURL { try? fileManager.removeItem(at: moved) }
                return
            }

            if case .failure(let error) = modelsDirResult {
                failDownload(taskID: taskId, target: entry, message: "Persistent model directory unavailable: \(error.localizedDescription)")
                return
            }

            guard let moved = movedURL else {
                failDownload(taskID: taskId, target: entry, message: "Could not move downloaded temporary file.")
                return
            }

            if let status = statusCode, !(200...299).contains(status) {
                failDownload(taskID: taskId, target: entry, message: "HTTP status \(status)", cleanupURLs: [moved])
                return
            }

            if let mime = mimeType?.lowercased(), mime.contains("text/html") || mime.contains("application/json") {
                failDownload(taskID: taskId, target: entry, message: "Unexpected response type \(mime)", cleanupURLs: [moved])
                return
            }

            switch await ModelFileIntegrity.validateDownloadedCatalogFileAsync(entry.model, at: moved) {
            case .success(let actualSize):
                guard targets[taskId]?.identity == entry.identity,
                      sessions[entry.identity]?.taskIdentifier == taskId,
                      terminationIntents[taskId] != .cancel
                else {
                    try? fileManager.removeItem(at: moved)
                    return
                }
                do {
                    try Self.atomicallyInstallValidatedStagingFile(moved, at: entry.destination)
                } catch {
                    failDownload(taskID: taskId, target: entry, message: "Could not atomically install downloaded file: \(error.localizedDescription)", cleanupURLs: [moved])
                    return
                }
                completeDownload(
                    taskID: taskId,
                    identity: entry.identity,
                    model: entry.model,
                    destination: entry.destination,
                    actualSize: actualSize
                )
            case .failure(let failure):
                failDownload(taskID: taskId, target: entry, message: failure.localizedDescription, cleanupURLs: [moved])
            }
        }
    }

    /// Handles errors from completed download tasks.
    ///
    /// Preserves resumable transport failures, distinguishes explicit pause/cancel intent, retries once without rejected resume data, and reserves terminal cleanup for non-retriable failures.
    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let taskId = task.taskIdentifier
        guard let error else { return }
        let nsError = error as NSError
        let resume = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        Task { @MainActor in
            guard let entry = targets[taskId] else {
                terminationIntents[taskId] = nil
                return
            }
            let disposition = ModelDownloadFailurePolicy.disposition(
                for: nsError,
                startedFromResumeData: entry.startedFromResumeData,
                explicitIntent: terminationIntents[taskId]
            )

            switch disposition {
            case .explicitPause:
                if let resume, !resume.isEmpty {
                    self.resumeData[entry.identity] = resume
                    persistResume(resume, for: entry.identity)
                }
                finishPaused(taskID: taskId, model: entry.model, identity: entry.identity, clearIntent: false)

            case .explicitCancel:
                detachTask(taskID: taskId, identity: entry.identity)
                self.resumeData[entry.identity] = nil
                clearPersistedResume(for: entry.identity)
                completionFanout.discard(for: entry.identity)
                clearSubscriberState(identity: entry.identity, clearProgress: true)

            case .retryWithoutResumeData:
                detachTask(taskID: taskId, identity: entry.identity)
                self.resumeData[entry.identity] = nil
                clearPersistedResume(for: entry.identity)
                updateProgress(
                    identity: entry.identity,
                    fallbackModel: entry.model,
                    fractionCompleted: 0,
                    bytesReceived: 0,
                    totalBytes: entry.model.sizeBytes,
                    state: .downloading
                )
                beginDownload(
                    model: entry.model,
                    identity: entry.identity,
                    destination: entry.destination,
                    downloadURL: entry.downloadURL,
                    resumeData: nil
                )

            case .retriableTransport:
                if let resume, !resume.isEmpty {
                    self.resumeData[entry.identity] = resume
                    persistResume(resume, for: entry.identity)
                }
                detachTask(taskID: taskId, identity: entry.identity)
                let hasUsableResumeData = resume?.isEmpty == false
                    || self.resumeData[entry.identity]?.isEmpty == false
                    || loadPersistedResume(for: entry.identity)?.isEmpty == false
                let state: DownloadProgress.State = hasUsableResumeData
                    ? .paused
                    : .failed("Network interruption; retry to continue.")
                updateProgress(
                    identity: entry.identity,
                    fallbackModel: entry.model,
                    fractionCompleted: nil,
                    bytesReceived: nil,
                    totalBytes: entry.model.sizeBytes,
                    state: state
                )
                NotificationCenter.default.post(
                    name: .modelDownloaderInfo,
                    object: nil,
                    userInfo: ["message": "Download interrupted for \(entry.model.name). Retry when connectivity returns."]
                )

            case .terminal:
                failDownload(taskID: taskId, target: entry, message: error.localizedDescription)
            }
        }
    }
}
